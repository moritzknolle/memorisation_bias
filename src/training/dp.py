"""DP-SGD configuration, including the non-private baseline (epsilon=inf).

So that the epsilon=inf baseline differs from the DP models only in the noise, it uses the same
jax_privacy training step (per-example gradients clipped at `clipping_norm`) with the noise
multiplier set to 0. jax_privacy rejects a zero noise multiplier, so `make_dp_config` builds this
config under `allow_zero_noise_multiplier()`, which skips only the `noise_multiplier > 0` check
(and the privacy accounting inside it).
"""

import contextlib
import math
from typing import Optional


def is_nonprivate(epsilon: Optional[float]) -> bool:
    """Whether `epsilon` denotes the non-private (noise-free) baseline."""
    return epsilon is not None and math.isinf(epsilon) and epsilon > 0


@contextlib.contextmanager
def allow_zero_noise_multiplier():
    """Lets `DPKerasConfig` accept `noise_multiplier=0.0` inside this block.

    Upstream rejects a non-positive noise multiplier, and computes the epsilon it implies
    to check it against the budget -- neither of which makes sense at sigma=0, where the
    budget is infinite by definition. The patch swaps the field out for None (the "not
    set" value, which upstream skips) for the duration of validation, so all the *other*
    checks -- batch size vs train size, positive step count, microbatch divisibility --
    still run against the real config.

    Restores the original validator on exit, including on error: the check must stay armed
    for the DP arms, where a zero noise multiplier would be a silent loss of the privacy
    guarantee.
    """
    from jax_privacy import keras_api  # type: ignore

    original = keras_api.DPKerasConfig._validate_params

    def validate_allowing_zero_noise(self) -> None:
        if self.noise_multiplier is not None and self.noise_multiplier == 0.0:
            # frozen dataclass: object.__setattr__ is the documented escape hatch, and the
            # value is put back before the caller ever sees the instance.
            object.__setattr__(self, "noise_multiplier", None)
            try:
                original(self)
            finally:
                object.__setattr__(self, "noise_multiplier", 0.0)
        else:
            original(self)

    keras_api.DPKerasConfig._validate_params = validate_allowing_zero_noise
    try:
        yield
    finally:
        keras_api.DPKerasConfig._validate_params = original


def make_dp_config(
    *,
    epsilon: float,
    delta: float,
    clipping_norm: float,
    batch_size: int,
    train_steps: int,
    train_size: int,
    seed: Optional[int] = None,
    microbatch_size: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
    rescale_to_unit_norm: bool = False,
):
    """Builds the `DPKerasConfig` for one privacy budget.

    Args:
        epsilon: privacy budget. `math.inf` selects the non-private baseline: same mechanism,
            noise multiplier set to 0 instead of calibrated.
        delta: delta of (eps, delta)-DP. Has no effect at eps=inf.
        clipping_norm: per-example L2 clipping norm. Also applied at eps=inf.
        batch_size: (physical) batch size handed to fit().
        train_steps: number of optimizer updates the run will perform.
        train_size: number of training examples.
        seed: seed for the noise PRNG. Irrelevant at eps=inf (stddev 0), passed anyway.
        microbatch_size: per-example gradient microbatch size, or None for one microbatch
            per batch. Memory/parallelism knob only; it does not change the update.
        gradient_accumulation_steps: must match the optimizer's, see
            `keras_api._validate_optimizer`.
        rescale_to_unit_norm: leave False, or the update is divided by clipping_norm.

    Returns:
        A `jax_privacy.keras_api.DPKerasConfig`, with `noise_multiplier` pinned to 0.0 for
        the non-private arm and left None (calibrated by jax_privacy at first use) for the
        DP arms.
    """
    from jax_privacy import keras_api  # type: ignore

    nonprivate = is_nonprivate(epsilon)
    if nonprivate and not math.isfinite(clipping_norm):
        # stddev is computed as noise_multiplier * l2_sensitivity / sqrt(accumulation),
        # and l2_sensitivity is clipping_norm / batch_size, so an infinite clipping norm
        # gives 0 * inf = nan and every gradient becomes nan on the first step. Pass a
        # clipping norm large enough never to bind instead (per-example gradient norms are
        # O(1) here) if a genuinely unclipped baseline is what is wanted.
        raise ValueError(
            "clipping_norm must be finite at epsilon=inf: the noise stddev is "
            "noise_multiplier * clipping_norm / batch_size, which is 0 * inf = nan. Use a "
            "large finite clipping norm for an effectively unclipped baseline."
        )

    kwargs = dict(
        epsilon=epsilon,
        delta=delta,
        clipping_norm=clipping_norm,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        rescale_to_unit_norm=rescale_to_unit_norm,
        train_steps=train_steps,
        train_size=train_size,
        seed=seed,
        microbatch_size=microbatch_size,
    )
    if not nonprivate:
        return keras_api.DPKerasConfig(**kwargs)
    with allow_zero_noise_multiplier():
        return keras_api.DPKerasConfig(noise_multiplier=0.0, **kwargs)
