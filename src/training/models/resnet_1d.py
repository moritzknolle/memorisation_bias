"""Residual baselines: a 1D ResNet for ECG and a ResNet-style MLP for tabular data.

The 1D ResNet is implemented as described in Wang et al., "Time Series Classification
from Scratch with Deep Neural Networks: A Strong Baseline", Data Mining and Knowledge
Discovery, 2019.

Normalisation defaults to GroupNorm rather than BatchNorm (`bn_fn` is still available as
a `norm=` argument), and every norm layer is pinned to float32 so mixed-precision
training does not compute the statistics in float16.

Both builders emit logits (no output activation).
"""

import keras  # type: ignore
from typing import Any, Tuple, Callable, Optional
from functools import partial

gn_10 = partial(
    keras.layers.GroupNormalization, axis=-1, groups=10, epsilon=1e-6, dtype="float32"
)
gn_16 = partial(
    keras.layers.GroupNormalization, axis=-1, groups=16, epsilon=1e-6, dtype="float32"
)
bn_fn = partial(
    keras.layers.BatchNormalization, momentum=0.9, epsilon=1e-6, dtype="float32"
)


def build_1d_resnet(
    input_shape: Tuple[int, ...] = (1000, 12),
    batch_size: Optional[int] = None,
    nb_classes: int = 5,
    nb_feature_maps: int = 64,
    norm: Callable = gn_16,
) -> keras.models.Model:
    """Three residual convolutional blocks over a 1D signal, then global average pooling.

    Reached via `get_model("resnet1d")` or `resnet1d_<nb_feature_maps>`.

    Args:
        input_shape: (timesteps, channels), e.g. (2500, 12) for 10 s of 12-lead ECG at 250 Hz.
        batch_size: Fixed batch size to build with, or None to leave it dynamic.
        nb_classes: Number of output units.
        nb_feature_maps: Channel width; blocks two and three use twice this.
        norm: Normalisation layer factory.

    Returns:
        keras.models.Model emitting logits of shape (batch, nb_classes).
    """
    input = keras.layers.Input(input_shape, batch_size=batch_size)
    # BLOCK 1
    x = keras.layers.Conv1D(filters=nb_feature_maps, kernel_size=8, padding="same")(
        input
    )
    x = norm()(x)
    x = keras.layers.Activation("leaky_relu")(x)
    x = keras.layers.Conv1D(filters=nb_feature_maps, kernel_size=5, padding="same")(x)
    x = norm()(x)
    x = keras.layers.Activation("leaky_relu")(x)
    x = keras.layers.Conv1D(filters=nb_feature_maps, kernel_size=3, padding="same")(x)
    x = norm()(x)
    shortcut_y = keras.layers.Conv1D(
        filters=nb_feature_maps, kernel_size=1, padding="same"
    )(input)
    shortcut_y = norm()(shortcut_y)
    output_block_1 = keras.layers.add([shortcut_y, x])
    output_block_1 = keras.layers.Activation("leaky_relu")(output_block_1)
    # BLOCK 2
    x = keras.layers.Conv1D(filters=nb_feature_maps * 2, kernel_size=8, padding="same")(
        output_block_1
    )
    x = norm()(x)
    x = keras.layers.Activation("leaky_relu")(x)
    x = keras.layers.Conv1D(filters=nb_feature_maps * 2, kernel_size=5, padding="same")(
        x
    )
    x = norm()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv1D(filters=nb_feature_maps * 2, kernel_size=3, padding="same")(
        x
    )
    x = norm()(x)
    shortcut_y = keras.layers.Conv1D(
        filters=nb_feature_maps * 2, kernel_size=1, padding="same"
    )(output_block_1)
    shortcut_y = norm()(shortcut_y)
    output_block_2 = keras.layers.add([shortcut_y, x])
    output_block_2 = keras.layers.Activation("leaky_relu")(output_block_2)
    # BLOCK 3
    x = keras.layers.Conv1D(filters=nb_feature_maps * 2, kernel_size=8, padding="same")(
        output_block_2
    )
    x = norm()(x)
    x = keras.layers.Activation("leaky_relu")(x)
    x = keras.layers.Conv1D(filters=nb_feature_maps * 2, kernel_size=5, padding="same")(
        x
    )
    x = norm()(x)
    x = keras.layers.Activation("leaky_relu")(x)
    x = keras.layers.Conv1D(filters=nb_feature_maps * 2, kernel_size=3, padding="same")(
        x
    )
    x = norm()(x)
    shortcut_y = norm()(output_block_2)
    output_block_3 = keras.layers.add([shortcut_y, x])
    output_block_3 = keras.layers.Activation("leaky_relu")(output_block_3)
    # classification head
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
    output = keras.layers.Dense(nb_classes, dtype="float32")(gap_layer)
    # construct model
    model = keras.models.Model(inputs=input, outputs=output)
    return model


class ResNetBlock(keras.Model):
    """Pre-norm residual block of two dense layers, used by `build_tabular_resnet`.

    Args:
        width: Output width of both dense layers and of the shortcut projection.
        dropout_rate: Dropout between the two dense layers; 0 disables it.
        norm: Normalisation layer factory, applied before the block (pre-norm).
    """

    def __init__(
        self,
        width: int,
        dropout_rate: float = 0.1,
        norm: Callable = gn_10,
    ):
        super().__init__()
        self.dense_1 = keras.layers.Dense(width)
        self.dense_2 = keras.layers.Dense(width)
        self.shortcut = keras.layers.Dense(width)
        self.dropout = (
            keras.layers.Dropout(dropout_rate)
            if dropout_rate > 0
            else keras.layers.Lambda(lambda x: x)
        )
        self.norm_fn = norm()

    def call(self, x):
        x = self.norm_fn(x)
        residual = self.shortcut(x)
        x = self.dense_1(x)
        x = keras.activations.relu(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        x += residual
        return x


def build_tabular_resnet(
    input_shape: Tuple[int, ...] = (64,),
    batch_size: Optional[int] = None,
    num_classes: int = 1,
    width: int = 300,
    depth: int = 5,
    dropout_rate: float = 0.1,
    norm: Callable = gn_10,
    input_stats: Optional[Tuple[Any, Any]] = None,
):
    """Fully connected ResNet for tabular inputs -- the MIMIC-IV-ED architecture.

    A projection to `width`, then `depth` residual blocks of it. Reached via
    `get_model("tabresnet_<width>_<depth>")`, e.g. the default `tabresnet_500_5`.

    Args:
        input_shape: (n_features,).
        batch_size: Fixed batch size to build with, or None to leave it dynamic.
        num_classes: Number of output units.
        width: Hidden width of every block.
        depth: Number of residual blocks.
        dropout_rate: Dropout inside each block.
        norm: Normalisation layer factory.
        input_stats: Optional (mean, variance) per feature, each of shape (n_features,).
            When given, a frozen float32 `Normalization` layer standardises the raw
            columns before the input projection. The stats must be computed once on the
            full historical training split, not per subset, so that IN and OUT models see
            identical preprocessing.

    Returns:
        keras.Model emitting logits of shape (batch, num_classes).
    """
    inputs = keras.Input(shape=input_shape, batch_size=batch_size)
    if input_stats is not None:
        mean, variance = input_stats
        # pinned to float32 for the same reason the norm layers are: under
        # mixed_float16 the standardisation itself must not lose precision, and the
        # raw columns span three orders of magnitude.
        x = keras.layers.Normalization(
            axis=-1, mean=mean, variance=variance, dtype="float32"
        )(inputs)
    else:
        x = inputs
    x = keras.layers.Dense(width)(x)
    for _ in range(depth):
        x = ResNetBlock(width, dropout_rate)(x)
    x = norm()(x)
    x = keras.activations.relu(x)
    outputs = keras.layers.Dense(num_classes)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
