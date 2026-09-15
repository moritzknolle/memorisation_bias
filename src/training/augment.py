"""Image augmentation, used by MIMIC-CXR only (the ECG and tabular runs train unaugmented).

`get_aug_fn(strength)` is the entry point: it maps a strength name to a single callable
that the tf.data pipeline maps over the training stream. The individual transforms are
*stateless* (they take an explicit seed tensor) so a run's augmentation is reproducible
from its seed rather than from tf's global RNG state.

Images arrive already ImageNet-normalised, so the fill value for rotations and shifts is
MIN_IMG_VAL rather than 0 -- filling with 0 would paint mid-grey into the corners.
"""

import math
from enum import Enum
from functools import partial
from typing import Callable, List, Optional

import keras  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore

MIN_IMG_VAL = -2.117 # minimum value an image in [0-1] can take after being transformed with ImageNet normalization


def stateless_random_rotate(
    img: tf.Tensor, seed: tf.Tensor, fill_constant: float, max_degrees=10, fillmode="CONSTANT",
):
    """
    Apply a random rotation to an image tensor. Assumes the image is in [-1, 1] range.
    XLA-compatible implementation that restores input dtype after rotation.

    Args:
        img (tf.Tensor): The input image tensor.
        seed (tf.Tensor): The seed tensor for randomness.
        max_degrees (int, optional): The maximum degrees for rotation. Defaults to 10.
        fillmode (str): Fill mode for pixels outside boundaries. Defaults to "CONSTANT".

    Returns:
        tf.Tensor: The rotated image tensor with same dtype as input.

    Note:
        Temporarily converts to float32 for rotation (required by ImageProjectiveTransformV3),
        then converts back to original dtype.
    """
    angle_degrees = tf.random.stateless_uniform(
        [1], seed=seed, minval=-float(max_degrees), maxval=float(max_degrees)
    )
    angle_radians = angle_degrees * (math.pi / 180.0)
    angle = angle_radians[0]

    # Build rotation transformation matrix
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    transform = tf.stack([cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0, 0.0, 0.0])
    transform = tf.expand_dims(transform, 0)

    # Store original dtype
    original_dtype = img.dtype

    # Expand and cast to float32 (required by ImageProjectiveTransformV3)
    expanded_image = tf.expand_dims(img, 0)
    needs_cast = original_dtype != tf.float32

    if needs_cast:
        expanded_image = tf.cast(expanded_image, tf.float32)

    output_shape = tf.shape(img)[:2]
    fill_value_tensor = tf.constant(fill_constant, dtype=tf.float32)

    # Apply rotation transformation with nearest neighbor interpolation
    rotated_image = tf.raw_ops.ImageProjectiveTransformV3(
        images=expanded_image,
        transforms=transform,
        output_shape=output_shape,
        fill_mode=fillmode,
        interpolation="BILINEAR",
        fill_value=fill_value_tensor,
    )

    rotated_image = tf.squeeze(rotated_image, axis=0)

    # Cast back to original dtype if needed
    if needs_cast:
        rotated_image = tf.cast(rotated_image, original_dtype)

    return rotated_image


def safe_stateless_random_contrast(
    img: tf.Tensor, seed: tf.Tensor, lower: float, upper: float
):
    """
    Apply random contrast adjustment to an image tensor with float32 casting.

    This wrapper ensures the operation runs in float32, as TensorFlow's AdjustContrastv2
    operation does not support float16 on CPU.

    Args:
        img (tf.Tensor): The input image tensor.
        seed (tf.Tensor): The seed tensor for randomness.
        lower (float): Lower bound for the random contrast factor.
        upper (float): Upper bound for the random contrast factor.

    Returns:
        tf.Tensor: The contrast-adjusted image tensor with same dtype as input.

    Note:
        Temporarily converts to float32 for contrast adjustment (required by CPU backend),
        then converts back to original dtype.
    """
    original_dtype = img.dtype
    needs_cast = original_dtype != tf.float32

    if needs_cast:
        img = tf.cast(img, tf.float32)

    img = tf.image.stateless_random_contrast(img, lower=lower, upper=upper, seed=seed)

    if needs_cast:
        img = tf.cast(img, original_dtype)

    return img


def safe_stateless_random_brightness(img: tf.Tensor, seed: tf.Tensor, max_delta: float):
    """
    Apply random brightness adjustment to an image tensor with float32 casting.

    This wrapper ensures the operation runs in float32, as TensorFlow's AdjustBrightness
    operation does not support float16 on CPU.

    Args:
        img (tf.Tensor): The input image tensor.
        seed (tf.Tensor): The seed tensor for randomness.
        max_delta (float): Maximum delta for the random brightness adjustment.

    Returns:
        tf.Tensor: The brightness-adjusted image tensor with same dtype as input.

    Note:
        Temporarily converts to float32 for brightness adjustment (required by CPU backend),
        then converts back to original dtype.
    """
    original_dtype = img.dtype
    needs_cast = original_dtype != tf.float32

    if needs_cast:
        img = tf.cast(img, tf.float32)

    img = tf.image.stateless_random_brightness(img, max_delta=max_delta, seed=seed)

    if needs_cast:
        img = tf.cast(img, original_dtype)

    return img


def random_pixel_shifts(
    img: tf.Tensor, seed: tf.Tensor, fill_constant: float, shift: float = 0.1, fillmode="CONSTANT"
):
    """
    Apply random pixel shifts to an image tensor.

    Args:
        img (tf.Tensor): The input image tensor.
        seed (tf.Tensor): The seed tensor for randomness.
        shift (float, optional): The shift factor. Defaults to 0.125.

    Returns:
        tf.Tensor: The image tensor with random pixel shifts.
    """
    x = img
    original_shape = img.shape
    shift = 0 if shift == 0.0 else int(shift * original_shape[1])
    x = tf.pad(
        x, [[shift] * 2, [shift] * 2, [0] * 2], mode=fillmode, constant_values=fill_constant
    )
    x = tf.image.stateless_random_crop(x, original_shape, seed=seed)
    return x


def create_random_aug_fn(aug_fn_list: List[Callable], rng: tf.random.Generator):
    """
    Create a random augmentation function from a list of augmentation functions.

    Args:
        aug_fn_list (List[Callable]): List of augmentation functions.
        rng (tf.random.Generator): Random number generator.

    Returns:
        Callable: The random augmentation function.
    """
    print(
        f"... creating augmentation function with {len(aug_fn_list)} augmentation(s): \n{aug_fn_list}"
    )

    def random_aug(x):
        if len(aug_fn_list) > 0:
            seeds = rng.make_seeds(len(aug_fn_list))
            for i, aug_fn in enumerate(aug_fn_list):
                x = aug_fn(x, seed=seeds[:, i])
        return x

    return random_aug


def get_aug_fn(aug_strength: str, fillmode="CONSTANT", fill_constant:float=MIN_IMG_VAL) -> Callable:
    """
    Get the augmentation function based on the augmentation strength.

    Args:
        aug_strength (str): The augmentation strength as a string.

    Returns:
        Callable: The augmentation function.

    Raises:
        ValueError: If the augmentation strength is invalid.
    """
    random_pixel_shift_fn = partial(random_pixel_shifts, fillmode=fillmode, shift=0.1, fill_constant=fill_constant)
    stateless_random_rotate_fn = partial(stateless_random_rotate, fillmode=fillmode, fill_constant=fill_constant, max_degrees=10)
    if aug_strength == "fliplr":
        augs = [tf.image.stateless_random_flip_left_right]
    elif aug_strength == "weak":
        augs = [tf.image.stateless_random_flip_left_right, random_pixel_shift_fn]
    elif aug_strength == "medium":
        augs = [
            tf.image.stateless_random_flip_left_right,
            random_pixel_shift_fn,
            stateless_random_rotate_fn,
        ]
    elif aug_strength == "cxr":
        dummy_random_center_crop_fn = lambda img, seed: tf.image.central_crop(
            img, central_fraction=0.875
        )
        augs = [
            tf.image.stateless_random_flip_left_right,
            stateless_random_rotate_fn,
            dummy_random_center_crop_fn,
        ]
    elif aug_strength == "center_crop":
        dummy_random_center_crop_fn = lambda img, seed: tf.image.central_crop(
            img, central_fraction=0.875
        )
        augs = [dummy_random_center_crop_fn]
    elif aug_strength == "cxr_strong":
        dummy_random_center_crop_fn = lambda img, seed: tf.image.central_crop(
            img, central_fraction=0.875
        )
        augs = [
            partial(safe_stateless_random_contrast, lower=0.5, upper=1.5),
            partial(safe_stateless_random_brightness, max_delta=0.5),
            tf.image.stateless_random_flip_left_right,
            stateless_random_rotate_fn,
            dummy_random_center_crop_fn,
        ]
    elif aug_strength == "strong":
        augs = [
            partial(safe_stateless_random_contrast, lower=0.75, upper=1.25),
            partial(safe_stateless_random_brightness, max_delta=0.25),
            tf.image.stateless_random_flip_left_right,
            random_pixel_shift_fn,
            stateless_random_rotate_fn,
        ]
    elif aug_strength == "extra_strong":
        stateless_random_rotate_fn = partial(
            stateless_random_rotate, fillmode=fillmode, max_degrees=20
        )
        augs = [
            partial(safe_stateless_random_contrast, lower=0.6, upper=1.4),
            partial(safe_stateless_random_brightness, max_delta=0.5),
            tf.image.stateless_random_flip_left_right,
            random_pixel_shift_fn,
            stateless_random_rotate_fn,
        ]
    elif aug_strength == "rotate":
        augs = [stateless_random_rotate_fn]
    elif aug_strength == "none":
        augs = []
    else:
        raise ValueError("Invalid augmentation strength.")
    random_aug_fn = create_random_aug_fn(
        aug_fn_list=augs, rng=tf.random.Generator.from_seed(420, alg="philox")
    )
    return random_aug_fn


def grayscale_to_rgb(img: tf.Tensor)-> tf.Tensor:
    """
    Preprocess a grayscale image for fine-tuning a pre-trained ImageNet model.

    Args:
        img (tf.Tensor): Grayscale image in [-1, 1] range.

    Returns:
        tf.Tensor: Three channel image preprocessed with ImageNet normalization.
    """
    IMAGENET_MEAN = 2 * keras.ops.convert_to_tensor([0.485, 0.456, 0.406]) - keras.ops.ones(
        3
    )  # since the normal IMAGENET_MEAN is for RGB images in [0, 1] range
    IMAGENET_STD = 2 * keras.ops.convert_to_tensor(
        [0.229, 0.224, 0.225]
    )  # scaling factor is constant so standard deviation remains the same
    img = keras.ops.repeat(img, 3, axis=-1)  # repeat color channel
    # the constants above are built at keras' floatx, so an image of any other float dtype
    # (float64 from numpy, for instance) would fail the subtraction on a dtype mismatch
    img = keras.ops.cast(img, keras.backend.floatx())
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img
