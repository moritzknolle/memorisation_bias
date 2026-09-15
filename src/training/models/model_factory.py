"""Model zoo, addressed by name so a run's architecture is a flag rather than an import.

`get_model(model_name, ...)` parses the name and builds the model. Families:

    vit1d_<size>_<patch>[_dp]   1D ViT for ECG. size in tiny/small/base/large/huge,
                                patch is the token length in samples (25 = 100 ms at
                                250 Hz). The _dp suffix sets dropout to 0, which is what
                                the DP runs use (they already inject noise of their own).
    vit_<size>_<patch>          2D ViT for images (vendored, see models/vit/readme.md).
    tabresnet_<width>_<blocks>  tabular ResNet, for MIMIC-IV-ED.
    resnet1d[_<featuremaps>]    1D ResNet for ECG.
    wrn_<depth>_<width>         wide ResNet.
    resnet50 / densenet{121,169,201} / efficientnet_<b0-b7>
                                keras.applications backbones; append _imagenet for
                                pretrained weights. A global-pool + linear head is added.
    small_cnn                   small convnet, for smoke tests.

All heads are linear: the models emit **logits**, and the loss (`from_logits=True`) and
the analysis apply the activation. Pass `seed` to make weight initialisation
reproducible -- it seeds keras globally, so identical (model, seed) pairs across the
leave-many-out runs differ only in their data subset.
"""

from functools import partial
from typing import Any, Callable, Optional, Tuple, Dict

import keras  # type: ignore

from .resnet_1d import build_1d_resnet, build_tabular_resnet
from .small_models import get_small_cnn
from .vit.vision_transformer import (
    vit_b8,
    vit_b16,
    vit_b32,
    vit_l16,
    vit_l32,
    vit_s8,
    vit_s16,
    vit_s32,
)
from .vit.vision_transformer_1d import (
    vit1d_base,
    vit1d_huge,
    vit1d_large,
    vit1d_small,
    vit1d_tiny,
)
from .wide_resnet import get_wide_resnet


def get_model(
    model_name: str,
    input_shape: Tuple[int, ...],
    num_classes: int,
    batch_size: Optional[int] = None,
    preprocessing_func: Optional[Callable] = None,
    seed: Optional[int] = None,
    freeze_bn_layers: bool = False,
    dropout_rate: Optional[float] = None,
    input_stats: Optional[Tuple[Any, Any]] = None,
):
    """
    Helper function that returns a function which creates the respective model given the model name.
    Args:
        model_name: str, name of the model
        input_shape: Tuple[int, ...], shape of the input
        num_classes: int, number of classes
        preprocessing_func: callable, preprocessing function to apply to the input image. Only effective for ViT models.
        seed: int, optional random seed for reproducible weight initialization
        freeze_bn_layers: bool, if True, freezes all Batch Normalization layers in the model
        dropout_rate: float, optional dropout rate override. 'tabresnet_*' only; None keeps
            the builder default.
        input_stats: optional (mean, variance) per input feature. 'tabresnet_*' only; when
            given, a frozen Normalization layer standardises the raw columns before the
            input projection. See build_tabular_resnet for why this is off by default.
    Returns:
        Callable: function that returns the corresponding model
    """
    # Set random seed for reproducible weight initialization
    if seed is not None:
        keras.utils.set_random_seed(seed)
    backbone_only = False
    for a in input_shape:
        if not isinstance(a, int):
            raise ValueError(
                "expected input_shape to be a tuple of integers, got: {}".format(
                    input_shape
                )
            )
    if model_name == "small_cnn":
        model = get_small_cnn(input_shape=input_shape, num_classes=num_classes)
    elif model_name.split("_")[0] == "wrn":
        depth = int(model_name.split("_")[1])
        width = int(model_name.split("_")[2])
        model = get_wide_resnet(
            depth=depth,
            width=width,
            input_shape=input_shape,
            num_classes=num_classes,
        )
    elif model_name == "resnet50":
        backbone = keras.applications.ResNet50(
            weights=None,
            include_top=False,
            input_shape=input_shape,
            classes=num_classes,
        )
        backbone_only = True
    elif model_name == "resnet50_imagenet":
        backbone = keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
            classes=num_classes,
        )
        backbone_only = True
    elif model_name == "densenet121":
        backbone = keras.applications.DenseNet121(
            weights=None,
            include_top=False,
            input_shape=input_shape,
            classes=num_classes,
        )
        backbone_only = True
    elif model_name == "densenet121_imagenet":
        backbone = keras.applications.DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
            classes=num_classes,
        )
        backbone_only = True
    elif model_name == "densenet169":
        backbone = keras.applications.DenseNet169(
            weights=None,
            include_top=False,
            input_shape=input_shape,
            classes=num_classes,
        )
        backbone_only = True
    elif model_name == "densenet169_imagenet":
        backbone = keras.applications.DenseNet169(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
            classes=num_classes,
        )
        backbone_only = True
    elif model_name == "densenet201":
        backbone = keras.applications.DenseNet201(
            weights=None,
            include_top=False,
            input_shape=input_shape,
            classes=num_classes,
        )
        backbone_only = True
    elif model_name == "densenet201_imagenet":
        backbone = keras.applications.DenseNet201(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
            classes=num_classes,
        )
        backbone_only = True
    elif model_name.split("_")[0] == "efficientnet":
        size = model_name.split("_")[1]
        if len(model_name.split("_")) > 2:
            weights = model_name.split("_")[2]
        else:
            weights = None
        if size == "b0":
            backbone = keras.applications.EfficientNetB0(
                weights=weights,
                include_top=False,
                input_shape=input_shape,
                classes=num_classes,
            )
        elif size == "b1":
            backbone = keras.applications.EfficientNetB1(
                weights=weights,
                include_top=False,
                input_shape=input_shape,
                classes=num_classes,
            )
        elif size == "b2":
            backbone = keras.applications.EfficientNetB2(
                weights=weights,
                include_top=False,
                input_shape=input_shape,
                classes=num_classes,
            )
        elif size == "b3":
            backbone = keras.applications.EfficientNetB3(
                weights=weights,
                include_top=False,
                input_shape=input_shape,
                classes=num_classes,
            )
        elif size == "b4":
            backbone = keras.applications.EfficientNetB4(
                weights=weights,
                include_top=False,
                input_shape=input_shape,
                classes=num_classes,
            )
        elif size == "b5":
            backbone = keras.applications.EfficientNetB5(
                weights=weights,
                include_top=False,
                input_shape=input_shape,
                classes=num_classes,
            )
        elif size == "b6":
            backbone = keras.applications.EfficientNetB6(
                weights=weights,
                include_top=False,
                input_shape=input_shape,
                classes=num_classes,
            )
        elif size == "b7":
            backbone = keras.applications.EfficientNetB7(
                weights=weights,
                include_top=False,
                input_shape=input_shape,
                classes=num_classes,
            )
        else:
            raise ValueError(f"Invalid EfficientNet variant {model_name}")
        backbone_only = True
    elif model_name.split("_")[0] == "resnet1d" or model_name == "resnet1d":
        nb_feature_maps = (
            int(model_name.split("_")[1]) if len(model_name.split("_")) > 1 else 64
        )
        model = build_1d_resnet(
            nb_classes=num_classes,
            input_shape=input_shape,
            batch_size=batch_size,
            nb_feature_maps=nb_feature_maps,
        )
    elif model_name.split("_")[0] == "vit1d":
        # Format: vit1d_[size]_[patch_size] or vit1d_[size]_[patch_size]_dp
        # Example: vit1d_tiny_100, vit1d_base_50, vit1d_small_25_dp
        # The _dp suffix disables dropout (sets it to 0.0), useful for DP training.
        parts = model_name.split("_")
        dp_mode = parts[-1] == "dp"
        if dp_mode:
            parts = parts[:-1]  # strip _dp suffix for parsing
        if len(parts) < 2:
            raise ValueError(
                f"Invalid 1D ViT model name {model_name}, expected format 'vit1d_[size]' or 'vit1d_[size]_[patch_size]'"
            )
        model_size = parts[1]
        patch_size = int(parts[2]) if len(parts) > 2 else 100
        dropout_override = 0.0 if dp_mode else None

        vit1d_kwargs = dict(
            input_shape=input_shape,
            patch_size=patch_size,
            classes=num_classes,
            activation="linear",
            include_top=True,
            dropout=dropout_override,
        )

        if model_size == "tiny":
            model = vit1d_tiny(**vit1d_kwargs)
        elif model_size == "small":
            model = vit1d_small(**vit1d_kwargs)
        elif model_size == "base":
            model = vit1d_base(**vit1d_kwargs)
        elif model_size == "large":
            model = vit1d_large(**vit1d_kwargs)
        elif model_size == "huge":
            model = vit1d_huge(**vit1d_kwargs)
        else:
            raise ValueError(
                f"Invalid 1D ViT size {model_size}, expected 'tiny', 'small', 'base', or 'large'"
            )
    elif model_name.split("_")[0] == "tabresnet":
        assert (
            len(model_name.split("_")) == 3
        ), f"Invalid model name {model_name}, expected format 'tabresnet_[width]_[n_blocks]'"
        parts = model_name.split("_")
        width_int = int(parts[1])
        n_blocks_int = int(parts[2])
        tabresnet_kwargs = {} if dropout_rate is None else {"dropout_rate": dropout_rate}
        model = build_tabular_resnet(
            input_shape=input_shape,
            batch_size=batch_size,
            width=width_int,
            depth=n_blocks_int,
            num_classes=num_classes,
            input_stats=input_stats,
            **tabresnet_kwargs,
        )

    elif model_name.split("_")[0] == "vit":
        if len(model_name.split("_")) != 3:
            raise ValueError(
                f"Invalid size {model_name}, expected format 'vit_[model_size]_[patch_size]'"
            )
        model_size = model_name.split("_")[1]
        patch_size = int(model_name.split("_")[2])
        if model_size == "small" and patch_size == 8:
            model = vit_s8(
                input_shape=input_shape,
                activation="linear",
                pretrained=False,  # Using non-pretrained weights for vit_s8 as pretrained weights are not available
                include_top=True,
                classes=num_classes,
                pretrained_top=False,
            )
        elif model_size == "small" and patch_size == 16:
            model = vit_s16(
                input_shape=input_shape,
                activation="linear",
                pretrained=True,
                include_top=True,
                classes=num_classes,
                pretrained_top=False,
            )
        elif model_size == "small" and patch_size == 32:
            model = vit_s32(
                input_shape=input_shape,
                activation="linear",
                pretrained=False,  # Using non-pretrained weights for vit_s32 as pretrained weights are not available
                include_top=True,
                classes=num_classes,
                pretrained_top=False,
            )
        elif model_size == "base" and patch_size == 8:
            model = vit_b8(
                input_shape=input_shape,
                activation="linear",
                pretrained=False,
                include_top=True,
                classes=num_classes,
                pretrained_top=False,
            )
        elif model_size == "base" and patch_size == 16:
            model = vit_b16(
                input_shape=input_shape,
                activation="linear",
                pretrained=True,
                include_top=True,
                classes=num_classes,
                pretrained_top=False,
            )
        elif model_size == "base" and patch_size == 32:
            model = vit_b32(
                input_shape=input_shape,
                activation="linear",
                pretrained=True,
                include_top=True,
                classes=num_classes,
                pretrained_top=False,
            )
        elif model_size == "large" and patch_size == 16:
            model = vit_l16(
                input_shape=input_shape,
                activation="linear",
                pretrained=True,
                include_top=True,
                classes=num_classes,
                pretrained_top=False,
            )
        elif model_size == "large" and patch_size == 32:
            model = vit_l32(
                input_shape=input_shape,
                activation="linear",
                pretrained=True,
                include_top=True,
                classes=num_classes,
                pretrained_top=False,
            )
        else:
            raise ValueError(
                f"Invalid ViT variant {model_name}, model variant: {model_size}, patch size: {patch_size}"
            )
    else:
        raise ValueError(f"Model {model_name} not found")
    if backbone_only:
        model = keras.Sequential(
            [
                backbone,
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(
                    num_classes,
                    activation="linear",
                    dtype="float32",
                    name="predictions",
                ),
            ]
        )
    # Apply preprocessing first (if needed)
    if preprocessing_func is not None:
        model = keras.Sequential(
            [
                keras.layers.Input(shape=input_shape, dtype="float16", name="input"),
                keras.layers.Lambda(preprocessing_func, name="preprocessing"),
                model,
            ]
        )

    # Freeze Batch Normalization layers if requested
    if freeze_bn_layers:
        def freeze_bn_recursive(layer):
            """Recursively freeze all BatchNormalization layers in a model."""
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
            # Recursively handle nested models
            if hasattr(layer, "layers"):
                for nested_layer in layer.layers:
                    freeze_bn_recursive(nested_layer)

        for layer in model.layers:
            freeze_bn_recursive(layer)

    return model
