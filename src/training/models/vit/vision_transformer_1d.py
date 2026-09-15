import keras
import typing_extensions as tx
import typing
from typing import Tuple

from . import layers

ConfigDict = tx.TypedDict(
    "ConfigDict",
    {
        "dropout": float,
        "mlp_dim": int,
        "num_heads": int,
        "num_layers": int,
        "hidden_size": int,
    },
)

CONFIG_TINY: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 768,
    "num_heads": 3,
    "num_layers": 4,
    "hidden_size": 192,
}

CONFIG_SMALL: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 1536,
    "num_heads": 6,
    "num_layers": 8,
    "hidden_size": 384,
}

CONFIG_BASE: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 12,
    "hidden_size": 768,
}

CONFIG_LARGE: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 4096,
    "num_heads": 16,
    "num_layers": 24,
    "hidden_size": 1024,
}

CONFIG_HUGE: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 5120,
    "num_heads": 16,
    "num_layers": 32,
    "hidden_size": 1280,
}


def build_model_1d(
    input_shape: Tuple[int, int],
    patch_size: int,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    name: str,
    mlp_dim: int,
    classes: int,
    dropout=0.1,
    activation="linear",
    include_top=True,
    representation_size=None,
):
    """Build a 1D ViT model for time series classification (e.g., ECG).

    Args:
        input_shape: The shape of the input time series (length, channels).
                     For example, (5000, 12) for 12-lead ECG with 5000 time steps.
        patch_size: The size of each patch along the time dimension.
        classes: Number of classes to classify into.
        num_layers: The number of transformer layers to use.
        hidden_size: The number of filters to use (embedding dimension).
        num_heads: The number of transformer heads.
        mlp_dim: The number of dimensions for the MLP output in the transformers.
        dropout: Fraction of the units to drop for dense layers.
        activation: The activation to use for the final layer.
        include_top: Whether to include the final classification layer.
        representation_size: The size of the representation prior to the
            classification layer. If None, no Dense layer is inserted.
    """
    assert input_shape[0] % patch_size == 0, (
        f"Input length must be a multiple of patch_size: {input_shape[0]} % {patch_size} != 0"
    )

    # Input layer: (length, channels)
    x = keras.layers.Input(shape=input_shape)

    # Patch embedding: Use Conv1D to extract patches
    # This converts (length, channels) -> (num_patches, hidden_size)
    y = keras.layers.Conv1D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="embedding",
    )(x)

    # y now has shape (batch, num_patches, hidden_size)

    # Add class token
    y = layers.ClassToken(name="class_token")(y)

    # Add positional embeddings
    y = layers.AddPositionEmbs(name="Transformer_posembed_input")(y)

    # Transformer blocks
    for n in range(num_layers):
        y, _ = layers.TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer_encoderblock_{n}",
        )(y)

    # Final layer normalization
    y = keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer_encoder_norm"
    )(y)

    # Extract class token (first token) - use slicing layer instead of Lambda
    y = layers.ExtractClassToken(name="ExtractToken")(y)

    # Optional representation layer
    if representation_size is not None:
        y = keras.layers.Dense(
            representation_size, name="pre_logits", activation="tanh"
        )(y)

    # Classification head
    if include_top:
        y = keras.layers.Dense(classes, name="head", activation=activation)(y)

    return keras.models.Model(inputs=x, outputs=y, name=name)


def vit1d_tiny(
    input_shape: Tuple[int, ...] = (5000, 12),
    patch_size: int = 100,
    classes: int = 6,
    activation: str = "linear",
    include_top: bool = True,
    dropout: typing.Optional[float] = None,
):
    """Build a tiny 1D ViT model for ECG classification.

    Args:
        input_shape: Shape of input (length, channels), e.g., (5000, 12) for 12-lead ECG.
        patch_size: Size of each patch along the time dimension.
        classes: Number of output classes.
        activation: Activation function for the output layer.
        include_top: Whether to include the classification head.

    Returns:
        A Keras model.
    """
    if len(input_shape) != 2:
        raise ValueError(
            f"Input shape must be of length 2 (length, channels), got {input_shape}"
        )
    config = {**CONFIG_TINY, **({"dropout": dropout} if dropout is not None else {})}
    model = build_model_1d(
        **config,
        name="vit1d-tiny",
        patch_size=patch_size,
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
    )
    return model


def vit1d_small(
    input_shape: Tuple[int, ...] = (5000, 12),
    patch_size: int = 100,
    classes: int = 6,
    activation: str = "linear",
    include_top: bool = True,
    dropout: typing.Optional[float] = None,
):
    """Build a small 1D ViT model for ECG classification.

    Args:
        input_shape: Shape of input (length, channels), e.g., (5000, 12) for 12-lead ECG.
        patch_size: Size of each patch along the time dimension.
        classes: Number of output classes.
        activation: Activation function for the output layer.
        include_top: Whether to include the classification head.
        dropout: Override for dropout rate. If None, uses config default (0.1).

    Returns:
        A Keras model.
    """
    if len(input_shape) != 2:
        raise ValueError(
            f"Input shape must be of length 2 (length, channels), got {input_shape}"
        )
    config = {**CONFIG_SMALL, **({"dropout": dropout} if dropout is not None else {})}
    model = build_model_1d(
        **config,
        name="vit1d-small",
        patch_size=patch_size,
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
    )
    return model


def vit1d_base(
    input_shape: Tuple[int, ...] = (5000, 12),
    patch_size: int = 100,
    classes: int = 6,
    activation: str = "linear",
    include_top: bool = True,
    dropout: typing.Optional[float] = None,
):
    """Build a base 1D ViT model for ECG classification.

    Args:
        input_shape: Shape of input (length, channels), e.g., (5000, 12) for 12-lead ECG.
        patch_size: Size of each patch along the time dimension.
        classes: Number of output classes.
        activation: Activation function for the output layer.
        include_top: Whether to include the classification head.
        dropout: Override for dropout rate. If None, uses config default (0.1).

    Returns:
        A Keras model.
    """
    if len(input_shape) != 2:
        raise ValueError(
            f"Input shape must be of length 2 (length, channels), got {input_shape}"
        )
    config = {**CONFIG_BASE, **({"dropout": dropout} if dropout is not None else {})}
    model = build_model_1d(
        **config,
        name="vit1d-base",
        patch_size=patch_size,
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
    )
    return model


def vit1d_large(
    input_shape: Tuple[int, ...] = (5000, 12),
    patch_size: int = 100,
    classes: int = 6,
    activation: str = "linear",
    include_top: bool = True,
    dropout: typing.Optional[float] = None,
):
    """Build a large 1D ViT model for ECG classification.

    Args:
        input_shape: Shape of input (length, channels), e.g., (5000, 12) for 12-lead ECG.
        patch_size: Size of each patch along the time dimension.
        classes: Number of output classes.
        activation: Activation function for the output layer.
        include_top: Whether to include the classification head.
        dropout: Override for dropout rate. If None, uses config default (0.1).

    Returns:
        A Keras model.
    """
    if len(input_shape) != 2:
        raise ValueError(
            f"Input shape must be of length 2 (length, channels), got {input_shape}"
        )
    config = {**CONFIG_LARGE, **({"dropout": dropout} if dropout is not None else {})}
    model = build_model_1d(
        **config,
        name="vit1d-large",
        patch_size=patch_size,
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
    )
    return model

def vit1d_huge(
    input_shape: Tuple[int, ...] = (5000, 12),
    patch_size: int = 100,
    classes: int = 6,
    activation: str = "linear",
    include_top: bool = True,
    dropout: typing.Optional[float] = None,
):
    """Build a huge 1D ViT model for ECG classification.

    Args:
        input_shape: Shape of input (length, channels), e.g., (5000, 12) for 12-lead ECG.
        patch_size: Size of each patch along the time dimension.
        classes: Number of output classes.
        activation: Activation function for the output layer.
        include_top: Whether to include the classification head.
        dropout: Override for dropout rate. If None, uses config default (0.1).

    Returns:
        A Keras model.
    """
    if len(input_shape) != 2:
        raise ValueError(
            f"Input shape must be of length 2 (length, channels), got {input_shape}"
        )
    config = {**CONFIG_HUGE, **({"dropout": dropout} if dropout is not None else {})}
    model = build_model_1d(
        **config,
        name="vit1d-huge",
        patch_size=patch_size,
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
    )
    return model