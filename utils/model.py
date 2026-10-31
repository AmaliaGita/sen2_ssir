"""Iimplementation of the Sen2-SSIR architecture."""

from pathlib import Path
from typing import Sequence, Tuple, Union

import tensorflow as tf
from tensorflow.keras import Model, layers

from .config import MODEL_NAME, RESIDUAL_SCALE


@tf.keras.utils.register_keras_serializable(package="Sen2-SSIR")
class ResidualScale(layers.Layer):
    """Residual Scaling"""

    def __init__(self, factor: float = RESIDUAL_SCALE, **kwargs):
        super().__init__(**kwargs)
        self.factor = float(factor)

    def call(self, inputs):
        return inputs * self.factor

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor})
        return config


def inception_module(inputs, filters: int = 32, name: str = "inception"):
    """Inception Module"""

    branch_1x1 = layers.Conv2D(
        filters,
        1,
        padding="same",
        activation="relu",
        kernel_initializer="he_uniform",
        name=f"{name}_direct_1x1",
    )(inputs)

    branch_3x3 = layers.Conv2D(
        filters,
        1,
        padding="same",
        activation="relu",
        kernel_initializer="he_uniform",
        name=f"{name}_reduce_3x3",
    )(inputs)
    branch_3x3 = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_uniform",
        name=f"{name}_conv_3x3",
    )(branch_3x3)

    branch_5x5 = layers.Conv2D(
        filters,
        1,
        padding="same",
        activation="relu",
        kernel_initializer="he_uniform",
        name=f"{name}_reduce_5x5",
    )(inputs)
    branch_5x5 = layers.Conv2D(
        filters,
        5,
        padding="same",
        activation="relu",
        kernel_initializer="he_uniform",
        name=f"{name}_conv_5x5",
    )(branch_5x5)

    return layers.Concatenate(axis=-1, name=f"{name}_concatenate")(
        [branch_1x1, branch_3x3, branch_5x5]
    )


def residual_dense_block(inputs, filters: int = 64, name: str = "rdb"):
    """Five-layer RDB"""

    conv_1 = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv_1")(inputs)
    conv_1 = layers.LeakyReLU(alpha=0.2, name=f"{name}_lrelu_1")(conv_1)
    dense_1 = layers.Concatenate(name=f"{name}_dense_1")([inputs, conv_1])

    conv_2 = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv_2")(dense_1)
    conv_2 = layers.LeakyReLU(alpha=0.2, name=f"{name}_lrelu_2")(conv_2)
    dense_2 = layers.Concatenate(name=f"{name}_dense_2")(
        [inputs, dense_1, conv_2]
    )

    conv_3 = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv_3")(dense_2)
    conv_3 = layers.LeakyReLU(alpha=0.2, name=f"{name}_lrelu_3")(conv_3)
    dense_3 = layers.Concatenate(name=f"{name}_dense_3")(
        [inputs, dense_1, dense_2, conv_3]
    )

    conv_4 = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv_4")(dense_3)
    conv_4 = layers.LeakyReLU(alpha=0.2, name=f"{name}_lrelu_4")(conv_4)
    dense_4 = layers.Concatenate(name=f"{name}_dense_4")(
        [inputs, dense_1, dense_2, dense_3, conv_4]
    )

    conv_5 = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv_5")(dense_4)
    conv_5 = ResidualScale(name=f"{name}_residual_scale")(conv_5)
    return layers.Add(name=f"{name}_residual_add")([inputs, conv_5])


def rrdb_module(inputs, filters: int = 64, name: str = "rrdb"):
    """Residual-in-residual module containing three RDBs."""

    features = inputs
    for block_index in range(1, 4):
        features = residual_dense_block(
            features,
            filters=filters,
            name=f"{name}_rdb_{block_index}",
        )
    features = ResidualScale(name=f"{name}_residual_scale")(features)
    return layers.Add(name=f"{name}_residual_add")([inputs, features])


def post_residual_module(
    rrdb_features,
    inception_features,
    filters: int = 64,
    name: str = "post_residual",
):
    """Fuse RRDB and initial Inception features"""

    features = layers.Conv2D(
        filters,
        3,
        padding="same",
        name=f"{name}_conv",
    )(rrdb_features)
    features = ResidualScale(name=f"{name}_scale")(features)
    return layers.Add(name=f"{name}_add")([inception_features, features])


def build_sen2_ssir(
    input_shapes: Sequence[Tuple[Union[int, None], Union[int, None], int]] = (
        (None, None, 4),
        (None, None, 6),
    ),
    initial_filters: int = 128,
    rrdb_filters: int = 64,
) -> Model:
    """Build Sen2-SSIR using spatially aligned fine and upsampled coarse inputs.

    The 2x coarse-band upsampling is deliberately performed by the shared
    preprocessing utilities.
    """

    if len(input_shapes) != 2:
        raise ValueError("Sen2-SSIR requires exactly two input shapes")
    if input_shapes[0][-1] != 4 or input_shapes[1][-1] != 6:
        raise ValueError("Sen2-SSIR expects four fine and six coarse bands")

    fine_input = layers.Input(shape=input_shapes[0], name="fine_bands")
    coarse_input = layers.Input(shape=input_shapes[1], name="upsampled_coarse_bands")

    concatenated = layers.Concatenate(axis=-1, name="input_concatenation")(
        [fine_input, coarse_input]
    )
    initial_features = layers.Conv2D(
        initial_filters,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_uniform",
        name="initial_convolution",
    )(concatenated)

    inception_features = inception_module(
        initial_features,
        filters=initial_filters // 4,
        name="inception_module",
    )
    inception_features = layers.Conv2D(
        rrdb_filters,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_uniform",
        name="inception_projection",
    )(inception_features)

    deep_features = rrdb_module(
        inception_features,
        filters=rrdb_filters,
        name="rrdb_module",
    )
    reconstructed_features = post_residual_module(
        deep_features,
        inception_features,
        filters=rrdb_filters,
        name="post_residual_module",
    )

    spectral_residual = layers.Conv2D(
        6,
        3,
        padding="same",
        kernel_initializer="he_uniform",
        name="spectral_reconstruction",
    )(reconstructed_features)
    output = layers.Add(name="spectral_feature_addition")(
        [coarse_input, spectral_residual]
    )

    return Model(
        inputs=[fine_input, coarse_input],
        outputs=output,
        name=MODEL_NAME,
    )


def load_sen2_ssir_weights(weights_path) -> Model:
    """Build Sen2-SSIR and load a compatible weights file."""

    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Sen2-SSIR weights not found: {weights_path}")
    model = build_sen2_ssir()
    model.load_weights(str(weights_path))
    return model
