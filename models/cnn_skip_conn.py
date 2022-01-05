import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .utilities.layers import conv_block
from .utilities.losses import custom_loss


def colorizer_model(
    input_shape,
    kernel_size,
    num_filters,
    num_colors,
    optimizer="adam",
    loss=custom_loss,
    lr=None,
):

    image_in = layers.Input(shape=input_shape)

    x_1 = conv_block(image_in, kernel_size, num_filters)
    x_2 = conv_block(x_1, kernel_size, 2 * num_filters)
    x_3 = conv_block(x_2, kernel_size, 2 * num_filters, pool=False)
    x_4 = conv_block(
        tf.concat([x_3, x_2], axis=-1),
        kernel_size,
        num_filters,
        pool=False,
        upsample=True,
    )
    x_5 = conv_block(
        tf.concat([x_4, x_1], axis=-1),
        kernel_size,
        num_colors,
        pool=False,
        upsample=True,
    )

    outputs = layers.Conv2D(
        filters=num_colors, kernel_size=kernel_size, padding="SAME"
    )(tf.concat([x_5, image_in], axis=-1))

    model = keras.Model(inputs=image_in, outputs=outputs)

    optimizer = keras.optimizers.get(optimizer)
    if lr is not None:
        optimizer.lr = lr

    model.compile(optimizer=optimizer, loss=loss, metrics=loss)

    return model
