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

    x = conv_block(image_in, kernel_size, num_filters)
    x = conv_block(x, kernel_size, 2 * num_filters)
    x = conv_block(x, kernel_size, 2 * num_filters, pool=False)
    x = conv_block(x, kernel_size, num_filters, pool=False, upsample=True)
    x = conv_block(x, kernel_size, num_colors, pool=False, upsample=True)

    outputs = layers.Conv2D(
        filters=num_colors, kernel_size=kernel_size, padding="SAME"
    )(x)

    model = keras.Model(inputs=image_in, outputs=outputs)

    optimizer = keras.optimizers.get(optimizer)
    if lr is not None:
        optimizer.lr = lr

    model.compile(optimizer=optimizer, loss=loss, metrics=loss)

    return model
