from tensorflow.keras import layers


def conv_block(
    image, kernel_size, num_filters, pool=True, batch_norm=True, upsample=False
):

    x = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, padding="SAME")(
        image
    )
    if pool:
        x = layers.MaxPool2D(pool_size=(2, 2))(x)
    elif upsample:
        x = layers.UpSampling2D(size=(2, 2))(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)

    return layers.Activation("relu")(x)
