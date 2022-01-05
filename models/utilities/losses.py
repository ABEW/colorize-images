import tensorflow as tf
from tensorflow import keras


def custom_loss(y_true, y_pred):
    predicted = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
    ground_truth = tf.reshape(
        y_true,
        (-1,),
    )

    return keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
        ground_truth, predicted
    )
