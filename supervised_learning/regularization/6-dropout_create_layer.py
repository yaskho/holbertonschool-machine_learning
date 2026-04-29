import tensorflow as tf

def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )

    A = layer(prev)

    if training:
        A = tf.nn.dropout(A, rate=1 - keep_prob)

    return A
