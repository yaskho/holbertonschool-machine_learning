#!/usr/bin/env python3
"""
Module to train a Transformer model for Portuguese to English translation.
"""
import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom Learning Rate Scheduler for Transformer.
    """

    def __init__(self, d_model, warmup_steps=4000):
        """
        Class constructor.
        """
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        Calculates learning rate value for current step.
        """
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """
    Computes sparse categorical crossentropy loss ignoring padded tokens.
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    """
    Computes sparse categorical accuracy ignoring padded tokens.
    """
    accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=-1), real.dtype))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


@tf.function
def train_step(transformer, optimizer, train_loss, train_accuracy, inp, tar):
    """
    Optimized single training step using compiled TensorFlow graph.
    """
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_mask, look_ahead_mask, dec_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions = transformer(
            inp, tar_inp,
            training=True,
            encoder_mask=enc_mask,
            look_ahead_mask=look_ahead_mask,
            decoder_mask=dec_mask
        )
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, transformer.trainable_variables)
    )

    acc = accuracy_function(tar_real, predictions)
    train_loss(loss)
    train_accuracy(acc)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a transformer model.

    Args:
        N: number of blocks in encoder and decoder
        dm: dimensionality of model
        h: number of heads
        hidden: number of hidden units in FC layers
        max_len: maximum number of tokens per sequence
        batch_size: batch size for training
        epochs: number of epochs to train for

    Returns:
        The trained Transformer model.
    """
    data = Dataset(batch_size, max_len)

    input_vocab_size = data.tokenizer_pt.vocab_size + 2
    target_vocab_size = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N, dm, h, hidden, input_vocab_size, target_vocab_size, max_len
    )

    learning_rate = CustomSchedule(dm)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    for epoch in range(epochs):
        train_loss.reset_state()
        train_accuracy.reset_state()

        for (batch, (inp, tar)) in enumerate(data.data_train):
            train_step(
                transformer, optimizer, train_loss, train_accuracy, inp, tar
            )

            if batch % 50 == 0:
                print(
                    f"Epoch {epoch + 1}, batch {batch}: "
                    f"loss {train_loss.result()} "
                    f"accuracy {train_accuracy.result()}"
                )

        print(
            f"Epoch {epoch + 1}: "
            f"loss {train_loss.result()} "
            f"accuracy {train_accuracy.result()}"
        )

    return transformer
