#!/usr/bin/env python3
"""Train a Transformer for Portuguese to English translation."""

import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Create and train a Transformer model.

    Args:
        N: Number of encoder and decoder blocks.
        dm: Dimensionality of the model.
        h: Number of attention heads.
        hidden: Number of hidden units in feed-forward layers.
        max_len: Maximum sequence length.
        batch_size: Training batch size.
        epochs: Number of training epochs.

    Returns:
        The trained Transformer model.
    """
    data = Dataset(batch_size, max_len)

    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_len
    )

    optimizer = tf.keras.optimizers.Adam(
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    warmup_steps = 4000

    def learning_rate(step):
        """Calculate the Transformer learning rate."""
        step = tf.cast(step, tf.float32)

        return (
            tf.math.rsqrt(tf.cast(dm, tf.float32))
            * tf.math.minimum(
                tf.math.rsqrt(step),
                step * tf.pow(
                    tf.cast(warmup_steps, tf.float32),
                    -1.5
                )
            )
        )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )

    def loss_function(real, pred):
        """Calculate loss while ignoring padding."""
        loss = loss_object(real, pred)

        mask = tf.cast(
            tf.not_equal(real, 0),
            loss.dtype
        )

        loss *= mask

        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def accuracy_function(real, pred):
        """Calculate accuracy while ignoring padding."""
        predictions = tf.argmax(
            pred, axis=-1,
            output_type=real.dtype
        )

        matches = tf.cast(
            tf.equal(real, predictions),
            tf.float32
        )

        mask = tf.cast(
            tf.not_equal(real, 0),
            tf.float32
        )

        matches *= mask

        return tf.reduce_sum(matches) / tf.reduce_sum(mask)

    for epoch in range(epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        batches = 0

        for batch, (inputs, target) in enumerate(
                data.data_train):

            target_input = target[:, :-1]
            target_real = target[:, 1:]

            encoder_mask, combined_mask, decoder_mask = \
                create_masks(inputs, target_input)

            with tf.GradientTape() as tape:
                predictions = transformer(
                    inputs,
                    target_input,
                    encoder_mask,
                    combined_mask,
                    decoder_mask
                )

                loss = loss_function(
                    target_real,
                    predictions
                )

            gradients = tape.gradient(
                loss,
                transformer.trainable_variables
            )

            optimizer.learning_rate = learning_rate(
                optimizer.iterations + 1
            )

            optimizer.apply_gradients(
                zip(gradients, transformer.trainable_variables)
            )

            accuracy = accuracy_function(
                target_real,
                predictions
            )

            total_loss += loss
            total_accuracy += accuracy
            batches += 1

            if batch % 50 == 0:
                print(
                    'Epoch {}, Batch {}: Loss {}, Accuracy {}'.format(
                        epoch + 1,
                        batch,
                        loss,
                        accuracy
                    )
                )

        print(
            'Epoch {}: Loss {}, Accuracy {}'.format(
                epoch + 1,
                total_loss / batches,
                total_accuracy / batches
            )
        )

    return transformer
