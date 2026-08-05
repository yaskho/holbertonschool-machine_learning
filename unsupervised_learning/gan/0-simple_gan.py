#!/usr/bin/env python3
"""Simple GAN module."""

import tensorflow as tf
from tensorflow import keras
import numpy as np


class Simple_GAN(keras.Model):
    """Simple GAN."""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        ...
        # your __init__ here

    def get_fake_sample(self, size=None, training=False):
        ...
        # your implementation

    def get_real_sample(self, size=None):
        ...
        # your implementation

    def train_step(self, useless_argument):
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)

                real_pred = self.discriminator(real_sample, training=True)
                fake_pred = self.discriminator(fake_sample, training=True)

                discr_loss = self.discriminator.loss(real_pred, fake_pred)

            gradients = tape.gradient(
                discr_loss,
                self.discriminator.trainable_variables
            )

            self.discriminator.optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            fake_pred = self.discriminator(fake_sample, training=True)

            gen_loss = self.generator.loss(fake_pred)

        gradients = tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )

        self.generator.optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss
        }
