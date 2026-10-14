#!/usr/bin/env python3
"""
Module containing the convolutional_GenDiscr function to build the
generator and discriminator for generating 16x16 face images.
"""

from tensorflow import keras


def convolutional_GenDiscr():
    """
    Builds a convolutional generator and discriminator network.

    Returns:
        tuple: (generator, discriminator) Keras models.
    """
    def get_generator():
        """
        Builds and returns the generator model.
        """
        inputs = keras.Input(shape=(16,))
        x = keras.layers.Dense(2048, activation='tanh')(inputs)
        x = keras.layers.Reshape((2, 2, 512))(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(
            64, kernel_size=3, padding='same', activation='tanh'
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(
            16, kernel_size=3, padding='same', activation='tanh'
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(
            1, kernel_size=3, padding='same', activation='tanh'
        )(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation('tanh')(x)

        return keras.Model(inputs, outputs, name="generator")

    def get_discriminator():
        """
        Builds and returns the discriminator model.
        """
        inputs = keras.Input(shape=(16, 16, 1))

        x = keras.layers.Conv2D(
            32, kernel_size=3, padding='same', activation='tanh'
        )(inputs)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        x = keras.layers.Conv2D(
            64, kernel_size=3, padding='same', activation='tanh'
        )(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        x = keras.layers.Conv2D(
            128, kernel_size=3, padding='same', activation='tanh'
        )(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        x = keras.layers.Conv2D(
            256, kernel_size=3, padding='same', activation='tanh'
        )(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation='tanh')(x)

        return keras.Model(inputs, outputs, name="discriminator")

    return get_generator(), get_discriminator()
