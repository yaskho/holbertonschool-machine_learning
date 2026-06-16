#!/usr/bin/env python3
"""Convolutional Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): shape of input (H, W, C)
        filters (list): filters for encoder conv layers
        latent_dims (tuple): shape of latent representation

    Returns:
        encoder, decoder, auto
    """

    # =======================
    # Encoder
    # =======================
    input_layer = keras.Input(shape=input_dims)
    x = input_layer

    for f in filters:
        x = keras.layers.Conv2D(
            f,
            (3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = keras.Model(inputs=input_layer, outputs=x)

    # =======================
    # Decoder
    # =======================
    latent_input = keras.Input(shape=latent_dims)
    x = latent_input

    reversed_filters = filters[::-1]

    # First decoder layers (all except last two convs)
    for f in reversed_filters[:-2]:
        x = keras.layers.Conv2D(
            f,
            (3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    # Second to last convolution (valid padding, no upsampling)
    x = keras.layers.Conv2D(
        reversed_filters[-2],
        (3, 3),
        padding='valid',
        activation='relu'
    )(x)

    # Last convolution (output layer)
    x = keras.layers.Conv2D(
        input_dims[2],
        (3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(inputs=latent_input, outputs=x)

    # =======================
    # Autoencoder
    # =======================
    encoded = encoder(input_layer)
    decoded = decoder(encoded)

    auto = keras.Model(inputs=input_layer, outputs=decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
