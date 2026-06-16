#!/usr/bin/env python3
"""Sparse Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder.

    Args:
        input_dims (int): input dimension
        hidden_layers (list): encoder hidden layers sizes
        latent_dims (int): latent space dimension
        lambtha (float): L1 regularization parameter

    Returns:
        encoder, decoder, auto
    """

    # Input
    input_layer = keras.Input(shape=(input_dims,))

    # Encoder
    x = input_layer
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    # Latent layer with L1 regularization (sparsity constraint)
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)

    encoder = keras.Model(inputs=input_layer, outputs=latent)

    # Decoder input
    latent_input = keras.Input(shape=(latent_dims,))

    x = latent_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    output = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(inputs=latent_input, outputs=output)

    # Full autoencoder
    encoded = encoder(input_layer)
    decoded = decoder(encoded)

    auto = keras.Model(inputs=input_layer, outputs=decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
