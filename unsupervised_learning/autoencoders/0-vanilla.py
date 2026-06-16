#!/usr/bin/env python3
"""Vanilla Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder.

    Args:
        input_dims (int): dimension of input data
        hidden_layers (list): number of nodes in encoder hidden layers
        latent_dims (int): dimension of latent space

    Returns:
        encoder: encoder model
        decoder: decoder model
        auto: full autoencoder model
    """

    # Input layer
    input_layer = keras.Input(shape=(input_dims,))

    # Encoder
    x = input_layer
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    latent = keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = keras.Model(inputs=input_layer, outputs=latent)

    # Decoder
    latent_input = keras.Input(shape=(latent_dims,))

    x = latent_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    output = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(inputs=latent_input, outputs=output)

    # Autoencoder
    auto_input = input_layer
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    auto = keras.Model(inputs=auto_input, outputs=decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
