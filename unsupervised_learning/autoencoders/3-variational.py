#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): dimension of input data
        hidden_layers (list): encoder hidden layers sizes
        latent_dims (int): dimension of latent space

    Returns:
        encoder, decoder, auto
    """

    # ======================
    # Encoder
    # ======================
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    # Mean and log variance
    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_sigma = keras.layers.Dense(latent_dims, activation=None)(x)

    # Reparameterization trick
    def sampling(args):
        mu, log_sigma = args
        epsilon = keras.backend.random_normal(shape=(keras.backend.shape(mu)[0],
                                                    latent_dims))
        return mu + keras.backend.exp(log_sigma / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mu, log_sigma])

    encoder = keras.Model(inputs, [z, mu, log_sigma])

    # ======================
    # Decoder
    # ======================
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, outputs)

    # ======================
    # Autoencoder
    # ======================
    outputs = decoder(encoder(inputs)[0])

    auto = keras.Model(inputs, outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
