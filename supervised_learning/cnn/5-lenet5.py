#!/usr/bin/env python3
"""LeNet-5 architecture using Keras"""
import tensorflow as tf
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 model

    Args:
        X: K.Input of shape (m, 28, 28, 1)

    Returns:
        compiled K.Model
    """

    init = K.initializers.he_normal(seed=0)

    # 1st Convolutional layer
    C1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    # 1st Pooling layer
    P1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(C1)

    # 2nd Convolutional layer
    C2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(P1)

    # 2nd Pooling layer
    P2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(C2)

    # Flatten
    F = K.layers.Flatten()(P2)

    # Fully connected layer 1
    FC1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(F)

    # Fully connected layer 2
    FC2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(FC1)

    # Output layer
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
    )(FC2)

    model = K.Model(inputs=X, outputs=output)

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
