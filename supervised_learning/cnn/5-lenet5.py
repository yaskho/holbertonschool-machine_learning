#!/usr/bin/env python3
"""LeNet-5 architecture using Keras"""
from tensorflow import keras as K


def lenet5(X):
    """Builds a modified LeNet-5 model"""

    init = K.initializers.he_normal(seed=0)

    C1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    P1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(C1)

    C2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(P1)

    P2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(C2)

    F = K.layers.Flatten()(P2)

    FC1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(F)

    FC2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(FC1)

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
