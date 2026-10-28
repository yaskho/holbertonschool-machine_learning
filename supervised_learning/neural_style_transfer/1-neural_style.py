
#!/usr/bin/env python3
"""Module for neural style transfer using VGG19."""

import numpy as np
import tensorflow as tf


class NST:
    """Performs neural style transfer using VGG19."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1):
        """Initialize the neural style transfer instance."""
        if (not isinstance(style_image, np.ndarray)
                or style_image.ndim != 3
                or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray)
                or content_image.ndim != 3
                or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(alpha, (int, float, np.number))
                or alpha < 0):
            raise TypeError("alpha must be a non-negative number")

        if (not isinstance(beta, (int, float, np.number))
                or beta < 0):
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.preprocess_image(style_image)
        self.content_image = self.preprocess_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()

    @staticmethod
    def preprocess_image(image):
        """Preprocess an image for the VGG19 model."""
        image = np.expand_dims(image, axis=0)
        return tf.keras.applications.vgg19.preprocess_input(image)

    def load_model(self):
        """Load the VGG19 model and return the selected layer outputs."""
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        outputs = [
            vgg.get_layer(name).output
            for name in self.style_layers
        ]
        outputs.append(vgg.get_layer(self.content_layer).output)

        model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
        model.trainable = False

        return model
