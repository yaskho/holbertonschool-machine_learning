#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class"""

    style_layers = [
        'block1_conv1', 'block2_conv1',
        'block3_conv1', 'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1):

        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")

        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        self.alpha = alpha
        self.beta = beta

        self.load_model()
        self.generate_features()

    def load_model(self):
        """Load VGG19 model"""
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        outputs = [
            vgg.get_layer(name).output
            for name in self.style_layers + [self.content_layer]
        ]

        self.model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

    def generate_features(self):
        """Extract style and content features"""

        # IMPORTANT FIX: ensure tensors are fed properly
        style_outputs = self.model(tf.convert_to_tensor(self.style_image))
        content_outputs = self.model(tf.convert_to_tensor(self.content_image))

        self.gram_style_features = [
            self.gram_matrix(style_outputs[i])
            for i in range(len(self.style_layers))
        ]

        self.content_feature = content_outputs[-1]

    @staticmethod
    def gram_matrix(input_layer):
        """Compute Gram matrix"""

        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape

        reshaped = tf.reshape(input_layer, (-1, c))

        gram = tf.matmul(reshaped, reshaped, transpose_a=True)

        return tf.expand_dims(
            gram / tf.cast(h * w, tf.float32),
            0
        )
