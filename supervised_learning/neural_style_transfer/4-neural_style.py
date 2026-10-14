#!/usr/bin/env python3
import tensorflow as tf
import numpy as np


class NST:
    """Neural Style Transfer class"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(alpha, (int, float))) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if (not isinstance(beta, (int, float))) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        self.alpha = alpha
        self.beta = beta

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales image to max size 512 and normalizes to [0,1]"""
        if (not isinstance(image, np.ndarray)
                or image.shape[-1] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w = image.shape[:2]
        max_dim = 512

        if h > w:
            new_h = max_dim
            new_w = int(w * max_dim / h)
        else:
            new_w = max_dim
            new_h = int(h * max_dim / w)

        image = tf.image.resize(
            image,
            (new_h, new_w),
            method='bicubic'
        )

        image = image / 255.0
        image = tf.expand_dims(image, axis=0)

        return tf.cast(image, tf.float32)

    def load_model(self):
        """Build VGG19 model"""
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        outputs = [
            vgg.get_layer(name).output
            for name in self.style_layers + [self.content_layer]
        ]

        self.model = tf.keras.Model(
            inputs=vgg.input,
            outputs=outputs
        )
        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """Computes Gram matrix"""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError(
                "input_layer must be a tensor of rank 4"
            )

        _, h, w, c = input_layer.shape

        features = tf.reshape(input_layer, (-1, c))
        gram = tf.matmul(features, features, transpose_a=True)

        return tf.expand_dims(
            gram / tf.cast(h * w, tf.float32),
            axis=0
        )

    def generate_features(self):
        """Extract style and content features"""
        style_output = self.model(self.style_image)
        content_output = self.model(self.content_image)

        self.gram_style_features = [
            self.gram_matrix(style_output[i])
            for i in range(len(self.style_layers))
        ]

        self.content_feature = content_output[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Computes style cost for a single layer"""
        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError(
                "style_output must be a tensor of rank 4"
            )

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or len(gram_target.shape) != 3):
            c = style_output.shape[-1]
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]"
            )

        _, h, w, c = style_output.shape

        gram_style = self.gram_matrix(style_output)

        # Correct normalized style cost
        cost = tf.reduce_mean(tf.square(gram_style - gram_target))

        return cost
