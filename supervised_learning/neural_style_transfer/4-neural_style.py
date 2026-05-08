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

        # Validate inputs
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

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.alpha = alpha
        self.beta = beta

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        self.load_model()
        self.generate_features()

    # =========================
    # MODEL
    # =========================
    def load_model(self):
        """Load VGG19 model for feature extraction"""
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

    # =========================
    # FEATURES
    # =========================
    def generate_features(self):
        """Extract style and content features"""

        style = tf.cast(self.style_image, tf.float32)
        content = tf.cast(self.content_image, tf.float32)

        outputs = self.model(style)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        self.gram_style_features = [
            self.gram_matrix(style_outputs[i])
            for i in range(len(self.style_layers))
        ]

        self.content_feature = content_output

    # =========================
    # GRAM MATRIX
    # =========================
    @staticmethod
    def gram_matrix(input_layer):
        """Compute Gram matrix"""

        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError(
                "input_layer must be a tensor of rank 4"
            )

        _, h, w, c = input_layer.shape

        reshaped = tf.reshape(input_layer, (-1, c))

        gram = tf.matmul(reshaped, reshaped, transpose_a=True)

        return tf.expand_dims(
            gram / tf.cast(h * w, tf.float32),
            0
        )

    # =========================
    # STYLE COST (TASK 4)
    # =========================
    def layer_style_cost(self, style_output, gram_target):
        """Calculates style cost for a single layer"""

        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError(
                "style_output must be a tensor of rank 4"
            )

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or len(gram_target.shape) != 3
                or gram_target.shape[0] != 1):
            raise TypeError(
                f"gram_target must be a tensor of shape "
                f"[1, {style_output.shape[-1]}, {style_output.shape[-1]}]"
            )

        _, _, _, c = style_output.shape

        gram_gen = self.gram_matrix(style_output)

        diff = gram_gen - gram_target

        cost = tf.reduce_sum(tf.square(diff))

        return cost / tf.cast(c ** 2, tf.float32)

    # =========================
    # IMAGE SCALING
    # =========================
    @staticmethod
    def scale_image(image):
        """Rescales image to max 512px and normalizes to [0,1]"""

        if (not isinstance(image, np.ndarray)
                or image.shape[-1] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w = image.shape[:2]

        max_side = 512
        scale = max_side / max(h, w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        image = tf.convert_to_tensor(image, dtype=tf.float32)

        image = tf.image.resize(
            image,
            (new_h, new_w),
            method='bicubic'
        )

        image = image / 255.0

        return image[tf.newaxis, ...]
