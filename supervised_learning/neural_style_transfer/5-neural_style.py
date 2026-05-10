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

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1):
        """Initialize NST"""

        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or
                content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        if (not isinstance(alpha, (int, float))
                or alpha < 0):
            raise TypeError(
                "alpha must be a non-negative number"
            )

        if (not isinstance(beta, (int, float))
                or beta < 0):
            raise TypeError(
                "beta must be a non-negative number"
            )

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        self.alpha = alpha
        self.beta = beta

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescale image"""

        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        h = image.shape[0]
        w = image.shape[1]

        scale = 512 / max(h, w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        image = tf.image.resize(
            image,
            (new_h, new_w),
            method=tf.image.ResizeMethod.BICUBIC
        )

        image = image / 255.0

        image = tf.clip_by_value(image, 0, 1)

        image = tf.expand_dims(image, axis=0)

        return image

    def load_model(self):
        """Loads VGG19 model"""

        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg.trainable = False

        outputs = [vgg.get_layer(name).output
                   for name in self.style_layers]

        outputs.append(
            vgg.get_layer(self.content_layer).output
        )

        self.model = tf.keras.models.Model(
            inputs=vgg.input,
            outputs=outputs
        )

        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates gram matrix"""

        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError(
                "input_layer must be a tensor of rank 4"
            )

        _, h, w, c = input_layer.shape

        tensor = tf.reshape(input_layer, [h * w, c])

        gram = tf.matmul(
            tensor,
            tensor,
            transpose_a=True
        )

        gram = gram / tf.cast(h * w, tf.float32)

        return tf.expand_dims(gram, axis=0)

    def generate_features(self):
        """Extract style and content features"""

        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )

        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)

        self.gram_style_features = [
            self.gram_matrix(style_output)
            for style_output in style_outputs[:-1]
        ]

        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculates style cost for one layer"""

        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError(
                "style_output must be a tensor of rank 4"
            )

        _, _, _, c = style_output.shape

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != (1, c, c)):
            raise TypeError(
                "gram_target must be a tensor of shape "
                "[1, {}, {}]".format(c, c)
            )

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_mean(
            tf.square(gram_style - gram_target)
        )

    def style_cost(self, style_outputs):
        """Calculates total style cost"""

        if (not isinstance(style_outputs, list) or
                len(style_outputs) != len(self.style_layers)):
            raise TypeError(
                "style_outputs must be a list with a length of {}"
                .format(len(self.style_layers))
            )

        weight = 1 / len(self.style_layers)

        style_cost = tf.constant(0.0)

        for i in range(len(style_outputs)):
            layer_cost = self.layer_style_cost(
                style_outputs[i],
                self.gram_style_features[i]
            )

            style_cost += weight * layer_cost

        return style_cost
