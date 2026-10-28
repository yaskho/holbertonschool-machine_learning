#!/usr/bin/env python3
"""Neural Style Transfer"""

import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Class constructor

        style_image: numpy.ndarray (h, w, 3), the style reference image
        content_image: numpy.ndarray (h, w, 3), the content reference image
        alpha: weight for content cost
        beta: weight for style cost
        """
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
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

    @staticmethod
    def scale_image(image):
        """Rescales an image so its largest side is 512 px and its pixel
        values are in the range [0, 1]

        image: numpy.ndarray (h, w, 3)
        Returns: tensor of shape (1, h_new, w_new, 3)
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        scale = 512 / max(h, w)
        new_h = int(scale * h)
        new_w = int(scale * w)

        image = tf.image.resize(image, (new_h, new_w),
                                method=tf.image.ResizeMethod.BICUBIC)
        image = tf.clip_by_value(image / 255, 0, 1)
        return tf.expand_dims(image, axis=0)

    def load_model(self):
        """Creates the model used to calculate cost, based on VGG19.

        MaxPooling layers are replaced by AveragePooling layers, and the
        outputs are the style layers followed by the content layer.
        The model is saved in the instance attribute model.
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.trainable = False

        x = vgg.input
        outputs = {}
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name)
            x = layer(x)
            outputs[layer.name] = x

        model_outputs = [outputs[name] for name in self.style_layers]
        model_outputs.append(outputs[self.content_layer])
        self.model = tf.keras.models.Model(inputs=vgg.input,
                                           outputs=model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of a layer output

        input_layer: tf.Tensor or tf.Variable of shape (1, h, w, c)
        Returns: tf.Tensor of shape (1, c, c) with the gram matrix
        """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        shape = tf.shape(input_layer)
        nb_locations = tf.cast(shape[1] * shape[2], tf.float32)
        return gram / nb_locations

    def generate_features(self):
        """Extracts the features used to calculate neural style cost

        Sets the public instance attributes:
            gram_style_features: list of gram matrices computed from the
                style layer outputs of the style image
            content_feature: content layer output of the content image
        """
        preprocess = tf.keras.applications.vgg19.preprocess_input
        style_input = preprocess(self.style_image * 255)
        content_input = preprocess(self.content_image * 255)

        style_outputs = self.model(style_input)
        content_outputs = self.model(content_input)

        self.gram_style_features = []
        for output in style_outputs[:-1]:
            self.gram_style_features.append(self.gram_matrix(output))
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer

        style_output: tf.Tensor of shape (1, h, w, c) containing the layer
            style output of the generated image
        gram_target: tf.Tensor of shape (1, c, c), the gram matrix of the
            target style output for that layer
        Returns: the layer's style cost
        """
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                list(gram_target.shape) != [1, c, c]):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))
