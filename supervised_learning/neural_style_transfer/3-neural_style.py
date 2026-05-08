#!/usr/bin/env python3
"""
Module containing the NST class for Neural Style Transfer
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    NST class to perform Neural Style Transfer using VGG-19
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block4_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for Neural Style Transfer

        Args:
            style_image: image used as style reference (np.ndarray)
            content_image: image used as content reference (np.ndarray)
            alpha: weight for content cost
            beta: weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
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
        """
        Rescales image pixels to [0, 1] and resizes to max dimension 512
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            h_new, w_new = 512, int(w * (512 / h))
        else:
            h_new, w_new = int(h * (512 / w)), 512

        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize(image, (h_new, w_new), method='bicubic')
        image = image / 255.0
        image = tf.clip_by_value(image, 0, 1)

        return image

    def load_model(self):
        """
        Loads the VGG19 model and sets the model attribute
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')

        # Set specific layers to non-trainable
        vgg.trainable = False

        style_outputs = [vgg.get_layer(name).output
                         for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_output]

        self.model = tf.keras.models.Model(vgg.input, model_outputs)

    @staticmethod
    def gram_matrix(input_tensor):
        """
        Calculates the gram matrix of a specific layer output
        """
        if not isinstance(input_tensor, (tf.Tensor, tf.Variable)) or \
           len(input_tensor.shape) != 4:
            raise TypeError("input_tensor must be a tensor of rank 4")

        # G = A * A_transpose
        # Einstein summation for matrix multiplication over the spatial dims
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return result / num_locations

    def generate_features(self):
        """
        Extracts features used to calculate neural style cost
        """
        # Preprocess images specifically for VGG19 (expected range [-1, 1])
        pre_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        pre_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        # Get style features from style image
        style_outputs = self.model(pre_style)[:-1]
        self.gram_style_features = [self.gram_matrix(style)
                                    for style in style_outputs]

        # Get content feature from content image
        content_output = self.model(pre_content)[-1]
        self.content_feature = content_output
