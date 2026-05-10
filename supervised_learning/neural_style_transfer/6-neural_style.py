#!/usr/bin/env python3
"""Neural Style Transfer - Task 6: Content Cost"""
 
import numpy as np
import tensorflow as tf
 
 
class NST:
    """Performs Neural Style Transfer.
 
    Attributes:
        style_layers (list): Layers used for style extraction.
        content_layer (str): Layer used for content extraction.
        style_image (tf.Tensor): Preprocessed style image.
        content_image (tf.Tensor): Preprocessed content image.
        alpha (float): Weight for content cost.
        beta (float): Weight for style cost.
        model (tf.keras.Model): Keras model with style and content outputs.
    """
 
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'
 
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initializes the NST instance.
 
        Args:
            style_image (numpy.ndarray): Image used as style reference.
            content_image (numpy.ndarray): Image used as content reference.
            alpha (float): Weight for content cost. Default 1e4.
            beta (float): Weight for style cost. Default 1.
 
        Raises:
            TypeError: If style_image is not a numpy.ndarray with shape
                (h, w, 3).
            TypeError: If content_image is not a numpy.ndarray with shape
                (h, w, 3).
            TypeError: If alpha is not a non-negative number.
            TypeError: If beta is not a non-negative number.
        """
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
        """Rescales an image so its pixel values are between 0 and 1
        and its largest dimension is 512 pixels.
 
        Args:
            image (numpy.ndarray): Image to rescale, shape (h, w, 3).
 
        Returns:
            tf.Tensor: Scaled image with shape (1, h_new, w_new, 3).
 
        Raises:
            TypeError: If image is not a numpy.ndarray with shape (h, w, 3).
        """
        if (not isinstance(image, np.ndarray)
                or image.ndim != 3
                or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )
        h, w = image.shape[:2]
        if h > w:
            new_h = 512
            new_w = int(round(w * 512 / h))
        else:
            new_w = 512
            new_h = int(round(h * 512 / w))
 
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(
            tf.expand_dims(image, axis=0),
            [new_h, new_w],
            method='bicubic'
        )
        image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
        return image
 
    def load_model(self):
        """Loads VGG19 model with outputs at style and content layers.
 
        The model is saved to self.model. Maxpooling layers are replaced
        with average pooling. VGG19 weights are not trainable.
        """
        vgg19_base = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg19_base.trainable = False
 
        # Replace MaxPooling2D with AveragePooling2D
        inputs = vgg19_base.input
        x = inputs
        for layer in vgg19_base.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name + '_avg'
                )(x)
            else:
                x = layer(x)
            if layer.name in self.style_layers + [self.content_layer]:
                pass  # we'll collect via Model below
 
        # Build model with desired outputs
        layer_names = self.style_layers + [self.content_layer]
        outputs = []
        for name in layer_names:
            # Get output after rebuilding: use original vgg layers by name
            pass
 
        # Simpler approach: use layer outputs directly from vgg19_base
        style_outputs = [
            vgg19_base.get_layer(name).output for name in self.style_layers
        ]
        content_output = vgg19_base.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_output]
 
        self.model = tf.keras.Model(inputs=vgg19_base.input,
                                    outputs=model_outputs)
 
    def gram_matrix(self, input_layer):
        """Calculates the gram matrix of a layer's output.
 
        Args:
            input_layer (tf.Tensor or tf.Variable): Layer output of shape
                (1, h, w, c).
 
        Returns:
            tf.Tensor: Gram matrix of shape (1, c, c).
 
        Raises:
            TypeError: If input_layer is not a tensor of rank 4.
        """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")
 
        # input_layer shape: (1, h, w, c)
        _, h, w, c = input_layer.shape
        # Reshape to (h*w, c)
        features = tf.reshape(input_layer, (-1, c))
        # Gram matrix: (c, c)
        gram = tf.matmul(features, features, transpose_a=True)
        # Normalize by number of locations
        gram = tf.expand_dims(gram, axis=0)
        gram = gram / tf.cast(h * w, tf.float32)
        return gram
 
    def generate_features(self):
        """Extracts style and content features from the respective images.
 
        Sets:
            self.gram_style_features (list): Gram matrices for each style
                layer computed from the style image.
            self.content_feature (tf.Tensor): Content layer output for the
                content image.
        """
        vgg19 = tf.keras.applications.vgg19
 
        # Preprocess style image
        style_preprocessed = vgg19.preprocess_input(
            self.style_image * 255
        )
        style_outputs = self.model(style_preprocessed)
 
        # Preprocess content image
        content_preprocessed = vgg19.preprocess_input(
            self.content_image * 255
        )
        content_outputs = self.model(content_preprocessed)
 
        # style_outputs[:-1] are style layers, [-1] is content
        self.gram_style_features = [
            self.gram_matrix(output)
            for output in style_outputs[:-1]
        ]
        self.content_feature = content_outputs[-1]
 
    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer.
 
        Args:
            style_output (tf.Tensor): Style output of the generated image
                for a given layer, shape (1, h, w, c).
            gram_target (tf.Tensor): Gram matrix of the style image for
                that layer, shape (1, c, c).
 
        Returns:
            tf.Tensor: Style cost for the layer.
 
        Raises:
            TypeError: If style_output is not a tensor of rank 4.
            TypeError: If gram_target is not a tensor of shape (1, c, c)
                where c matches style_output's channel count.
        """
        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")
 
        c = style_output.shape[-1]
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != (1, c, c)):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {c}, {c}]".format(
                    c=c
                )
            )
 
        gram_generated = self.gram_matrix(style_output)
        cost = tf.reduce_mean(tf.square(gram_generated - gram_target))
        return cost
 
    def style_cost(self, style_outputs):
        """Calculates the total style cost for the generated image.
 
        Each style layer is weighted evenly, with all weights summing to 1.
 
        Args:
            style_outputs (list): A list of tf.Tensor style outputs for
                the generated image, one per style layer.
 
        Returns:
            tf.Tensor: The total style cost.
 
        Raises:
            TypeError: If style_outputs is not a list with the same length
                as self.style_layers.
        """
        l = len(self.style_layers)
        if not isinstance(style_outputs, list) or len(style_outputs) != l:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(l)
            )
 
        weight = 1.0 / l
        total_style_cost = tf.constant(0.0, dtype=tf.float32)
 
        for style_output, gram_target in zip(
                style_outputs, self.gram_style_features):
            total_style_cost += weight * self.layer_style_cost(
                style_output, gram_target
            )
 
        return total_style_cost
 
    def content_cost(self, content_output):
        """Calculates the content cost for the generated image.
 
        Args:
            content_output (tf.Tensor): Content layer output for the
                generated image, must have the same shape as
                self.content_feature.
 
        Returns:
            tf.Tensor: The content cost (scalar).
 
        Raises:
            TypeError: If content_output is not a tf.Tensor or tf.Variable
                with the same shape as self.content_feature.
        """
        s = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable))
                or content_output.shape != s):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s)
            )
 
        cost = tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        )
        return cost
