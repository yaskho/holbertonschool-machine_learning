#!/usr/bin/env python3
"""Neural Style Transfer"""

import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1,
                 var=10):
        """Class constructor

        style_image: numpy.ndarray (h, w, 3), the style reference image
        content_image: numpy.ndarray (h, w, 3), the content reference image
        alpha: weight for content cost
        beta: weight for style cost
        var: weight for variational cost
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
        if not isinstance(var, (int, float)) or var < 0:
            raise TypeError("var must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var
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

    def style_cost(self, style_outputs):
        """Calculates the style cost for the generated image

        style_outputs: list of tf.Tensor style outputs for the generated
            image, one per style layer
        Returns: the style cost
        """
        length = len(self.style_layers)
        if not isinstance(style_outputs, list) or \
                len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length))

        weight = 1 / length
        cost = 0
        for style_output, gram_target in zip(style_outputs,
                                             self.gram_style_features):
            cost += weight * self.layer_style_cost(style_output, gram_target)
        return cost

    def content_cost(self, content_output):
        """Calculates the content cost for the generated image

        content_output: tf.Tensor containing the content output for the
            generated image
        Returns: the content cost
        """
        shape = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != shape):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(shape))

        return tf.reduce_mean(tf.square(content_output -
                                        self.content_feature))

    @staticmethod
    def variational_cost(generated_image):
        """Calculates the variational cost for the generated image

        generated_image: tf.Tensor of shape (1, nh, nw, 3) containing the
            generated image
        Returns: the variational cost
        """
        return tf.reduce_sum(tf.image.total_variation(generated_image))

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image

        generated_image: tf.Tensor of shape (1, nh, nw, 3) containing the
            generated image, with values in the range [0, 1]
        Returns: (J, J_content, J_style, J_var)
            J: the total cost
            J_content: the content cost
            J_style: the style cost
            J_var: the variational cost
        """
        shape = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != shape):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(shape))

        preprocess = tf.keras.applications.vgg19.preprocess_input
        outputs = self.model(preprocess(generated_image * 255))

        J_content = self.content_cost(outputs[-1])
        J_style = self.style_cost(outputs[:-1])
        J_var = self.variational_cost(generated_image)
        J = self.alpha * J_content + self.beta * J_style + self.var * J_var
        return J, J_content, J_style, J_var

    def compute_grads(self, generated_image):
        """Calculates the gradients for the generated image

        generated_image: tf.Tensor or tf.Variable of shape (1, nh, nw, 3)
            containing the generated image
        Returns: gradients, J_total, J_content, J_style, J_var
            gradients: tf.Tensor with the gradients of the total cost with
                respect to the generated image
            J_total: the total cost
            J_content: the content cost
            J_style: the style cost
            J_var: the variational cost
        """
        shape = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != shape):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(shape))

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style, J_var = self.total_cost(
                generated_image)

        gradients = tape.gradient(J_total, generated_image)
        return gradients, J_total, J_content, J_style, J_var

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generates the neural style transferred image

        iterations: number of iterations to perform gradient descent over
        step: if not None, the step at which to print information about
            the training, including the final iteration
        lr: learning rate for gradient descent
        beta1: beta1 parameter for Adam optimization
        beta2: beta2 parameter for Adam optimization
        Returns: generated_image, cost
            generated_image: the best generated image
            cost: the best cost
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError(
                    "step must be positive and less than iterations")
        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                             beta_1=beta1,
                                             beta_2=beta2)
        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            grads, J_total, J_content, J_style, J_var = self.compute_grads(
                generated_image)

            if J_total < best_cost:
                best_cost = float(J_total)
                best_image = generated_image.numpy()

            if step is not None and (i % step == 0 or i == iterations):
                print("Cost at iteration {}: {}, content {}, style {}, "
                      "var {}".format(i, float(J_total), float(J_content),
                                      float(J_style), float(J_var)))

            optimizer.apply_gradients([(grads, generated_image)])
            generated_image.assign(tf.clip_by_value(generated_image, 0, 1))

        return best_image[0], best_cost
