
def load_model(self):
    """Load the VGG19 model for neural style transfer."""
    vgg = tf.keras.applications.VGG19(
        include_top=False,
        weights='imagenet'
    )

    vgg.trainable = False

    # Rebuild VGG19 using average pooling instead of max pooling.
    inputs = tf.keras.Input(shape=(None, None, 3))
    x = inputs
    outputs = {}

    for layer in vgg.layers[1:]:
        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            x = tf.keras.layers.AveragePooling2D(
                pool_size=layer.pool_size,
                strides=layer.strides,
                padding=layer.padding,
                name=layer.name
            )(x)
        else:
            x = layer(x)

        outputs[layer.name] = x

    selected_outputs = [
        outputs[name] for name in self.style_layers
    ]
    selected_outputs.append(outputs[self.content_layer])

    self.model = tf.keras.Model(
        inputs=inputs,
        outputs=selected_outputs
    )
    self.model.trainable = False
