#!/usr/bin/env python3
"""Classify CIFAR-10 images using transfer learning."""

from tensorflow import keras as K


def preprocess_data(X, Y):
    """Preprocess CIFAR-10 images and labels."""
    X_p = X.astype('float32')
    Y_p = K.utils.to_categorical(Y.reshape(-1), 10)
    return X_p, Y_p


def build_model():
    """Build a classifier using a frozen ResNet50V2 application."""
    inputs = K.Input(shape=(32, 32, 3))

    resized = K.layers.Lambda(
        lambda x: K.image.resize(x, (224, 224)),
        name='resize_images'
    )(inputs)

    processed = K.layers.Lambda(
        lambda x: K.applications.resnet_v2.preprocess_input(x),
        name='preprocess_images'
    )(resized)

    base = K.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    base.trainable = False

    features = base(processed, training=False)
    features = K.layers.GlobalAveragePooling2D()(features)

    classifier = K.Sequential([
        K.layers.Dense(512, activation='relu'),
        K.layers.Dropout(0.4),
        K.layers.Dense(10, activation='softmax')
    ], name='classifier')

    outputs = classifier(features)
    model = K.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base, classifier


def train_model():
    """Train the classification head and save the compiled model."""
    K.utils.set_random_seed(42)

    (X_train, Y_train), (X_test, Y_test) = (
        K.datasets.cifar10.load_data()
    )

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    model, base, classifier = build_model()

    # Cache the frozen application's features only once.
    feature_model = K.Model(
        inputs=model.input,
        outputs=model.layers[-1].input
    )

    train_features = feature_model.predict(
        X_train, batch_size=128, verbose=1
    )
    test_features = feature_model.predict(
        X_test, batch_size=128, verbose=1
    )

    # Train the classifier directly on the cached features.
    classifier.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    classifier.fit(
        train_features,
        Y_train,
        batch_size=128,
        epochs=40,
        validation_split=0.1,
        callbacks=[
            K.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=6,
                restore_best_weights=True
            ),
            K.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ],
        verbose=1
    )

    # Evaluate the complete model on the held-out test set.
    model.evaluate(X_test, Y_test, batch_size=128, verbose=1)

    # Save the complete, compiled model, including preprocessing.
    model.save('cifar10.h5')


if __name__ == '__main__':
    train_model()
