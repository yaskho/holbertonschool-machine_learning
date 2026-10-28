#!/usr/bin/env python3
"""Module for performing Bayesian Optimization using GPyOpt."""
import GPyOpt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def load_data():
    """
    Loads and preprocesses the MNIST dataset for training and validation.

    Returns:
        tuple: (X_train, Y_train), (X_val, Y_val)
    """
    (X_train, Y_train), (X_val, Y_val) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
    X_val = X_val.reshape(-1, 784).astype('float32') / 255.0
    Y_train = tf.keras.utils.to_categorical(Y_train, 10)
    Y_val = tf.keras.utils.to_categorical(Y_val, 10)
    return (X_train, Y_train), (X_val, Y_val)


# Global dataset variables
(X_train, Y_train), (X_val, Y_val) = load_data()


def build_and_train_model(bounds_vector):
    """
    Objective function for GPyOpt. Builds, trains, and evaluates model.

    Args:
        bounds_vector (numpy.ndarray): 2D array of shape (1, 5) containing:
            - learning_rate: float
            - num_units: float (cast to int)
            - dropout_rate: float
            - l2_reg: float
            - batch_size: float (cast to int power of 2)

    Returns:
        float: Loss value to be minimized (1 - validation accuracy).
    """
    learning_rate = float(bounds_vector[0, 0])
    num_units = int(bounds_vector[0, 1])
    dropout_rate = float(bounds_vector[0, 2])
    l2_reg = float(bounds_vector[0, 3])
    batch_size = int(bounds_vector[0, 4])

    # Construct model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            num_units,
            activation='relu',
            input_shape=(784,),
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        ),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Checkpoint filename encoding hyperparameter configuration
    chkpt_name = (
        f"ckpt_lr{learning_rate:.4f}_units{num_units}_"
        f"drop{dropout_rate:.2f}_l2{l2_reg:.4f}_bs{batch_size}.h5"
    )

    # Callbacks: Early Stopping & Best Checkpoint Saver
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=chkpt_name,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=5,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    # Return satisficing objective metric to minimize (1 - best val_accuracy)
    best_val_acc = max(history.history['val_accuracy'])
    return 1.0 - best_val_acc


def optimize_model():
    """
    Configures and runs Bayesian Optimization using GPyOpt across 5 params.
    """
    # Define hyperparameter search domain
    bounds = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
        {'name': 'num_units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
        {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
    ]

    # Initialize GPyOpt Bayesian Optimizer
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=build_and_train_model,
        domain=bounds,
        acquisition_type='EI',
        exact_feval=True
    )

    # Run optimization for maximum 30 iterations
    optimizer.run_optimization(max_iter=30)

    # Plot convergence
    optimizer.plot_convergence(filename='convergence_plot.png')

    # Save summary report to 'bayes_opt.txt'
    with open('bayes_opt.txt', 'w') as f:
        f.write("--- Bayesian Optimization Report ---\n")
        f.write(f"Optimized Hyperparameters (x_opt):\n")
        f.write(f"  Learning Rate: {optimizer.x_opt[0]:.6f}\n")
        f.write(f"  Number of Units: {int(optimizer.x_opt[1])}\n")
        f.write(f"  Dropout Rate: {optimizer.x_opt[2]:.4f}\n")
        f.write(f"  L2 Regularization Weight: {optimizer.x_opt[3]:.6f}\n")
        f.write(f"  Batch Size: {int(optimizer.x_opt[4])}\n\n")
        f.write(f"Best Satisficing Loss (1 - val_acc): {optimizer.fx_opt:.6f}\n")
        f.write(f"Best Validation Accuracy: {(1.0 - optimizer.fx_opt):.4%}\n")


if __name__ == '__main__':
    optimize_model()
