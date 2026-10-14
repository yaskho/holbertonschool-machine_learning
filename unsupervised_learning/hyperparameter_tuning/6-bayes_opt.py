Here is the complete implementation for 6-bayes_opt.py, followed by a complete blog post draft ready to publish on Medium or LinkedIn.

Python Script (6-bayes_opt.py)
Python
#!/usr/bin/env python3
"""
Bayesian Optimization with GPyOpt
"""
import GPyOpt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, regularizers


def load_data():
    """
    Loads and preprocesses the MNIST dataset

    Returns:
        X_train, Y_train, X_val, Y_val preprocessed data arrays
    """
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_val = x_val.reshape(-1, 784).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    return x_train, y_train, x_val, y_val


X_train, Y_train, X_val, Y_val = load_data()


def build_and_train_model(x_sample):
    """
    Builds, trains, and evaluates a neural network based on hyperparameters

    Parameters:
        x_sample: 2D numpy array of hyperparameter values [1, 5]

    Returns:
        Validation loss (float) as the objective value to minimize
    """
    lr = float(x_sample[0, 0])
    num_units = int(x_sample[0, 1])
    dropout_rate = float(x_sample[0, 2])
    l2_reg = float(x_sample[0, 3])
    batch_size = int(x_sample[0, 4])

    model = models.Sequential([
        layers.Dense(
            num_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            input_shape=(784,)
        ),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    chkpt_filename = (
        f"chkpt_lr{lr:.4f}_units{num_units}_"
        f"drop{dropout_rate:.2f}_l2{l2_reg:.5f}_bs{batch_size}.h5"
    )

    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            filepath=chkpt_filename,
            monitor='val_loss',
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=15,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=0
    )

    val_loss = float(min(history.history['val_loss']))
    return val_loss


def optimize_hyperparameters():
    """
    Performs Bayesian Optimization using GPyOpt on 5 hyperparameters
    """
    domain = [
        {'name': 'learning_rate', 'type': 'continuous',
         'domain': (0.0001, 0.01)},
        {'name': 'num_units', 'type': 'discrete',
         'domain': (32, 64, 128, 256)},
        {'name': 'dropout_rate', 'type': 'continuous',
         'domain': (0.1, 0.5)},
        {'name': 'l2_reg', 'type': 'continuous',
         'domain': (0.00001, 0.001)},
        {'name': 'batch_size', 'type': 'discrete',
         'domain': (32, 64, 128)}
    ]

    optimizer = GPyOpt.methods.BayesianOptimization(
        f=build_and_train_model,
        domain=domain,
        acquisition_type='EI',
        exact_feval=True
    )

    optimizer.run_optimization(max_iter=30)
    optimizer.plot_convergence('bayes_opt_convergence.png')

    best_x = optimizer.x_opt
    best_fx = optimizer.fx_opt

    report = (
        "Bayesian Optimization Report\n"
        "============================\n"
        f"Best Validation Loss: {best_fx:.5f}\n"
        f"Optimal Learning Rate: {best_x[0]:.6f}\n"
        f"Optimal Number of Units: {int(best_x[1])}\n"
        f"Optimal Dropout Rate: {best_x[2]:.4f}\n"
        f"Optimal L2 Regularization Weight: {best_x[3]:.6f}\n"
        f"Optimal Batch Size: {int(best_x[4])}\n"
    )

    with open('bayes_opt.txt', 'w') as f:
        f.write(report)


if __name__ == '__main__':
    optimize_hyperparameters()
