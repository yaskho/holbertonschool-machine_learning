# Databricks and MLflow: Diabetes Prediction with a Neural Network

This project is my answer to the task **"0. Databricks"** of the ML life cycle project. I trained a neural network on Databricks, tracked every run with **MLflow**, and packaged the trained model as an **MLflow Model**.

## Summary

| Requirement | What I did |
|---|---|
| Pick a model | A neural network (multi-layer perceptron) that predicts diabetes from the Pima Indians Diabetes dataset |
| New MLflow experiment | Experiment `diabetes-neural-network`, created from the notebook with `mlflow.set_experiment` |
| New cluster | Serverless compute (see the note below) |
| Add data | `diabetes.csv` uploaded to the Unity Catalog volume `workspace.default.diabetes_data` |
| Track parameters and metrics | Three runs, each logging its parameters, the training loss at every iteration, and test metrics |
| MLflow Model | The full pipeline is logged with a signature and an input example, reloaded with `mlflow.pyfunc`, and registered in Unity Catalog as `workspace.default.diabetes_neural_network` (version 1) |

## Note on the environment: Free Edition instead of Community Edition

The task asks for the Databricks Community Edition, but Databricks has replaced it with **Databricks Free Edition**. Free Edition runs on **serverless compute only** and does not allow creating custom clusters, so I used serverless compute in place of a manually created cluster (screenshot 1). Free Edition also restricts direct access to the DBFS root, so the data is stored in a Unity Catalog **volume** and read from a `/Volumes/...` path.

## The model

The model is one scikit-learn pipeline, so the preprocessing is packaged together with the network:

1. **Median imputation** of impossible zeros in glucose, blood pressure, skin thickness, insulin and BMI (zeros mean "missing" in this dataset). It is fitted on the training set only.
2. **Standard scaling** of all features.
3. **Neural network** (`MLPClassifier`) trained with the Adam optimizer.

The data is split 80% / 20% (stratified, `random_state=42`). Three configurations were trained and compared:

| Run | Hidden layers | Initial learning rate | L2 penalty (alpha) |
|---|---|---|---|
| `mlp_8_lr0.001` | (8) | 0.001 | 0.0001 |
| `mlp_16-8_lr0.001` | (16, 8) | 0.001 | 0.0001 |
| `mlp_32-16_lr0.005` | (32, 16) | 0.005 | 0.001 |

## What MLflow tracks

- **Parameters:** hidden layers, initial learning rate, alpha, maximum iterations, test size, random state, zero imputation method.
- **Metrics:** `train_loss` (at every iteration), `train_accuracy`, `test_accuracy`, `test_precision`, `test_recall`, `test_f1`, `test_roc_auc`, `n_iterations`.
- **Artifacts:** `loss_curve.png`, `confusion_matrix.png`, and the packaged `model` (with its `MLmodel` file, signature and input example).
- **Tags:** project name, model type, dataset.

### steps:

### 1. Compute


### 2. Data


### 3. Training output


### 4. Comparison of the runs


### 5. MLflow experiment


### 6. One run in detail


### 7. Artifacts and MLflow Model


### 8. Registered model


### 9. Packaged model in use


## How to reproduce

1. Create a Databricks Free Edition account.
2. In **Catalog**, open `workspace` then `default`, and create a **volume** named `diabetes_data`. Upload `diabetes.csv` to it.
3. In **Workspace**, import `databricks_mlflow_diabetes.py` (it becomes a notebook with its cells already split).
4. Attach the notebook to **Serverless** compute and click **Run all**.
5. Open **Experiments**, then `diabetes-neural-network`, to see the runs, and **Catalog**, then **Models**, to see the registered model.

If your volume has another name, edit the `DATA_PATH` line in the notebook.

## Files

- `databricks_mlflow_diabetes.py`: the Databricks notebook (source format).
- `screenshots/`: the screenshots shown above.

## What I learned

- The ML life cycle does not stop at training: tracking parameters and metrics makes each experiment comparable and reproducible.
- Packaging the preprocessing together with the model means anyone can load it and predict from raw values.
- Platforms change: Community Edition was replaced by Free Edition, so my setup (serverless compute, volumes instead of DBFS) differs from the original task description.
- A "file not found" error was simply a missing upload to the volume, a reminder to check data paths first.