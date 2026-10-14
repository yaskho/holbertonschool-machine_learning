# Databricks notebook source
# MAGIC %md
# MAGIC # Diabetes prediction with a neural network, tracked with MLflow
# MAGIC
# MAGIC This notebook trains a small **neural network** (a multi-layer perceptron) on the Pima Indians Diabetes dataset
# MAGIC and uses **MLflow** to:
# MAGIC
# MAGIC 1. create an **experiment**,
# MAGIC 2. track the **parameters** and **metrics** of every run (including the loss at every iteration),
# MAGIC 3. save plots as **artifacts**,
# MAGIC 4. package the trained pipeline as an **MLflow Model**,
# MAGIC 5. load that packaged model back and use it for predictions.
# MAGIC
# MAGIC **Before running:** upload `diabetes.csv` to a Unity Catalog volume (see the setup steps) and attach the notebook to compute.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration and MLflow experiment
# MAGIC
# MAGIC Change `DATA_PATH` if your catalog, schema or volume has another name.

# COMMAND ----------

DATA_PATH = "/Volumes/workspace/default/diabetes_data/diabetes.csv"
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Databricks experiment names must be absolute workspace paths
username = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT_NAME = f"/Users/{username}/diabetes-neural-network"

experiment = mlflow.set_experiment(EXPERIMENT_NAME)  # creates the experiment if it does not exist
print("Experiment:", EXPERIMENT_NAME)
print("Experiment id:", experiment.experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load and split the data
# MAGIC
# MAGIC In this dataset, zeros in glucose, blood pressure, skin thickness, insulin and BMI are impossible values: they mean
# MAGIC "missing". They are replaced by the median **inside the model pipeline** (fitted on the training set only), so the
# MAGIC packaged MLflow Model accepts raw values.

# COMMAND ----------

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df["Outcome"].value_counts(normalize=True).round(3).to_dict())

X = df.drop(columns="Outcome").astype("float64")  # float columns keep the MLflow schema simple
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, "Test:", X_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The model: preprocessing + neural network in one pipeline

# COMMAND ----------

def build_pipeline(hidden_layers, learning_rate_init, alpha):
    """Median imputation of impossible zeros, scaling, then a neural network."""
    preprocess = ColumnTransformer(
        [("impute_zeros", SimpleImputer(missing_values=0, strategy="median"), ZERO_COLS)],
        remainder="passthrough")
    return Pipeline([
        ("preprocess", preprocess),
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=hidden_layers,
                              learning_rate_init=learning_rate_init,
                              alpha=alpha, max_iter=500, random_state=42)),
    ])


def evaluate(pipeline):
    """Return the metrics to track and the test predictions."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "train_accuracy": accuracy_score(y_train, pipeline.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train several runs and track everything with MLflow
# MAGIC
# MAGIC Each configuration is one MLflow **run**. Compare them afterwards in the Experiments page.

# COMMAND ----------

CONFIGS = [
    {"hidden_layers": (8,), "learning_rate_init": 0.001, "alpha": 0.0001},
    {"hidden_layers": (16, 8), "learning_rate_init": 0.001, "alpha": 0.0001},
    {"hidden_layers": (32, 16), "learning_rate_init": 0.005, "alpha": 0.001},
]

for cfg in CONFIGS:
    run_name = "mlp_" + "-".join(str(n) for n in cfg["hidden_layers"]) + f"_lr{cfg['learning_rate_init']}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"project": "MedRoute AI", "model_type": "MLPClassifier",
                         "dataset": "Pima Indians Diabetes"})

        # 1. Parameters
        mlflow.log_params({
            "hidden_layers": str(cfg["hidden_layers"]),
            "learning_rate_init": cfg["learning_rate_init"],
            "alpha": cfg["alpha"],
            "max_iter": 500,
            "test_size": 0.2,
            "random_state": 42,
            "zero_imputation": "median",
        })

        # 2. Train
        pipeline = build_pipeline(**cfg)
        pipeline.fit(X_train, y_train)
        mlp = pipeline.named_steps["mlp"]

        # 3. Metrics: loss at every iteration, then the final scores
        for step, loss in enumerate(mlp.loss_curve_):
            mlflow.log_metric("train_loss", float(loss), step=step)
        metrics, y_pred = evaluate(pipeline)
        metrics["n_iterations"] = mlp.n_iter_
        mlflow.log_metrics(metrics)

        # 4. Artifacts: plots
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(mlp.loss_curve_)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Training loss")
        ax.set_title(run_name)
        mlflow.log_figure(fig, "loss_curve.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False)
        ax.set_title("Confusion matrix (test set)")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # 5. Package the pipeline as an MLflow Model
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(pipeline, artifact_path="model",
                                 signature=signature, input_example=X_train.head(3))

        print(f"{run_name}: test accuracy={metrics['test_accuracy']:.3f}, "
              f"recall={metrics['test_recall']:.3f}, f1={metrics['test_f1']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compare the runs and pick the best one

# COMMAND ----------

runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], order_by=["metrics.test_f1 DESC"])
columns = ["run_id", "tags.mlflow.runName", "params.hidden_layers", "params.learning_rate_init",
           "metrics.test_accuracy", "metrics.test_recall", "metrics.test_f1", "metrics.test_roc_auc"]
display(runs[columns])

best_run_id = runs.iloc[0]["run_id"]
print("Best run:", best_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Load the packaged MLflow Model and use it
# MAGIC
# MAGIC The model is loaded from the run's artifacts, exactly as a deployment would do. It accepts raw patient values.

# COMMAND ----------

model_uri = f"runs:/{best_run_id}/model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

sample = X_test.head(5)
predictions = loaded_model.predict(sample)
print(pd.DataFrame({"actual": y_test.head(5).values, "predicted": predictions}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. (Optional) Register the model in Unity Catalog
# MAGIC
# MAGIC Registering gives the model a name and versions. If your workspace does not allow it, the cell prints the reason
# MAGIC and the rest of the notebook is not affected.

# COMMAND ----------

try:
    mlflow.set_registry_uri("databricks-uc")
    registered = mlflow.register_model(model_uri, "workspace.default.diabetes_neural_network")
    print("Registered:", registered.name, "version", registered.version)
except Exception as error:
    print("Registration skipped:", error)
