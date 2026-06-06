import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Hapus environment MLflow yang mungkin mengganggu
os.environ.pop("MLFLOW_TRACKING_URI", None)
os.environ.pop("MLFLOW_REGISTRY_URI", None)

# Paksa tracking ke folder lokal
TRACKING_DIR = os.path.abspath("mlruns")
mlflow.set_tracking_uri(f"file://{TRACKING_DIR}")

print("Tracking URI:", mlflow.get_tracking_uri())


def main():

    # Buat experiment jika belum ada
    experiment_name = "Wine Quality Basic"

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment dibuat: {experiment_id}")

    mlflow.set_experiment(experiment_name)

    # Load dataset
    df = pd.read_csv("winequality_preprocessing.csv")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, "model")

        print("RUN ID:", run.info.run_id)

        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)


if __name__ == "__main__":
    main()