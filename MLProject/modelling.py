import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    # Paksa MLflow menggunakan local file store
    os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
    mlflow.set_tracking_uri("file:./mlruns")

    print("Tracking URI:", mlflow.get_tracking_uri())

    # Membuat experiment
    mlflow.set_experiment("Wine Quality Basic")

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

        # Simpan model ke MLflow
        mlflow.sklearn.log_model(model, "model")

        print("RUN ID:", run.info.run_id)

        # Simpan run id untuk workflow berikutnya
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)


if __name__ == "__main__":
    main()