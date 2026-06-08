import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():

    experiment_name = "Wine Quality Basic"

    mlflow.set_experiment(experiment_name)

    print("Tracking URI:", mlflow.get_tracking_uri())

    df = pd.read_csv("winequality_preprocessing.csv")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    with mlflow.start_run() as run:

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        with open("experiment_id.txt", "w") as f:
            f.write(run.info.experiment_id)

        print("RUN_ID =", run.info.run_id)
        print("EXP_ID =", run.info.experiment_id)
        print("ACCURACY =", accuracy)


if __name__ == "__main__":
    main()