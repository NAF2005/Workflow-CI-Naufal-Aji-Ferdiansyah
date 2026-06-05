import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =========================
# Load Dataset
# =========================
df = pd.read_csv("winequality_preprocessing.csv")

target_column = "quality"  # ganti sesuai target

X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MLflow Local Tracking
# =========================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine Quality Basic")

# Autolog
mlflow.sklearn.autolog()

# =========================
# Training
# =========================
with mlflow.start_run():

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    print(f"Accuracy: {score:.4f}")