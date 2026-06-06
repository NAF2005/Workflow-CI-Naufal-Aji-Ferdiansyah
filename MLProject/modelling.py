import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    # SOLUSI PASTI: Paksa MLflow mencatat ke folder lokal (bukan ke server/port 5000)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Wine Quality Basic")
    
    # Load dataset
    df = pd.read_csv('winequality_preprocessing.csv')
    X = df.drop('quality', axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Autolog untuk pencatatan otomatis
    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan model secara eksplisit ke dalam folder "model"
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()