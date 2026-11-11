from pathlib import Path
import json
import pickle
import sys

import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
PKL_PATH = ARTIFACT_DIR / "iris_knn_k4.pkl"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    # Guardamos el histórico original en CSV para reproducir/auditar el entrenamiento.
    historical = X.copy()
    historical["target"] = y
    historical["species"] = y.map(dict(enumerate(iris.target_names)))
    historical.to_csv(DATA_DIR / "iris_historico.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # El escalador y el clasificador quedan unidos para evitar escalar distinto en producción.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=4)),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    bundle = {
        "pipeline": pipeline,
        "feature_names": list(X.columns),
        "target_names": list(iris.target_names),
        "model_name": "KNeighborsClassifier",
        "k": 4,
        "random_state": 42,
        "test_size": 0.2,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
    }
    with PKL_PATH.open("wb") as file:
        pickle.dump(bundle, file)

    print(f"Modelo guardado: {PKL_PATH}")
    print(f"Exactitud en test: {bundle['accuracy']:.3f}")
    print(classification_report(y_test, predictions, target_names=iris.target_names))
    print(f"Histórico separado: {DATA_DIR / 'iris_historico.csv'}")


if __name__ == "__main__":
    main()
