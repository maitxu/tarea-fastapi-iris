from pathlib import Path
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "iris_knn_k4.pkl"

app = FastAPI(title="API Iris KNN", description="Clasificación de flores Iris con KNN, k=4", version="1.0.0")

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0, description="Largo del sépalo en cm")
    sepal_width: float = Field(gt=0, description="Ancho del sépalo en cm")
    petal_length: float = Field(gt=0, description="Largo del pétalo en cm")
    petal_width: float = Field(gt=0, description="Ancho del pétalo en cm")


def load_bundle():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No se encontró {MODEL_PATH}. Ejecuta python train_model.py")
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)

bundle = load_bundle()

@app.get("/")
def home():
    return {"mensaje": "API Iris lista", "documentacion": "/docs", "k": bundle["k"]}

@app.get("/health")
def health():
    return {"estado": "ok", "modelo": bundle["model_name"], "k": bundle["k"]}

@app.post("/predict")
def predict(data: IrisInput):
    values = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    predicted_index = int(bundle["pipeline"].predict(values)[0])
    distances, _ = bundle["pipeline"].named_steps["knn"].kneighbors(values)
    return {
        "clase_id": predicted_index,
        "especie": bundle["target_names"][predicted_index],
        "distancias_vecinos": distances[0].tolist(),
        "k": bundle["k"],
    }
