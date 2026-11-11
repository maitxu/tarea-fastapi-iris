# Iris KNN k=4 con API FastAPI

Ejemplo didáctico de entrenamiento y predicción sobre el conjunto Iris. El modelo usa `StandardScaler` y `KNeighborsClassifier(n_neighbors=4)` dentro de un único `Pipeline`, serializado en `artifacts/iris_knn_k4.pkl`.

## 1. Preparar el entorno

Requiere Python 3.11 recomendado. Desde esta carpeta:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Entrenar y crear los archivos

```bash
python train_model.py
```

El comando genera el PKL y `data/iris_historico.csv`, e imprime la exactitud sobre un conjunto de prueba reservado.

## 3. Iniciar y probar la API

```bash
uvicorn app:app --reload
```

Abre <http://127.0.0.1:8000/docs>. Prueba `POST /predict` con:

```json
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
```

También existen `GET /` y `GET /health`.

## 4. Ejecutar con Docker

Primero entrena el modelo para crear el PKL, luego:

```bash
docker build -t iris-knn-api .
docker run --rm -p 8000:8000 iris-knn-api
```

Visita <http://localhost:8000/docs>.

## Archivos y datos históricos

El PKL contiene el pipeline ajustado (incluido el escalador), nombres y orden de características, nombres de clases, hiperparámetro k y métricas/versiones informativas. Eso basta para inferencia en este ejemplo. El archivo `iris_historico.csv` conserva aparte los 150 registros para reproducibilidad, auditoría o reentrenamiento; no se carga al hacer predicciones. En un proyecto real conviene controlar su acceso y versionarlo aparte si es necesario.

Usa el mismo entorno de versiones al cargar el PKL. No cargues archivos pickle de fuentes no confiables: pueden ejecutar código durante su lectura. Para producción también deben añadirse validación, control de errores y gestión de versiones del modelo.

## Relación con la tarea adjunta

La tarea adjunta pide una API sencilla de dos endpoints, Swagger, ngrok, Colab, GitHub y un diagrama; señala que entrenar o integrar IA no es necesario. Este ejemplo añade un modelo KNN a ese recorrido. Para cumplir la rúbrica, todavía habría que demostrar las llamadas desde Colab a través de ngrok, publicar el repo y aportar el diagrama/capturas.
