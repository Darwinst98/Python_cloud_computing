# Importar las dependencias necesarias
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

# Crear una instancia de FastAPI
app = FastAPI()

# Cargar el modelo de diabetes desde un archivo
with open('diabetes_model2.pkl', 'rb') as file:
    diabetes_ml_model = pickle.load(file)

# Definir el modelo de datos para la entrada del usuario utilizando Pydantic
class InputData(BaseModel):
    Pregnancies: int
    PlasmaGlucose: int
    DiastolicBloodPressure: int
    TricepsThickness: int
    SerumInsulin: int
    BMI: float
    DiabetesPedigree: float
    Age: int

# Configurar el middleware CORS para permitir solicitudes desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Reemplazar con el origen del frontend
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Definir el endpoint para la ruta raíz "/"
@app.get("/")
async def root():
    return {"message": "Welcome to the Diabetes Prediction API!"}

# Definir el endpoint para realizar predicciones utilizando el método POST
@app.post("/predict/")
async def predict_diabetes(input_data: InputData):
    # Convertir los datos de entrada a enteros
    input_data_dict = input_data.dict()
    input_data_dict = {k: int(v) for k, v in input_data_dict.items()}
    
    # Crear una nueva instancia de InputData con los valores convertidos
    input_data_int = InputData(**input_data_dict)

    return predict(input_data_int)

# Definir el endpoint para realizar predicciones desde un JSON utilizando el método POST
@app.post("/predict_json/")
async def predict_diabetes_from_json(input_data: dict):
    try:
        # Crear una instancia de InputData a partir del JSON recibido
        input_data_model = InputData(**input_data)
        return predict(input_data_model)
    except Exception as e:
        return {"error": str(e)}

# Función para realizar la predicción utilizando el modelo cargado
def predict(input_data: InputData):
    # Convertir los datos de entrada a un array de numpy
    input_values = np.array([[
        input_data.Pregnancies,
        input_data.PlasmaGlucose,
        input_data.DiastolicBloodPressure,
        input_data.TricepsThickness,
        input_data.SerumInsulin,
        input_data.BMI,
        input_data.DiabetesPedigree,
        input_data.Age
    ]])

    # Realizar la predicción utilizando el modelo cargado
    prediction = diabetes_ml_model.predict(input_values)

    # Devolver la predicción como respuesta
    return {"prediction": int(prediction[0])}  # Cambiar "Prediccion" a "prediction"
