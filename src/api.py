from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Habilita CORS para toda la aplicación

# Cargar el modelo de diabetes desde un archivo
with open('diabetes_model2.pkl', 'rb') as file:
    diabetes_ml_model = pickle.load(file)

# Definir el modelo de datos para la entrada del usuario
class InputData:
    def __init__(self, Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness, SerumInsulin, BMI, DiabetesPedigree, Age):
        self.Pregnancies = Pregnancies
        self.PlasmaGlucose = PlasmaGlucose
        self.DiastolicBloodPressure = DiastolicBloodPressure
        self.TricepsThickness = TricepsThickness
        self.SerumInsulin = SerumInsulin
        self.BMI = BMI
        self.DiabetesPedigree = DiabetesPedigree
        self.Age = Age

# Definir el endpoint para realizar predicciones utilizando el método POST
@app.route('/predict/', methods=['POST'])
def predict_diabetes():
    input_data = request.json
    input_data_model = InputData(**input_data)

    # Convertir los datos de entrada a un array de numpy
    input_values = np.array([[
        input_data_model.Pregnancies,
        input_data_model.PlasmaGlucose,
        input_data_model.DiastolicBloodPressure,
        input_data_model.TricepsThickness,
        input_data_model.SerumInsulin,
        input_data_model.BMI,
        input_data_model.DiabetesPedigree,
        input_data_model.Age
    ]])

    # Realizar la predicción utilizando el modelo cargado
    prediction = diabetes_ml_model.predict(input_values)

    # Devolver la predicción como respuesta
    if prediction[0] == 0:
        return jsonify({"prediction": "No tiene diabetes"})
    elif prediction[0] == 1:
        return jsonify({"prediction": "Tiene diabetes"})
    else:
        return jsonify({"error": "Predicción no válida"})

if __name__ == '__main__':
    app.run(debug=True)
