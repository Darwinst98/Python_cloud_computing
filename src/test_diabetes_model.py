import pickle
import pandas as pd
import numpy as np
from typing import List

# Cargando el modelo:
diabetes_ml_model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Definiendo la función de predicción:
# Orden de los valores: ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']
def predict(input_values: List[float]):
    
    # Creando un numpy array bidimensional
    # Un numpy array es un contenedor eficiente en memoria que permite realizar operaciones numéricas rápidas
    features = [np.array(input_values)]
    
    # Creando un dataframe a partir del array bidimensional
    features_df = pd.DataFrame(features)
    
    # Generando las predicciones
    prediction_values = diabetes_ml_model.predict_proba(features_df)
    print('prediction_values', prediction_values)

    # Determinando la predicción final
    final_prediction = np.argmax(prediction_values)    
    
    return True if final_prediction==1 else False


if __name__ == '__main__':
    positive_case_1 = (5, 114, 101, 43, 70, 36.49531966, 0.079190164, 38)
    positive_case_2 = (9, 103, 78, 25, 304, 29.58219193, 1.282869847, 43)
    positive_case_3 = (9,104,68,42,40,51.85540108,0.182937824,21)

    negative_case_1 = (0,171,80,34,23,43.50972593,1.213191354,21)
    negative_case_2 = (0,109,56,44,26,20.21133193,0.780654857,26)
    negative_case_3 = (0,133,47,19,227,21.94135672,0.174159779,21)

    
    assert predict(positive_case_1) == True
    assert predict(positive_case_2) == True
    assert predict(positive_case_3) == True

    assert predict(negative_case_1) == False
    assert predict(negative_case_1) == False
    assert predict(negative_case_3) == False