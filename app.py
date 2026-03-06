import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -----------------------------
# Título de la aplicación
# -----------------------------
st.title("Predictor de Riesgo de Diabetes")

st.write(
    """
    Esta aplicación utiliza un modelo de Machine Learning (Random Forest)
    entrenado con el dataset **Pima Indians Diabetes** para estimar
    la probabilidad de que un paciente tenga diabetes.
    """
)

# -----------------------------
# Cargar dataset
# -----------------------------
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
data = pd.read_csv(url)

# -----------------------------
# Separar variables
# -----------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# -----------------------------
# División entrenamiento / test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Entrenar modelo
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# Evaluar modelo
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Precisión del modelo: **{accuracy:.2f}**")

# -----------------------------
# Inputs del usuario
# -----------------------------
st.header("Ingrese los datos del paciente")

pregnancies = st.number_input("Número de embarazos", 0, 20, 1)
glucose = st.number_input("Nivel de glucosa", 0, 200, 120)
blood_pressure = st.number_input("Presión arterial", 0, 140, 70)
skin_thickness = st.number_input("Espesor de piel", 0, 100, 20)
insulin = st.number_input("Nivel de insulina", 0, 900, 79)
bmi = st.number_input("BMI (Índice de masa corporal)", 0.0, 70.0, 25.0)
dpf = st.number_input("Historial familiar de diabetes", 0.0, 3.0, 0.5)
age = st.number_input("Edad", 1, 120, 30)

# -----------------------------
# Predicción
# -----------------------------
if st.button("Predecir riesgo de diabetes"):

    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"Alto riesgo de diabetes (probabilidad: {probability:.2%})")
    else:
        st.success(f"Bajo riesgo de diabetes (probabilidad: {probability:.2%})")

# -----------------------------
# Importancia de variables
# -----------------------------
st.header("Importancia de variables")

importances = model.feature_importances_
features = X.columns

fig, ax = plt.subplots()
ax.barh(features, importances)
ax.set_xlabel("Importancia")
ax.set_title("Importancia de variables en el modelo")

st.pyplot(fig)
