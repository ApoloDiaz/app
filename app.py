%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px # Import plotly express for dynamic plots

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Diabetes Risk Predictor",
                   initial_sidebar_state="expanded")

# --- Custom CSS for professional styling ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stContainer {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSuccess {
        background-color: #e6ffe6; /* Light green for success */
        color: #006600;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
    }
    .stError {
        background-color: #ffe6e6; /* Light red for error */
        color: #cc0000;
        border-left: 5px solid #f44336;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Título de la aplicación
# -----------------------------
with st.container():
    st.title("Predictor de Riesgo de Diabetes")

    st.write(
        """
        Esta aplicación utiliza un modelo de Machine Learning (Random Forest)
        entrenado con el dataset **Pima Indians Diabetes** para estimar
        la probabilidad de que un paciente tenga diabetes.
        """
    )

# -----------------------------
# Cargar dataset (moved inside a container for styling)
# -----------------------------
# No need to display this directly, but model training is here.
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

with st.container():
    st.markdown("<h3 style='color: #333333;'>Información del Modelo</h3>", unsafe_allow_html=True)
    st.write(f"Precisión del modelo: **{accuracy:.2f}**")

# -----------------------------
# Inputs del usuario y Predicción
# -----------------------------
with st.container():
    st.header("Ingrese los datos del paciente")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Número de embarazos", 0, 20, 1)
        glucose = st.number_input("Nivel de glucosa", 0, 200, 120)
        blood_pressure = st.number_input("Presión arterial", 0, 140, 70)
        skin_thickness = st.number_input("Espesor de piel", 0, 100, 20)
    with col2:
        insulin = st.number_input("Nivel de insulina", 0, 900, 79)
        bmi = st.number_input("BMI (Índice de masa corporal)", 0.0, 70.0, 25.0)
        dpf = st.number_input("Historial familiar de diabetes", 0.0, 3.0, 0.5)
        age = st.number_input("Edad", 1, 120, 30)

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
with st.container():
    st.header("Importancia de variables")

    importances = model.feature_importances_
    features = X.columns

    # Create a DataFrame for Plotly
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=True) # Sort for better visualization

    # Create interactive bar chart using Plotly Express
    fig = px.bar(importance_df, x='Importance', y='Feature',
                 title='Importancia de variables en el modelo',
                 labels={'Importance': 'Importancia', 'Feature': 'Variable'},
                 color='Importance', color_continuous_scale=px.colors.sequential.Viridis)

    st.plotly_chart(fig, use_container_width=True) # Display the Plotly chart
