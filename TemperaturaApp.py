import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.write("# Predicción de Temperatura en México")
st.image("quiz.jpg", caption="Predicción de temperatura")
st.header("Selecciona los datos")

# Cargar datos primero
data = pd.read_csv("MexicoTemperatures.csv", encoding="latin-1")

# Normalizar nombres de columnas para evitar KeyError
data.columns = [col.strip().capitalize() for col in data.columns]

def user_input_features():
    Year = st.number_input(
        "Año:",
        min_value=1700,
        max_value=2025,
        value=2000,
        step=1,
    )

    Month = st.number_input(
        "Mes (1-12):",
        min_value=1,
        max_value=12,
        value=1,
        step=1,
    )

    City = st.text_input("Ingr_
