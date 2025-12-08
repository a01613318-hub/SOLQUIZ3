import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.write("# Predicción de Temperatura en México")
st.image("quiz.jpg", caption="Predicción de temperatura")
st.header("Selecciona los datos")

# Cargar CSV
data = pd.read_csv("MexicoTemperatures.csv", encoding="latin-1")

# Normalizar columnas
data.columns = [col.strip().lower() for col in data.columns]

# Mantener solo columnas numéricas para el modelo
# year, month, averagetemperature deben existir
data = data.dropna(subset=['year', 'month', 'averagetemperature'])

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

    City = st.text_input("Ingresa el nombre de la ciudad:")

    if 'City' in data.columns:
        filtrado = data[data['City'].str.lower() == City.lower()]
        if not filtrado.empty:
            st.write("Datos encontrados:")
            st.write(filtrado.iloc[0])
        else:
            st.write("Ciudad no encontrada en el dataset.")

    user_input_data = {
        "Year": Year,
        "Month": Month
        # city ya no se usa como feature
    }

    return pd.DataFrame(user_input_data, index=[0])

df_input = user_input_features()

# Modelo SOLO con columnas numéricas
X = data[['Year', 'Month', 'AverageTemperature']]
y = data['AverageTemperatureUncertainty']

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1613726
)

# Entrenar
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predecir
prediccion = modelo.predict(df_input)[0]

st.subheader("Predicción de temperatura")
st.write(f"La temperatura estimada es: **{prediccion:.2f} °C**")
