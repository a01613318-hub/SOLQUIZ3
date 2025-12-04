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

# Normalizar nombres de columnas: quitar espacios y pasar a minúsculas
data.columns = [col.strip().lower() for col in data.columns]

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

    # Filtrar ciudad usando la columna normalizada 'city'
    if 'city' in data.columns:
        filtrado = data[data['city'].str.lower() == City.lower()]
        if not filtrado.empty:
            st.write("Datos encontrados:")
            st.write(filtrado.iloc[0])
        else:
            st.write("Ciudad no encontrada en el dataset.")
    else:
        st.write("El dataset no tiene una columna 'city'.")

    user_input_data = {
        "year": Year,
        "month": Month,
        "city": City,
    }

    features = pd.DataFrame(user_input_data, index=[0])
    return features

df_input = user_input_features()

# Asegurarse que las columnas coincidan con minúsculas
X = data[['year', 'month', 'city']]
y = data['averagetemperature']  # también en minúsculas

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1613726
)

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicción
prediccion = modelo.predict(df_input)[0]

st.subheader("Predicción de temperatura")
st.write(f"La temperatura estimada es: **{prediccion:.2f} °C**")
