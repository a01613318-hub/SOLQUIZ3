import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor  # O el modelo que uses
import joblib  # Si quieres cargar un modelo entrenado

st.title("Predicción de Temperatura")

# Función para capturar entradas del usuario
def user_input_features():
    # Selección de ciudad
    City = st.selectbox("Selecciona la ciudad", ["Ciudad1", "Ciudad2", "Ciudad3"])
    
    # Año y mes
    Year = st.number_input("Año", min_value=2000, max_value=2050, value=2025)
    Month = st.number_input("Mes", min_value=1, max_value=12, value=12)
    
    # Crear DataFrame con las entradas
    data = {
        "City": City,
        "Year": Year,
        "Month": Month
    }
    df = pd.DataFrame(data, index=[0])
    
    # Convertir ciudad a código numérico
    df["City_num"] = df["City"].astype('category').cat.codes
    
    return df

# Obtener los datos del usuario
df = user_input_features()

st.subheader("Entradas del usuario")
st.write(df)

# Preparar las variables para el modelo
X = df[["Year", "Month", "City_num"]]

# Cargar modelo (opcional) o entrenar uno de ejemplo
# modelo = joblib.load("modelo_temperatura.pkl")

# Para ejemplo, entrenamos un modelo dummy
# Nota: En producción, reemplazar con tu modelo real
dummy_data = pd.DataFrame({
    "Year": [2023, 2024, 2025],
    "Month": [1, 6, 12],
    "City_num": [0, 1, 2],
    "Temp": [20, 25, 30]
})
model = DecisionTreeRegressor()
model.fit(dummy_data[["Year", "Month", "City_num"]], dummy_data["Temp"])

# Predicción
prediction = model.predict(X)

st.subheader("Predicción de Temperatura")
st.write(f"La temperatura estimada es: {prediction[0]:.2f} °C")

