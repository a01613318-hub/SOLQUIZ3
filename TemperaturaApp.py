import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.write("# Predicción de Temperatura en México")
st.image("quiz.jpg", caption="Predicción de temperatura")

data = pd.read_csv("MexicoTemperatures.csv", encoding="latin-1")


data.columns = data.columns.str.strip()

encoder = LabelEncoder()
data["City_encoded"] = encoder.fit_transform(data["City"])

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

    filtrado = data[data['City'].str.lower() == City.strip().lower()]

    if not filtrado.empty:
        st.write(filtrado.iloc[0])
        # Convertir la ciudad ingresada al valor numérico correspondiente
        City_encoded = encoder.transform([City.strip()])[0]
    else:
        st.write("")
        City_encoded = -1  # Valor inválido para evitar crash

    user_input_data = {
        "Year": Year,
        "Month": Month,
        "City_encoded": City_encoded,
    }

    return pd.DataFrame(user_input_data, index=[0])

df_input = user_input_features()

X = data[["Year", "Month", "City_encoded"]]
y = data["AverageTemperature"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1613318
)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

if df_input["City_encoded"].iloc[0] != -1:
    prediccion = modelo.predict(df_input)[0]
    st.subheader("Predicción de temperatura")
    st.write(f"La temperatura estimada es: **{prediccion:.2f} °C**")
else:
    st.write(" ")
