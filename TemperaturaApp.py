import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.write("Predicción de temperatura")
st.image("quiz.jpg", caption="Modelo predictivo")

st.header("Datos de entrada")

def user_input_features():

    Year = st.number_input("Año:", min_value=1700, max_value=2025, value=2000)
    Month = st.number_input("Mes (1-12):", min_value=1, max_value=12, value=1)
    City = st.text_input("Ciudad:")

    user_input_data = {
        "Year": Year,
        "Month": Month,
        "City": City
    }

    features = pd.DataFrame(user_input_data, index=[0])
    return features

df_input = user_input_features()

data = pd.read_csv("MexicoTemperatures.csv", encoding="latin-1")

X = data[["Year", "Month"]]
y = data["AverageTemperature"]

model = LinearRegression()
model.fit(X, y)

df_model = df_input[["Year", "Month"]]

prediccion = model.predict(df_model)[0]

st.subheader("Predicción de temperatura estimada")
st.write(f"**Temperatura estimada:** {prediccion:.2f} °C")
