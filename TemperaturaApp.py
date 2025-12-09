import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.write("Predicción de temperatura")
st.image("quiz.jpg", caption="Modelo predictivo")

st.header("Datos de entrada")

def user_input_features():
    Year = st.number_input("Año:", min_value=1700, max_value=2025, value=2000)
    Month = st.number_input("Mes (1-12):", min_value=1, max_value=12, value=1)
    City = st.text_input("Ciudad:")

    return pd.DataFrame({"Year": Year, "Month": Month, "City": City}, index=[0])

df_input = user_input_features()

datos = pd.read_csv("MexicoTemperatures.csv", encoding="latin-1")

X = datos[["Year", "Month"]]
y = datos["AverageTemperature"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1613318
)

model = LinearRegression()
model.fit(X_train, y_train)

prediccion = model.predict(df_input[["Year", "Month"]])[0]

st.subheader("Predicción de temperatura estimada")
st.write(f"**Temperatura estimada:** {prediccion:.2f} °C")
