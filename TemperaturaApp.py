import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

st.title("Predicción de Temperatura")

def user_input_features():
    # Selección de ciudad
    City = st.selectbox("Selecciona la ciudad", ["Ciudad1", "Ciudad2", "Ciudad3"])
    
  
    Year = st.number_input("Año", min_value=2000, max_value=2050, value=2025)
    Month = st.number_input("Mes", min_value=1, max_value=12, value=12)
    

    df = pd.DataFrame({"City": [City], "Year": [Year], "Month": [Month]})
    return df

df = user_input_features()

st.subheader("Entradas del usuario")
st.write(df)


df_encoded = pd.get_dummies(df, columns=["City"])

st.subheader("Datos preparados para el modelo")
st.write(df_encoded)

dummy_data = pd.DataFrame({
    "Year": [2023, 2024, 2025],
    "Month": [1, 6, 12],
    "City_Ciudad1": [1, 0, 0],
    "City_Ciudad2": [0, 1, 0],
    "City_Ciudad3": [0, 0, 1],
    "Temp": [20, 25, 30]
})

X_dummy = dummy_data.drop("Temp", axis=1)
y_dummy = dummy_data["Temp"]

model = DecisionTreeRegressor()
model.fit(X_dummy, y_dummy)

for col in X_dummy.columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0  # agregar columna faltante como 0

df_encoded = df_encoded[X_dummy.columns]

prediction = model.predict(df_encoded)

st.subheader("Predicción de Temperatura")
st.write(f"La temperatura estimada es: {prediction[0]:.2f} °C")

prediccion = modelo.predict(df)[0]

st.subheader("Predicción de temperatura")
st.write(f"La temperatura estimada es: **{prediccion:.2f} °C**")

