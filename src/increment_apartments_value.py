import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# --- Cargar modelo y datos ---
modelo = joblib.load("/workspaces/FINAL-PROJECT-ML-Wendy-2025-DS/src/modelo_vivienda.joblib")
df = pd.read_csv("/workspaces/FINAL-PROJECT-ML-Wendy-2025-DS/src/datos_historicos.csv")

# --- Configuración de la app ---
st.title("Predicción de Precio por m² de Viviendas en España")
st.write("Modelo: Árbol de Decisión | Datos históricos hasta 2023 | Pronóstico hasta 2030")

# --- Opciones ---
años_hist = sorted(df["year"].unique())
año_min, año_max = min(años_hist), max(años_hist)
año_pred = st.slider("Selecciona el año a pronosticar", año_max + 1, 2030, 2025)

comunidades = ["Todas"] + sorted(df["comunidad"].unique())
comunidad_sel = st.selectbox("Selecciona la comunidad autónoma", comunidades)

# --- Preparar datos para predicción ---
if comunidad_sel == "Todas":
    comunidades_pred = sorted(df["comunidad"].unique())
else:
    comunidades_pred = [comunidad_sel]

df_pred = pd.DataFrame({
    "comunidad": comunidades_pred,
    "year": [año_pred] * len(comunidades_pred)
})

# Convertir variables categóricas si el modelo lo requiere
X_pred = pd.get_dummies(df_pred)
modelo_cols = modelo.feature_names_in_
for col in modelo_cols:
    if col not in X_pred.columns:
        X_pred[col] = 0
X_pred = X_pred[modelo_cols]

# --- Predicción ---
predicciones = modelo.predict(X_pred)

# --- Calcular métricas ---
X_hist = pd.get_dummies(df[["comunidad", "year"]])
for col in modelo_cols:
    if col not in X_hist.columns:
        X_hist[col] = 0
X_hist = X_hist[modelo_cols]
y_hist = df["precio"]
confiabilidad = modelo.score(X_hist, y_hist) * 100  # %

ultimo_año = año_max
df_ultimo = df[df["year"] == ultimo_año]

# --- Crear DataFrame de resultados ---
resultados = []
for i, comunidad in enumerate(comunidades_pred):
    precio_pred = predicciones[i]
    precio_actual = df_ultimo[df_ultimo["comunidad"] == comunidad]["precio"].mean()
    crecimiento = ((precio_pred - precio_actual) / precio_actual) * 100 if precio_actual else np.nan
    recomendacion = "Recomiendo COMPRA" if crecimiento < 0 else "Recomiendo VENTA"
    resultados.append([comunidad, año_pred, precio_pred, confiabilidad, crecimiento, recomendacion])

df_resultados = pd.DataFrame(resultados, columns=["Comunidad", "Año", "Precio Predicho", "Confiabilidad (%)", "Crecimiento (%)", "Recomendación"])

# --- Mostrar resultados ---
st.subheader(f"Pronóstico para el año {año_pred}")
st.dataframe(df_resultados)

# --- Gráfico ---
st.subheader("Gráfico de precios por año")
fig, ax = plt.subplots(figsize=(10, 5))

if comunidad_sel == "Todas":
    for comunidad in comunidades_pred:
        datos_com = df[df["comunidad"] == comunidad]
        ax.plot(datos_com["year"], datos_com["precio"], label=comunidad)
        ax.scatter(año_pred, predicciones[comunidades_pred.index(comunidad)], color="red")
else:
    datos_com = df[df["comunidad"] == comunidad_sel]
    ax.plot(datos_com["year"], datos_com["precio"], label=comunidad_sel, color="blue")
    ax.scatter(año_pred, predicciones[0], color="red")

ax.set_title("Precio por m² según año")
ax.set_xlabel("Año")
ax.set_ylabel("Precio €/m²")
ax.legend()
st.pyplot(fig)

# --- Botón para descargar gráfico ---
buffer_img = BytesIO()
fig.savefig(buffer_img, format="png")
st.download_button(
    label="📥 Descargar gráfico como PNG",
    data=buffer_img.getvalue(),
    file_name=f"grafico_{año_pred}.png",
    mime="image/png"
)

# --- Botón para descargar resultados en Excel ---
buffer_excel = BytesIO()
with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
    df_resultados.to_excel(writer, index=False, sheet_name="Predicciones")
st.download_button(
    label="📥 Descargar resultados en Excel",
    data=buffer_excel.getvalue(),
    file_name=f"predicciones_{año_pred}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
