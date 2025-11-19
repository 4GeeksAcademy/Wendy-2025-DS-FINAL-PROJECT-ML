# OBJECTIVE:
# Develop a web application that, shows the percentage growth in property values (€/m2), by area. With heat maps and confidence levels; for decicions-making regarding, and investment in real estate.

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import json
from sklearn.base import BaseEstimator
from typing import Tuple

# Título de la app
st.title("Predicción del aumento del valor de vivienda en España")

# Cargar el modelo
model = joblib.load("/workspaces/FINAL-PROJECT-ML-Wendy-2025-DS/src/models/app_web_pipeline.pkl")

# Cargar el archivo geojson
with open("spain-communities.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# Obtener las comunidades autónomas desde el modelo
comunidades = model.named_steps['preprocessor'].transformers_[0][1].categories_[0]

# Selección de comunidad autónoma
comunidad = st.selectbox("Selecciona la comunidad autónoma", comunidades)

# Selección de años al futuro
años = st.slider("Selecciona cuántos años al futuro quieres ver", min_value=1, max_value=50, value=5)

# Crear el DataFrame de entrada para la predicción
input_df = pd.DataFrame({
    "Comunidad Autónoma": [comunidad],
    "Años al futuro": [años]
})

# Fun
class HousingGrowthPipeline(BaseEstimator):
    def __init__(self, model, base_df):
        self.model = model
        self.base_df = base_df

    def predict(self, future_years: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        last_year = pd.to_datetime(self.base_df["year"]).dt.year.max()
        predictions = []
        for year_offset in future_years:
            future_year = last_year + year_offset
            X_future = pd.DataFrame({
                "year": [future_year],
                "precio": [self.base_df["precio"].mean()],
                "comunidad": [self.base_df["comunidad"].mode()[0]],
                "name": [self.base_df["name"].mode()[0]]
            })
            pred = self.model.predict(X_future)[0]
            predictions.append(pred)
        predictions = np.array(predictions)
        confidence = 1 / (1 + 0.05 * future_years)
        return predictions, confidence

    def get_region_growth(self, years: int) -> pd.DataFrame:
        last_year = pd.to_datetime(self.base_df["year"]).dt.year.max()
        future_year = last_year + years
        region_predictions = []
        for region, group in self.base_df.groupby("comunidad"):
            last_row = group.iloc[-1]
            last_price = last_row["precio"]
            X_future = pd.DataFrame({
                "year": [future_year],
                "precio": [last_price],
                "comunidad": [region],
                "name": [last_row["name"]]
            })
            pred_price = self.model.predict(X_future)[0]
            growth = ((pred_price - last_price) / last_price) * 100
            region_predictions.append({"region": region, "growth": growth})
        return pd.DataFrame(region_predictions)

# Realizar la predicción
prediccion = model.predict(input_df)[0]

# Mostrar la predicción
st.metric(label="Predicción de aumento del valor de vivienda (%)", value=f"{prediccion:.2f}%")

# Crear DataFrame para todas las comunidades para el mapa de calor
df_mapa = pd.DataFrame({
    "Comunidad Autónoma": comunidades,
    "Años al futuro": [años] * len(comunidades)
})
df_mapa["Predicción"] = model.predict(df_mapa)

# Crear el mapa de calor
fig = px.choropleth(
    df_mapa,
    geojson=geojson_data,
    locations="Comunidad Autónoma",
    featureidkey="properties.name",
    color="Predicción",
    color_continuous_scale=["green", "yellow", "red"],
    scope="europe",
    title="Mapa de calor del crecimiento del valor de vivienda"
)
fig.update_geos(fitbounds="locations", visible=False)

# Mostrar el mapa
st.plotly_chart(fig)

# Note:
st.info("Nota: El nivel de confianza disminuye a medida que se predicen más años en el futuro.")
