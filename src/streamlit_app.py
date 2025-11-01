# OBJECTIVE:
# Develop a web application that, shows the percentage growth in property values (€/m2), by area. With heat maps and confidence levels; for decicions-making regarding, and investment in real estate.
import utils
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import geopandas as gpd
from datetime import datetime
from utils import HousingGrowthPipeline

# App configuration:
st.set_page_config(page_title="Predicción del valor de viviendas en España", layout="wide")
st.title("🏠 Predicción del crecimiento del valor de viviendas en España")
# Pipeline:
pipeline_path = "src/models/app_web_pipeline.pkl"
pipeline = joblib.load(pipeline_path)
# Future year selction:
years = st.slider("Selecciona el número de años al futuro para la predicción:", 1, 50, 10)
# Predictions:
future_years = np.arange(1, years + 1)
predictions, confidence_intervals = pipeline.predict(future_years)
# Data frame creation for visualisation:
df_pred = pd.DataFrame({
    "Año": future_years + datetime.now().year,
    "Predicción (%)": predictions,
    "Confianza": confidence_intervals
})
# Prediction chart:
st.subheader("📈 Predicción del crecimiento del valor de viviendas")
fig_pred = px.line(df_pred, x="Año", y="Predicción (%)", title="Predicción del crecimiento")
fig_pred.add_scatter(x=df_pred["Año"], y=df_pred["Confianza"], mode="lines", name="Nivel de confianza")
st.plotly_chart(fig_pred, use_container_width=True)
# Display current confidence level in a box:
st.markdown(
    f"<div style='border:1px solid #ccc; padding:10px; width:300px; background:#f9f9f9;'>"
    f"<b>Nivel de confianza actual:</b> {df_pred['Confianza'].iloc[-1]:.2f}</div>",
    unsafe_allow_html=True
)
# Heat map by region:
st.subheader("🗺️ Mapa de calor del crecimiento por región")
# Load geographical data:
geojson_path = "data/spain_regions.geojson"
geo_df = gpd.read_file(geojson_path)
# Obtain growth data by region from the pipeline:
pipeline
values_df = pipeline.get_region_growth(years)
geo_df = geo_df.merge(values_df, on="region")
# Create heat map:
fig_map = px.choropleth(
    geo_df,
    geojson=geo_df.geometry,
    locations=geo_df.index,
    color="growth",
    color_continuous_scale="YlOrRd",
    title="Mapa de calor del crecimiento por región",
    projection="mercator"
)
fig_map.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig_map, use_container_width=True)
# Note:
st.info("Nota: El nivel de confianza disminuye a medida que se predicen más años en el futuro.")
