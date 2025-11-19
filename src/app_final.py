import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import json

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predicción de Precios de Vivienda en España",
    page_icon="🏠",
    layout="wide"
)

# --- Funciones de Carga ---
@st.cache_data
def cargar_modelo():
    """Carga el pipeline completo (preprocesamiento + modelo)."""
    return joblib.load('/workspaces/FINAL-PROJECT-ML-Wendy-2025-DS/src/modelo_vivienda.joblib')

@st.cache_data
def cargar_datos():
    """Carga los datos históricos y renombra columnas para visualización."""
    df = pd.read_csv('/workspaces/FINAL-PROJECT-ML-Wendy-2025-DS/src/datos_historicos.csv')
    df = df.rename(columns={
        'comunidad': 'Comunidad Autónoma',
        'year': 'Año',
        'precio': 'Precio_Medio_m2'
    })
    return df

@st.cache_data
def cargar_geojson():
    """Carga el archivo GeoJSON para el mapa."""
    try:
        with open('/workspaces/FINAL-PROJECT-ML-Wendy-2025-DS/data/spain_communities.geojson', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("No se encontró el archivo 'spain_communities.geojson'. El mapa no se mostrará.")
        return None

# --- Carga de Datos y Modelo ---
modelo = cargar_modelo()
df_historico = cargar_datos()
geojson = cargar_geojson()
comunidades = sorted(df_historico['Comunidad Autónoma'].unique())

# --- Barra Lateral ---
st.sidebar.header('Parámetros de Entrada')

comunidad_seleccionada = st.sidebar.selectbox(
    'Selecciona una Comunidad Autónoma',
    options=comunidades
)

ano_futuro = st.sidebar.number_input(
    'Selecciona un año para la predicción',
    min_value=2024,
    max_value=2030,
    value=2025,
    step=1
)

# --- Predicción ---
if st.sidebar.button('Realizar Predicción'):
    # Crear DataFrame con columnas originales
    input_data = pd.DataFrame({
        'year': [ano_futuro],
        'comunidad': [comunidad_seleccionada]
    })

    # Predicción directa con el pipeline
    prediccion_porc = modelo.predict(input_data)[0]

    # --- Mostrar Resultados ---
    st.header(f"Resultados para {comunidad_seleccionada} en {ano_futuro}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predicción de Incremento Anual")
        st.metric(
            label="Variación Anual Estimada",
            value=f"{prediccion_porc:.2f}%",
            delta=f"{prediccion_porc:.2f}% vs año anterior"
        )

    with col2:
        st.subheader("Rango de Confianza Simple")
        margen_error = 1.5
        rango_inferior = prediccion_porc - margen_error
        rango_superior = prediccion_porc + margen_error
        st.info(f"El incremento podría variar entre **{rango_inferior:.2f}%** y **{rango_superior:.2f}%**.")

    # --- Gráfico de Proyección ---
    st.subheader("Proyección del Precio del m²")
    
    df_comunidad = df_historico[df_historico['Comunidad Autónoma'] == comunidad_seleccionada].copy()
    ultimo_precio = df_comunidad.sort_values('Año', ascending=False).iloc[0]['Precio_Medio_m2']
    precio_predicho = ultimo_precio * (1 + prediccion_porc / 100)
    
    df_prediccion = pd.DataFrame({
        'Año': [ano_futuro],
        'Precio_Medio_m2': [precio_predicho],
        'Tipo': ['Predicción']
    })
    
    df_comunidad['Tipo'] = 'Histórico'
    df_plot = pd.concat([df_comunidad[['Año', 'Precio_Medio_m2', 'Tipo']], df_prediccion], ignore_index=True)
    
    fig_proyeccion = px.line(
        df_plot,
        x='Año',
        y='Precio_Medio_m2',
        color='Tipo',
        markers=True,
        labels={'Precio_Medio_m2': 'Precio Medio (€/m²)', 'Año': 'Año'},
        title=f'Evolución y Proyección de Precios en {comunidad_seleccionada}',
        color_discrete_map={'Histórico': 'blue', 'Predicción': 'red'}
    )
    fig_proyeccion.update_layout(legend_title_text='Datos')
    st.plotly_chart(fig_proyeccion, width="stretch")

# --- Visualizaciones Generales ---
st.markdown("---")
st.header("Visualizaciones Generales")

# Serie Temporal
st.subheader("Evolución Histórica por Comunidad Autónoma")
fig_historico = px.line(
    df_historico,
    x='Año',
    y='Precio_Medio_m2',
    color='Comunidad Autónoma',
    title='Precio Medio del m² por Comunidad Autónoma (2011-2023)',
    labels={'Precio_Medio_m2': 'Precio Medio (€/m²)', 'Año': 'Año'}
)
st.plotly_chart(fig_historico, width="stretch")

# --- Mapa ---
if geojson:
    st.subheader(f"Mapa de Precios Estimados para {ano_futuro}")
    
    df_mapa_input = pd.DataFrame({
        'year': [ano_futuro] * len(comunidades),
        'comunidad': comunidades
    })

    # Predicción directa con el pipeline
    incrementos_mapa = modelo.predict(df_mapa_input)
    
    df_mapa_output = pd.DataFrame({
        'Comunidad Autónoma': comunidades,
        'Incremento_Estimado_Porc': incrementos_mapa
    })
    
    ultimos_precios = df_historico.loc[df_historico.groupby('Comunidad Autónoma')['Año'].idxmax()]
    df_mapa_output = pd.merge(df_mapa_output, ultimos_precios[['Comunidad Autónoma', 'Precio_Medio_m2']], on='Comunidad Autónoma', how='left')
    df_mapa_output = df_mapa_output.rename(columns={'Precio_Medio_m2': 'Ultimo_Precio_Conocido'})
    
    df_mapa_output['Precio_Estimado_m2'] = df_mapa_output['Ultimo_Precio_Conocido'] * (1 + df_mapa_output['Incremento_Estimado_Porc'] / 100)
    
    fig_mapa = px.choropleth(
        df_mapa_output,
        geojson=geojson,
        locations='Comunidad Autónoma',
        featureidkey="properties.name",
        color='Precio_Estimado_m2',
        color_continuous_scale="Viridis",
        scope="europe",
        labels={'Precio_Estimado_m2':'Precio Estimado (€/m²)'}
    )
    fig_mapa.update_geos(fitbounds="locations", visible=False)
    fig_mapa.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    st.plotly_chart(fig_mapa, width="stretch")
else:
    st.warning("El mapa no se puede mostrar porque falta el archivo `spain_communities.geojson`.")