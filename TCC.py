# Importar as bibliotecas
import pandas as pd
import geopandas as gpd
import json
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import folium
from streamlit_folium import st_folium
import datetime
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.colors as mcolors

st.set_page_config(layout="wide", page_title="Análise Climática por Município")


def obter_shapefile_municipios(cod_uf):
    url = f"https://servicodados.ibge.gov.br/api/v4/malhas/estados/{cod_uf}?formato=application/json&intrarregiao=Municipio&qualidade=intermediaria"
    response = requests.get(url)
    if response.status_code == 200:
        municipios = gpd.read_file(response.text)
        return municipios
    else:
        print("Erro:", response.status_code, response.text)


def obter_municipios_por_estado(uf: str):
    url = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"
    response = requests.get(url)
    if response.status_code == 200:
        dados = response.json()
        municipios = [{
            'codigo_ibge': mun['id'],
            'municipio': mun['nome'],
            'uf': uf.upper()
        } for mun in dados]
        return pd.DataFrame(municipios)
    else:
        print(f"Erro {response.status_code}: {response.text}")
        return pd.DataFrame()


# Título do APP
st.title("🌍 Análise da Temperatura e da Precipitação por Município")
st.markdown("Explore os dados climáticos de temperatura e precipitação dos municípios de Mato Grosso do Sul. 📊🌦️")

# Fixar estado como Mato Grosso do Sul
uf_selecionado = 'MS'
cod_uf = '50'

# Obter o shapefile do MS
gdf = obter_shapefile_municipios(cod_uf)

# Setar o CRS
gdf = gdf.set_crs(epsg=4674)

# Obter municípios do MS
df_mun = obter_municipios_por_estado(uf_selecionado)

# Criando um selectbox para escolher a cidade (79 municípios em ordem alfabética)
cidade_selecionada = st.sidebar.selectbox("Escolha uma cidade:", sorted(df_mun['municipio']))

# Selecionar o código IBGE da cidade a partir da cidade_selecionada
geocod = str(df_mun[df_mun['municipio'] == cidade_selecionada]['codigo_ibge'].to_list()[0])

# Selecionar o GeoDataFrame
gdf = gdf[gdf.codarea == geocod]

# Obter as coordenadas
long_x = gdf.geometry.centroid.x.values[0]
lat_y = gdf.geometry.centroid.y.values[0]

# Criar o mapa com Folium
mapa = folium.Map(location=[lat_y, long_x], zoom_start=10)

# Exibir no Streamlit
st.header("Município selecionado")

# Adicionar a camada do Município
folium.GeoJson(
    data=gdf,
    name='Município',
    tooltip=folium.GeoJsonTooltip(
        fields=['codarea'],
        aliases=['Código município: '],
        localize=True
    ),
    style_function=lambda x: {
        'fillColor': 'white',
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.6
    }
).add_to(mapa)

# Exibir o mapa
st_folium(mapa, use_container_width=True, height=500)

# Definir data mínima e máxima
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2025, 2, 28)

# Criar o seletor de intervalo de datas no sidebar
data_range = st.sidebar.date_input(
    "Selecione o intervalo de datas:",
    value=(start_date, end_date),
    min_value=datetime.date(2000, 1, 1),
    max_value=datetime.date(2025, 12, 31),
)

# Verificar se o usuário selecionou um intervalo válido
if isinstance(data_range, tuple) and len(data_range) == 2:
    start_date = data_range[0].strftime("%Y%m%d")
    end_date = data_range[1].strftime("%Y%m%d")

    st.sidebar.write(f"**Data de Início:** {start_date}")
    st.sidebar.write(f"**Data de Fim:** {end_date}")

    # Definir os parâmetros do EndPoint
    variavel = 'PRECTOTCORR,T2M'

    # URL NASA Power
    endpoint_nasa_power = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters={variavel}&community=SB&longitude={long_x}&latitude={lat_y}&start={start_date}&end={end_date}&format=JSON"

    # Aplicar a requisição e obter o conteúdo
    req_power = requests.get(endpoint_nasa_power).content

    # Carregar o conteúdo como json
    json_power = json.loads(req_power)

    # Converter json para DataFrame
    df = pd.DataFrame(json_power['properties']['parameter'])

    # Renomear colunas
    df.rename(columns={'PRECTOTCORR': 'prec', 'T2M': 'temp'}, inplace=True)

    # Convertendo o índice para datetime
    df.index = pd.to_datetime(df.index)

    # Extrair o mês e o ano
    df['month'] = df.index.month
    df['year'] = df.index.year

    # Calcular a média, desvio padrão e soma por ano e mês
    df_mean = df.groupby(['year', 'month']).mean()
    df_std = df.groupby(['year', 'month']).std()
    df_sum = df.groupby(['year', 'month']).sum()

    # --- Gráfico de Precipitação ---
    dfp = df_sum.reset_index()

    fig = px.line(
        dfp, x="month", y="prec", color="year",
        markers=True, title="Precipitação Mensal por Ano",
        labels={"month": "Mês", "prec": "Precipitação acumulada", "year": "Ano"},
        color_discrete_sequence=px.colors.sequential.Blues
    )

    fig.update_layout(
        xaxis=dict(
            title="Mês",
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
        ),
        yaxis=dict(title="Precipitação acumulada (mm/mês)"),
        legend_title="Ano",
        template="plotly_white"
    )

    st.plotly_chart(fig)

    # --- Gráfico de Temperatura ---
    dft = df_mean.reset_index()

    fig = px.line(
        dft, x="month", y="temp", color="year",
        markers=True, title="Temperatura Média Mensal por Ano",
        labels={"month": "Mês", "temp": "Temperatura", "year": "Ano"},
        color_discrete_sequence=px.colors.sequential.Reds
    )

    fig.update_layout(
        xaxis=dict(
            title="Mês",
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
        ),
        yaxis=dict(title="Temperatura média (°C)"),
        legend_title="Ano",
        template="plotly_white"
    )

    st.plotly_chart(fig)

else:
    st.sidebar.warning("Por favor, selecione um intervalo válido de datas.")
