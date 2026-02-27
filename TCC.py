# =============================================================================
# 🌍 Análise Climática por Município — Versão Melhorada
# =============================================================================
# Melhorias implementadas:
#   1. Mancha de precipitação: grade de pontos NASA POWER + Interpolação IDW/Kriging
#   2. Modelo ML: Prophet (Facebook) para previsão de precipitação e temperatura
#   3. Métricas de avaliação do modelo (MAE, RMSE)
#   4. Organização em abas (tabs) para melhor UX
#   5. Cache de requisições para melhor performance
# =============================================================================

# ── Bibliotecas ──────────────────────────────────────────────────────────────
import pandas as pd
import geopandas as gpd
import json
import requests
import io
import numpy as np
import datetime
import warnings
warnings.filterwarnings("ignore")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

import folium
import branca.colormap as cm
from streamlit_folium import st_folium

# Interpolação espacial
from scipy.spatial import cKDTree          # IDW
from scipy.interpolate import griddata     # fallback scipy
try:
    from pykrige.ok import OrdinaryKriging  # Kriging — pip install pykrige
    PYKRIGE_OK = True
except ImportError:
    PYKRIGE_OK = False

import plotly.express as px
import plotly.graph_objects as go

# Prophet — instale com: pip install prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

import streamlit as st

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Análise Climática por Município",
    page_icon="🌦️"
)

# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def obter_shapefile_municipios(cod_uf: str):
    """Baixa o shapefile do estado via API IBGE."""
    url = (
        f"https://servicodados.ibge.gov.br/api/v4/malhas/estados/{cod_uf}"
        "?formato=application/json&intrarregiao=Municipio&qualidade=intermediaria"
    )
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        return gpd.read_file(io.BytesIO(response.content))
    st.error(f"Erro ao baixar shapefile: {response.status_code}")
    return gpd.GeoDataFrame()


@st.cache_data(show_spinner=False)
def obter_municipios_por_estado(uf: str) -> pd.DataFrame:
    """Retorna DataFrame com código IBGE e nome dos municípios do estado."""
    url = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        dados = response.json()
        return pd.DataFrame([
            {"codigo_ibge": m["id"], "municipio": m["nome"], "uf": uf.upper()}
            for m in dados
        ])
    st.error(f"Erro ao obter municípios: {response.status_code}")
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def obter_dados_nasa(lat: float, lon: float, start: str, end: str,
                     variavel: str = "PRECTOTCORR,T2M") -> pd.DataFrame:
    """
    Consulta a API NASA POWER para um ponto (lat/lon) e retorna DataFrame
    com colunas: prec, temp, month, year.
    """
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={variavel}&community=SB"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start}&end={end}&format=JSON"
    )
    resp = requests.get(url, timeout=60)
    j = json.loads(resp.content)
    df = pd.DataFrame(j["properties"]["parameter"])
    df.rename(columns={"PRECTOTCORR": "prec", "T2M": "temp"}, inplace=True)
    df.index = pd.to_datetime(df.index)
    df["month"] = df.index.month
    df["year"] = df.index.year
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO: MANCHA DE PRECIPITAÇÃO (GRADE DE PONTOS)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def obter_grade_precipitacao(
    lat_c: float, lon_c: float,
    start: str, end: str,
    n_pontos: int = 5
) -> list[tuple[float, float, float]]:
    """
    Cria uma grade regular de n_pontos x n_pontos ao redor do centróide
    e consulta a precipitação acumulada de cada ponto via NASA POWER.

    Retorna lista de tuplas (lat, lon, prec_total).

    ⚠️  Cada ponto faz uma requisição HTTP — mantenha n_pontos ≤ 5 para não
    sobrecarregar a API (máximo 25 requisições).
    """
    delta = 0.3  # graus de raio ao redor do centróide
    lats = np.linspace(lat_c - delta, lat_c + delta, n_pontos)
    lons = np.linspace(lon_c - delta, lon_c + delta, n_pontos)

    pontos = []
    total = n_pontos * n_pontos
    barra = st.progress(0, text="Baixando grade de precipitação...")

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            try:
                url = (
                    "https://power.larc.nasa.gov/api/temporal/daily/point"
                    f"?parameters=PRECTOTCORR&community=SB"
                    f"&longitude={lon:.4f}&latitude={lat:.4f}"
                    f"&start={start}&end={end}&format=JSON"
                )
                resp = requests.get(url, timeout=60)
                dados = json.loads(resp.content)
                serie = dados["properties"]["parameter"]["PRECTOTCORR"]
                prec_total = sum(v for v in serie.values() if v != -999.0)
                pontos.append((lat, lon, prec_total))
            except Exception:
                pontos.append((lat, lon, 0.0))

            progresso = (i * n_pontos + j + 1) / total
            barra.progress(progresso, text=f"Grade: {i * n_pontos + j + 1}/{total} pontos")

    barra.empty()
    return pontos


# ══════════════════════════════════════════════════════════════════════════════
# INTERPOLAÇÃO ESPACIAL — IDW e Kriging Ordinário
# ══════════════════════════════════════════════════════════════════════════════

def interpolar_idw(
    lons_pts: np.ndarray,
    lats_pts: np.ndarray,
    valores: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    potencia: float = 2.0,
) -> np.ndarray:
    """
    Inverse Distance Weighting (IDW).

    Cada célula da grade recebe uma média ponderada dos pontos amostrados,
    onde o peso é 1 / distância^potencia.

    Parâmetros
    ----------
    lons_pts, lats_pts : coordenadas dos pontos amostrados (1-D)
    valores            : precipitação acumulada de cada ponto (1-D)
    lon_grid, lat_grid : grades 2-D de destino (meshgrid)
    potencia           : expoente da distância (default = 2 → clássico IDW²)

    Retorna
    -------
    grade 2-D com valores interpolados
    """
    pts_src = np.column_stack([lons_pts, lats_pts])
    pts_dst = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    # Distâncias euclidianas (graus — suficiente para municípios pequenos)
    arvore = cKDTree(pts_src)
    dists, idx = arvore.query(pts_dst, k=len(pts_src))

    # Caso ponto coincida exatamente com amostrado
    dists = np.where(dists == 0, 1e-10, dists)
    pesos = 1.0 / dists ** potencia
    z_interp = np.sum(pesos * valores[idx], axis=1) / np.sum(pesos, axis=1)

    return z_interp.reshape(lon_grid.shape)


def interpolar_kriging(
    lons_pts: np.ndarray,
    lats_pts: np.ndarray,
    valores: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    variogram_model: str = "spherical",
) -> np.ndarray:
    """
    Kriging Ordinário via pykrige.

    O Kriging estima não só o valor interpolado mas também a variância de
    estimativa (incerteza). Aqui retornamos apenas a estimativa (z_pred).

    Parâmetros
    ----------
    variogram_model : 'spherical' | 'exponential' | 'gaussian' | 'linear'
                      Controla como a correlação espacial decai com a distância.

    Retorna
    -------
    grade 2-D com valores estimados pelo Kriging
    """
    if not PYKRIGE_OK:
        raise ImportError("pykrige não instalado. Execute: pip install pykrige")

    ok = OrdinaryKriging(
        lons_pts, lats_pts, valores,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
        nlags=6,
    )
    z_pred, _ = ok.execute(
        "grid",
        np.unique(lon_grid[0]),   # vetor único de lons
        np.unique(lat_grid[:, 0]),  # vetor único de lats
    )
    return np.array(z_pred)


def grade_interpolada_para_geojson(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    z_grid: np.ndarray,
    colormap,
) -> dict:
    """
    Converte a grade interpolada em GeoJSON de polígonos (pixels) coloridos,
    prontos para renderizar no Folium via GeoJson layer.

    Cada pixel vira um retângulo com fill = cor do colormap.
    """
    dlat = abs(lat_grid[1, 0] - lat_grid[0, 0]) / 2
    dlon = abs(lon_grid[0, 1] - lon_grid[0, 0]) / 2

    features = []
    rows, cols = z_grid.shape
    for i in range(rows):
        for j in range(cols):
            v = float(z_grid[i, j])
            lat = float(lat_grid[i, j])
            lon = float(lon_grid[i, j])
            cor = colormap(v)
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon - dlon, lat - dlat],
                        [lon + dlon, lat - dlat],
                        [lon + dlon, lat + dlat],
                        [lon - dlon, lat + dlat],
                        [lon - dlon, lat - dlat],
                    ]],
                },
                "properties": {"prec": round(v, 1), "cor": cor},
            })
    return {"type": "FeatureCollection", "features": features}


def criar_mapa_interpolado(
    gdf_mun,
    pontos_grade: list,
    metodo: str = "IDW",
    resolucao_grid: int = 60,
    potencia_idw: float = 2.0,
    variogram_model: str = "spherical",
) -> folium.Map:
    """
    Gera mapa Folium com mancha de precipitação interpolada (IDW ou Kriging).

    Fluxo
    -----
    1. Extrai arrays de coordenadas e valores dos pontos amostrados
    2. Cria grade densa (resolucao_grid × resolucao_grid)
    3. Interpola via IDW ou Kriging Ordinário
    4. Converte grade → GeoJSON de pixels coloridos
    5. Adiciona ao mapa com colorbar (LinearColormap do branca)
    6. Plota pontos amostrados como marcadores
    """
    lats_pts = np.array([p[0] for p in pontos_grade])
    lons_pts = np.array([p[1] for p in pontos_grade])
    valores  = np.array([p[2] for p in pontos_grade], dtype=float)

    # Grade de destino dentro do bbox dos pontos amostrados
    lat_min, lat_max = lats_pts.min(), lats_pts.max()
    lon_min, lon_max = lons_pts.min(), lons_pts.max()

    lon_vec = np.linspace(lon_min, lon_max, resolucao_grid)
    lat_vec = np.linspace(lat_min, lat_max, resolucao_grid)
    lon_grid, lat_grid = np.meshgrid(lon_vec, lat_vec)

    # ── Interpolação ──────────────────────────────────────────────────────────
    if metodo == "Kriging" and PYKRIGE_OK:
        z_grid = interpolar_kriging(lons_pts, lats_pts, valores, lon_grid, lat_grid, variogram_model)
    else:
        if metodo == "Kriging" and not PYKRIGE_OK:
            st.warning("pykrige não encontrado — usando IDW como fallback. (`pip install pykrige`)")
        z_grid = interpolar_idw(lons_pts, lats_pts, valores, lon_grid, lat_grid, potencia_idw)

    # ── Colormap ──────────────────────────────────────────────────────────────
    vmin, vmax = float(z_grid.min()), float(z_grid.max())
    colormap = cm.LinearColormap(
        colors=["#313695", "#4575b4", "#74add1", "#abd9e9",
                "#e0f3f8", "#ffffbf", "#fee090", "#fdae61",
                "#f46d43", "#d73027", "#a50026"],
        vmin=vmin, vmax=vmax,
        caption="Precipitação acumulada (mm)",
    )

    # ── Mapa base ─────────────────────────────────────────────────────────────
    lat_c = gdf_mun.geometry.centroid.y.values[0]
    lon_c = gdf_mun.geometry.centroid.x.values[0]
    mapa = folium.Map(location=[lat_c, lon_c], zoom_start=10, tiles="CartoDB positron")

    # ── Camada interpolada ────────────────────────────────────────────────────
    geojson_pixels = grade_interpolada_para_geojson(lon_grid, lat_grid, z_grid, colormap)

    folium.GeoJson(
        data=geojson_pixels,
        name=f"Precipitação ({metodo})",
        style_function=lambda feat: {
            "fillColor": feat["properties"]["cor"],
            "color":     "none",
            "weight":    0,
            "fillOpacity": 0.75,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["prec"],
            aliases=["Prec. (mm):"],
            localize=True,
        ),
    ).add_to(mapa)

    # ── Contorno do município ─────────────────────────────────────────────────
    folium.GeoJson(
        data=gdf_mun,
        name="Município",
        style_function=lambda _: {
            "fillColor": "none",
            "color":     "#0d47a1",
            "weight":    2.5,
            "fillOpacity": 0,
        },
    ).add_to(mapa)

    # ── Pontos amostrados ─────────────────────────────────────────────────────
    for lat, lon, prec in pontos_grade:
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="black",
            weight=1.5,
            fill=True,
            fill_color=colormap(prec),
            fill_opacity=1.0,
            popup=folium.Popup(f"<b>Ponto amostrado</b><br>Prec: {prec:.1f} mm", max_width=200),
            tooltip=f"{prec:.1f} mm",
        ).add_to(mapa)

    colormap.add_to(mapa)
    folium.LayerControl().add_to(mapa)
    return mapa


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO: MODELO PROPHET
# ══════════════════════════════════════════════════════════════════════════════

def treinar_prophet(df: pd.DataFrame, coluna: str, periodos_futuros: int = 12,
                    freq: str = "MS") -> tuple:
    """
    Treina um modelo Prophet em dados mensais.

    Parâmetros
    ----------
    df : DataFrame com índice datetime e coluna 'prec' ou 'temp'
    coluna : 'prec' ou 'temp'
    periodos_futuros : meses a prever além do período histórico
    freq : frequência ('MS' = início do mês)

    Retorna
    -------
    (modelo, forecast_df, df_treino_prophet, metricas_dict)
    """
    # Agregar para mensal
    if freq == "MS":
        ts = df[coluna].resample("MS").sum() if coluna == "prec" \
             else df[coluna].resample("MS").mean()
    else:
        ts = df[coluna].resample(freq).mean()

    # Formato exigido pelo Prophet: colunas 'ds' e 'y'
    df_prophet = ts.reset_index()
    df_prophet.columns = ["ds", "y"]
    df_prophet = df_prophet.dropna()

    # Separar treino (80%) e teste (20%)
    split = int(len(df_prophet) * 0.8)
    df_train = df_prophet.iloc[:split]
    df_test = df_prophet.iloc[split:]

    # ── Modelo ──────────────────────────────────────────────────────────────
    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative" if coluna == "prec" else "additive",
        changepoint_prior_scale=0.1,        # regularização (evita overfitting)
        seasonality_prior_scale=10.0,
    )

    # Variáveis de regressão adicionais podem ser adicionadas aqui com
    # modelo.add_regressor('nome_variavel')

    modelo.fit(df_train)

    # ── Previsão ─────────────────────────────────────────────────────────────
    future = modelo.make_future_dataframe(
        periods=len(df_test) + periodos_futuros, freq=freq
    )
    forecast = modelo.predict(future)

    # ── Métricas no conjunto de teste ─────────────────────────────────────────
    y_pred_test = forecast.set_index("ds").loc[df_test["ds"].values, "yhat"].values
    y_true_test = df_test["y"].values
    mae = mean_absolute_error(y_true_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))

    metricas = {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "n_treino": len(df_train),
        "n_teste": len(df_test),
    }

    return modelo, forecast, df_prophet, metricas


def plotar_prophet(forecast: pd.DataFrame, df_historico: pd.DataFrame,
                   coluna: str, titulo: str) -> go.Figure:
    """Cria gráfico Plotly interativo com histórico + previsão + IC."""
    fig = go.Figure()

    # Intervalo de confiança
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(100,149,237,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Intervalo de Confiança (80%)",
        showlegend=True,
    ))

    # Previsão
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        mode="lines",
        line=dict(color="royalblue", width=2),
        name="Previsão (Prophet)",
    ))

    # Dados históricos
    fig.add_trace(go.Scatter(
        x=df_historico["ds"], y=df_historico["y"],
        mode="markers",
        marker=dict(color="black", size=5),
        name="Dados históricos",
    ))

    unidade = "mm/mês" if coluna == "prec" else "°C"
    fig.update_layout(
        title=titulo,
        xaxis_title="Data",
        yaxis_title=f"{'Precipitação' if coluna == 'prec' else 'Temperatura'} ({unidade})",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════

# ── Dicionário UF → Código IBGE ───────────────────────────────────────────────
dict_uf = {
    "AC": "12", "AL": "27", "AM": "13", "AP": "16", "BA": "29",
    "CE": "23", "DF": "53", "ES": "32", "GO": "52", "MA": "21",
    "MG": "31", "MS": "50", "MT": "51", "PA": "15", "PB": "25",
    "PE": "26", "PI": "22", "PR": "41", "RJ": "33", "RN": "24",
    "RO": "11", "RR": "14", "RS": "43", "SC": "42", "SE": "28",
    "SP": "35", "TO": "17",
}

# ── Título ────────────────────────────────────────────────────────────────────
st.title("🌍 Análise Climática por Município")
st.markdown(
    "Explore dados de **temperatura** e **precipitação**, visualize a "
    "**mancha espacial** de chuva e veja previsões com **Machine Learning** (Prophet)."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")

    uf_selecionado = st.selectbox("Estado:", sorted(dict_uf.keys()), index=list(sorted(dict_uf.keys())).index("MS"))

    with st.spinner("Carregando municípios..."):
        df_mun = obter_municipios_por_estado(uf_selecionado)

    cidade_selecionada = st.selectbox(
        "Município:", sorted(df_mun["municipio"]), key="cidade"
    )

    st.markdown("---")

    # Intervalo de datas
    data_range = st.date_input(
        "Intervalo de datas:",
        value=(datetime.date(2020, 1, 1), datetime.date(2025, 2, 28)),
        min_value=datetime.date(2000, 1, 1),
        max_value=datetime.date(2025, 12, 31),
    )

    st.markdown("---")

    # Parâmetros do Prophet
    st.subheader("🤖 Parâmetros do Modelo ML")
    meses_previsao = st.slider(
        "Meses a prever além do histórico:", 6, 36, 12, step=6
    )
    variavel_ml = st.radio(
        "Variável a prever:", ["Precipitação", "Temperatura"]
    )

    # Mancha de precipitação
    st.markdown("---")
    st.subheader("🗺️ Mancha de Precipitação")
    n_pontos_grade = st.slider(
        "Resolução da grade (n×n pontos):", 3, 5, 3, step=1,
        help="Cada ponto faz uma requisição à NASA POWER. Grades maiores são mais lentas."
    )
    metodo_interp = st.radio(
        "Método de interpolação:",
        ["IDW", "Kriging"],
        help="IDW = Inverse Distance Weighting (sempre disponível). Kriging requer `pip install pykrige`.",
    )
    resolucao_grid = st.slider("Resolução da grade interpolada:", 30, 120, 60, step=10,
                               help="Número de células em cada eixo. Mais células = imagem mais suave.")

    # Parâmetros específicos por método
    if metodo_interp == "IDW":
        potencia_idw = st.slider("Potência IDW (p):", 1.0, 4.0, 2.0, step=0.5,
                                 help="p=1 suaviza mais; p=4 valoriza os pontos mais próximos.")
        variogram_model = "spherical"  # não usado no IDW
    else:
        potencia_idw = 2.0             # não usado no Kriging
        variogram_model = st.selectbox(
            "Modelo de variograma:",
            ["spherical", "exponential", "gaussian", "linear"],
            help="Controla como a correlação espacial decai com a distância.",
        )

    gerar_mancha = st.button("🌧️ Gerar Mancha de Precipitação", use_container_width=True)

# ── Validações ────────────────────────────────────────────────────────────────
if not (isinstance(data_range, tuple) and len(data_range) == 2):
    st.sidebar.warning("Selecione um intervalo de datas válido.")
    st.stop()

start_date = data_range[0].strftime("%Y%m%d")
end_date = data_range[1].strftime("%Y%m%d")

# ── Carregar dados geográficos ────────────────────────────────────────────────
with st.spinner("Carregando shapefile..."):
    gdf_estado = obter_shapefile_municipios(dict_uf[uf_selecionado])
    gdf_estado = gdf_estado.set_crs(epsg=4674, allow_override=True)

geocod = str(df_mun[df_mun["municipio"] == cidade_selecionada]["codigo_ibge"].iloc[0])
gdf_mun = gdf_estado[gdf_estado["codarea"] == geocod]

long_x = gdf_mun.geometry.centroid.x.values[0]
lat_y  = gdf_mun.geometry.centroid.y.values[0]

# ── Carregar dados climáticos ─────────────────────────────────────────────────
with st.spinner("Consultando NASA POWER..."):
    df = obter_dados_nasa(lat_y, long_x, start_date, end_date)

# ══════════════════════════════════════════════════════════════════════════════
# ABAS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Mapa do Município",
    "📊 Análise Climática",
    "🌧️ Mancha de Precipitação",
    "🤖 Previsão com ML (Prophet)",
])

# ─────────────────────────────────────────────────────────────────────────────
# ABA 1 — MAPA DO MUNICÍPIO
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"📍 {cidade_selecionada} — {uf_selecionado}")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Latitude (centróide)", f"{lat_y:.4f}°")
    col_info2.metric("Longitude (centróide)", f"{long_x:.4f}°")
    col_info3.metric("Código IBGE", geocod)

    mapa_base = folium.Map(location=[lat_y, long_x], zoom_start=10, tiles="CartoDB positron")
    folium.GeoJson(
        data=gdf_mun,
        name="Município",
        tooltip=folium.GeoJsonTooltip(
            fields=["codarea"], aliases=["Código IBGE: "], localize=True
        ),
        style_function=lambda _: {
            "fillColor": "#bbdefb",
            "color": "#1a237e",
            "weight": 2,
            "fillOpacity": 0.5,
        },
    ).add_to(mapa_base)

    folium.Marker(
        location=[lat_y, long_x],
        popup=f"{cidade_selecionada}",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(mapa_base)

    st_folium(mapa_base, use_container_width=True, height=500)

# ─────────────────────────────────────────────────────────────────────────────
# ABA 2 — ANÁLISE CLIMÁTICA
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Precipitação e Temperatura Históricas")

    # Métricas resumo
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precipitação total", f"{df['prec'].sum():.0f} mm")
    col2.metric("Temp. média", f"{df['temp'].mean():.1f} °C")
    col3.metric("Temp. máxima", f"{df['temp'].max():.1f} °C")
    col4.metric("Temp. mínima", f"{df['temp'].min():.1f} °C")

    st.markdown("---")

    df_sum  = df.groupby(["year", "month"]).sum(numeric_only=True).reset_index()
    df_mean = df.groupby(["year", "month"]).mean(numeric_only=True).reset_index()

    # ── Precipitação ──────────────────────────────────────────────────────────
    fig_prec = px.line(
        df_sum, x="month", y="prec", color="year",
        markers=True, title="💧 Precipitação Mensal Acumulada por Ano",
        labels={"month": "Mês", "prec": "Precipitação (mm)", "year": "Ano"},
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig_prec.update_layout(
        xaxis=dict(
            tickmode="array", tickvals=list(range(1, 13)),
            ticktext=["Jan","Fev","Mar","Abr","Mai","Jun",
                      "Jul","Ago","Set","Out","Nov","Dez"],
        ),
        yaxis_title="Precipitação acumulada (mm/mês)",
        legend_title="Ano",
        template="plotly_white",
    )
    st.plotly_chart(fig_prec, use_container_width=True)

    # ── Temperatura ───────────────────────────────────────────────────────────
    fig_temp = px.line(
        df_mean, x="month", y="temp", color="year",
        markers=True, title="🌡️ Temperatura Média Mensal por Ano",
        labels={"month": "Mês", "temp": "Temperatura (°C)", "year": "Ano"},
        color_discrete_sequence=px.colors.sequential.Reds_r,
    )
    fig_temp.update_layout(
        xaxis=dict(
            tickmode="array", tickvals=list(range(1, 13)),
            ticktext=["Jan","Fev","Mar","Abr","Mai","Jun",
                      "Jul","Ago","Set","Out","Nov","Dez"],
        ),
        yaxis_title="Temperatura média (°C)",
        legend_title="Ano",
        template="plotly_white",
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # ── Boxplot mensal ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Distribuição Mensal (todos os anos)")
    df["mes_nome"] = df["month"].apply(
        lambda m: ["Jan","Fev","Mar","Abr","Mai","Jun",
                   "Jul","Ago","Set","Out","Nov","Dez"][m - 1]
    )
    df["mes_ordem"] = df["month"]

    col_box1, col_box2 = st.columns(2)
    with col_box1:
        fig_box_p = px.box(
            df, x="mes_nome", y="prec", category_orders={"mes_nome": ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]},
            title="Distribuição da Precipitação Diária por Mês",
            color_discrete_sequence=["#1565c0"],
        )
        fig_box_p.update_layout(template="plotly_white", xaxis_title="Mês", yaxis_title="mm/dia")
        st.plotly_chart(fig_box_p, use_container_width=True)

    with col_box2:
        fig_box_t = px.box(
            df, x="mes_nome", y="temp", category_orders={"mes_nome": ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]},
            title="Distribuição da Temperatura Diária por Mês",
            color_discrete_sequence=["#c62828"],
        )
        fig_box_t.update_layout(template="plotly_white", xaxis_title="Mês", yaxis_title="°C")
        st.plotly_chart(fig_box_t, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ABA 3 — MANCHA DE PRECIPITAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🌧️ Mancha Espacial de Precipitação — Interpolação Espacial")

    col_desc, col_metodo = st.columns([3, 1])
    with col_desc:
        st.markdown(
            f"""
            Consulta **{n_pontos_grade}×{n_pontos_grade} pontos** ao redor de
            **{cidade_selecionada}** na API NASA POWER e aplica **{metodo_interp}**
            para gerar uma superfície contínua de precipitação acumulada.

            | Método | Princípio | Quando usar |
            |--------|-----------|-------------|
            | **IDW** | Média ponderada pelo inverso da distância | Sempre disponível; rápido; bom para dados uniformes |
            | **Kriging** | Geoestatístico; considera estrutura espacial via variograma | Mais preciso quando há correlação espacial; requer `pykrige` |
            """
        )
    with col_metodo:
        st.info(f"**Método ativo:** {metodo_interp}\n\n**Potência/Variograma:** "
                f"{potencia_idw if metodo_interp == 'IDW' else variogram_model}")

    if gerar_mancha:
        with st.spinner("⬇️ Baixando pontos da NASA POWER..."):
            pontos = obter_grade_precipitacao(
                lat_y, long_x, start_date, end_date, n_pontos=n_pontos_grade
            )

        st.success(f"✅ {len(pontos)} pontos coletados. Interpolando com {metodo_interp}...")

        # ── Tabela de pontos amostrados ───────────────────────────────────────
        with st.expander("📋 Pontos amostrados (valores brutos)", expanded=False):
            df_pontos = pd.DataFrame(pontos, columns=["Latitude", "Longitude", "Precipitação (mm)"])
            st.dataframe(
                df_pontos.style.format({
                    "Latitude": "{:.4f}", "Longitude": "{:.4f}", "Precipitação (mm)": "{:.1f}"
                }).background_gradient(subset=["Precipitação (mm)"], cmap="Blues"),
                use_container_width=True,
                hide_index=True,
            )

        # ── Mapa interpolado ──────────────────────────────────────────────────
        with st.spinner(f"🎨 Gerando superfície {metodo_interp}..."):
            mapa_interp = criar_mapa_interpolado(
                gdf_mun, pontos,
                metodo=metodo_interp,
                resolucao_grid=resolucao_grid,
                potencia_idw=potencia_idw,
                variogram_model=variogram_model,
            )

        st_folium(mapa_interp, use_container_width=True, height=580)

        # ── Estatísticas ──────────────────────────────────────────────────────
        precs = [p[2] for p in pontos]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Máxima (pts amostrados)", f"{max(precs):.1f} mm")
        c2.metric("Mínima (pts amostrados)", f"{min(precs):.1f} mm")
        c3.metric("Média (pts amostrados)",  f"{np.mean(precs):.1f} mm")
        c4.metric("Desvio padrão",           f"{np.std(precs):.1f} mm")

        # ── Variograma empírico (apenas Kriging) ──────────────────────────────
        if metodo_interp == "Kriging" and PYKRIGE_OK:
            st.markdown("---")
            st.markdown("### 📐 Variograma Empírico vs Modelo Ajustado")
            st.markdown(
                """
                O variograma descreve **como a variância entre pontos cresce com a distância**.
                O Kriging ajusta um modelo matemático a essa curva para usar como função de peso
                durante a interpolação.
                """
            )
            lats_pts = np.array([p[0] for p in pontos])
            lons_pts = np.array([p[1] for p in pontos])
            valores  = np.array([p[2] for p in pontos], dtype=float)
            ok_plot = OrdinaryKriging(
                lons_pts, lats_pts, valores,
                variogram_model=variogram_model,
                verbose=False, enable_plotting=False, nlags=6,
            )
            lags    = ok_plot.lags
            gamma   = ok_plot.semivariance

            fig_vario = go.Figure()
            fig_vario.add_trace(go.Scatter(
                x=lags, y=gamma, mode="markers+lines",
                name="Variograma empírico",
                marker=dict(size=8, color="#1565c0"),
            ))
            fig_vario.update_layout(
                title=f"Variograma Empírico — modelo: {variogram_model}",
                xaxis_title="Distância (graus)",
                yaxis_title="Semivariância",
                template="plotly_white",
            )
            st.plotly_chart(fig_vario, use_container_width=True)

    else:
        st.info("⬅️ Configure os parâmetros na barra lateral e clique em **Gerar Mancha de Precipitação**.")
        mapa_vazio = folium.Map(location=[lat_y, long_x], zoom_start=10, tiles="CartoDB positron")
        folium.GeoJson(
            data=gdf_mun,
            style_function=lambda _: {"fillColor": "#e3f2fd", "color": "#1a237e", "weight": 2},
        ).add_to(mapa_vazio)
        st_folium(mapa_vazio, use_container_width=True, height=500)

# ─────────────────────────────────────────────────────────────────────────────
# ABA 4 — PREVISÃO COM PROPHET
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🤖 Previsão com Prophet (Facebook / Meta)")

    with st.expander("📖 Como funciona o Prophet?", expanded=False):
        st.markdown(
            """
            **Prophet** é um modelo de previsão de séries temporais desenvolvido pelo
            Facebook/Meta (Taylor & Letham, 2017). Ele é especialmente adequado para
            dados climáticos porque:

            - **Decompõe** a série em: tendência + sazonalidade anual + feriados (opcional)
            - **Lida bem** com dados faltantes e mudanças abruptas de tendência (*changepoints*)
            - **Não exige** pré-processamento complexo
            - Oferece **intervalos de incerteza** automáticos

            **Referências GitHub relacionadas:**
            - [facebook/prophet](https://github.com/facebook/prophet) — repositório oficial
            - [climate-forecasting-prophet](https://github.com/topics/climate-forecasting) — projetos de previsão climática
            - Estudos usando Prophet para precipitação: e.g., Bui et al. (2020), Mouatadid & Adamowski (2017)

            **Parâmetros importantes:**
            | Parâmetro | O que controla |
            |-----------|---------------|
            | `changepoint_prior_scale` | Flexibilidade da tendência (0.05–0.5) |
            | `seasonality_mode` | `additive` (efeito constante) vs `multiplicative` (proporcional) |
            | `yearly_seasonality` | Captura ciclos anuais (verão/inverno) |
            """
        )

    coluna = "prec" if variavel_ml == "Precipitação" else "temp"
    unidade = "mm/mês" if coluna == "prec" else "°C"
    emoji  = "💧" if coluna == "prec" else "🌡️"

    with st.spinner(f"Treinando modelo Prophet para {variavel_ml}..."):
        modelo, forecast, df_hist_prophet, metricas = treinar_prophet(
            df, coluna=coluna, periodos_futuros=meses_previsao
        )

    # ── Métricas do modelo ────────────────────────────────────────────────────
    st.markdown("### 📐 Métricas de Avaliação (conjunto de teste)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{metricas['MAE']} {unidade}",
              help="Erro Absoluto Médio — quanto o modelo erra em média")
    m2.metric("RMSE", f"{metricas['RMSE']} {unidade}",
              help="Raiz do Erro Quadrático Médio — penaliza erros grandes")
    m3.metric("Amostras treino", metricas["n_treino"])
    m4.metric("Amostras teste", metricas["n_teste"])

    st.markdown("---")

    # ── Gráfico principal ─────────────────────────────────────────────────────
    fig_forecast = plotar_prophet(
        forecast, df_hist_prophet, coluna,
        f"{emoji} Previsão de {variavel_ml} — {cidade_selecionada} ({meses_previsao} meses à frente)"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # ── Componentes do modelo (sazonalidade e tendência) ──────────────────────
    st.markdown("### 🔍 Decomposição do Modelo (Componentes)")
    fig_comp, axes = plt.subplots(2, 1, figsize=(12, 6))
    modelo.plot_components(forecast, ax=axes)
    plt.tight_layout()
    st.pyplot(fig_comp, use_container_width=True)
    plt.close()

    # ── Tabela de previsões futuras ────────────────────────────────────────────
    st.markdown("### 📅 Tabela de Previsões Futuras")
    fut_mask = forecast["ds"] > df_hist_prophet["ds"].max()
    df_futuro = forecast[fut_mask][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    df_futuro.columns = ["Data", "Previsão", "Limite Inferior (80%)", "Limite Superior (80%)"]
    df_futuro["Data"] = df_futuro["Data"].dt.strftime("%b/%Y")

    st.dataframe(
        df_futuro.style.format({
            "Previsão": "{:.2f}",
            "Limite Inferior (80%)": "{:.2f}",
            "Limite Superior (80%)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Interpretação ─────────────────────────────────────────────────────────
    with st.expander("💡 Como interpretar os resultados?"):
        st.markdown(
            f"""
            - **Linha azul** = previsão pontual (valor mais provável)
            - **Área sombreada** = intervalo de confiança de 80% (20% de chance do valor real ficar fora)
            - **MAE de {metricas['MAE']} {unidade}** = em média, o modelo erra esse valor no conjunto de teste
            - **RMSE de {metricas['RMSE']} {unidade}** = versão mais severa do MAE (penaliza erros grandes)
            - A **decomposição** mostra a tendência de longo prazo e o padrão sazonal anual identificados pelo modelo

            > ⚠️ Previsões climáticas em escala mensal têm incerteza inerente. Use os resultados como
            > referência exploratória, não como previsão operacional.
            """
        )

# ── Rodapé ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Dados: [NASA POWER](https://power.larc.nasa.gov/) · "
    "Malha municipal: [IBGE](https://servicodados.ibge.gov.br) · "
    "Modelo: [Prophet — Facebook/Meta](https://facebook.github.io/prophet/)"
)
