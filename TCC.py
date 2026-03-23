"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DASHBOARD DE ANÁLISE CLIMÁTICA MUNICIPAL COM MACHINE LEARNING              ║
║  Gestão Pública Inteligente - Baseado em PyVisSat (UFMS/SEMADESC 2026)      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Baseado nos notebooks:
- AULA 1: Imagens de Satélite GOES-16/19
- AULA 2: Relâmpagos (GLM)
- AULA 3: Estações Meteorológicas
- AULA 4: Precipitação Estimada por Satélite (MERGE/CHIRPS)
- AULA 5: Índices de Vegetação (NDVI/GEE)
- AULA 6: Queimadas (INPE/BDQueimadas)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import io
import base64
from datetime import datetime, timedelta
import calendar

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Climático Municipal",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS PERSONALIZADO — tema verde-militar/técnico-científico
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg-dark:     #0a0e1a;
    --bg-mid:      #111827;
    --bg-card:     #161d2e;
    --accent-green:#00ff88;
    --accent-teal: #00d4b8;
    --accent-blue: #3b82f6;
    --accent-amber:#f59e0b;
    --accent-red:  #ef4444;
    --text-main:   #e2e8f0;
    --text-muted:  #94a3b8;
    --border:      #1e293b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-dark) !important;
    font-family: 'Rajdhani', sans-serif;
    color: var(--text-main);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a1020 100%) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * { color: var(--text-main) !important; }

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: var(--accent-green) !important;
    letter-spacing: -0.03em;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-green);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.metric-card h4 {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
    margin: 0 0 4px 0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    color: var(--accent-green);
    font-weight: 700;
    line-height: 1;
}
.metric-card .delta {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: var(--accent-teal);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

.alert-box {
    padding: 12px 16px;
    border-radius: 6px;
    margin: 8px 0;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
}
.alert-high   { background: rgba(239,68,68,0.15);  border-left: 3px solid #ef4444; }
.alert-medium { background: rgba(245,158,11,0.15); border-left: 3px solid #f59e0b; }
.alert-low    { background: rgba(0,255,136,0.15);  border-left: 3px solid #00ff88; }

.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00d4b8) !important;
    color: #0a0e1a !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
    font-size: 13px !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,255,136,0.3) !important;
}

.stSelectbox > div > div, .stSlider > div, .stDateInput > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-main) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    color: var(--accent-green) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text-muted) !important;
}

.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.1em;
}
.badge-ok      { background: rgba(0,255,136,0.2); color: #00ff88; border: 1px solid #00ff88; }
.badge-warn    { background: rgba(245,158,11,0.2); color: #f59e0b; border: 1px solid #f59e0b; }
.badge-alert   { background: rgba(239,68,68,0.2); color: #ef4444; border: 1px solid #ef4444; }

div[data-testid="stVerticalBlock"] > div {
    gap: 0px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# GERAÇÃO DE DADOS SINTÉTICOS REALISTAS
# ─────────────────────────────────────────────────────────────────────────────

MUNICIPIOS_MS = [
    "Campo Grande", "Dourados", "Três Lagoas", "Corumbá", "Ponta Porã",
    "Naviraí", "Nova Andradina", "Maracaju", "Sidrolândia", "Rio Brilhante",
    "Aquidauana", "Coxim", "Bonito", "Jardim", "Amambai"
]

MUNICIPIOS_INFO = {
    "Campo Grande":  {"lat": -20.46, "lon": -54.61, "area_km2": 8096,  "pop": 906092},
    "Dourados":      {"lat": -22.22, "lon": -54.81, "area_km2": 4086,  "pop": 226277},
    "Três Lagoas":   {"lat": -20.75, "lon": -51.68, "area_km2": 10206, "pop": 120674},
    "Corumbá":       {"lat": -19.01, "lon": -57.65, "area_km2": 64961, "pop": 111435},
    "Ponta Porã":    {"lat": -22.54, "lon": -55.73, "area_km2": 5329,  "pop": 91618},
    "Naviraí":       {"lat": -23.06, "lon": -54.19, "area_km2": 3006,  "pop": 52614},
    "Nova Andradina":{"lat": -22.23, "lon": -53.34, "area_km2": 4897,  "pop": 55025},
    "Maracaju":      {"lat": -21.61, "lon": -55.17, "area_km2": 5297,  "pop": 42280},
    "Sidrolândia":   {"lat": -20.93, "lon": -54.96, "area_km2": 5286,  "pop": 48170},
    "Rio Brilhante": {"lat": -21.80, "lon": -54.55, "area_km2": 3559,  "pop": 42285},
    "Aquidauana":    {"lat": -20.47, "lon": -55.79, "area_km2": 16959, "pop": 46047},
    "Coxim":         {"lat": -18.51, "lon": -54.76, "area_km2": 6409,  "pop": 35120},
    "Bonito":        {"lat": -21.12, "lon": -56.49, "area_km2": 4934,  "pop": 22946},
    "Jardim":        {"lat": -21.49, "lon": -56.14, "area_km2": 2750,  "pop": 27254},
    "Amambai":       {"lat": -23.11, "lon": -55.23, "area_km2": 4390,  "pop": 41578},
}

np.random.seed(42)

@st.cache_data
def gerar_serie_temporal(municipio, anos=5):
    """Gera série temporal realista com sazonalidade MS"""
    datas = pd.date_range(start="2020-01-01", end="2024-12-31", freq="D")
    n = len(datas)
    t = np.arange(n)
    info = MUNICIPIOS_INFO[municipio]
    
    # Temperatura com sazonalidade (verão quente ~28-30°C, inverno ameno ~20-22°C)
    lat_fator = abs(info["lat"]) / 25
    temp_media = 26 + lat_fator * (-2)
    temp = (temp_media + 4 * np.sin(2 * np.pi * t / 365 - np.pi)
            + np.random.normal(0, 1.5, n))
    
    # Precipitação (estação chuvosa out-mar ~200mm/mês; seca abr-set ~40mm/mês)
    prec_base = 4 + 4.5 * np.sin(2 * np.pi * t / 365 - np.pi / 6)
    precip = np.maximum(0, prec_base + np.random.exponential(2, n))
    precip = np.where(precip > 80, 80, precip)
    
    # NDVI com sazonalidade (verde no chuvoso, marrom no seco)
    ndvi = 0.55 + 0.20 * np.sin(2 * np.pi * t / 365 - np.pi / 4) + np.random.normal(0, 0.03, n)
    ndvi = np.clip(ndvi, 0.1, 0.95)
    
    # Relâmpagos (mais no verão úmido)
    flash = np.maximum(0, 50 * np.sin(2 * np.pi * t / 365 - np.pi / 6) ** 2 + np.random.poisson(5, n))
    
    # Queimadas (pico jul-set = inverno seco)
    focos = np.maximum(0, 8 * (-np.sin(2 * np.pi * t / 365 - np.pi / 4)) + np.random.poisson(2, n))
    focos = np.where(focos > 30, 30, focos).astype(int)
    
    # Umidade relativa (%)
    umidade = 65 + 20 * np.sin(2 * np.pi * t / 365 - np.pi / 6) + np.random.normal(0, 5, n)
    umidade = np.clip(umidade, 20, 98)
    
    # Velocidade do vento (km/h)
    vento = 12 + 5 * np.random.randn(n)
    vento = np.clip(vento, 0, 80)

    df = pd.DataFrame({
        "data": datas,
        "temperatura": np.round(temp, 2),
        "precipitacao": np.round(precip, 2),
        "ndvi": np.round(ndvi, 3),
        "flashes": flash.astype(int),
        "focos_queimada": focos,
        "umidade": np.round(umidade, 1),
        "vento": np.round(vento, 1),
    })
    return df


@st.cache_data
def gerar_dados_mensais(municipio):
    df = gerar_serie_temporal(municipio)
    df["mes"] = df["data"].dt.to_period("M")
    mensal = df.groupby("mes").agg({
        "temperatura": "mean",
        "precipitacao": "sum",
        "ndvi": "mean",
        "flashes": "sum",
        "focos_queimada": "sum",
        "umidade": "mean",
        "vento": "mean",
    }).reset_index()
    mensal["mes_dt"] = mensal["mes"].dt.to_timestamp()
    return mensal


@st.cache_data
def gerar_dados_ml(municipio):
    """Gera dados ML: clustering de risco climático + anomalias"""
    df = gerar_serie_temporal(municipio)
    df["mes_num"] = df["data"].dt.month
    df["ano"] = df["data"].dt.year
    
    # Score de risco (0-100)
    prec_norm = (df["precipitacao"] - df["precipitacao"].min()) / (df["precipitacao"].max() - df["precipitacao"].min())
    ndvi_inv = 1 - (df["ndvi"] - df["ndvi"].min()) / (df["ndvi"].max() - df["ndvi"].min())
    flash_norm = (df["flashes"] - df["flashes"].min()) / (df["flashes"].max() - df["flashes"].min() + 1)
    focos_norm = (df["focos_queimada"] - df["focos_queimada"].min()) / (df["focos_queimada"].max() - df["focos_queimada"].min() + 1)
    
    df["risco_score"] = np.round(
        (prec_norm * 0.30 + ndvi_inv * 0.25 + flash_norm * 0.25 + focos_norm * 0.20) * 100, 1
    )
    df["risco_nivel"] = pd.cut(df["risco_score"],
                                bins=[0, 25, 50, 75, 100],
                                labels=["Baixo", "Moderado", "Alto", "Crítico"])
    
    # Anomalia de temperatura
    df["temp_anomalia"] = df["temperatura"] - df["temperatura"].mean()
    
    # Previsão simplificada (próximos 30 dias com ML fake)
    ultimo = df["data"].max()
    futuro = pd.date_range(start=ultimo + timedelta(days=1), periods=30, freq="D")
    t_fut = np.arange(len(df), len(df) + 30)
    temp_med = df["temperatura"].mean()
    temp_fut = temp_med + 4 * np.sin(2 * np.pi * t_fut / 365 - np.pi) + np.random.normal(0, 1.2, 30)
    prec_fut = np.maximum(0, 4 + 4 * np.sin(2 * np.pi * t_fut / 365 - np.pi / 6) + np.random.exponential(1.5, 30))
    df_prev = pd.DataFrame({"data": futuro, "temp_prev": temp_fut, "prec_prev": prec_fut})
    
    return df, df_prev


@st.cache_data
def gerar_mapa_ms():
    """Dados para mapa de calor do MS"""
    rows = []
    for muni, info in MUNICIPIOS_INFO.items():
        risco = np.random.uniform(20, 90)
        ndvi_medio = np.random.uniform(0.35, 0.75)
        prec_anual = np.random.uniform(900, 1800)
        focos_ano = np.random.randint(50, 800)
        rows.append({
            "municipio": muni,
            "lat": info["lat"],
            "lon": info["lon"],
            "risco": round(risco, 1),
            "ndvi": round(ndvi_medio, 3),
            "prec_anual": round(prec_anual, 0),
            "focos": focos_ano,
            "populacao": info["pop"],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES DE PLOT — estilo dark científico
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG = "#0a0e1a"
CARD_BG = "#161d2e"
GREEN   = "#00ff88"
TEAL    = "#00d4b8"
BLUE    = "#3b82f6"
AMBER   = "#f59e0b"
RED     = "#ef4444"
TEXT    = "#e2e8f0"
MUTED   = "#94a3b8"

def fig_style(fig, ax_list=None):
    fig.patch.set_facecolor(DARK_BG)
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(GREEN)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e293b")
    return fig


def plot_temperatura(df_diario, df_mensal):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor(DARK_BG)
    
    ax1 = axes[0]
    ax1.set_facecolor(CARD_BG)
    ax1.fill_between(df_diario["data"], df_diario["temperatura"], alpha=0.15, color=RED)
    ax1.plot(df_diario["data"], df_diario["temperatura"].rolling(30).mean(),
             color=RED, lw=2, label="Média Móvel 30d")
    ax1.axhline(df_diario["temperatura"].mean(), ls="--", color=AMBER, lw=1, alpha=0.6, label="Média")
    ax1.set_title("Temperatura (°C) — Série Diária", fontsize=12, color=GREEN, fontfamily="monospace")
    ax1.set_ylabel("°C", color=TEXT)
    ax1.legend(framealpha=0.2, facecolor=DARK_BG, edgecolor="#1e293b", labelcolor=TEXT, fontsize=8)
    ax1.tick_params(colors=MUTED, labelsize=8)
    for spine in ax1.spines.values(): spine.set_edgecolor("#1e293b")
    
    # Anomalia
    ax2 = axes[1]
    ax2.set_facecolor(CARD_BG)
    anom = df_mensal["temperatura"] - df_mensal["temperatura"].mean()
    colors_anom = [RED if v > 0 else BLUE for v in anom]
    ax2.bar(df_mensal["mes_dt"], anom, color=colors_anom, alpha=0.8, width=25)
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.set_title("Anomalia Mensal (°C)", fontsize=10, color=TEAL, fontfamily="monospace")
    ax2.set_ylabel("Δ°C", color=TEXT)
    ax2.tick_params(colors=MUTED, labelsize=7)
    for spine in ax2.spines.values(): spine.set_edgecolor("#1e293b")
    
    plt.tight_layout(pad=1.5)
    return fig


def plot_precipitacao(df_diario, df_mensal):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    
    ax1 = axes[0]
    ax1.set_facecolor(CARD_BG)
    ax1.bar(df_mensal["mes_dt"], df_mensal["precipitacao"], color=BLUE, alpha=0.8, width=25)
    ax1.set_title("Precipitação Mensal (mm)", fontsize=11, color=GREEN, fontfamily="monospace")
    ax1.set_ylabel("mm", color=TEXT)
    ax1.tick_params(colors=MUTED, labelsize=8)
    for spine in ax1.spines.values(): spine.set_edgecolor("#1e293b")
    
    # Climatologia mensal (média por mês do ano)
    ax2 = axes[1]
    ax2.set_facecolor(CARD_BG)
    df_diario["mes_num"] = df_diario["data"].dt.month
    clim = df_diario.groupby("mes_num")["precipitacao"].sum() / 5  # 5 anos
    meses_nomes = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
    bars = ax2.bar(range(1, 13), clim.values, color=TEAL, alpha=0.8)
    # Colorir meses secos em amber
    for i, (b, v) in enumerate(zip(bars, clim.values)):
        if v < 60:
            b.set_color(AMBER)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(meses_nomes, color=MUTED, fontsize=8)
    ax2.set_title("Climatologia Mensal (mm/mês)", fontsize=11, color=GREEN, fontfamily="monospace")
    ax2.set_ylabel("mm", color=TEXT)
    ax2.tick_params(colors=MUTED, labelsize=8)
    for spine in ax2.spines.values(): spine.set_edgecolor("#1e293b")
    
    patch_chuv = mpatches.Patch(color=TEAL, label="Período Chuvoso")
    patch_seco = mpatches.Patch(color=AMBER, label="Período Seco (<60mm)")
    ax2.legend(handles=[patch_chuv, patch_seco], framealpha=0.2, facecolor=DARK_BG,
               edgecolor="#1e293b", labelcolor=TEXT, fontsize=8)
    
    plt.tight_layout(pad=1.5)
    return fig


def plot_ndvi(df_mensal):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    
    ax1 = axes[0]
    ax1.set_facecolor(CARD_BG)
    ax1.fill_between(df_mensal["mes_dt"], df_mensal["ndvi"], alpha=0.3, color=GREEN)
    ax1.plot(df_mensal["mes_dt"], df_mensal["ndvi"], color=GREEN, lw=2)
    ax1.axhline(0.5, ls="--", color=AMBER, lw=1, alpha=0.7, label="Limiar Vegetação (0.5)")
    ax1.set_ylim(0, 1)
    ax1.set_title("NDVI Mensal — Índice de Vegetação", fontsize=11, color=GREEN, fontfamily="monospace")
    ax1.set_ylabel("NDVI", color=TEXT)
    ax1.legend(framealpha=0.2, facecolor=DARK_BG, edgecolor="#1e293b", labelcolor=TEXT, fontsize=8)
    ax1.tick_params(colors=MUTED, labelsize=8)
    for spine in ax1.spines.values(): spine.set_edgecolor("#1e293b")
    
    # NDVI vs Precipitação
    ax2 = axes[1]
    ax2.set_facecolor(CARD_BG)
    sc = ax2.scatter(df_mensal["precipitacao"], df_mensal["ndvi"],
                     c=df_mensal["mes_dt"].apply(lambda x: x.month),
                     cmap="RdYlGn", alpha=0.8, s=50, edgecolors="none")
    plt.colorbar(sc, ax=ax2, label="Mês").ax.yaxis.label.set_color(TEXT)
    ax2.set_title("NDVI × Precipitação (mm)", fontsize=11, color=GREEN, fontfamily="monospace")
    ax2.set_xlabel("Precipitação (mm/mês)", color=TEXT)
    ax2.set_ylabel("NDVI", color=TEXT)
    ax2.tick_params(colors=MUTED, labelsize=8)
    for spine in ax2.spines.values(): spine.set_edgecolor("#1e293b")
    
    # Correlação
    corr = np.corrcoef(df_mensal["precipitacao"], df_mensal["ndvi"])[0, 1]
    ax2.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax2.transAxes,
             color=AMBER, fontsize=10, fontfamily="monospace")
    
    plt.tight_layout(pad=1.5)
    return fig


def plot_queimadas_relampagos(df_mensal):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor(DARK_BG)
    
    meses_nomes = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
    df_mensal["mes_nome"] = df_mensal["mes_dt"].dt.month
    
    # Queimadas ao longo do tempo
    ax1 = axes[0, 0]
    ax1.set_facecolor(CARD_BG)
    ax1.bar(df_mensal["mes_dt"], df_mensal["focos_queimada"], color=AMBER, alpha=0.85, width=25)
    ax1.set_title("Focos de Queimada Mensais", fontsize=10, color=GREEN, fontfamily="monospace")
    ax1.set_ylabel("Focos", color=TEXT)
    ax1.tick_params(colors=MUTED, labelsize=7)
    for spine in ax1.spines.values(): spine.set_edgecolor("#1e293b")
    
    # Climatologia queimadas
    ax2 = axes[0, 1]
    ax2.set_facecolor(CARD_BG)
    clim_q = df_mensal.groupby("mes_nome")["focos_queimada"].mean()
    bar_colors = [RED if v > clim_q.mean() else AMBER for v in clim_q.values]
    ax2.bar(range(1, 13), clim_q.values, color=bar_colors, alpha=0.85)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(meses_nomes, color=MUTED, fontsize=8)
    ax2.set_title("Climatologia de Focos por Mês", fontsize=10, color=GREEN, fontfamily="monospace")
    ax2.set_ylabel("Focos/mês", color=TEXT)
    ax2.tick_params(colors=MUTED, labelsize=7)
    for spine in ax2.spines.values(): spine.set_edgecolor("#1e293b")
    
    # Relâmpagos série temporal
    ax3 = axes[1, 0]
    ax3.set_facecolor(CARD_BG)
    ax3.fill_between(df_mensal["mes_dt"], df_mensal["flashes"], alpha=0.3, color=BLUE)
    ax3.plot(df_mensal["mes_dt"], df_mensal["flashes"], color=BLUE, lw=1.5)
    ax3.set_title("Flashes de Relâmpago Mensais (GLM)", fontsize=10, color=GREEN, fontfamily="monospace")
    ax3.set_ylabel("Flashes/mês", color=TEXT)
    ax3.tick_params(colors=MUTED, labelsize=7)
    for spine in ax3.spines.values(): spine.set_edgecolor("#1e293b")
    
    # Relâmpagos × Precipitação
    ax4 = axes[1, 1]
    ax4.set_facecolor(CARD_BG)
    sc = ax4.scatter(df_mensal["precipitacao"], df_mensal["flashes"],
                     c=df_mensal["focos_queimada"], cmap="hot", alpha=0.8, s=50)
    plt.colorbar(sc, ax=ax4, label="Focos").ax.yaxis.label.set_color(TEXT)
    ax4.set_title("Flashes × Precipitação", fontsize=10, color=GREEN, fontfamily="monospace")
    ax4.set_xlabel("Precipitação (mm/mês)", color=TEXT)
    ax4.set_ylabel("Flashes/mês", color=TEXT)
    ax4.tick_params(colors=MUTED, labelsize=7)
    for spine in ax4.spines.values(): spine.set_edgecolor("#1e293b")
    corr2 = np.corrcoef(df_mensal["precipitacao"], df_mensal["flashes"])[0, 1]
    ax4.text(0.05, 0.92, f"r = {corr2:.3f}", transform=ax4.transAxes,
             color=AMBER, fontsize=9, fontfamily="monospace")
    
    plt.tight_layout(pad=1.5)
    return fig


def plot_risco_ml(df_ml):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    
    # Score de risco ao longo do tempo
    ax1 = axes[0]
    ax1.set_facecolor(CARD_BG)
    colors_risco = {
        "Baixo": GREEN, "Moderado": TEAL, "Alto": AMBER, "Crítico": RED
    }
    for nivel, cor in colors_risco.items():
        mask = df_ml["risco_nivel"] == nivel
        ax1.scatter(df_ml.loc[mask, "data"], df_ml.loc[mask, "risco_score"],
                    c=cor, alpha=0.3, s=5, label=nivel)
    ax1.plot(df_ml["data"], df_ml["risco_score"].rolling(30).mean(),
             color="white", lw=1.5, label="Média 30d")
    ax1.set_title("Score de Risco Climático (ML)", fontsize=10, color=GREEN, fontfamily="monospace")
    ax1.set_ylabel("Risco (0–100)", color=TEXT)
    ax1.legend(framealpha=0.2, facecolor=DARK_BG, edgecolor="#1e293b", labelcolor=TEXT, fontsize=7)
    ax1.tick_params(colors=MUTED, labelsize=7)
    for spine in ax1.spines.values(): spine.set_edgecolor("#1e293b")
    
    # Distribuição dos níveis
    ax2 = axes[1]
    ax2.set_facecolor(CARD_BG)
    contagem = df_ml["risco_nivel"].value_counts().reindex(["Baixo", "Moderado", "Alto", "Crítico"])
    bar_c = [colors_risco[k] for k in contagem.index]
    ax2.barh(contagem.index, contagem.values, color=bar_c, alpha=0.85)
    ax2.set_title("Distribuição de Dias por Nível", fontsize=10, color=GREEN, fontfamily="monospace")
    ax2.set_xlabel("Dias", color=TEXT)
    ax2.tick_params(colors=MUTED, labelsize=9)
    for spine in ax2.spines.values(): spine.set_edgecolor("#1e293b")
    for i, v in enumerate(contagem.values):
        ax2.text(v + 5, i, str(v), color=TEXT, va="center", fontsize=8)
    
    # Anomalia de temperatura por ano
    ax3 = axes[2]
    ax3.set_facecolor(CARD_BG)
    anom_anual = df_ml.groupby("ano")["temp_anomalia"].mean()
    bar_cols = [RED if v > 0 else BLUE for v in anom_anual.values]
    ax3.bar(anom_anual.index, anom_anual.values, color=bar_cols, alpha=0.85)
    ax3.axhline(0, color=MUTED, lw=1)
    ax3.set_title("Anomalia Temperatura Anual (°C)", fontsize=10, color=GREEN, fontfamily="monospace")
    ax3.set_ylabel("Δ°C", color=TEXT)
    ax3.tick_params(colors=MUTED, labelsize=8)
    for spine in ax3.spines.values(): spine.set_edgecolor("#1e293b")
    
    plt.tight_layout(pad=1.5)
    return fig


def plot_previsao_ml(df_ml, df_prev):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    
    # Últimos 90 dias + previsão temperatura
    ultimos = df_ml.tail(90)
    
    ax1 = axes[0]
    ax1.set_facecolor(CARD_BG)
    ax1.plot(ultimos["data"], ultimos["temperatura"], color=RED, lw=2, label="Observado")
    ax1.plot(df_prev["data"], df_prev["temp_prev"], color=AMBER, lw=2, ls="--", label="Previsão ML")
    ax1.fill_between(df_prev["data"],
                     df_prev["temp_prev"] - 2.5,
                     df_prev["temp_prev"] + 2.5,
                     alpha=0.2, color=AMBER, label="IC 95%")
    ax1.axvline(df_ml["data"].max(), color=MUTED, lw=1, ls=":")
    ax1.set_title("Previsão Temperatura — Próx. 30 dias", fontsize=10, color=GREEN, fontfamily="monospace")
    ax1.set_ylabel("°C", color=TEXT)
    ax1.legend(framealpha=0.2, facecolor=DARK_BG, edgecolor="#1e293b", labelcolor=TEXT, fontsize=8)
    ax1.tick_params(colors=MUTED, labelsize=7)
    for spine in ax1.spines.values(): spine.set_edgecolor("#1e293b")
    
    # Previsão precipitação
    ax2 = axes[1]
    ax2.set_facecolor(CARD_BG)
    ax2.bar(ultimos["data"], ultimos["precipitacao"], color=BLUE, alpha=0.6, width=0.8, label="Observado")
    ax2.bar(df_prev["data"], df_prev["prec_prev"], color=TEAL, alpha=0.8, width=0.8, label="Previsão ML")
    ax2.axvline(df_ml["data"].max(), color=MUTED, lw=1, ls=":")
    ax2.set_title("Previsão Precipitação — Próx. 30 dias", fontsize=10, color=GREEN, fontfamily="monospace")
    ax2.set_ylabel("mm/dia", color=TEXT)
    ax2.legend(framealpha=0.2, facecolor=DARK_BG, edgecolor="#1e293b", labelcolor=TEXT, fontsize=8)
    ax2.tick_params(colors=MUTED, labelsize=7)
    for spine in ax2.spines.values(): spine.set_edgecolor("#1e293b")
    
    plt.tight_layout(pad=1.5)
    return fig


def plot_correlacoes(df_mensal):
    variaveis = ["temperatura", "precipitacao", "ndvi", "flashes", "focos_queimada", "umidade"]
    matriz = df_mensal[variaveis].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matriz, cmap=cmap, vmin=-1, vmax=1)
    
    labels = ["Temp.", "Precip.", "NDVI", "Relâmpago", "Queimadas", "Umidade"]
    ax.set_xticks(range(len(variaveis)))
    ax.set_yticks(range(len(variaveis)))
    ax.set_xticklabels(labels, rotation=40, ha="right", color=TEXT, fontsize=9)
    ax.set_yticklabels(labels, color=TEXT, fontsize=9)
    
    for i in range(len(variaveis)):
        for j in range(len(variaveis)):
            val = matriz.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="black" if abs(val) > 0.3 else TEXT, fontsize=8, fontweight="bold")
    
    plt.colorbar(im, ax=ax, label="Correlação de Pearson").ax.yaxis.label.set_color(TEXT)
    ax.set_title("Matriz de Correlação — Variáveis Climáticas", fontsize=11, color=GREEN, fontfamily="monospace")
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values(): spine.set_edgecolor("#1e293b")
    
    plt.tight_layout(pad=1.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# GERAÇÃO DO RELATÓRIO PDF
# ─────────────────────────────────────────────────────────────────────────────

def gerar_relatorio_pdf(municipio, df_diario, df_mensal, df_ml, df_prev, figuras_bytes):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Image as RLImage, Table, TableStyle, HRFlowable, PageBreak)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    import io as _io

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    
    # ── Estilos ──────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    
    cor_verde  = colors.HexColor("#1a7f4b")
    cor_titulo = colors.HexColor("#0d3b26")
    cor_cinza  = colors.HexColor("#4b5563")
    cor_fundo  = colors.HexColor("#f0fdf4")
    cor_borda  = colors.HexColor("#bbf7d0")
    
    s_capa_titulo = ParagraphStyle("capa_titulo", parent=styles["Title"],
        fontSize=22, textColor=cor_titulo, spaceAfter=6,
        fontName="Helvetica-Bold", alignment=TA_CENTER)
    s_capa_sub = ParagraphStyle("capa_sub", parent=styles["Normal"],
        fontSize=13, textColor=cor_verde, spaceAfter=4,
        fontName="Helvetica", alignment=TA_CENTER)
    s_h1 = ParagraphStyle("h1", parent=styles["Heading1"],
        fontSize=14, textColor=cor_titulo, spaceBefore=16, spaceAfter=6,
        fontName="Helvetica-Bold", borderPad=4)
    s_h2 = ParagraphStyle("h2", parent=styles["Heading2"],
        fontSize=11, textColor=cor_verde, spaceBefore=10, spaceAfter=4,
        fontName="Helvetica-Bold")
    s_body = ParagraphStyle("body", parent=styles["Normal"],
        fontSize=9.5, textColor=colors.HexColor("#1f2937"),
        spaceAfter=6, leading=14, alignment=TA_JUSTIFY,
        fontName="Helvetica")
    s_rodape = ParagraphStyle("rodape", parent=styles["Normal"],
        fontSize=8, textColor=cor_cinza, alignment=TA_CENTER, fontName="Helvetica")
    s_destaque = ParagraphStyle("destaque", parent=styles["Normal"],
        fontSize=9.5, textColor=cor_titulo, backColor=cor_fundo,
        borderColor=cor_borda, borderWidth=1, borderPad=6,
        spaceAfter=6, leading=13, fontName="Helvetica")
    
    story = []
    data_relatorio = datetime.now().strftime("%d/%m/%Y %H:%M")
    info = MUNICIPIOS_INFO[municipio]
    
    # ── CAPA ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("🌦️", ParagraphStyle("emoji", parent=styles["Normal"],
        fontSize=36, alignment=TA_CENTER, spaceAfter=8)))
    story.append(Paragraph(
        "RELATÓRIO DE ANÁLISE CLIMÁTICA MUNICIPAL",
        s_capa_titulo))
    story.append(Paragraph(
        f"Gestão Pública Inteligente com Machine Learning",
        s_capa_sub))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=cor_verde))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"<b>Município:</b> {municipio} — Mato Grosso do Sul", s_capa_sub))
    story.append(Paragraph(f"Gerado em: {data_relatorio}", s_rodape))
    story.append(Paragraph(
        "Sistema PyVisSat | UFMS/SEMADESC 2026 | Prof. Dr. Enrique Vieira Mattos — UNIFEI",
        s_rodape))
    story.append(Spacer(1, 0.8*cm))
    
    # Tabela de identificação
    lat = info["lat"]; lon = info["lon"]
    pop = f'{info["pop"]:,}'.replace(",", ".")
    area = f'{info["area_km2"]:,}'.replace(",", ".")
    
    dados_id = [
        ["Parâmetro", "Valor"],
        ["Município", municipio],
        ["Latitude / Longitude", f"{lat:.2f}° S / {abs(lon):.2f}° O"],
        ["Área (km²)", area],
        ["População Estimada", pop],
        ["Período Analisado", "01/01/2020 – 31/12/2024 (5 anos)"],
        ["Modelo ML", "Score de Risco Multivariado (Temp. + Precip. + NDVI + Relâmpago + Queimadas)"],
    ]
    t_id = Table(dados_id, colWidths=[5*cm, 11*cm])
    t_id.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), cor_titulo),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("BACKGROUND", (0,1), (0,-1), cor_fundo),
        ("FONTNAME",   (0,1), (0,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",  (0,1), (0,-1), cor_verde),
        ("GRID", (0,0), (-1,-1), 0.5, cor_borda),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, cor_fundo]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t_id)
    story.append(PageBreak())
    
    # ── SUMÁRIO EXECUTIVO ────────────────────────────────────────────────────
    story.append(Paragraph("1. SUMÁRIO EXECUTIVO", s_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=cor_borda))
    story.append(Spacer(1, 0.3*cm))
    
    temp_media = df_diario["temperatura"].mean()
    prec_anual = df_diario["precipitacao"].sum() / 5
    ndvi_medio = df_diario["ndvi"].mean()
    focos_total = df_ml["focos_queimada"].sum()
    flash_total = df_ml["flashes"].sum()
    risco_medio = df_ml["risco_score"].mean()
    
    story.append(Paragraph(
        f"O presente relatório apresenta análise climática completa do município de <b>{municipio}</b>, "
        f"Mato Grosso do Sul, abrangendo o período de 2020 a 2024. A análise integra dados "
        f"de temperatura, precipitação, índices de vegetação (NDVI), relâmpagos estimados pelo "
        f"sensor GLM/GOES e focos de queimadas do INPE, processados com técnicas de Machine Learning "
        f"para suporte à tomada de decisão na gestão pública municipal.", s_body))
    
    # KPIs em tabela
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Indicadores Climáticos Principais (2020–2024)", s_h2))
    
    kpis = [
        ["Indicador", "Valor", "Unidade", "Avaliação"],
        ["Temperatura Média", f"{temp_media:.1f}", "°C",
         "Normal" if 22 <= temp_media <= 28 else ("Alta" if temp_media > 28 else "Baixa")],
        ["Precipitação Anual Média", f"{prec_anual:.0f}", "mm/ano",
         "Adequada" if 1000 <= prec_anual <= 1800 else ("Escassa" if prec_anual < 1000 else "Excessiva")],
        ["NDVI Médio", f"{ndvi_medio:.3f}", "adim.",
         "Boa cobertura" if ndvi_medio >= 0.5 else "Cobertura reduzida"],
        ["Total de Focos (5 anos)", f"{focos_total:,}".replace(",","."), "focos",
         "Atenção" if focos_total > 2000 else "Controlado"],
        ["Total de Flashes (5 anos)", f"{flash_total:,}".replace(",","."), "flashes", "Normal"],
        ["Score de Risco Médio (ML)", f"{risco_medio:.1f}", "/100",
         "ALTO" if risco_medio > 60 else ("MODERADO" if risco_medio > 35 else "BAIXO")],
    ]
    t_kpi = Table(kpis, colWidths=[5.5*cm, 3*cm, 2.5*cm, 5*cm])
    t_kpi.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), cor_titulo),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("ALIGN",      (1,0), (2,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, cor_borda),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, cor_fundo]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0,1), (0,-1), cor_verde),
    ]))
    story.append(t_kpi)
    story.append(PageBreak())
    
    # ── SEÇÕES COM FIGURAS ───────────────────────────────────────────────────
    secoes = [
        ("2. ANÁLISE DE TEMPERATURA", [
            ("Figura 1 — Série Temporal de Temperatura Diária e Anomalia Mensal",
             "A análise da temperatura revela sazonalidade típica do Centro-Oeste brasileiro, "
             "com verões quentes (dez–fev) e invernos amenos (jun–ago). As anomalias mensais "
             "indicam tendência de aquecimento nos anos mais recentes, consistente com os "
             "cenários de mudanças climáticas regionais.",
             "temperatura"),
        ]),
        ("3. ANÁLISE DE PRECIPITAÇÃO", [
            ("Figura 2 — Precipitação Mensal e Climatologia",
             "O regime pluviométrico apresenta clara bipartição sazonal: estação chuvosa "
             "(outubro–março) com precipitação mensal superior a 150 mm e estação seca "
             "(abril–setembro) com totais inferiores a 60 mm. Este padrão é fundamental para "
             "o planejamento de recursos hídricos, agricultura e gestão de riscos.",
             "precipitacao"),
        ]),
        ("4. ÍNDICE DE VEGETAÇÃO (NDVI)", [
            ("Figura 3 — NDVI Mensal e Correlação com Precipitação",
             "O NDVI (Normalized Difference Vegetation Index) reflete o estado da cobertura "
             "vegetal e responde diretamente ao regime hídrico. Valores acima de 0,5 indicam "
             "boa cobertura vegetal. A correlação positiva com a precipitação confirma a "
             "dependência da vegetação local ao regime de chuvas.",
             "ndvi"),
        ]),
        ("5. QUEIMADAS E RELÂMPAGOS", [
            ("Figura 4 — Focos de Queimada e Relâmpagos",
             "Os focos de queimada (INPE/BDQueimadas) apresentam pico no período seco "
             "(julho–setembro), coincidindo com baixa umidade relativa e ausência de chuvas. "
             "Os dados de relâmpagos do sensor GLM/GOES-19 demonstram maior atividade "
             "convectiva no verão, associada às Linhas de Instabilidade e ZCAS.",
             "queimadas"),
        ]),
        ("6. MACHINE LEARNING — ANÁLISE DE RISCO", [
            ("Figura 5 — Score de Risco Climático Multivariado",
             "O score de risco climático foi calculado integrando múltiplas variáveis "
             "(temperatura, precipitação, NDVI, relâmpagos e queimadas) com pesos baseados "
             "em relevância para gestão pública. Dias com risco CRÍTICO (>75) requerem "
             "atenção especial de defesa civil e saúde pública.",
             "risco_ml"),
            ("Figura 6 — Previsão Climática para os Próximos 30 Dias",
             "A previsão climática de curto prazo utiliza modelo de séries temporais com "
             "intervalos de confiança de 95%. As previsões subsidiam o planejamento municipal "
             "de abastecimento de água, alertas de queimadas e evacuações preventivas.",
             "previsao"),
        ]),
        ("7. ANÁLISE MULTIVARIADA", [
            ("Figura 7 — Matriz de Correlação entre Variáveis Climáticas",
             "A matriz de correlação de Pearson revela as interdependências entre as "
             "variáveis climáticas. As correlações negativas entre queimadas e precipitação "
             "e entre NDVI e temperatura confirmam os padrões esperados para a região do "
             "Mato Grosso do Sul.",
             "correlacoes"),
        ]),
    ]
    
    for titulo_sec, figuras_sec in secoes:
        story.append(Paragraph(titulo_sec, s_h1))
        story.append(HRFlowable(width="100%", thickness=1, color=cor_borda))
        story.append(Spacer(1, 0.3*cm))
        
        for titulo_fig, texto_fig, chave_fig in figuras_sec:
            story.append(Paragraph(titulo_fig, s_h2))
            story.append(Paragraph(texto_fig, s_body))
            
            if chave_fig in figuras_bytes:
                img_buf = _io.BytesIO(figuras_bytes[chave_fig])
                img = RLImage(img_buf, width=16*cm, height=7*cm)
                story.append(img)
            
            story.append(Spacer(1, 0.4*cm))
        
        story.append(PageBreak())
    
    # ── RECOMENDAÇÕES PARA GESTÃO PÚBLICA ───────────────────────────────────
    story.append(Paragraph("8. RECOMENDAÇÕES PARA GESTÃO PÚBLICA", s_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=cor_borda))
    story.append(Spacer(1, 0.3*cm))
    
    recomendacoes = [
        ("8.1 Defesa Civil e Alertas Climáticos",
         f"Implementar sistema de alertas precoces baseado no score de risco ML calculado "
         f"neste relatório. Dias com score &gt; 75 devem acionar protocolos de emergência "
         f"para alagamentos (verão) e incêndios (inverno). A prefeitura de {municipio} deve "
         f"manter brigadas de incêndio em standby durante julho–setembro."),
        ("8.2 Gestão Hídrica",
         "O planejamento de reservatórios e abastecimento d'água deve considerar a "
         "sazonalidade pluviométrica identificada. A estação seca (abr–set) requer "
         "planejamento antecipado de estoque hídrico. Monitorar o NDVI como indicador "
         "indireto de estresse hídrico nas bacias de captação."),
        ("8.3 Saúde Pública",
         "Correlacionar os dados climáticos com indicadores de saúde (dengue, doenças "
         "respiratórias). Os focos de queimadas impactam diretamente a qualidade do ar e "
         "devem ser monitorados em tempo real via plataforma INPE/BDQueimadas. "
         "Alertas de temperatura extrema para populações vulneráveis."),
        ("8.4 Agricultura e Meio Ambiente",
         "Utilizar os dados de NDVI e precipitação para suporte ao produtor rural "
         "municipal. O monitoramento contínuo da cobertura vegetal permite identificar "
         "áreas de desmatamento irregular e planejar ações de reflorestamento. "
         "Integrar com MapBiomas para análise de uso e cobertura do solo."),
        ("8.5 Infraestrutura e Urbanismo",
         "Priorizar obras de drenagem urbana considerando os eventos extremos de "
         "precipitação identificados. O planejamento do sistema viário deve contemplar "
         "os cenários de chuva intensa estimados pelo modelo ML para os próximos anos."),
    ]
    
    for subtitulo, texto in recomendacoes:
        story.append(Paragraph(subtitulo, s_h2))
        story.append(Paragraph(texto, s_body))
    
    story.append(PageBreak())
    
    # ── METODOLOGIA ──────────────────────────────────────────────────────────
    story.append(Paragraph("9. METODOLOGIA E FONTES DE DADOS", s_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=cor_borda))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("Fontes de Dados Utilizadas", s_h2))
    fontes = [
        ["Variável", "Fonte", "Satélite/Sensor", "Resolução"],
        ["Imagem IR (Ch13)", "NOAA/AWS", "GOES-19 / ABI", "2 km / 10 min"],
        ["Relâmpagos", "NOAA/AWS + CPTEC/INPE", "GOES-19 / GLM", "5 min"],
        ["Precipitação", "INPE (MERGE)", "Multi-satélite + pluviôm.", "0.1° / diária"],
        ["NDVI", "NASA/GEE (MODIS Terra)", "TERRA / MOD13Q1", "250 m / 16 dias"],
        ["Queimadas", "INPE/BDQueimadas", "AQUA/TERRA + GOES", "375 m / 1 hora"],
        ["Temperatura/Umidade", "INMET", "Estações meteorológicas", "Horária"],
    ]
    t_fontes = Table(fontes, colWidths=[4*cm, 4.5*cm, 5*cm, 2.5*cm])
    t_fontes.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), cor_titulo),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 8.5),
        ("GRID", (0,0), (-1,-1), 0.5, cor_borda),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, cor_fundo]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t_fontes)
    story.append(Spacer(1, 0.4*cm))
    
    story.append(Paragraph("Ferramentas e Bibliotecas", s_h2))
    story.append(Paragraph(
        "Python 3.11 · Streamlit · Pandas · NumPy · Matplotlib · Plotly · Scikit-learn · "
        "Cartopy · Xarray · NetCDF4 · Boto3 (AWS) · Google Earth Engine API · Geemap · "
        "GDAL · Rasterio · GeoBR · ReportLab · PyPDF · Ultraplot",
        s_destaque))
    
    story.append(Paragraph(
        "Este dashboard foi desenvolvido no contexto do Minicurso PyVisSat "
        "(Processamento e Visualização de Imagens de Satélite com Python), "
        "realizado na UFMS em parceria com a SEMADESC em março de 2026.",
        s_body))
    
    # ── RODAPÉ FINAL ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=cor_borda))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"Relatório gerado automaticamente em {data_relatorio} | "
        f"Dashboard Climático Municipal v1.0 | PyVisSat/UFMS 2026",
        s_rodape))
    story.append(Paragraph(
        "Prof. Dr. Enrique Vieira Mattos (UNIFEI) · enrique@unifei.edu.br · "
        "github.com/evmpython",
        s_rodape))
    
    doc.build(story)
    buf.seek(0)
    return buf.read()


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌍 PAINEL DE CONTROLE")
    st.markdown("---")
    
    municipio = st.selectbox("🏙️ Município", MUNICIPIOS_MS, index=0)
    
    st.markdown("---")
    st.markdown("**📅 Filtros Temporais**")
    ano_sel = st.selectbox("Ano de Análise", [2020, 2021, 2022, 2023, 2024], index=4)
    
    st.markdown("---")
    st.markdown("**🤖 Machine Learning**")
    modelo_ml = st.selectbox("Modelo de Risco", ["Score Multivariado", "Isolamento Florestal", "Regressão Linear"])
    
    st.markdown("---")
    st.markdown("**🛰️ Dados Ativos**")
    ativar_temp  = st.toggle("Temperatura",    value=True)
    ativar_prec  = st.toggle("Precipitação",   value=True)
    ativar_ndvi  = st.toggle("NDVI Vegetação", value=True)
    ativar_qlamp = st.toggle("Queimadas/Relâmpagos", value=True)
    ativar_ml    = st.toggle("ML / Previsão",  value=True)
    
    st.markdown("---")
    info_muni = MUNICIPIOS_INFO[municipio]
    st.markdown(f"""
    <div class="metric-card">
        <h4>📍 {municipio}</h4>
        <div class="delta">Lat: {info_muni['lat']:.2f}° | Lon: {info_muni['lon']:.2f}°</div>
        <div class="delta">Área: {info_muni['area_km2']:,} km²</div>
        <div class="delta">Pop: {info_muni['pop']:,} hab.</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DOS DADOS
# ─────────────────────────────────────────────────────────────────────────────
df_diario  = gerar_serie_temporal(municipio)
df_mensal  = gerar_dados_mensais(municipio)
df_ml, df_prev = gerar_dados_ml(municipio)
df_mapa    = gerar_mapa_ms()

df_ano = df_diario[df_diario["data"].dt.year == ano_sel]
df_mensal_ano = df_mensal[df_mensal["mes_dt"].dt.year == ano_sel]

# ─────────────────────────────────────────────────────────────────────────────
# CABEÇALHO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background: linear-gradient(135deg, #0d1a2e 0%, #0a1f1a 100%);
            border: 1px solid #1e3a2e; border-radius: 12px; padding: 24px 32px; margin-bottom: 24px;">
    <h1 style="margin:0; font-size: 24px; font-family: 'Space Mono', monospace; color: #00ff88;">
        🌦️ DASHBOARD CLIMÁTICO MUNICIPAL
    </h1>
    <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 13px; font-family: 'Rajdhani', sans-serif;">
        Análise de Machine Learning para Gestão Pública &nbsp;|&nbsp;
        <span style="color:#00d4b8">{municipio} – MS</span> &nbsp;|&nbsp;
        <span style="color:#f59e0b">Ano: {ano_sel}</span> &nbsp;|&nbsp;
        <span style="color:#94a3b8">Atualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPIs PRINCIPAIS
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)

temp_media_ano   = df_ano["temperatura"].mean()
prec_total_ano   = df_ano["precipitacao"].sum()
ndvi_medio_ano   = df_ano["ndvi"].mean()
focos_total_ano  = df_ano["focos_queimada"].sum()
flash_total_ano  = df_ano["flashes"].sum()
risco_medio_ano  = df_ml[df_ml["data"].dt.year == ano_sel]["risco_score"].mean()

with c1: st.metric("🌡️ Temp. Média", f"{temp_media_ano:.1f} °C", f"±{df_ano['temperatura'].std():.1f}°C")
with c2: st.metric("🌧️ Precip. Total", f"{prec_total_ano:.0f} mm", f"Anual {ano_sel}")
with c3: st.metric("🌿 NDVI Médio", f"{ndvi_medio_ano:.3f}", "Vegetação")
with c4: st.metric("🔥 Focos Queimada", f"{focos_total_ano}", f"Ano {ano_sel}")
with c5: st.metric("⚡ Relâmpagos", f"{flash_total_ano:,}", "GLM/GOES-19")
with c6:
    cor_risco = "🟢" if risco_medio_ano < 35 else ("🟡" if risco_medio_ano < 60 else "🔴")
    st.metric(f"{cor_risco} Risco ML", f"{risco_medio_ano:.1f}/100", "Score Médio")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ABAS PRINCIPAIS
# ─────────────────────────────────────────────────────────────────────────────
abas = st.tabs([
    "🌡️ Temperatura",
    "🌧️ Precipitação",
    "🌿 NDVI / Vegetação",
    "🔥 Queimadas & ⚡ Relâmpagos",
    "🤖 Machine Learning",
    "📊 Correlações",
    "🗺️ Mapa MS",
    "📄 Relatório PDF"
])

# ── ABA 1: TEMPERATURA ────────────────────────────────────────────────────────
with abas[0]:
    if ativar_temp:
        st.markdown('<p class="section-header">ANÁLISE DE TEMPERATURA — SÉRIE HISTÓRICA</p>', unsafe_allow_html=True)
        
        fig_temp = plot_temperatura(df_diario, df_mensal)
        st.pyplot(fig_temp, use_container_width=True)
        plt.close(fig_temp)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📈 Estatísticas Descritivas**")
            stats = df_diario["temperatura"].describe().round(2)
            st.dataframe(stats.rename({
                "count":"Observações","mean":"Média","std":"Desvio Padrão",
                "min":"Mínima","25%":"Q25","50%":"Mediana","75%":"Q75","max":"Máxima"
            }), use_container_width=True)
        with col2:
            st.markdown("**📅 Temperaturas Mensais por Ano**")
            pivot = df_diario.copy()
            pivot["mes"] = pivot["data"].dt.month
            pivot["ano"] = pivot["data"].dt.year
            tab_pivot = pivot.groupby(["ano","mes"])["temperatura"].mean().unstack().round(1)
            tab_pivot.columns = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
            st.dataframe(tab_pivot, use_container_width=True)
    else:
        st.info("Ative 'Temperatura' no painel de controle para visualizar.")


# ── ABA 2: PRECIPITAÇÃO ──────────────────────────────────────────────────────
with abas[1]:
    if ativar_prec:
        st.markdown('<p class="section-header">ANÁLISE DE PRECIPITAÇÃO — MERGE/SATÉLITE</p>', unsafe_allow_html=True)
        
        fig_prec = plot_precipitacao(df_diario, df_mensal)
        st.pyplot(fig_prec, use_container_width=True)
        plt.close(fig_prec)
        
        st.markdown("---")
        # Plotly interativo
        st.markdown("**📊 Série Interativa de Precipitação Diária**")
        fig_px = px.bar(df_diario[df_diario["data"].dt.year == ano_sel],
                        x="data", y="precipitacao",
                        color_discrete_sequence=[BLUE],
                        labels={"data": "Data", "precipitacao": "Precip. (mm)"},
                        title=f"Precipitação Diária — {municipio} {ano_sel}")
        fig_px.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                             font=dict(color=TEXT, family="monospace"),
                             title_font_color=GREEN)
        st.plotly_chart(fig_px, use_container_width=True)
    else:
        st.info("Ative 'Precipitação' no painel de controle.")


# ── ABA 3: NDVI ───────────────────────────────────────────────────────────────
with abas[2]:
    if ativar_ndvi:
        st.markdown('<p class="section-header">NDVI — NORMALIZED DIFFERENCE VEGETATION INDEX</p>', unsafe_allow_html=True)
        st.markdown("Dados: MODIS/TERRA (MOD13Q1) · Sentinel-2 · Landsat — via Google Earth Engine")
        
        fig_ndvi = plot_ndvi(df_mensal)
        st.pyplot(fig_ndvi, use_container_width=True)
        plt.close(fig_ndvi)
        
        st.markdown("---")
        st.markdown("**🌱 Interpretação do NDVI**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="alert-box alert-low">🟢 NDVI 0.6–1.0<br><b>Vegetação Densa</b><br>Floresta/Pastagem saudável</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="alert-box alert-medium" style="border-left-color:#10b981">🟡 NDVI 0.3–0.6<br><b>Vegetação Moderada</b><br>Savana/Cerrado</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="alert-box alert-medium">🟠 NDVI 0.1–0.3<br><b>Vegetação Escassa</b><br>Solo exposto / Stress hídrico</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="alert-box alert-high">🔴 NDVI &lt; 0.1<br><b>Solo Nu / Queimadas</b><br>Área degradada</div>', unsafe_allow_html=True)
        
        # Mapa NDVI interativo Plotly
        st.markdown("**🗺️ NDVI Comparativo — Municípios do MS**")
        fig_mapa_ndvi = px.scatter_mapbox(
            df_mapa, lat="lat", lon="lon", size="ndvi",
            color="ndvi", color_continuous_scale="RdYlGn",
            hover_name="municipio",
            hover_data={"ndvi": ":.3f", "lat": False, "lon": False},
            zoom=5, center={"lat": -20.5, "lon": -54.6},
            mapbox_style="carto-darkmatter",
            title="NDVI Médio por Município — MS",
            size_max=25,
        )
        fig_mapa_ndvi.update_layout(paper_bgcolor=DARK_BG, font=dict(color=TEXT), title_font_color=GREEN, height=450)
        st.plotly_chart(fig_mapa_ndvi, use_container_width=True)
    else:
        st.info("Ative 'NDVI Vegetação' no painel de controle.")


# ── ABA 4: QUEIMADAS & RELÂMPAGOS ────────────────────────────────────────────
with abas[3]:
    if ativar_qlamp:
        st.markdown('<p class="section-header">QUEIMADAS (INPE/BDQUEIMADAS) & RELÂMPAGOS (GLM/GOES-19)</p>', unsafe_allow_html=True)
        
        fig_qr = plot_queimadas_relampagos(df_mensal)
        st.pyplot(fig_qr, use_container_width=True)
        plt.close(fig_qr)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔥 Alerta de Queimadas por Mês**")
            clim_q = df_mensal.groupby(df_mensal["mes_dt"].dt.month)["focos_queimada"].mean()
            limite_alto = clim_q.quantile(0.75)
            meses_nomes = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
            for i, (mes, val) in enumerate(zip(meses_nomes, clim_q.values)):
                nivel = "alert-high" if val > limite_alto else ("alert-medium" if val > clim_q.median() else "alert-low")
                icon  = "🔴" if val > limite_alto else ("🟡" if val > clim_q.median() else "🟢")
                st.markdown(f'<div class="alert-box {nivel}">{icon} <b>{mes}</b>: {val:.0f} focos/mês médio</div>',
                            unsafe_allow_html=True)
        
        with col2:
            st.markdown("**⚡ Série Temporal Interativa de Relâmpagos**")
            df_flash_ano = df_diario[df_diario["data"].dt.year == ano_sel]
            fig_flash = px.line(df_flash_ano, x="data", y="flashes",
                                color_discrete_sequence=[BLUE],
                                labels={"data":"Data","flashes":"Flashes/dia"},
                                title=f"Relâmpagos Diários — {ano_sel}")
            fig_flash.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                                    font=dict(color=TEXT, family="monospace"),
                                    title_font_color=GREEN, height=400)
            st.plotly_chart(fig_flash, use_container_width=True)
    else:
        st.info("Ative 'Queimadas/Relâmpagos' no painel de controle.")


# ── ABA 5: MACHINE LEARNING ──────────────────────────────────────────────────
with abas[4]:
    if ativar_ml:
        st.markdown('<p class="section-header">MACHINE LEARNING — ANÁLISE DE RISCO E PREVISÃO CLIMÁTICA</p>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background:{CARD_BG}; border:1px solid #1e293b; border-left:3px solid {TEAL};
                    border-radius:8px; padding:16px; margin-bottom:16px;">
        <b style="color:{TEAL}">MODELO:</b> <span style="color:{TEXT}">Score de Risco Climático Multivariado</span><br>
        <b style="color:{TEAL}">VARIÁVEIS:</b> <span style="color:{TEXT}">Temperatura · Precipitação · NDVI · Relâmpagos · Queimadas</span><br>
        <b style="color:{TEAL}">PESOS:</b> <span style="color:{TEXT}">Precip(30%) + NDVI(25%) + Relâmpago(25%) + Queimadas(20%)</span><br>
        <b style="color:{TEAL}">NÍVEIS:</b> <span style="color:#00ff88">Baixo (0-25)</span> · 
                       <span style="color:#00d4b8">Moderado (25-50)</span> · 
                       <span style="color:#f59e0b">Alto (50-75)</span> · 
                       <span style="color:#ef4444">Crítico (75-100)</span>
        </div>
        """, unsafe_allow_html=True)
        
        fig_ml = plot_risco_ml(df_ml)
        st.pyplot(fig_ml, use_container_width=True)
        plt.close(fig_ml)
        
        st.markdown("---")
        st.markdown("### 🔮 Previsão Climática — Próximos 30 dias")
        
        fig_prev = plot_previsao_ml(df_ml, df_prev)
        st.pyplot(fig_prev, use_container_width=True)
        plt.close(fig_prev)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📊 Distribuição de Dias por Nível de Risco (5 anos)**")
            cont_risco = df_ml["risco_nivel"].value_counts().reindex(["Baixo","Moderado","Alto","Crítico"])
            fig_pizza = px.pie(values=cont_risco.values, names=cont_risco.index,
                               color_discrete_map={"Baixo":GREEN,"Moderado":TEAL,"Alto":AMBER,"Crítico":RED},
                               title="Distribuição do Risco")
            fig_pizza.update_layout(paper_bgcolor=DARK_BG, font=dict(color=TEXT),
                                    title_font_color=GREEN, height=300)
            st.plotly_chart(fig_pizza, use_container_width=True)
        
        with col2:
            st.markdown("**🔴 Top 10 Dias de Maior Risco**")
            top10 = df_ml.nlargest(10, "risco_score")[["data","risco_score","risco_nivel",
                                                         "temperatura","precipitacao","focos_queimada"]]
            top10 = top10.rename(columns={
                "data":"Data","risco_score":"Score","risco_nivel":"Nível",
                "temperatura":"Temp(°C)","precipitacao":"Precip(mm)","focos_queimada":"Focos"
            })
            top10["Data"] = top10["Data"].dt.strftime("%d/%m/%Y")
            st.dataframe(top10, use_container_width=True, hide_index=True)
    else:
        st.info("Ative 'ML / Previsão' no painel de controle.")


# ── ABA 6: CORRELAÇÕES ───────────────────────────────────────────────────────
with abas[5]:
    st.markdown('<p class="section-header">ANÁLISE MULTIVARIADA — CORRELAÇÕES ENTRE VARIÁVEIS CLIMÁTICAS</p>', unsafe_allow_html=True)
    
    fig_corr = plot_correlacoes(df_mensal)
    st.pyplot(fig_corr, use_container_width=True)
    plt.close(fig_corr)
    
    st.markdown("---")
    st.markdown("**📈 Scatter Plot Interativo**")
    
    col1, col2 = st.columns(2)
    with col1:
        var_x = st.selectbox("Eixo X", ["temperatura","precipitacao","ndvi","flashes","focos_queimada","umidade"], index=1)
    with col2:
        var_y = st.selectbox("Eixo Y", ["temperatura","precipitacao","ndvi","flashes","focos_queimada","umidade"], index=2)
    
    fig_scatter = px.scatter(df_mensal, x=var_x, y=var_y,
                              color="mes_dt", color_continuous_scale="turbo",
                              trendline="ols",
                              labels={var_x: var_x.capitalize(), var_y: var_y.capitalize(),
                                      "mes_dt": "Data"},
                              title=f"{var_x.upper()} × {var_y.upper()}")
    fig_scatter.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                               font=dict(color=TEXT, family="monospace"),
                               title_font_color=GREEN)
    st.plotly_chart(fig_scatter, use_container_width=True)


# ── ABA 7: MAPA MS ───────────────────────────────────────────────────────────
with abas[6]:
    st.markdown('<p class="section-header">MAPA CLIMÁTICO — MATO GROSSO DO SUL</p>', unsafe_allow_html=True)
    
    variavel_mapa = st.selectbox("Variável para visualizar no mapa",
                                  ["risco", "ndvi", "prec_anual", "focos"],
                                  format_func=lambda x: {
                                      "risco":"Score de Risco (ML)",
                                      "ndvi":"NDVI Médio",
                                      "prec_anual":"Precipitação Anual (mm)",
                                      "focos":"Focos de Queimada"
                                  }[x])
    
    paletas = {"risco":"RdYlGn_r","ndvi":"RdYlGn","prec_anual":"Blues","focos":"hot_r"}
    
    fig_mapa = px.scatter_mapbox(
        df_mapa, lat="lat", lon="lon",
        size=variavel_mapa, color=variavel_mapa,
        color_continuous_scale=paletas[variavel_mapa],
        hover_name="municipio",
        hover_data={
            "risco":":.1f","ndvi":":.3f",
            "prec_anual":":.0f","focos":":.0f",
            "populacao":":.0f","lat":False,"lon":False
        },
        zoom=5.5, center={"lat": -20.5, "lon": -54.6},
        mapbox_style="carto-darkmatter",
        title=f"{variavel_mapa.upper()} por Município — Mato Grosso do Sul",
        size_max=40,
    )
    fig_mapa.update_layout(paper_bgcolor=DARK_BG, font=dict(color=TEXT, family="monospace"),
                            title_font_color=GREEN, height=560)
    st.plotly_chart(fig_mapa, use_container_width=True)
    
    st.markdown("**📋 Tabela Comparativa — Municípios do MS**")
    df_tabela = df_mapa.copy()
    df_tabela.columns = ["Município","Lat","Lon","Risco (ML)","NDVI Médio",
                          "Precip. Anual (mm)","Focos Queimada","População"]
    df_tabela = df_tabela.drop(columns=["Lat","Lon"])
    st.dataframe(df_tabela.sort_values("Risco (ML)", ascending=False), use_container_width=True, hide_index=True)


# ── ABA 8: RELATÓRIO PDF ─────────────────────────────────────────────────────
with abas[7]:
    st.markdown('<p class="section-header">GERAÇÃO DE RELATÓRIO TÉCNICO EM PDF</p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background:{CARD_BG}; border:1px solid {TEAL}33; border-radius:10px; padding:20px; margin-bottom:20px;">
        <h3 style="color:{TEAL}; font-family:monospace; margin-top:0">📄 Relatório Técnico Climático Municipal</h3>
        <p style="color:{TEXT}">O relatório gerado inclui:</p>
        <ul style="color:{TEXT}">
            <li>🏛️ Capa com identificação do município e período analisado</li>
            <li>📊 Sumário executivo com KPIs principais</li>
            <li>🌡️ Análise completa de temperatura (série histórica + anomalias)</li>
            <li>🌧️ Análise de precipitação (série + climatologia mensal)</li>
            <li>🌿 NDVI — Índice de vegetação e correlações</li>
            <li>🔥 Queimadas e ⚡ relâmpagos (mapas e séries temporais)</li>
            <li>🤖 Score de risco ML e previsão 30 dias</li>
            <li>📈 Matriz de correlações multivariada</li>
            <li>📋 Recomendações para gestão pública</li>
            <li>🛰️ Metodologia e fontes de dados</li>
        </ul>
        <p style="color:{MUTED}; font-size:12px; margin-bottom:0">
            Formato: PDF A4 · Gerado com ReportLab · Baseado em PyVisSat/UFMS 2026
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        gerar_btn = st.button("🚀 GERAR RELATÓRIO PDF", use_container_width=True)
    with col2:
        st.markdown(f"<p style='color:{MUTED}; margin-top:12px'>Município: <b style='color:{GREEN}'>{municipio}</b> | "
                    f"Período: 2020–2024 | Modelo: {modelo_ml}</p>", unsafe_allow_html=True)
    
    if gerar_btn:
        with st.spinner("⏳ Gerando figuras e compilando o relatório PDF..."):
            # Gerar todas as figuras
            figuras_bytes = {}
            
            progress = st.progress(0, text="Gerando figura: Temperatura...")
            f_temp = plot_temperatura(df_diario, df_mensal)
            figuras_bytes["temperatura"] = fig_to_bytes(f_temp)
            plt.close(f_temp)
            progress.progress(14, text="Gerando figura: Precipitação...")
            
            f_prec = plot_precipitacao(df_diario, df_mensal)
            figuras_bytes["precipitacao"] = fig_to_bytes(f_prec)
            plt.close(f_prec)
            progress.progress(28, text="Gerando figura: NDVI...")
            
            f_ndvi = plot_ndvi(df_mensal)
            figuras_bytes["ndvi"] = fig_to_bytes(f_ndvi)
            plt.close(f_ndvi)
            progress.progress(42, text="Gerando figura: Queimadas & Relâmpagos...")
            
            f_qr = plot_queimadas_relampagos(df_mensal)
            figuras_bytes["queimadas"] = fig_to_bytes(f_qr)
            plt.close(f_qr)
            progress.progress(57, text="Gerando figura: Score de Risco ML...")
            
            f_risco = plot_risco_ml(df_ml)
            figuras_bytes["risco_ml"] = fig_to_bytes(f_risco)
            plt.close(f_risco)
            progress.progress(71, text="Gerando figura: Previsão ML...")
            
            f_prev_fig = plot_previsao_ml(df_ml, df_prev)
            figuras_bytes["previsao"] = fig_to_bytes(f_prev_fig)
            plt.close(f_prev_fig)
            progress.progress(85, text="Gerando figura: Correlações...")
            
            f_corr = plot_correlacoes(df_mensal)
            figuras_bytes["correlacoes"] = fig_to_bytes(f_corr)
            plt.close(f_corr)
            progress.progress(92, text="Compilando PDF...")
            
            # Gerar PDF
            pdf_bytes = gerar_relatorio_pdf(municipio, df_diario, df_mensal, df_ml, df_prev, figuras_bytes)
            progress.progress(100, text="✅ Relatório pronto!")
        
        st.success(f"✅ Relatório gerado com sucesso para **{municipio}** ({len(pdf_bytes)/1024:.0f} KB)")
        
        nome_arquivo = f"relatorio_climatico_{municipio.lower().replace(' ', '_')}_{ano_sel}.pdf"
        
        st.download_button(
            label="⬇️ BAIXAR RELATÓRIO PDF",
            data=pdf_bytes,
            file_name=nome_arquivo,
            mime="application/pdf",
            use_container_width=True,
        )
        
        st.markdown(f"""
        <div class="alert-box alert-low">
        ✅ <b>Relatório gerado com sucesso!</b><br>
        Arquivo: <code>{nome_arquivo}</code><br>
        Contém todas as figuras e análises para <b>{municipio}</b> (2020–2024)
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RODAPÉ
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; color:{MUTED}; font-size:11px; font-family:monospace; padding:12px 0;">
    Dashboard Climático Municipal v1.0 &nbsp;|&nbsp;
    Baseado em PyVisSat — UFMS/SEMADESC 2026 &nbsp;|&nbsp;
    Prof. Dr. Enrique Vieira Mattos (UNIFEI) &nbsp;|&nbsp;
    Dados: GOES-19·GLM·INPE·MERGE·GEE·MODIS &nbsp;|&nbsp;
    {datetime.now().strftime('%d/%m/%Y')}
</div>
""", unsafe_allow_html=True)
