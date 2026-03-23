# 🌦️ Dashboard Climático Municipal com Machine Learning
**Gestão Pública Inteligente — PyVisSat/UFMS/SEMADESC 2026**

---

## Como executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Rodar o dashboard
streamlit run app_clima_municipal.py
```

Acesse em: `http://localhost:8501`

---

## Fontes de dados integradas (dos notebooks do curso)

| Aula | Dado | Fonte |
|------|------|-------|
| 1 | Imagens de Satélite IR (Ch13) | GOES-19/ABI — NOAA/AWS |
| 2 | Relâmpagos (GLM) | GOES-19/GLM — NOAA/AWS + CPTEC/INPE |
| 3 | Estações Meteorológicas | INMET |
| 4 | Precipitação estimada | MERGE/INPE + CHIRPS |
| 5 | NDVI / Índices de Vegetação | MODIS·Sentinel-2·Landsat via GEE |
| 6 | Focos de Queimada | INPE/BDQueimadas |

---

## Estrutura do app

- **7 abas analíticas** + **1 aba de relatório PDF**
- **ML Score de Risco** multivariado (0–100)
- **Previsão 30 dias** com intervalos de confiança
- **Relatório PDF completo** com todas as figuras e recomendações

---

*Teruo Allyson Yamada — UFMS | yamada.teruo@ufms.br*
*Prof. Dr. Enrique Vieira Mattos — UNIFEI | enrique@unifei.edu.br*
