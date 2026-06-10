<div align="center">

# Anomaly Detection in Industrial Time Series
### SKAB — Skoltech Anomaly Benchmark

Detecção de anomalias em sensores industriais usando Isolation Forest e análise de séries temporais multivariadas.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](#)
[![sklearn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-F7931E?style=flat&logo=scikitlearn&logoColor=white)](#)
[![Kaggle](https://img.shields.io/badge/Kaggle-SKAB%20Dataset-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab)
[![Status](https://img.shields.io/badge/Status-Completo-4CAF50?style=flat)](#)

</div>

---

## O Problema

Equipamentos industriais — bombas, válvulas, sensores de pressão e temperatura — geram séries temporais contínuas. **Anomalias nesses sinais indicam falhas iminentes, perdas operacionais e riscos de segurança.** Detectá-las automaticamente, antes que causem paradas não planejadas, é um dos problemas mais relevantes em indústria 4.0 e manutenção preditiva.

**Perguntas que este projeto responde:**
- Como detectar automaticamente comportamentos anômalos em dados de sensores?
- Quais sensores são mais importantes para identificar falhas?
- Qual algoritmo performa melhor sem supervisão completa (poucos dados rotulados)?

---

## O Dataset — SKAB

O **Skoltech Anomaly Benchmark (SKAB)** é o benchmark de referência para detecção de anomalias em séries temporais industriais, desenvolvido pelo Instituto Skoltech (Rússia) e publicado no Kaggle.

| Característica | Valor |
|---------------|-------|
| Fonte | Circuito de bomba d'água industrial (IIoT) |
| Sensores | 8 features (acelerômetro, pressão, temperatura, corrente, tensão, fluxo) |
| Anomalias | Rotuladas com causa física conhecida (válvulas fechadas) |
| Problema | Outlier detection + Changepoint detection |
| Referência | Citado em papers de ML/IEEE, disponível no paperswithcode.com |

---

## Resultado

![Série temporal com anomalias detectadas](img/05_anomaly_detection_series.png)

*Série temporal do acelerômetro com anomalias reais (vermelho) e detectadas pelo Isolation Forest (roxo) — e o Anomaly Score com threshold dinâmico.*

---

## Metodologia

```
Dados brutos (CSV) → Normalização (StandardScaler)
       ↓
Baseline: Z-Score estatístico
       ↓
Modelo principal: Isolation Forest (200 estimadores)
       ↓
Avaliação: F1-Score · AUC-ROC · FAR · MAR
       ↓
Visualização: série temporal · matriz de confusão · feature importance · ROC curve
```

### Por que Isolation Forest?

O Isolation Forest é o algoritmo de referência do leaderboard oficial do SKAB. Ele é especialmente adequado para dados industriais porque:
- Funciona bem com dados desbalanceados (anomalias raras)
- Não requer que as anomalias sejam rotuladas no treino
- É interpretável via feature importance
- Escala bem para dados multivariados

---

## Resultados

![Matriz de confusão](img/06_confusion_matrix.png)

| Modelo | F1-Score | AUC-ROC |
|--------|----------|---------|
| Z-Score Baseline | — | — |
| **Isolation Forest** | **—** | **—** |

*Métricas atualizadas após execução do notebook.*

---

## Feature Importance

![Feature importance](img/07_feature_importance.png)

Os sensores mais relevantes para a detecção de anomalias foram identificados via permutation importance no Isolation Forest.

---

## Como Rodar

```bash
# 1. Clone o repositório
git clone https://github.com/bpreschad-gif/anomaly-detection-skab.git
cd anomaly-detection-skab

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Baixe o dataset
# Acesse: https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab-teaser
# Salve em: data/raw/skab_teaser.csv

# 4. Execute os notebooks em ordem
python src/01_eda_skab.py
python src/02_anomaly_detection_skab.py
```

---

## Estrutura do Repositório

```
anomaly-detection-skab/
├── data/
│   ├── raw/                    ← dataset original (não versionado)
│   └── processed/              ← dados tratados
├── src/
│   ├── 01_eda_skab.py          ← EDA completa
│   └── 02_anomaly_detection_skab.py  ← modelos e avaliação
├── img/                        ← gráficos de resultado
│   ├── 01_distribuicao_anomalias.png
│   ├── 02_todos_sensores.png
│   ├── 03_correlacao_sensores.png
│   ├── 04_rolling_statistics.png
│   ├── 05_anomaly_detection_series.png
│   ├── 06_confusion_matrix.png
│   ├── 07_feature_importance.png
│   └── 08_roc_curve.png
├── requirements.txt
└── README.md
```

---

## Habilidades Demonstradas

`Python` · `scikit-learn` · `Isolation Forest` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Séries Temporais` · `Anomaly Detection` · `StandardScaler` · `ROC-AUC` · `F1-Score`

---

## Referências

- 📓 [Notebook no Kaggle](https://www.kaggle.com/brunapreschadt) — versão interativa
- 📊 [Dataset SKAB](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab) — Katser & Kozitsin, 2020
- 📄 [Leaderboard oficial SKAB](https://github.com/waico/SKAB) — paperswithcode.com

---

<div align="center">
Feito por <a href="https://github.com/bpreschac-gif">Bruna Preschadt</a> ·
<a href="https://www.linkedin.com/in/bruna-preschadt-de-oliveira-1550ab1ab/">LinkedIn</a> ·
<a href="https://www.kaggle.com/brunapreschadt">Kaggle</a>
</div>
