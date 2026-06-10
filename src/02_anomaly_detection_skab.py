# =============================================================
# SKAB — Skoltech Anomaly Benchmark
# Notebook 02: Anomaly Detection — Isolation Forest + LSTM AE
# Autora: Bruna Preschadt de Oliveira
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'normal': '#2196F3', 'anomaly': '#E53935', 'pred': '#9C27B0'}

# ── 1. CARREGAMENTO E PREPARAÇÃO ──────────────────────────────
df = pd.read_csv('data/raw/skab_teaser.csv', index_col='datetime', parse_dates=True)
sensors = [c for c in df.columns if c not in ['anomaly', 'changepoint']]

X = df[sensors].values
y = df['anomaly'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Shape X: {X_scaled.shape}")
print(f"Anomalias: {y.sum()} ({y.mean()*100:.2f}%)")

# ── 2. BASELINE — Z-SCORE ─────────────────────────────────────
print("\n📊 MODELO 1: Baseline Z-Score")
z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10))
z_max = z_scores.max(axis=1)
threshold_z = np.percentile(z_max, 95)
y_pred_zscore = (z_max > threshold_z).astype(int)
f1_zscore = f1_score(y, y_pred_zscore, zero_division=0)
print(f"F1-Score Z-Score: {f1_zscore:.4f}")

# ── 3. ISOLATION FOREST ───────────────────────────────────────
print("\n🌳 MODELO 2: Isolation Forest")
contamination = y.mean()  # usa proporção real de anomalias

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_scaled)

y_pred_if = iso_forest.predict(X_scaled)
y_pred_if = (y_pred_if == -1).astype(int)  # -1 = anomalia no sklearn

scores_if = -iso_forest.score_samples(X_scaled)  # maior = mais anômalo

f1_if = f1_score(y, y_pred_if, zero_division=0)
auc_if = roc_auc_score(y, scores_if)

print(f"F1-Score Isolation Forest: {f1_if:.4f}")
print(f"AUC-ROC Isolation Forest:  {auc_if:.4f}")
print("\nClassification Report:")
print(classification_report(y, y_pred_if, target_names=['Normal', 'Anomalia'], zero_division=0))

# ── 4. FEATURE IMPORTANCE (Isolation Forest) ──────────────────
importances = []
for i in range(X_scaled.shape[1]):
    X_perm = X_scaled.copy()
    np.random.shuffle(X_perm[:, i])
    score_perm = -iso_forest.score_samples(X_perm).mean()
    importances.append(score_perm)

importances = np.array(importances)
importances = (importances - importances.min()) / (importances.max() - importances.min())
feat_imp = pd.Series(importances, index=sensors).sort_values(ascending=True)

# ── 5. VISUALIZAÇÕES PRINCIPAIS ───────────────────────────────

# 5.1 Série temporal com anomalias detectadas
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

sensor_main = 'Accelerometer1RMS'

# Real
axes[0].plot(df.index, df[sensor_main], color='#90A4AE', linewidth=0.8, alpha=0.9)
axes[0].scatter(df.index[y==1], df[sensor_main][y==1],
                color=COLORS['anomaly'], s=25, label='Anomalia real', zorder=5)
axes[0].set_title('Anomalias REAIS (ground truth)', fontsize=11, fontweight='bold')
axes[0].legend()

# Isolation Forest
axes[1].plot(df.index, df[sensor_main], color='#90A4AE', linewidth=0.8, alpha=0.9)
axes[1].scatter(df.index[y_pred_if==1], df[sensor_main][y_pred_if==1],
                color=COLORS['pred'], s=25, label=f'Isolation Forest (F1={f1_if:.3f})', zorder=5)
axes[1].set_title('Anomalias detectadas — Isolation Forest', fontsize=11, fontweight='bold')
axes[1].legend()

# Anomaly score
axes[2].plot(df.index, scores_if, color='#FF7043', linewidth=1.0, label='Anomaly Score')
threshold_val = np.percentile(scores_if, (1 - contamination) * 100)
axes[2].axhline(threshold_val, color=COLORS['anomaly'], linestyle='--',
                linewidth=1.5, label=f'Threshold ({threshold_val:.3f})')
axes[2].fill_between(df.index, threshold_val, scores_if,
                     where=scores_if >= threshold_val,
                     alpha=0.3, color=COLORS['anomaly'])
axes[2].set_title('Anomaly Score — Isolation Forest', fontsize=11, fontweight='bold')
axes[2].legend()
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

plt.suptitle('SKAB — Detecção de Anomalias com Isolation Forest', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('img/05_anomaly_detection_series.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Salvo: img/05_anomaly_detection_series.png")

# 5.2 Matriz de confusão
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (y_pred, title, color) in zip(axes, [
    (y_pred_zscore, f'Z-Score Baseline\nF1={f1_zscore:.3f}', 'Blues'),
    (y_pred_if, f'Isolation Forest\nF1={f1_if:.3f}  AUC={auc_if:.3f}', 'Purples')
]):
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=ax,
                xticklabels=['Normal', 'Anomalia'],
                yticklabels=['Normal', 'Anomalia'],
                linewidths=0.5, annot_kws={'size': 14})
    ax.set_xlabel('Predito', fontsize=11)
    ax.set_ylabel('Real', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

plt.suptitle('Matriz de Confusão — Comparação de Modelos', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('img/06_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Salvo: img/06_confusion_matrix.png")

# 5.3 Feature importance
fig, ax = plt.subplots(figsize=(10, 5))
colors_imp = ['#E53935' if v > feat_imp.median() else '#2196F3' for v in feat_imp.values]
feat_imp.plot(kind='barh', ax=ax, color=colors_imp, edgecolor='white')
ax.set_title('Feature Importance — Isolation Forest\n(vermelho = mais importante para detecção)', fontsize=12, fontweight='bold')
ax.set_xlabel('Importância relativa')
ax.axvline(feat_imp.median(), color='gray', linestyle='--', linewidth=1, label='Mediana')
ax.legend()
plt.tight_layout()
plt.savefig('img/07_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Salvo: img/07_feature_importance.png")

# 5.4 ROC Curve
fig, ax = plt.subplots(figsize=(7, 6))
fpr, tpr, _ = roc_curve(y, scores_if)
ax.plot(fpr, tpr, color=COLORS['pred'], linewidth=2, label=f'Isolation Forest (AUC = {auc_if:.3f})')
ax.plot([0,1], [0,1], 'k--', linewidth=1, label='Random classifier')
ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS['pred'])
ax.set_xlabel('False Alarm Rate', fontsize=11)
ax.set_ylabel('Detection Rate', fontsize=11)
ax.set_title('ROC Curve — SKAB Anomaly Detection', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig('img/08_roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Salvo: img/08_roc_curve.png")

# ── 6. RESUMO FINAL ───────────────────────────────────────────
print("\n" + "="*55)
print("📋 RESUMO DE RESULTADOS — SKAB ANOMALY DETECTION")
print("="*55)
print(f"{'Modelo':<25} {'F1-Score':>10} {'AUC-ROC':>10}")
print("-"*55)
print(f"{'Z-Score Baseline':<25} {f1_zscore:>10.4f} {'—':>10}")
print(f"{'Isolation Forest':<25} {f1_if:>10.4f} {auc_if:>10.4f}")
print("="*55)
print("\n✅ Modelos concluídos! Próximo: 03_results_visualization_skab.py")
