# ==============================================================================
# UNIVERSIDAD ESTATAL AMAZÓNICA (UEA) - FACULTAD DE TI
# ASIGNATURA: MINERÍA DE DATOS (UEA-L-UFPTI-009)
# PROYECTO: MANTENIMIENTO PREDICTIVO RADAR INDRA LTR-20
# AUTORES: ERICK ROSERO HERRERA & MIGUEL TIMBIANO CAPITO
# REPOSOTORIO: https://github.com/ErickRosero/Mantenimiento-Radar-LTR-20.git
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------------------
# 1. GENERACIÓN DEL DATASET SINTÉTICO BASADO EN TELEMETRÍA BITE LTR-20
# ------------------------------------------------------------------------------
np.random.seed(42)
n_samples = 1350

data = {
    'id_registro': np.arange(1, n_samples + 1),
    'temperatura_gabinete_c': np.random.normal(loc=42.5, scale=5.0, size=n_samples),
    'voltaje_fuente_dc': np.random.normal(loc=28.0, scale=0.8, size=n_samples),
    'vswr_antena': np.random.uniform(low=1.05, high=2.30, size=n_samples),
    'potencia_transmision_kw': np.random.normal(
        loc=10.0, scale=1.2, size=n_samples
    ),
    'horas_transmision_acum': np.random.randint(
        low=100, high=5000, size=n_samples
    ),
    'humedad_interna_pct': np.random.uniform(
        low=20.0, high=70.0, size=n_samples
    ),
}

df = pd.DataFrame(data)

# Introducir nulos y duplicados intencionales segun guial
df.loc[df.sample(10, random_state=42).index, 'humedad_interna_pct'] = np.nan
duplicados = df.sample(14, random_state=42)
df = pd.concat([df, duplicados], ignore_index=True)

print(f'=== REPORTE INICIAL ===')
print(f'Total de registros iniciales: {len(df)}')
print(f'Duplicados detectados: {df.duplicated().sum()}')
print(f'Valores nulos por columna:\n{df.isnull().sum()}\n')

# ------------------------------------------------------------------------------
# 2. PREPROCESAMIENTO Y LIMPIEZA DE DATOS
# ------------------------------------------------------------------------------
# A. Eliminacion de duplicados
df = df.drop_duplicates().reset_index(drop=True)

# B. Imputacion de valores nulos con la mediana
df['humedad_interna_pct'].fillna(
    df['humedad_interna_pct'].median(), inplace=True
)

# C. Tratamiento de Outliers mediante Winsorization
limit_vswr = df['vswr_antena'].quantile(0.99)
df['vswr_antena'] = np.where(
    df['vswr_antena'] > limit_vswr, limit_vswr, df['vswr_antena']
)

# D. Feature Engineering (Variables Derivadas)
df['Indice_Estres_Termico_LTR20'] = df['temperatura_gabinete_c'] * (
    df['horas_transmision_acum'] / 1000.0
)
df['Ratio_VSWR_Potencia'] = df['vswr_antena'] / (
    df['potencia_transmision_kw'] + 1.0
)

# E. Generacion de la Variable Objetivo (Etiqueta de Salud del Hardware)
condiciones = [
    (df['Indice_Estres_Termico_LTR20'] < 120) & (df['vswr_antena'] < 1.35),
    (df['Indice_Estres_Termico_LTR20'] >= 120)
    & (df['Indice_Estres_Termico_LTR20'] < 190),
    (df['Indice_Estres_Termico_LTR20'] >= 190) | (df['vswr_antena'] >= 1.65),
]
clases = [0, 1, 2]  # 0: Normal, 1: Alerta, 2: Riesgo de Fallo
df['estado_operativo'] = np.select(condiciones, clases, default=0)

# ------------------------------------------------------------------------------
# 3. PREPARACIÓN DE MATRICES Y ESCALAMIENTO
# ------------------------------------------------------------------------------
features = [
    'temperatura_gabinete_c',
    'voltaje_fuente_dc',
    'vswr_antena',
    'potencia_transmision_kw',
    'horas_transmision_acum',
    'humedad_interna_pct',
    'Indice_Estres_Termico_LTR20',
    'Ratio_VSWR_Potencia',
]

X = df[features]
y = df['estado_operativo']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# ------------------------------------------------------------------------------
# 4. MODELADO Y EVALUACIÓN
# ------------------------------------------------------------------------------

# MODELO 1: K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
sil_score = silhouette_score(X_scaled, clusters)

# MODELO 2: Arbol de Decision (Decision Tree)
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

# MODELO 3: Random Forest Classifier (Ensamblaje)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
y_prob_rf = rf_clf.predict_proba(X_test)
auc_rf = roc_auc_score(y_test, y_prob_rf, multi_class='ovr')

# Validation Cruzada (10-Fold CV) para Random Forest
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_clf, X_scaled, y, cv=cv10, scoring='accuracy')

print(f'=== RESULTADOS DE MODELADO ===')
print(f'Silhouette Score (K-Means): {sil_score:.4f}')
print(f'Accuracy (Árbol de Decisión): {acc_dt * 100:.2f}%')
print(f'Accuracy (Random Forest): {acc_rf * 100:.2f}%')
print(f'AUC (Random Forest): {auc_rf:.4f}')
print(
    f'10-Fold CV Promedio (Random Forest): {cv_scores.mean() * 100:.2f}% +/- {cv_scores.std() * 100:.2f}%\n'
)

print('=== MATRIZ DE CONFUSIÓN (RANDOM FOREST) ===')
print(confusion_matrix(y_test, y_pred_rf))

print('\n=== REPORTE DE CLASIFICACIÓN (RANDOM FOREST) ===')
print(
    classification_report(
        y_test, y_pred_rf, target_names=['Normal', 'Alerta', 'Riesgo Fallo']
    )
)