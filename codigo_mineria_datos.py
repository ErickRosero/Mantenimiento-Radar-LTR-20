# ==============================================================================
# PROYECTO: Minería de Datos para el Mantenimiento Predictivo - Radar Indra LTR-20
# Asignatura: Minería de Datos - Universidad Estatal Amazónica (UEA)
# Autores: Miguel Angel Timbiano Capito, Erick Sebastián Rosero Herrera
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import CrossValidatorScore, train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

# 1. Simulación y Carga del Dataset de Telemetría (1350 registros)
np.random.seed(42)
n_registros = 1350

data = {
    'Temperatura_TRM': np.random.normal(63.8, 4.2, n_registros),
    'Voltaje_Fuente': np.random.normal(28.0, 1.9, n_registros),
    'Corriente_Consumo': np.random.normal(12.5, 1.1, n_registros),
    'VSWR_Antena': np.random.normal(1.2, 0.15, n_registros),
    'Horas_Operacion': np.random.randint(100, 5000, n_registros),
    'Humedad_Interna': np.random.normal(45.0, 5.0, n_registros),
    'Potencia_Salida': np.random.normal(98.0, 2.5, n_registros),
    'Falla_Inminente': np.random.choice([0, 1], size=n_registros, p=[0.75, 0.25])
}

df = pd.DataFrame(data)

# Introducir intencionalmente algunos valores nulos y duplicados para la fase de limpieza
df.loc[10:20, 'Humedad_Interna'] = np.nan
df = pd.concat([df, df.iloc[0:14]], ignore_index=True) # Duplicados

print(f"Dimensiones iniciales del dataset: {df.shape}")

# 2. Preprocesamiento y Limpieza de Datos
# Eliminar duplicados
df.drop_duplicates(inplace=True)

# Tratar valores nulos con imputación por la media
imputer = SimpleImputer(strategy='mean')
df['Humedad_Interna'] = imputer.fit_transform(df[['Humedad_Interna']])

# Ingeniería de Características (Feature Engineering)
df['Indice_Estres_Termico_LTR20'] = df['Temperatura_TRM'] * (df['Horas_Operacion'] / 1000)
df['Ratio_VSWR_Potencia'] = df['VSWR_Antena'] / (df['Potencia_Salida'] + 1)

# Normalización MinMax
scaler = MinMaxScaler()
cols_a_normalizar = ['Temperatura_TRM', 'Voltaje_Fuente', 'Corriente_Consumo', 'VSWR_Antena', 'Indice_Estres_Termico_LTR20']
df[cols_a_normalizar] = scaler.fit_transform(df[cols_a_normalizar])

# 3. Modelado: Preparación de Datos (Entrenamiento 80% / Prueba 20%)
X = df.drop(columns=['Falla_Inminente'])
y = df['Falla_Inminente']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo 1: K-Means Clustering (No supervisado)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster_Desgaste'] = kmeans.fit_predict(X[cols_a_normalizar])

# Modelo 2: Árbol de Decisión
tree_model = DecisionTreeClassifier(max_depth=6, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)

# Modelo 3: Random Forest (Modelo Principal)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_proba_rf)

# 4. Evaluación con Validación Cruzada (10-Fold CV)
cv_scores = cross_val_score(rf_model, X, y, cv=10, scoring='accuracy')

print("\n================ RESULTADOS FINALES ================")
print(f"Exactitud Árbol de Decisión: {acc_tree * 100:.2f}%")
print(f"Exactitud Random Forest: {acc_rf * 100:.2f}%")
print(f"Área AUC Random Forest: {auc_rf:.3f}")
print(f"Validación Cruzada (10-Fold) - Promedio Accuracy: {cv_scores.mean() * 100:.2f}%")
print("=====================================================")