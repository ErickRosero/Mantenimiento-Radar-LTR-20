# Mantenimiento-Radar-LTR-20
Práctica experimental de Minería de Datos/Radar Indra LTR-20
# Minería de Datos Aplicada al Mantenimiento Predictivo del Radar Indra LTR-20

Repositorio correspondiente al componente práctico-experimental de la asignatura de **Minería de Datos** de la Universidad Estatal Amazónica (UEA).

## Autores
* Miguel Angel Timbiano Capito
* Erick Sebastián Rosero Herrera

---

## 📋 Descripción del Proyecto
Este proyecto implementa modelos de aprendizaje automático (Machine Learning) orientados al mantenimiento preventivo y la detección temprana de fallas en los subsistemas electrónicos del radar tridimensional móvil **Indra LTR-20**. 

Se aplicó la metodología **CRISP-DM** para el procesamiento de datos telemétricos provenientes del sistema de autodiagnóstico (BITE), integrando técnicas de:
1. **Estadística descriptiva y preprocesamiento** (limpieza de nulos, eliminación de duplicados y *winsorization* de valores atípicos).
2. **Ingeniería de características** (creación del `Indice_Estres_Termico_LTR20`).
3. **Modelado predictivo y de agrupamiento** (K-Means Clustering, Árboles de Decisión y Random Forest).
4. **Validación cruzada** (10-Fold Cross-Validation).

---

## 📂 Estructura del Repositorio
* `dataset_radar_ltr20.csv` -> Conjunto de datos simulado con 1,350 registros telemétricos.
* `codigo_mineria_datos.py` -> Script en Python con la ejecución de los algoritmos de Scikit-Learn y Pandas.
* `informe_practica.pdf` -> Documento oficial del informe presentado en la UEA.

---

## 🚀 Resultados Principales
El modelo de **Random Forest** demostró el mejor rendimiento con una exactitud (*Accuracy*) del **94.2%** y un área bajo la curva AUC de **0.965**, validando su alta capacidad para anticipar fallas críticas en los módulos de radiofrecuencia del radar.
