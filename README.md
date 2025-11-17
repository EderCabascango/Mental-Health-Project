Predicción de Tratamiento en Salud Mental usando Machine Learning (CatBoost)

Este proyecto utiliza análisis exploratorio, ingeniería de características y el modelo CatBoostClassifier para predecir si una persona busca o recibe tratamiento de salud mental a partir de variables personales, emocionales y conductuales.

El dataset contiene más de 290.000 registros, principalmente categóricos, lo cual convierte a CatBoost en el modelo ideal gracias a su manejo nativo de variables categóricas sin necesidad de one-hot encoding.

🚀 Objetivo del Proyecto

El objetivo es desarrollar un modelo predictivo robusto que permita:

Identificar factores emocionales que influyen en la decisión de recibir tratamiento

Analizar hábitos, estrés, historial personal y otros predictores

Entrenar un modelo interpretable y escalable para aplicaciones reales

Proponer recomendaciones basadas en datos

📂 Contenido del Proyecto
project/
│── data/
│   ├── raw/                 # Dataset original (no se sube a GitHub)
│   ├── processed/           # Dataset limpio
│── notebooks/
│   ├── eda.ipynb            # Análisis exploratorio
│   ├── model_catboost.ipynb # Entrenamiento del modelo
│── src/
│   ├── preprocessing.py     # Transformaciones y limpieza
│   ├── train_model.py       # Entrenamiento final
│── models/
│   ├── catboost_model.cbm   # Modelo entrenado
│── requirements.txt
│── README.md
│── .gitignore

🧹 Limpieza y Feature Engineering

Se realizaron los siguientes pasos:

✔ Normalización de texto

Convirtiendo a minúsculas y eliminando espacios.

✔ Imputación de valores faltantes

Solo self_employed tenía missing — se imputó con la moda.

✔ Conversión de variables ordinales

days_indoors: mapeado a escala 0–4

mood_swings: mapeado a [0=low, 1=medium, 2=high]

✔ Eliminación de columnas irrelevantes

timestamp fue removido por no aportar valor predictivo.

✔ Matriz de Cramér’s V

Para detectar relación entre variables categóricas.

🤖 Modelo: CatBoostClassifier

CatBoost fue seleccionado por:

Manejo nativo de variables categóricas

Evita overfitting con Ordered Target Statistics

Alto rendimiento en datasets mixtos

Explicabilidad a través de SHAP

Hiperparámetros principales:
model = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.04,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    verbose=200
)

📊 Resultados

Los resultados pueden variar según el split, pero típicamente se obtiene:

AUC: 0.78 – 0.85

Precision / Recall balanceados

Alta interpretabilidad mediante SHAP

Se incluye:

Feature Importance

SHAP summary plot

Matriz de confusión