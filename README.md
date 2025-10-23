# Predicción de Enfermedades Cardíacas con Machine Learning

Este proyecto implementa un sistema de predicción de enfermedades cardíacas utilizando técnicas de minería de datos y machine learning. El objetivo es desarrollar una herramienta que ayude a los profesionales de la salud en el diagnóstico temprano de problemas cardíacos.

## Descripción del Proyecto

El sistema analiza datos clínicos de pacientes para predecir la presencia de enfermedades cardíacas. Utiliza un modelo de Support Vector Machine (SVM) optimizado que alcanza una precisión del 84% en la predicción.

## Datos Utilizados

- **Fuente**: Dataset UCI Heart Disease con datos de múltiples centros médicos
- **Registros**: 918 pacientes
- **Variables**: 14 características clínicas incluyendo edad, presión arterial, colesterol, ECG, y pruebas cardíacas especializadas
- **Centros**: Cleveland, Hungary, Switzerland, VA Long Beach

## Componentes del Proyecto

### 1. Análisis de Calidad de Datos (`01_Calidad_Datos.ipynb`)
- Diagnóstico completo de las 5 dimensiones de calidad de datos
- Generación de reporte HTML con ydata-profiling
- Limpieza y preparación de datos
- Eliminación de valores faltantes y anomalías
- Resultado: Dataset limpio con 100% completitud

### 2. Modelo Predictivo (`02_Modelo_Predictivo.ipynb`)
- Implementación de 5 algoritmos de machine learning:
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Multi-Layer Perceptron (MLP)
  - Support Vector Machine (SVM)
  - Random Forest
- Evaluación comparativa de modelos
- Optimización de hiperparámetros con GridSearchCV
- Selección del mejor modelo (SVM)

### 3. Aplicación Web (`app_streamlit.py`)
- Interfaz gráfica desarrollada con Streamlit
- Formulario interactivo para entrada de datos del paciente
- Predicción en tiempo real
- Visualización de resultados con recomendaciones médicas
- Información detallada del modelo

## Resultados del Modelo

El modelo SVM optimizado presenta las siguientes métricas de rendimiento:

- **Accuracy**: 84.06%
- **Precision**: 83.95%
- **Recall**: 88.31%
- **F1-Score**: 0.8608
- **ROC-AUC**: 0.89

## Instalación y Uso

### Requisitos
```bash
pip install -r requirements.txt
```

### Ejecutar la Aplicación
```bash
streamlit run app_streamlit.py
```

La aplicación estará disponible en `http://localhost:8501`

## Estructura del Proyecto

```
Mineria-Calidad-Datos/
├── 01_Calidad_Datos.ipynb          # Análisis de calidad de datos
├── 02_Modelo_Predictivo.ipynb      # Desarrollo del modelo predictivo
├── app_streamlit.py                # Aplicación web
├── requirements.txt                 # Dependencias
├── README.md                       # Este archivo
├── datasets/
│   ├── heart_disease_uci.csv       # Dataset original
│   └── heart_disease_clean.csv     # Dataset limpio
├── models/
│   ├── best_model.pkl             # Modelo entrenado
│   ├── scaler.pkl                  # Normalizador
│   ├── label_encoders.pkl         # Codificadores
│   └── model_info.pkl             # Información del modelo
└── reporte/
    └── heart_disease_profiling_report.html  # Reporte de perfilado
```

## Características Técnicas

- **Algoritmo**: Support Vector Machine (SVM)
- **Preprocesamiento**: StandardScaler + LabelEncoder
- **Validación**: 5-fold cross validation
- **Optimización**: GridSearchCV
- **División de datos**: 70% entrenamiento, 15% validación, 15% test

## Variables Predictoras

El modelo utiliza 14 variables clínicas:

**Datos Demográficos:**
- Edad del paciente
- Sexo
- Centro médico de origen

**Síntomas y Signos:**
- Tipo de dolor torácico
- Presión arterial en reposo
- Colesterol sérico
- Glucosa en ayunas

**Pruebas Cardíacas:**
- Resultados del ECG en reposo
- Frecuencia cardíaca máxima
- Angina inducida por ejercicio
- Depresión ST
- Pendiente del segmento ST
- Número de vasos principales
- Resultado de talasemia


**Integrantes:**
- Isaac Echeverri Ospina
- Samuel Mauricio Arango Arias
- Juan Felipe Fernández Grajales

