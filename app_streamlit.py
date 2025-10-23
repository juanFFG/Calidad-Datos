"""
Aplicación de Despliegue - Predicción de Enfermedades Cardíacas
Sistema de apoyo al diagnóstico basado en Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Configuración de la página:
st.set_page_config(
    page_title="Sistema de Predicción de Enfermedades Cardíacas",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "Sistema de apoyo al diagnóstico médico desarrollado con Machine Learning"
    }
)

# Estilos CSS:
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .prediction-box {
        padding: 2.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .prediction-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .positive {
        background: linear-gradient(135deg, #ef5350 0%, #e53935 100%);
        border: 4px solid #b71c1c;
        animation: pulse-red 2s infinite;
        color: white;
        font-weight: bold;
    }
    .negative {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        border: 4px solid #1b5e20;
        animation: pulse-green 2s infinite;
        color: white;
        font-weight: bold;
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 4px 6px rgba(244, 67, 54, 0.3); }
        50% { box-shadow: 0 4px 6px rgba(244, 67, 54, 0.6); }
        100% { box-shadow: 0 4px 6px rgba(244, 67, 54, 0.3); }
    }
    @keyframes pulse-green {
        0% { box-shadow: 0 4px 6px rgba(76, 175, 80, 0.3); }
        50% { box-shadow: 0 4px 6px rgba(76, 175, 80, 0.6); }
        100% { box-shadow: 0 4px 6px rgba(76, 175, 80, 0.3); }
    }
    .metric-card {
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #0d47a1;
        box-shadow: 0 3px 6px rgba(0,0,0,0.3);
        border: 2px solid #1565c0;
        color: white;
    }
    .section-header {
        background: linear-gradient(90deg, #0d47a1, #1565c0);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .input-group {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 2px solid #1976d2;
    }
    .stButton > button {
        background: linear-gradient(45deg, #1565c0, #0d47a1);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(13, 71, 161, 0.5);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(13, 71, 161, 0.7);
        background: linear-gradient(45deg, #0d47a1, #01579b);
    }
    
    /* Solución simple para texto borroso */
    .stSelectbox, .stNumberInput, .stCheckbox, .stSlider {
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
    }
    
    .success-box {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        border: 3px solid #1b5e20;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.3);
        color: white;
        font-weight: 500;
    }
    .warning-box {
        background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%);
        border: 3px solid #e65100;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.3);
        color: white;
        font-weight: 500;
    }
    .info-box {
        background: linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%);
        border: 3px solid #0d47a1;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.3);
        color: white;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Cargar modelos y componentes:
@st.cache_resource
def load_models():
    """Carga los modelos y componentes guardados"""
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('models/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, encoders, model_info
    except Exception as e:
        st.error(f"Error al cargar los modelos: {str(e)}")
        return None, None, None, None

# Cargar modelos:
model, scaler, label_encoders, model_info = load_models()

# Título y descripción
st.markdown('<h1 class="main-title">🫀 Sistema de Predicción de Enfermedades Cardíacas</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Modelo de Machine Learning para apoyo en el diagnóstico clínico</p>', unsafe_allow_html=True)

# Barra de progreso y estado del sistema
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if model is not None:
            st.success("✅ Modelo Cargado")
        else:
            st.error("❌ Error al cargar modelo")
    
    with col2:
        if scaler is not None:
            st.success("✅ Preprocesador Listo")
        else:
            st.error("❌ Error en preprocesador")
    
    with col3:
        if label_encoders is not None:
            st.success("✅ Sistema Operativo")
        else:
            st.error("❌ Sistema con errores")

# Información del modelo en sidebar:

with st.sidebar:
    st.markdown('<div class="section-header">📊 Información del Modelo</div>', unsafe_allow_html=True)
    
    if model_info:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🤖 Algoritmo", model_info['model_name'])
            st.metric("🎯 Accuracy", f"{model_info['test_accuracy']:.2%}")
        with col2:
            st.metric("⚖️ F1-Score", f"{model_info['test_f1']:.4f}")
            st.metric("📈 ROC-AUC", f"{model_info['test_roc_auc']:.4f}")
        
        st.markdown("---")
        st.markdown("**🔧 Hiperparámetros Optimizados**")
        for param, value in model_info['best_params'].items():
            st.markdown(f"• **{param}**: `{value}`")
        
        # Barra de confianza del modelo:
        st.markdown("---")
        st.markdown("**🎯 Confianza del Modelo**")
        confidence = model_info['test_accuracy'] * 100
        st.progress(confidence / 100)
        st.caption(f"Confianza: {confidence:.1f}%")
    

# Tabs principales:
tab1, tab2 = st.tabs(["Predicción", "Acerca del Modelo"])

with tab1:
    st.markdown('<div class="section-header">📝 Ingrese los Datos del Paciente</div>', unsafe_allow_html=True)
    
    # Información adicional:
    st.markdown("""
    <div class="info-box">
        <strong>💡 Instrucciones:</strong><br>
        Complete todos los campos con la información clínica del paciente. 
        Los campos marcados con <span style="color: red;">*</span> son obligatorios.
    </div>
    """, unsafe_allow_html=True)
    
    # Crear formulario en columnas con mejor organización:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("**👤 Datos Demográficos**")
        age = st.number_input("Edad (años)", min_value=0, max_value=120, value=50, help="Edad del paciente en años")
        sex = st.selectbox("Sexo", ["Male", "Female"], help="Sexo del paciente")
        dataset = st.selectbox("Centro Médico", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"], help="Centro médico de origen")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("**🫀 Síntomas y Signos**")
        cp = st.selectbox(
            "Tipo de Dolor Torácico",
            ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
            help="Tipo de dolor torácico reportado"
        )
        trestbps = st.number_input("Presión Arterial (mmHg)", min_value=80, max_value=220, value=120, help="Presión arterial sistólica en reposo")
        chol = st.number_input("Colesterol Sérico (mg/dl)", min_value=100, max_value=600, value=200, help="Nivel de colesterol en sangre")
        fbs = st.checkbox("Glucosa en Ayunas > 120 mg/dl", help="Azúcar en sangre en ayunas mayor a 120 mg/dl")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("**🔬 Pruebas Cardíacas**")
        restecg = st.selectbox(
            "Resultados ECG en Reposo",
            ["normal", "st-t abnormality", "lv hypertrophy"],
            help="Resultados del electrocardiograma en reposo"
        )
        thalch = st.number_input("Frecuencia Cardíaca Máxima", min_value=60, max_value=220, value=150, help="Frecuencia cardíaca máxima alcanzada")
        exang = st.checkbox("Angina por Ejercicio", help="Presencia de angina inducida por ejercicio")
        oldpeak = st.number_input("Depresión ST", min_value=-3.0, max_value=10.0, value=0.0, step=0.1, 
                                  help="Depresión ST inducida por ejercicio relativo al reposo")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Fila adicional para variables especializadas:
    st.markdown('<div class="section-header">🔬 Pruebas Especializadas</div>', unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("**📈 Segmento ST**")
        slope = st.selectbox(
            "Pendiente del Segmento ST",
            ["upsloping", "flat", "downsloping"],
            help="Pendiente del segmento ST del ejercicio máximo"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("**🩸 Fluoroscopia**")
        ca = st.slider("Vasos Principales", min_value=0, max_value=3, value=0,
                      help="Número de vasos principales coloreados por fluoroscopia")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown("**🧬 Talasemia**")
        thal = st.selectbox(
            "Resultado Talasemia",
            ["normal", "fixed defect", "reversable defect"],
            help="Resultado de la prueba de talasemia"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Botón de predicción:
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        if st.button("🔮 Realizar Predicción", type="primary", width="stretch"):
            if model is None:
                st.error("Error: No se pudieron cargar los modelos. Verifique que los archivos .pkl existan en la carpeta 'models/'")
            else:
                # Crear DataFrame con los datos ingresados:
                input_data = pd.DataFrame({
                    'age': [age],
                    'sex': [sex],
                    'dataset': [dataset],
                    'cp': [cp],
                    'trestbps': [trestbps],
                    'chol': [chol],
                    'fbs': [fbs],
                    'restecg': [restecg],
                    'thalch': [thalch],
                    'exang': [exang],
                    'oldpeak': [oldpeak],
                    'slope': [slope],
                    'ca': [float(ca)],
                    'thal': [thal]
                })
            
            try:
                # Codificar variables categóricas:
                for col in ['sex', 'cp', 'restecg', 'slope', 'thal', 'dataset']:
                    if col in label_encoders:
                        input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
                
                # Convertir booleanos a numérico:
                input_data['fbs'] = input_data['fbs'].astype(int)
                input_data['exang'] = input_data['exang'].astype(int)
                
                # Asegurar el orden correcto de columnas:
                input_data = input_data[model_info['feature_columns']]
                
                # Escalar datos:
                input_scaled = scaler.transform(input_data)
                
                # Realizar predicción:
                prediction = model.predict(input_scaled)[0]
                
                # Mostrar resultado:
                st.markdown("---")
                st.markdown('<div class="section-header">🎯 Resultado de la Predicción</div>', unsafe_allow_html=True)
                
                if prediction == 1:
                    st.markdown('<div class="prediction-box positive">', unsafe_allow_html=True)
                    st.markdown("### ⚠️ **RIESGO DE ENFERMEDAD CARDÍACA DETECTADO**")
                    st.markdown("El modelo indica presencia de enfermedad cardíaca")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="warning-box">
                        <strong>🚨 Recomendación Médica:</strong><br>
                        Se sugiere consultar inmediatamente con un cardiólogo para evaluación adicional 
                        y seguimiento médico especializado.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-box negative">', unsafe_allow_html=True)
                    st.markdown("### ✅ **SIN RIESGO DE ENFERMEDAD CARDÍACA**")
                    st.markdown("El modelo no indica presencia de enfermedad cardíaca")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="success-box">
                        <strong>💚 Recomendación:</strong><br>
                        Mantener hábitos de vida saludables y continuar con controles médicos regulares.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Mostrar datos del paciente con mejor diseño:
                st.markdown("---")
                st.markdown('<div class="section-header">📋 Resumen de Datos Clínicos</div>', unsafe_allow_html=True)
                
                # Crear un resumen más legible:
                summary_data = {
                    'Parámetro': ['Edad', 'Sexo', 'Centro Médico', 'Dolor Torácico', 'Presión Arterial', 
                                'Colesterol', 'Glucosa Ayunas', 'ECG Reposo', 'Frecuencia Cardíaca Máx',
                                'Angina Ejercicio', 'Depresión ST', 'Pendiente ST', 'Vasos Principales', 'Talasemia'],
                    'Valor': [f"{age} años", sex, dataset, cp, f"{trestbps} mmHg", 
                            f"{chol} mg/dl", "Sí" if fbs else "No", restecg, f"{thalch} bpm",
                            "Sí" if exang else "No", f"{oldpeak}", slope, f"{ca}", thal]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, width="stretch", hide_index=True)
                
            except Exception as e:
                st.error(f"Error al realizar la predicción: {str(e)}")
                st.error("Verifique que todos los datos estén correctos y que los modelos sean compatibles.")

with tab2:
    st.markdown('<div class="section-header">🔬 Información Detallada del Modelo</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 🧠 Desarrollo del Modelo")
        st.markdown("""
        <div class="info-box">
            <strong>📊 Algoritmo:</strong> Support Vector Machine (SVM)<br>
            <strong>📈 Datos de Entrenamiento:</strong> 918 pacientes de múltiples centros médicos<br>
            <strong>🔧 Optimización:</strong> GridSearchCV con validación cruzada
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Proceso de Desarrollo:**")
        st.markdown("""
        1. **Limpieza de Datos**: 100% completitud alcanzada
        2. **Análisis Exploratorio**: Identificación de patrones y correlaciones
        3. **Feature Engineering**: Preparación de variables predictoras
        4. **Entrenamiento**: Comparación de 5 algoritmos diferentes
        5. **Selección**: SVM elegido por mejor rendimiento
        6. **Optimización**: Ajuste fino de hiperparámetros
        7. **Validación**: Test en conjunto independiente
        """)
        
        st.markdown("**Características Técnicas:**")
        st.markdown("""
        - **Variables Predictoras**: 14 características clínicas
        - **Preprocesamiento**: StandardScaler + LabelEncoder
        - **Validación**: 5-fold cross validation
        - **Optimización**: Maximización de F1-Score
        - **Robustez**: Validación en múltiples centros médicos
        """, unsafe_allow_html=False)
    
    with col_right:
        st.markdown("### 📈 Métricas de Rendimiento")
        
        if model_info:
            # Métricas principales:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🎯 Accuracy", f"{model_info['test_accuracy']:.2%}")
                st.caption("84% de predicciones correctas")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("⚖️ F1-Score", f"{model_info['test_f1']:.2%}")
                st.caption("Balance entre precisión y sensibilidad")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📈 ROC-AUC", f"{model_info['test_roc_auc']:.2%}")
                st.caption("Excelente capacidad discriminativa")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🎯 Precision", "83.95%")
                st.caption("84% de predicciones positivas correctas")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Tabla de métricas:
            st.markdown("**📊 Métricas Detalladas:**")
            metrics_data = {
                'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Valor': [
                    f"{model_info['test_accuracy']:.2%}",
                    "83.95%",
                    "88.31%",
                    f"{model_info['test_f1']:.2%}",
                    f"{model_info['test_roc_auc']:.2%}"
                ],
                'Interpretación': [
                    'Predicciones correctas',
                    'Predicciones positivas correctas',
                    'Casos positivos detectados',
                    'Balance precision-recall',
                    'Capacidad discriminativa'
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, width="stretch", hide_index=True)
        
        st.markdown("---")
        st.subheader("Variables Predictoras")
        st.markdown("""
        El modelo utiliza las siguientes características clínicas:
        - **Demográficas**: Edad, sexo
        - **Síntomas**: Tipo de dolor torácico, angina por ejercicio
        - **Signos vitales**: Presión arterial, frecuencia cardíaca máxima
        - **Pruebas de laboratorio**: Colesterol, glucosa en ayunas
        - **Pruebas cardíacas**: ECG, depresión ST, pendiente ST, fluoroscopia, talasemia
        """)
    

