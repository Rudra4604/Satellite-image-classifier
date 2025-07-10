import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import plotly.express as px
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean professional CSS
st.markdown("""
<style>
    .main {
        padding: 2rem 1rem;
    }
    
    .header-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #7f8c8d;
        font-weight: 400;
    }
    
    .card {
        background: white;
        border-radius: 8px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        margin-bottom: 2rem;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .upload-area {
        border: 2px dashed #cbd5e0;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #4299e1;
        background: #ebf8ff;
    }
    
    .result-container {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        border-left: 4px solid #4299e1;
        margin: 1rem 0;
    }
    
    .prediction-badge {
        background: #4299e1;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        margin: 1rem 0;
    }
    
    .confidence-score {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin: 1rem 0;
    }
    
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .info-item {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    .info-item h4 {
        color: #2d3748;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .info-item p {
        color: #718096;
        font-size: 0.9rem;
        margin: 0;
    }
    
    .scores-container {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    .score-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #f7fafc;
    }
    
    .score-item:last-child {
        border-bottom: none;
    }
    
    .score-label {
        font-weight: 500;
        color: #2d3748;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .score-value {
        font-weight: 600;
        color: #4299e1;
    }
    
    .score-bar {
        width: 100px;
        height: 6px;
        background: #e2e8f0;
        border-radius: 3px;
        overflow: hidden;
        margin-left: 1rem;
    }
    
    .score-fill {
        height: 100%;
        background: #4299e1;
        border-radius: 3px;
        transition: width 0.5s ease;
    }
    
    .image-info {
        background: #f7fafc;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .image-info-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .image-info-item:last-child {
        border-bottom: none;
    }
    
    .stButton > button {
        background: #4299e1;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #3182ce;
        transform: translateY(-1px);
    }
    
    .feature-list {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .feature-item {
        padding: 1rem;
        background: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .divider {
        height: 1px;
        background: #e2e8f0;
        margin: 3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model configuration
MODEL_URL = "https://drive.google.com/uc?export=download&id=1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"
MODEL_PATH = "Modelenv.v1.h5"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None
    return load_model(MODEL_PATH)

# Load model
model = download_and_load_model()

# Class information
class_info = {
    'Cloudy': {'emoji': '☁️', 'color': '#87CEEB', 'description': 'Cloud formations and atmospheric conditions'},
    'Desert': {'emoji': '🏜️', 'color': '#F4A460', 'description': 'Arid regions and desert terrain'},
    'Green_Area': {'emoji': '🌿', 'color': '#32CD32', 'description': 'Vegetation, forests, and green spaces'},
    'Water': {'emoji': '💧', 'color': '#4682B4', 'description': 'Water bodies including rivers, lakes, and oceans'}
}

class_names = list(class_info.keys())

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🛰️ Satellite Image Classifier</div>
    <div class="header-subtitle">Advanced AI-powered terrain classification from satellite imagery</div>
</div>
""", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">📤 Upload Image</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a satellite image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a satellite image for classification"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_resized = image.resize((256, 256))
        
        # Display image
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Image information
        st.markdown("""
        <div class="image-info">
            <div class="image-info-item">
                <span><strong>Dimensions:</strong></span>
                <span>{} × {} pixels</span>
            </div>
            <div class="image-info-item">
                <span><strong>File Size:</strong></span>
                <span>{:.1f} KB</span>
            </div>
            <div class="image-info-item">
                <span><strong>Format:</strong></span>
                <span>{}</span>
            </div>
        </div>
        """.format(
            image.size[0], 
            image.size[1], 
            uploaded_file.size / 1024,
            uploaded_file.type.split('/')[-1].upper()
        ), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">🔍 Classification Results</div>
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None and model is not None:
        with st.spinner("Analyzing image..."):
            # Preprocess image
            img_array = img_to_array(image_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = model.predict(img_array, verbose=0)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)
            
            # Display main result
            predicted_info = class_info[predicted_class]
            st.markdown(f"""
            <div class="result-container">
                <div style="font-size: 3rem; margin-bottom: 1rem;">{predicted_info['emoji']}</div>
                <div class="prediction-badge">{predicted_class}</div>
                <div class="confidence-score">{confidence * 100:.1f}% Confidence</div>
                <p style="color: #718096; margin: 0;">{predicted_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed scores
            st.markdown("""
            <div class="scores-container">
                <h4 style="margin-bottom: 1rem; color: #2d3748;">Detailed Scores</h4>
            """, unsafe_allow_html=True)
            
            for i, class_name in enumerate(class_names):
                score = prediction[i] * 100
                class_data = class_info[class_name]
                
                st.markdown(f"""
                <div class="score-item">
                    <div class="score-label">
                        <span>{class_data['emoji']}</span>
                        <span>{class_name}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div class="score-bar">
                            <div class="score-fill" style="width: {score}%;"></div>
                        </div>
                        <div class="score-value">{score:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Visualization
            st.markdown("### Classification Breakdown")
            
            # Create clean chart
            df = pd.DataFrame({
                'Class': [f"{class_info[name]['emoji']} {name}" for name in class_names],
                'Confidence': prediction * 100
            })
            
            fig = px.bar(
                df, 
                x='Class', 
                y='Confidence',
                color='Confidence',
                color_continuous_scale='Blues',
                title="Confidence Scores by Class"
            )
            
            fig.update_layout(
                showlegend=False,
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family="Arial",
                title_font_size=16,
                xaxis_title="Classification",
                yaxis_title="Confidence (%)"
            )
            
            fig.update_traces(
                texttemplate='%{y:.1f}%',
                textposition='outside'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #718096;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🛰️</div>
            <h3 style="color: #2d3748; margin-bottom: 1rem;">Ready to Classify</h3>
            <p>Upload a satellite image to get started with AI-powered terrain classification.</p>
        </div>
        """, unsafe_allow_html=True)

# Divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Features section
st.markdown("### Key Features")

st.markdown("""
<div class="feature-list">
    <div class="feature-item">
        <div class="feature-icon">🤖</div>
        <h4>AI-Powered</h4>
        <p>Advanced deep learning model with high accuracy</p>
    </div>
    <div class="feature-item">
        <div class="feature-icon">⚡</div>
        <h4>Fast Processing</h4>
        <p>Real-time image classification in seconds</p>
    </div>
    <div class="feature-item">
        <div class="feature-icon">🎯</div>
        <h4>Multi-Class Detection</h4>
        <p>Identifies 4 different terrain types</p>
    </div>
    <div class="feature-item">
        <div class="feature-icon">📊</div>
        <h4>Detailed Analytics</h4>
        <p>Comprehensive confidence scores and visualizations</p>
    </div>
</div>
""", unsafe_allow_html=True)

# About section
with st.expander("About This Application"):
    st.markdown("""
    **Satellite Image Classifier** uses advanced machine learning to automatically identify and classify terrain types in satellite imagery.
    
    **Supported Classifications:**
    - **Cloudy**: Cloud formations and weather patterns
    - **Desert**: Arid regions and sandy terrain  
    - **Green Area**: Vegetation, forests, and agricultural land
    - **Water**: Rivers, lakes, oceans, and water bodies
    
    **Technical Specifications:**
    - Model: Convolutional Neural Network (CNN)
    - Input Size: 256×256 pixels
    - Framework: TensorFlow/Keras
    - Supported Formats: JPG, JPEG, PNG
    - Processing Time: < 1 second per image
    
    The model provides confidence scores for each classification to help you understand the certainty of predictions.
    """)

# Error handling
if model is None:
    st.error("Unable to load the classification model. Please refresh the page and try again.")
    st.stop()