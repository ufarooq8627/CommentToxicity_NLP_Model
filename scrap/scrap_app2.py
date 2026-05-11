"""
Comment Toxicity Detection - Streamlit Web Application

Interactive web app for real-time toxicity detection using
a trained Bidirectional LSTM deep learning model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import re
import string
import os
import time

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Comment Toxicity Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark/Purple Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Main App Background */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #1e1b4b 0%, #312e81 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #3730a3;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .main-header-left {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 600; color: #fff;}
    .main-header p { margin: 0.2rem 0 0 0; color: #a5b4fc; font-size: 1rem; }
    
    .model-badge {
        background: rgba(255,255,255,0.1);
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Cards */
    .dark-card {
        background: #111827;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #1f2937;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        height: 100%;
    }
    
    .metric-box {
        background: #1f2937;
        border: 1px solid #374151;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .metric-title { font-size: 0.8rem; color: #9ca3af; margin-bottom: 5px; }
    .metric-value { font-size: 1.2rem; font-weight: 600; color: #fff; }
    .metric-sub { font-size: 0.75rem; color: #10b981; }

    /* Progress bars custom styling */
    .stProgress > div > div > div > div {
        background-color: #ef4444;
    }

    /* Custom Button Styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6366f1 0%, #d946ef 100%);
        border: none;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(217, 70, 239, 0.4);
    }
    

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        color: #fff !important;
        border-bottom-color: #818cf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MAX_LEN = 200
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Helper Functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r"={2,}", " ", text)
    text = re.sub(r":{2,}", " ", text)
    text = re.sub(r"\\n|\\t|\\r", " ", text)
    text = re.sub(r"\n|\t|\r", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/toxicity_model.keras")
    with open("model/tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

@st.cache_data
def load_metrics():
    if os.path.exists("model/metrics.json"):
        with open("model/metrics.json", "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_training_data():
    if os.path.exists("train.csv"):
        return pd.read_csv("train.csv")
    return None

def predict_toxicity(text, model, tokenizer):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0]
    return {label: float(pred) for label, pred in zip(LABELS, prediction)}

def create_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        number = {'suffix': "%", 'font': {'color': 'white', 'size': 40}},
        title = {'text': "Toxicity Score", 'font': {'color': '#9ca3af', 'size': 16}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "white", 'thickness': 0.1},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': "#10b981"}, # Green
                {'range': [30, 60], 'color': "#f59e0b"}, # Yellow
                {'range': [60, 100], 'color': "#ef4444"} # Red
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score * 100
            }
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )
    return fig


# Header
st.markdown("""
<div class="main-header">
    <div class="main-header-left">
        <h1 style="font-size: 3rem;">🛡️</h1>
        <div>
            <h1>Comment Toxicity Detector</h1>
            <p>AI-powered real-time toxicity detection using Deep Learning (BiLSTM)</p>
        </div>
    </div>
    <div class="model-badge">
        🔵 Model: BiLSTM
    </div>
</div>
""", unsafe_allow_html=True)

# Load Model
try:
    model, tokenizer = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Model not found. Please run `python train_model.py` first.\n\nError: {e}")

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Settings")
    threshold = st.slider("Toxicity Threshold", 0.0, 1.0, 0.5, 0.05)
    st.caption(f"Scores ≥ {threshold:.0%} flag as toxic.")
    
    st.markdown("---")
    st.markdown("### ℹ️ Architecture")
    st.markdown("""
    - Embedding
    - SpatialDropout1D
    - BiLSTM (64)
    - GlobalMaxPool1D
    - Dense (64) + Dropout
    - Dense (6, Sigmoid)
    """)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Bulk Prediction", "📊 Dashboard"])

# Tab 1: Single Prediction
with tab1:
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown('<div class="dark-card">', unsafe_allow_html=True)
        st.markdown("### 💬 Analyze a Comment")
        st.caption("Try a sample or enter your own comment to analyze")
        
        sample = st.selectbox("Or try a sample:", [
            "— Select a sample —",
            "Thank you for your help, this article is really well written!",
            "You are a complete idiot, go kill yourself nobody likes you",
            "This is the worst garbage I have ever read, you stupid moron",
            "The edit was reverted because it violated community policy",
            "I hate you and everyone like you, you disgusting piece of trash"
        ])

        default_text = sample if sample != "— Select a sample —" else ""

        user_input = st.text_area(
            "Enter a comment to analyze:",
            value=default_text,
            placeholder="Type your comment here...",
            height=150
        )

        analyze_btn = st.button("✨ Analyze", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dark-card">', unsafe_allow_html=True)
        st.markdown("### 🛡️ Prediction Result")
        
        if analyze_btn and user_input and model_loaded:
            start_time = time.time()
            with st.spinner("Analyzing..."):
                scores = predict_toxicity(user_input, model, tokenizer)
            process_time = time.time() - start_time
            
            max_score = max(scores.values())
            is_toxic = max_score >= threshold
            
            # Gauge Chart
            st.plotly_chart(create_gauge_chart(max_score), use_container_width=True)
            
            # Verdict
            if is_toxic:
                st.error(f"🚨 **Toxic** - This comment is likely to be toxic. (High Severity)")
            elif max_score >= (threshold * 0.6):
                st.warning(f"⚠️ **Warning** - This comment has mild toxicity indicators.")
            else:
                st.success(f"✅ **Safe** - This comment appears to be safe and non-toxic.")
                
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-box"><div class="metric-title">Confidence</div><div class="metric-value">{max_score:.1%}</div><div class="metric-sub">{"Very High" if max_score > 0.8 else "Moderate"}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown('<div class="metric-box"><div class="metric-title">Model</div><div class="metric-value">BiLSTM</div><div class="metric-sub" style="color:#818cf8">Deep Learning</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-box"><div class="metric-title">Time Taken</div><div class="metric-value">{process_time:.2f}s</div><div class="metric-sub">Fast</div></div>', unsafe_allow_html=True)
                
            st.markdown("<br>#### Probability Breakdown", unsafe_allow_html=True)
            
            # Custom styled progress bars for breakdown
            for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                col_name, col_prog, col_val = st.columns([3, 6, 1])
                with col_name:
                    st.write(f"{label.replace('_', ' ').title()}")
                with col_prog:
                    # Streamlit progress bar
                    st.progress(min(score, 1.0))
                with col_val:
                    st.write(f"{score:.0%}")

        else:
            st.info("👈 Enter a comment on the left and click Analyze to see the results here.")
            # Show empty placeholder
            st.plotly_chart(create_gauge_chart(0), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# Tab 2: Bulk Prediction
with tab2:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.subheader("Bulk Comment Analysis")
    st.markdown("Upload a CSV file with a `comment_text` column to analyze multiple comments.")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file and model_loaded:
        df_upload = pd.read_csv(uploaded_file)

        if 'comment_text' not in df_upload.columns:
            st.error("❌ CSV must contain a `comment_text` column!")
        else:
            st.info(f"📄 Loaded {len(df_upload)} comments")

            if st.button("🚀 Predict All", type="primary"):
                with st.spinner(f"Analyzing {len(df_upload)} comments..."):
                    cleaned_texts = df_upload['comment_text'].astype(str).apply(clean_text).tolist()
                    sequences = tokenizer.texts_to_sequences(cleaned_texts)
                    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

                    predictions = model.predict(padded, batch_size=256, verbose=0)

                    results_df = pd.DataFrame(predictions, columns=LABELS)
                    output_df = pd.concat([df_upload.reset_index(drop=True), results_df], axis=1)

                    output_df['is_toxic'] = (results_df.max(axis=1) >= threshold).astype(int)

                st.success(f"✅ Analysis complete!")

                toxic_count = output_df['is_toxic'].sum()
                clean_count = len(output_df) - toxic_count
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Comments", len(output_df))
                with col2:
                    st.metric("🚨 Toxic", int(toxic_count))
                with col3:
                    st.metric("✅ Clean", int(clean_count))

                st.dataframe(output_df, use_container_width=True)

                csv = output_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="toxicity_predictions.csv",
                    mime="text/csv"
                )
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Dashboard
with tab3:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.subheader("Data Insights & Model Performance")

    metrics = load_metrics()
    if metrics:
        st.markdown("#### 📈 Model Performance")

        auc_scores = metrics.get('auc_roc_scores', {})
        if auc_scores:
            cols = st.columns(len(LABELS))
            for i, label in enumerate(LABELS):
                with cols[i]:
                    score = auc_scores.get(label, 0)
                    if score:
                        st.markdown(f"""
                        <div class="metric-box" style="margin-bottom: 15px;">
                            <div class="metric-title">{label.replace('_', ' ').title()}</div>
                            <div class="metric-value" style="color: #818cf8;">{score:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)

            overall = auc_scores.get('overall_macro', 0)
            if overall:
                st.markdown(f"**Overall Macro AUC-ROC: `{overall:.4f}`**")

        history = metrics.get('training_history', {})
        if history:
            st.markdown("#### 📉 Training History")
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                # Dark theme matplotlib
                fig.patch.set_facecolor('#111827')
                ax.set_facecolor('#111827')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                
                ax.plot(history['loss'], label='Train Loss', color='#818cf8', linewidth=2)
                ax.plot(history['val_loss'], label='Val Loss', color='#f43f5e', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training vs Validation Loss')
                legend = ax.legend(facecolor='#1f2937', edgecolor='#374151', labelcolor='white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#374151')
                ax.spines['left'].set_color('#374151')
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                fig.patch.set_facecolor('#111827')
                ax.set_facecolor('#111827')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                
                ax.plot(history['accuracy'], label='Train Acc', color='#10b981', linewidth=2)
                ax.plot(history['val_accuracy'], label='Val Acc', color='#38bdf8', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.set_title('Training vs Validation Accuracy')
                legend = ax.legend(facecolor='#1f2937', edgecolor='#374151', labelcolor='white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#374151')
                ax.spines['left'].set_color('#374151')
                plt.tight_layout()
                st.pyplot(fig)

    else:
        st.info("📌 Train the model first by running `python train_model.py` to see the dashboard.")
    st.markdown('</div>', unsafe_allow_html=True)
