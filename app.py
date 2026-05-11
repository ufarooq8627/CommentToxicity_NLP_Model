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

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Comment Toxicity Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem; }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card h3 { margin: 0; color: #2d3436; font-size: 0.9rem; font-weight: 500; }
    .metric-card h2 { margin: 0.3rem 0 0 0; color: #6c5ce7; font-size: 1.8rem; font-weight: 700; }

    .toxic-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-safe { background: #00b894; color: white; }
    .badge-warning { background: #fdcb6e; color: #2d3436; }
    .badge-danger { background: #e17055; color: white; }
    .badge-critical { background: #d63031; color: white; }

    .result-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MAX_LEN = 200
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
LABEL_EMOJIS = {
    'toxic': '☠️', 'severe_toxic': '💀', 'obscene': '🤬',
    'threat': '⚠️', 'insult': '😡', 'identity_hate': '🚫'
}


# Helper Functions
def clean_text(text):
    """Clean raw comment text."""
    text = text.lower()
    # remove Wikipedia markup (== headers ==, ::: indents, etc.)
    text = re.sub(r"={2,}", " ", text)
    text = re.sub(r":{2,}", " ", text)
    # remove escaped newlines and tabs
    text = re.sub(r"\\n|\\t|\\r", " ", text)
    text = re.sub(r"\n|\t|\r", " ", text)
    # remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove numbers
    text = re.sub(r"\d+", "", text)
    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_severity_badge(score):
    """Return HTML badge based on toxicity score."""
    if score < 0.3:
        return f'<span class="toxic-badge badge-safe">{score:.1%} Safe</span>'
    elif score < 0.5:
        return f'<span class="toxic-badge badge-warning">{score:.1%} Mild</span>'
    elif score < 0.7:
        return f'<span class="toxic-badge badge-danger">{score:.1%} Toxic</span>'
    else:
        return f'<span class="toxic-badge badge-critical">{score:.1%} Highly Toxic</span>'


@st.cache_resource
def load_model():
    """Load the trained model and tokenizer."""
    model = tf.keras.models.load_model("model/toxicity_model.keras")
    with open("model/tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


@st.cache_data
def load_metrics():
    """Load saved evaluation metrics."""
    if os.path.exists("model/metrics.json"):
        with open("model/metrics.json", "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_training_data():
    """Load the training data for EDA."""
    if os.path.exists("train.csv"):
        return pd.read_csv("train.csv")
    return None


def predict_toxicity(text, model, tokenizer):
    """Predict toxicity scores for a given text."""
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0]
    return {label: float(pred) for label, pred in zip(LABELS, prediction)}


# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ Comment Toxicity Detector</h1>
    <p>AI-powered real-time toxicity detection using Deep Learning (BiLSTM)</p>
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
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app uses a **Bidirectional LSTM** deep learning model
    to detect toxic comments in real-time.

    **Toxicity Labels:**
    - ☠️ Toxic
    - 💀 Severe Toxic
    - 🤬 Obscene
    - ⚠️ Threat
    - 😡 Insult
    - 🚫 Identity Hate

    **Model Architecture:**
    - Embedding → SpatialDropout1D
    - Bidirectional LSTM (64 units)
    - GlobalMaxPooling → Dense → Sigmoid

    **Built with:** TensorFlow, Streamlit
    """)

    st.markdown("---")
    st.markdown("### 🎛️ Settings")
    threshold = st.slider("Toxicity Threshold", 0.0, 1.0, 0.5, 0.05)
    st.caption(f"Comments with any score ≥ {threshold:.0%} will be flagged as toxic.")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Bulk Prediction", "📊 Dashboard"])

# Tab 1: Single Prediction
with tab1:
    # WHY WE BUILT THIS TAB:
    # This section allows moderators or users to test the model by typing in a single raw comment.
    # It proves the model works in "real-time" and visually breaks down exactly which toxic categories
    # were triggered using progress bars. It shows the practical, interactive side of our Deep Learning model.
    st.subheader("Analyze a Comment")

    # 1. Sample selection at the top
    sample = st.selectbox("Or try a sample:", [
        "— Select a sample —",
        "Thank you for your help, this article is really well written!",
        "You are a complete idiot, go kill yourself nobody likes you",
        "This is the worst garbage I have ever read, you stupid moron",
        "The edit was reverted because it violated community policy",
        "I hate you and everyone like you, you disgusting piece of trash"
    ])

    # If a sample is selected, pre-fill the text area with it
    default_text = sample if sample != "— Select a sample —" else ""

    # 2. Text area in the middle
    user_input = st.text_area(
        "Enter a comment to analyze:",
        value=default_text,
        placeholder="Type or paste a comment here...",
        height=120
    )

    # 3. Analyze button at the bottom
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze_btn and user_input and model_loaded:
        with st.spinner("Analyzing..."):
            scores = predict_toxicity(user_input, model, tokenizer)

        # Overall verdict using dynamic threshold
        max_score = max(scores.values())
        if max_score >= threshold:
            st.error(f"🚨 This comment is likely **toxic**! (Score: {max_score:.1%} ≥ Threshold: {threshold:.1%})")
        elif max_score >= (threshold * 0.6): # Mild warning if it's getting close to threshold
            st.warning(f"⚠️ This comment has **mild toxicity** indicators. (Score: {max_score:.1%})")
        else:
            st.success(f"✅ This comment appears to be **safe and non-toxic**.")

        # Score breakdown
        st.markdown("#### Toxicity Breakdown")
        cols = st.columns(3)
        for i, (label, score) in enumerate(scores.items()):
            with cols[i % 3]:
                emoji = LABEL_EMOJIS.get(label, '📊')
                color = '#00b894' if score < 0.3 else '#fdcb6e' if score < 0.5 else '#e17055' if score < 0.7 else '#d63031'
                st.markdown(f"**{emoji} {label.replace('_', ' ').title()}**")
                st.progress(min(score, 1.0))
                st.markdown(get_severity_badge(score), unsafe_allow_html=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ['#00b894' if s < 0.3 else '#fdcb6e' if s < 0.5 else '#e17055' if s < 0.7 else '#d63031' for s in scores.values()]
        bars = ax.barh(list(scores.keys()), list(scores.values()), color=colors, height=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Toxicity Score")
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

# Tab 2: Bulk Prediction
with tab2:
    # WHY WE BUILT THIS TAB:
    # Single predictions are great for testing, but real companies (like YouTube or Reddit) need to
    # process millions of comments at once. This tab demonstrates "batch inference", predicting on
    # an entire CSV file cleanly and efficiently. It solves the core business problem of automated scale.
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
                    # Batch prediction — processes all at once on GPU for maximum speed
                    cleaned_texts = df_upload['comment_text'].astype(str).apply(clean_text).tolist()
                    sequences = tokenizer.texts_to_sequences(cleaned_texts)
                    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

                    # Single GPU call for all predictions (much faster than one-by-one)
                    predictions = model.predict(padded, batch_size=256, verbose=0)

                    results_df = pd.DataFrame(predictions, columns=LABELS)
                    output_df = pd.concat([df_upload.reset_index(drop=True), results_df], axis=1)

                    # Add overall toxic flag using the dynamic threshold
                    output_df['is_toxic'] = (results_df.max(axis=1) >= threshold).astype(int)

                st.success(f"✅ Analysis complete!")

                # Summary metrics
                toxic_count = output_df['is_toxic'].sum()
                clean_count = len(output_df) - toxic_count
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Comments", len(output_df))
                with col2:
                    st.metric("🚨 Toxic", int(toxic_count))
                with col3:
                    st.metric("✅ Clean", int(clean_count))

                # Show results
                st.dataframe(output_df, use_container_width=True)

                # Download button
                csv = output_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="toxicity_predictions.csv",
                    mime="text/csv"
                )

# Tab 3: Dashboard
with tab3:
    # WHY WE BUILT THIS TAB:
    # A model without metrics is a black box. This dashboard serves as the translation layer
    # between Data Scientists and Business Stakeholders. It visualizes the AUC-ROC scores, training loss,
    # and original data distributions so the tutor can instantly verify the model's actual mathematical footprint.
    st.subheader("Data Insights & Model Performance")

    # --- Model Metrics ---
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
                        <div class="metric-card">
                            <h3>{LABEL_EMOJIS.get(label, '📊')} {label.replace('_', ' ').title()}</h3>
                            <h2>{score:.3f}</h2>
                        </div>
                        """, unsafe_allow_html=True)

            overall = auc_scores.get('overall_macro', 0)
            if overall:
                st.markdown(f"**Overall Macro AUC-ROC: `{overall:.4f}`**")

        # Training history
        history = metrics.get('training_history', {})
        if history:
            st.markdown("#### 📉 Training History")
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(history['loss'], label='Train Loss', color='#6c5ce7', linewidth=2)
                ax.plot(history['val_loss'], label='Val Loss', color='#e17055', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training vs Validation Loss')
                ax.legend()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(history['accuracy'], label='Train Acc', color='#00b894', linewidth=2)
                ax.plot(history['val_accuracy'], label='Val Acc', color='#0984e3', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.set_title('Training vs Validation Accuracy')
                ax.legend()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)

        # Classification report
        report = metrics.get('classification_report', {})
        if report:
            st.markdown("#### 📋 Classification Report")
            report_data = {
                label: {
                    'Precision': report[label]['precision'],
                    'Recall': report[label]['recall'],
                    'F1-Score': report[label]['f1-score'],
                    'Support': report[label]['support']
                }
                for label in LABELS if label in report
            }
            if report_data:
                st.dataframe(pd.DataFrame(report_data).T, use_container_width=True)

    # --- EDA Section ---
    df_eda = load_training_data()
    if df_eda is not None:
        st.markdown("---")
        st.markdown("#### 📊 Dataset Insights")

        col1, col2 = st.columns(2)

        with col1:
            # Class distribution
            fig, ax = plt.subplots(figsize=(6, 4))
            label_counts = df_eda[LABELS].sum().sort_values(ascending=True)
            colors = sns.color_palette("viridis", len(LABELS))
            bars = ax.barh(label_counts.index, label_counts.values, color=colors, height=0.6)
            ax.set_xlabel("Count")
            ax.set_title("Toxicity Label Distribution")
            for bar, count in zip(bars, label_counts.values):
                ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                        f'{count:,}', va='center', fontweight='bold', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            # Multi-label overlap
            fig, ax = plt.subplots(figsize=(6, 4))
            corr = df_eda[LABELS].corr()
            sns.heatmap(corr, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f',
                       ax=ax, square=True, linewidths=0.5)
            ax.set_title("Label Correlation Heatmap")
            plt.tight_layout()
            st.pyplot(fig)

        # Dataset stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Comments", f"{len(df_eda):,}")
        with col2:
            st.metric("Toxic Comments", f"{df_eda['toxic'].sum():,}")
        with col3:
            st.metric("Toxicity Rate", f"{df_eda['toxic'].mean():.1%}")
        with col4:
            multi_label = (df_eda[LABELS].sum(axis=1) > 1).sum()
            st.metric("Multi-label", f"{multi_label:,}")

    elif not metrics:
        st.info("📌 Train the model first by running `python train_model.py` to see the dashboard.")

