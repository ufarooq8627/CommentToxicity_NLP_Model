# 🛡️ Comment Toxicity Detection with Deep Learning

**🔴 Live Demo:** [https://commenttoxicitynlpmodel.streamlit.app/](https://commenttoxicitynlpmodel.streamlit.app/)

A real-time comment toxicity detection system powered by a **Bidirectional LSTM** deep learning model, deployed as an interactive **Streamlit** web application.

## 📋 Project Overview

Online communities face challenges from toxic comments including harassment, hate speech, and offensive language. This project builds an automated system that analyzes text input and predicts toxicity across 6 categories:

| Label | Description |
|-------|-------------|
| **Toxic** | Generally toxic or rude |
| **Severe Toxic** | Extremely toxic content |
| **Obscene** | Obscene language |
| **Threat** | Threatening language |
| **Insult** | Insulting language |
| **Identity Hate** | Hate based on identity |

## 🏗️ Architecture

```
Input Text → Clean Text → Tokenize → Pad Sequences
    → Embedding (128d)
    → SpatialDropout1D (0.3)
    → Bidirectional LSTM (64 units)
    → GlobalMaxPooling1D
    → Dense (64, ReLU) → Dropout (0.3)
    → Dense (6, Sigmoid) → Multi-label Predictions
```

## 🚀 Setup & Installation

### 1. Clone / Download the project

```bash
cd CommentToxicity
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download `train.csv` and `test.csv`. However you can find the dataset here -[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and place them in the project root.

### 4. Train the model

```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Train a BiLSTM model (~10 epochs with early stopping)
- Save model artifacts to the `model/` directory

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 📁 Project Structure

```
CommentToxicity/
├── train.csv                 # Training dataset
├── test.csv                  # Test dataset
├── app.py                    # Streamlit web application
├── comment_toxicity.ipynb    # Jupyter notebook (EDA & experiments)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── model/                    # Generated after training
    ├── toxicity_model.keras  # Trained Keras model
    ├── tokenizer.pickle      # Fitted tokenizer
    └── metrics.json          # Evaluation metrics
```

## 🌐 Streamlit App Features

1. **🔍 Single Prediction** — Enter any comment and get real-time toxicity scores with visual breakdown
2. **📁 Bulk Prediction** — Upload a CSV file with comments, get predictions for all rows, and download results
3. **📊 Dashboard** — View data insights (class distribution, correlation heatmap), model performance (AUC-ROC, classification report), and training history charts

## 📊 Model Performance

The model is evaluated using:
- **Classification Report** (Precision, Recall, F1-Score per label)
- **AUC-ROC Score** (per label and macro average)

Results are saved in `model/metrics.json` and displayed in the Streamlit dashboard.

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — Deep learning model
- **Streamlit** — Web application
- **scikit-learn** — Evaluation metrics
- **imbalanced-learn** — SMOTE for class balancing
- **pandas, numpy** — Data processing
- **matplotlib, seaborn** — Visualizations

## 📝 License

This project is for educational purposes as part of the GUVI Deep Learning curriculum.
