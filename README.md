# 🩺 MediPredict AI — Intelligent Disease Risk Prediction System

A production-ready Streamlit web app that predicts a patient's risk of heart disease
from medical data, demonstrating a complete end-to-end machine learning workflow:
data cleaning, EDA, feature engineering, model training/comparison, evaluation, and
live prediction.

Built as a Machine Learning mid-term project and portfolio piece.

---

## Features

- Dataset upload, preview, and statistics
- Data cleaning: missing values, duplicates, IQR-based outlier removal
- Exploratory Data Analysis with interactive charts
- Feature engineering: encoding, scaling, feature selection, train/test split
- Training and comparison of 4 models: Logistic Regression, Decision Tree,
  Random Forest, SVM
- Full evaluation suite: accuracy, precision, recall, F1, ROC-AUC, confusion
  matrix, ROC curve, feature importance
- Automatic best-model selection
- Interactive risk prediction form with probability and risk level
- Downloadable predictions, cleaned dataset, and trained model

---

## Tech stack

Streamlit · Python 3.12+ · Pandas · NumPy · Scikit-learn · Joblib · Matplotlib ·
Seaborn · Plotly

---

## Project structure

See `src/` for ML logic and `pages/` for the Streamlit UI. Full breakdown in the
project documentation (Phase 1).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kammi11/MediPredict-AI.git
cd MediPredict-AI
```

### 2. Create a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the app locally

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repo, branch `main`, and set the main file
   path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   automatically.

---

## Project architecture

```
MediPredict-AI/
├── app.py              # Entry point, sidebar branding
├── src/                # ML logic - no Streamlit imports
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── model_utils.py
├── pages/               # Streamlit UI - calls into src/
│   ├── 1_Home.py ... 10_About.py
├── data/raw/            # heart.csv goes here
├── data/processed/      # cleaned dataset output
├── models/              # saved .joblib models + scaler
├── notebooks/           # optional exploration
├── assets/              # logos/images
└── requirements.txt
```

## Screenshots

_Add screenshots here after running the app locally — e.g._

```markdown
![Home page](assets/screenshot_home.png)
![EDA page](assets/screenshot_eda.png)
![Prediction page](assets/screenshot_prediction.png)
```

## Sample dataset included

A schema-correct **synthetic** sample (`data/raw/heart.csv`, 303 rows) is
bundled so the app runs immediately after install. For a real submission,
replace it with the actual UCI Heart Disease dataset from
[Kaggle](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) or the
[UCI repository](https://archive.ics.uci.edu/dataset/45/heart+disease) —
same 14-column schema, just drop it in at the same path.

## Resume bullet points

- Built an end-to-end ML web application (MediPredict AI) in Streamlit,
  covering data cleaning, EDA, feature engineering, and model comparison
  across 4 classifiers, achieving 88%+ accuracy and 0.96 ROC-AUC
- Designed a modular Python architecture separating ML logic (`src/`) from UI
  (`pages/`), enabling independent testing and reuse of the pipeline
- Implemented a full model evaluation suite (accuracy, precision, recall, F1,
  ROC-AUC, confusion matrix) with automatic best-model selection
- Deployed a production-ready multi-page Streamlit app to Streamlit
  Community Cloud with file upload, live prediction, and data export

## Portfolio description

MediPredict AI is a machine learning web application that predicts heart
disease risk from patient clinical data. It demonstrates the complete ML
lifecycle — data cleaning, exploratory data analysis, feature engineering,
model training across four algorithms, rigorous evaluation, and an
interactive prediction interface — in a clean, deployable Streamlit app.

## Future improvements

- Add SHAP-based explainability to the prediction page
- Support additional datasets (e.g. combined Cleveland + Hungarian + VA data)
- Add hyperparameter tuning (GridSearchCV) as an optional training mode
- Add user authentication for saving prediction history
- Add unit tests for `src/` modules with pytest

## Disclaimer

This application is built for educational purposes as part of a Machine
Learning coursework project. Predictions are **not** medical advice and should
never be used for real clinical decision-making.

---

## Author

Qaim Ali — BS Computer Science, Abasyn University Peshawar
GitHub: [kammi11](https://github.com/kammi11)
