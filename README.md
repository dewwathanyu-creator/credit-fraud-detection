# Credit Card Fraud Detection

A machine learning pipeline that identifies fraudulent financial transactions from a dataset of **6.3 million records**.  
Built with scikit-learn and deployed as an interactive web app using Streamlit.

---

## Project Overview

| | |
|---|---|
| **Dataset** | Simulated financial transactions (PaySim) |
| **Records** | 6,362,620 transactions |
| **Target** | `isFraud` — binary classification |
| **Challenge** | Severe class imbalance (fraud = 0.13% of data) |
| **Final Model** | Random Forest |

---

## Workflow

```
Data Loading  ->  EDA  ->  Feature Engineering  ->  Model Training  ->  Evaluation  ->  Web App
```

### Key Findings from EDA
- Fraud occurs **only** in `TRANSFER` and `CASH_OUT` transaction types
- In nearly all fraud cases, the sender's balance is completely drained to zero
- No single raw feature is strongly correlated with fraud — tree-based models handle this better than linear ones

### Feature Engineering
Three new features were created to capture fraud signals:

| Feature | Description |
|---|---|
| `balanceDiffOrig` | Money that left the sender's account |
| `balanceDiffDest` | Money that arrived at the receiver's account |
| `isAccountDrained` | Flag: 1 if sender balance was fully emptied to zero |

---

## Model Comparison

Both models use the same preprocessing pipeline (StandardScaler + OneHotEncoder) and handle class imbalance via `class_weight="balanced"`.

| Model | Recall (Fraud) | Precision (Fraud) | F1 (Fraud) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.94 | 0.02 | 0.04 | ~0.97 | ~0.30 |
| **Random Forest** | **0.92** | **0.77** | **0.84** | **0.9976** | **0.9445** |

> **Why Recall matters here:** Missing a real fraud (False Negative) is far more costly than flagging a legitimate transaction for review (False Positive).  
> Random Forest was selected — it achieves a much better Precision-Recall balance and leverages the engineered features more effectively.

---

## Project Structure

```
Credit Fraud/
├── Analysis.ipynb        # Full EDA + model training notebook
├── fraud_detection.py    # Streamlit web app
├── train_model.py        # Script to retrain and save the model
├── requirements.txt      # Python dependencies
└── README.md
```

> Note: `AIML Dataset.csv` and `fraud_detection_pipeline.pkl` are excluded from this repo via `.gitignore` due to file size.  
> Run `python train_model.py` after downloading the dataset to regenerate the model.

---

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (requires AIML Dataset.csv in the same folder)
python train_model.py

# 3. Launch the web app
streamlit run fraud_detection.py
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-web_app-red?logo=streamlit&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data-150458?logo=pandas&logoColor=white)
