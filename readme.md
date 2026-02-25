# Churn Prediction & Data Drift

A machine learning project that predicts customer churn for a service-based company (telecom / SaaS / subscription). The main challenge is that customer behavior changes over time — so the pipeline is built to handle **data drift** and avoid temporal leakage.

---

## What this project does

- Detects drift in the data over time (target, features, and missing values)
- Builds a clean ML pipeline that trains on past data and tests on future data
- Compares Logistic Regression vs Random Forest
- Finds the best retraining window automatically using backtesting
- Optimizes the decision threshold based on real business costs

---

## How to run it

1. Download or clone this repo
2. Open `churn_prediction.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com)
3. Make sure `customer_churn.csv` is in the same folder
4. Run all cells — everything is generated automatically

**Requirements:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Results

| Model | AUC | Accuracy |
|-------|-----|----------|
| Logistic Regression | ~0.87 | ~0.79 |
| Random Forest | ~0.86 | ~0.78 |

Best retraining window: **3 periods** — recent data matters more than older history when drift is present.
