# Churn Prediction & Data Drift — Offline MLOps

Final project for the Data Science / MLOps course. Builds an end-to-end churn prediction pipeline for a service-based company, with a focus on data drift, temporal evaluation, and business-oriented optimization.

---

## Project Structure

```
churn-prediction-mlops/
│
├── churn_prediction.ipynb        # Main notebook — all 8 steps
├── customer_churn.csv            # Dataset (5,000 rows, 24 periods)
├── Churn_Prediction_Report.docx  # Full project report (5 pages)
│
└── figures/
    ├── fig1_churn_rate_per_period.png
    ├── fig2_covariate_drift.png
    ├── fig3_early_vs_recent.png
    ├── fig4_missing_mnar.png
    ├── fig5_lr_performance.png
    ├── fig6_fixed_vs_rolling.png
    ├── fig7_window_selection.png
    ├── fig8_rf_vs_lr.png
    ├── fig9_feature_importance.png
    └── fig10_business_optimization.png
```

---

## Steps Covered

| Step | Description |
|------|-------------|
| 1 | Data loading & quality checks |
| 2 | Drift-oriented EDA (target, covariate, MNAR) |
| 3 | Leakage-free preprocessing pipeline |
| 4 | Logistic Regression baseline |
| 5 | Fixed vs rolling retraining policies |
| 6 | Automatic window selection via backtesting |
| 7 | Random Forest comparison |
| 8 | Business cost optimization & threshold tuning |

---

## How to Run

1. Clone the repo
2. Make sure `customer_churn.csv` is in the same folder as the notebook
3. Open `churn_prediction.ipynb` in Jupyter or Google Colab
4. Run all cells — figures are generated automatically

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Results Summary

| Model | Mean AUC | Mean Accuracy |
|-------|----------|---------------|
| Logistic Regression | ~0.87 | ~0.79 |
| Random Forest | ~0.86 | ~0.78 |

- Best training window: **3 periods** (identified via backtesting)
- Optimal decision threshold: **< 0.50** (cost-sensitive, FN = €200, FP = €20)
