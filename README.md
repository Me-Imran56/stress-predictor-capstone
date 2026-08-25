# Daily Stress Level Predictor — Neurofive ML Track Capstone

## Problem Statement

Chronic stress is a major driver of burnout and long-term health problems, but most people
never get a formal stress assessment until something has already gone wrong. Meanwhile,
wearables and fitness apps already collect exactly the kind of data that correlates with
stress — sleep duration, sleep quality, resting heart rate, daily activity — but rarely use
it to flag stress risk directly.

**Can a model predict a person's stress level (1–10) from everyday, easily-trackable
lifestyle habits alone — without a clinical questionnaire?**

## Dataset

[Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
— 374 individuals, 13 columns: sleep duration/quality, physical activity, occupation, BMI
category, blood pressure, heart rate, daily steps, sleep disorder, and a self-reported
Stress Level (1–10) target. **Note: this dataset is synthetic** (documented as such by its
source) — see the "Honest Limitations" section below for what that means for these results.

## Approach

1. **Clean:** fixed inconsistent BMI labels ("Normal" vs "Normal Weight"), treated missing
   `Sleep Disorder` values as "None" (not actually missing), and split the `Blood Pressure`
   string column into numeric systolic/diastolic columns.
2. **EDA:** found Sleep Quality and Sleep Duration both correlate strongly (negatively) with
   Stress Level; occupation and BMI category also visibly shift the stress distribution.
3. **Feature engineering:** added `Sleep_Efficiency` (quality ÷ duration), `Activity_per_1000_Steps`
   (activity intensity relative to daily movement), and `Pulse_Pressure` (systolic − diastolic,
   a simple cardiovascular strain proxy).
4. **Modeling:** built a single `Pipeline` (`ColumnTransformer` with `StandardScaler` +
   `OneHotEncoder`) and trained three regressors — Linear Regression, Random Forest, XGBoost.
5. **Evaluation:** compared RMSE, MAE, and R² on a held-out test set; picked the best by R².
6. **Deployment:** saved the winning pipeline with `joblib` and built a Streamlit app so
   anyone can enter their own habits and get a live stress estimate.

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 0.3403 | 0.2021 | 0.9629 |
| Random Forest | 0.1388 | 0.0431 | 0.9938 |
| **XGBoost (best)** | **0.0472** | **0.0104** | **0.9993** |

**Top features (XGBoost):** `Sleep_Efficiency` (by far the strongest), `Quality of Sleep`,
`Sleep Duration`, `Gender`, `Heart Rate`.

## Honest Limitations

R² = 0.9993 is unusually high, and it's important not to overclaim it. This dataset is
explicitly synthetic, and the dominant feature (`Sleep_Efficiency`) is derived directly from
two of the raw inputs — this suggests the data was generated with a fairly direct mathematical
relationship a tree-based model can learn almost exactly. **These numbers would not hold on
real human data**, where stress is shaped by far more than sleep/activity metrics (work,
relationships, finances, mental health history). A production version trained on real
wearable data should expect a much more modest R² (realistically 0.4–0.7), and the value would
come from directionally flagging elevated risk, not precise scores. The pipeline and app
architecture here carry over directly to real data — only the performance ceiling would change.

## Case Study: Real-World Value

A wellness or fitness app could plug a model like this directly into data it already collects,
surfacing a simple "your stress signals are trending up this week" nudge without requiring a
survey — useful for the user (early self-awareness) and valuable for the business (an
engagement-driving feature that differentiates a health app). The same clean → engineer
features → compare models → deploy pipeline is directly reusable for adjacent problems
companies already pay for: burnout risk scoring for HR platforms, risk tiers for health
insurers, or engagement dashboards for corporate wellness programs.

## Live App

**[[Live Demo ->]](https://daily-stress-predictor56.streamlit.app/)**

## How to Run

### Notebook
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
jupyter notebook stress_prediction.ipynb
```

### Streamlit app (locally)
```bash
pip install -r requirements.txt
streamlit run app.py
```



## Files

| File | Purpose |
|---|---|
| `stress_prediction.ipynb` | Full workflow: problem definition → clean → EDA → feature engineering → train → evaluate → save |
| `app.py` | Streamlit web app for live predictions |
| `stress_pipeline.joblib` | Saved best model (XGBoost, wrapped in a preprocessing pipeline) |
| `sleep_data.csv` | Dataset |
| `requirements.txt` | Dependencies for the app |
| `*.png` | Charts generated in the notebook (EDA, correlation heatmap, feature importance, predicted vs. actual) |
