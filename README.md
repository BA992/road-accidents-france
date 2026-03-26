# Road Accidents France - EDA & ML

Analysis of French road accident data from the ONISR (data.gouv.fr).

## Overview
End-to-end data science project on governmental road accident data:
- **Data Cleaning**: merging and cleaning 4 raw tables (~120,000 rows)
- **EDA**: exploratory analysis with bias identification and interactive visualizations
- **ML**: basic predictive modeling of injury severity with temporal validation (2023→2024)

## Stack
Python, Pandas, Scikit-learn, Plotly Express, PyTorch

## Data Source
[data.gouv.fr - Bases de données annuelles des accidents corporels](https://www.data.gouv.fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024)

## Results

All models trained on 2023 data, evaluated on 2024 data (temporal validation).

| Model               | Accuracy | Macro F1 |
|---------------------|----------|----------|
| Dummy Classifier    | 0.41     | 0.15     |
| Logistic Regression | 0.39     | 0.31     |
| Neural Network      | 0.64     | 0.45     |
| Random Forest       | 0.65     | 0.47     |

Random Forest and Neural Network perform comparably — consistent with the literature 
showing tree-based models and neural networks are competitive on tabular data.

The key finding of this project is the consistently low recall on fatalities across 
all models, including a dedicated binary classifier (Killed vs. Rest). This is not 
a modeling failure — it reflects a fundamental data limitation: the most predictive 
factors for road fatalities (blood alcohol level, actual speed, driver fatigue) are 
absent from the public BAAC dataset. No model tuning can compensate for missing data.
