# Road Accidents France - EDA & ML

Analysis of French road accident data from the ONISR (data.gouv.fr).

## Overview
End-to-end data science project on governmental road accident data:
- **Data Cleaning**: merging and cleaning 4 raw tables (~120,000 rows)
- **EDA**: exploratory analysis with bias identification and interactive visualizations
- **ML**: predictive modeling of injury severity with temporal validation (2023→2024)

## Stack
Python, Pandas, Scikit-learn, Plotly Express

## Data Source
[data.gouv.fr - Bases de données annuelles des accidents corporels](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-2005-a-2022/)

## Results
Random Forest trained on 2023 data, tested on 2024 data.
- Accuracy: 65% vs 41% for a naive baseline
- Low recall on fatalities explained by absence of key features 
  (blood alcohol, actual speed) in public data
