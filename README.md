# Road Accidents France — Data Science Project

End-to-end data science project on French road accident data (BAAC dataset, source: [data.gouv.fr](https://www.data.gouv.fr)).

The goal is to predict injury severity from accident features, using temporal validation: models are trained on 2023 data and tested on 2024 data — simulating a real production setting.

## Structure

| Notebook | Description |
|---|---|
| `cleaning.ipynb` | Raw data ingestion, merging of 4 government tables, missing value handling, feature engineering |
| `eda.ipynb` | Exploratory analysis — peak hours, seasonality, severity by road type, vehicle, age group, geographic heatmap, chi-squared tests |
| `ml.ipynb` | Model comparison (Logistic Regression, Random Forest with hyperparameter tuning, PyTorch neural network), feature importance, binary classifier on fatalities |


## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Dummy Classifier (baseline) | 0.41 | 0.15 |
| Logistic Regression | 0.39 | 0.31 |
| Random Forest (default) | 0.65 | 0.47 |
| Random Forest (tuned) | 0.65 | **0.50** |
| Neural Network (PyTorch) | 0.57 | 0.47 |

## Key findings

- Random Forest and Neural Network perform comparably (Macro F1 ~0.47-0.50), both significantly outperforming the linear baseline
- Hyperparameter tuning yields a meaningful gain on the "Killed" class specifically (F1 from 0.06 to 0.15)
- Recall on fatalities remains near zero across all models — even in a dedicated binary classifier (recall: 0.01)
- This is a **data limitation**, not a modeling failure: the most discriminating factors for road fatalities (blood alcohol level, actual speed, driver fatigue) are absent from the public dataset
- Chi-squared tests confirm statistically significant associations between severity and variables like road type, vehicle type, and age — but the signal is too weak for reliable prediction

## Technical stack

- **Python** — Pandas, NumPy, Scikit-learn, PyTorch, Plotly, SciPy
- **ML** — Logistic Regression, Random Forest, RandomizedSearchCV, feed-forward neural network with early stopping
- **Visualization** — interactive charts (Plotly), density heatmap (Mapbox)

## What I learned

Beyond the technical implementation, this project taught me that understanding the *limits* of a model matters as much as optimizing it. The binary classifier experiment was deliberately designed to rule out modeling issues and isolate the data as the root cause — a reasoning process I found as interesting as the modeling itself.
