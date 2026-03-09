# Impact Coefficient Predictor

> **Note:** The trained LightGBM model file is not included in this repository.

A LightGBM-based desktop tool for single-case and batch prediction of runway-bridge impact coefficients.

## Features

- Single-case prediction
- Batch CSV prediction
- Output of predicted impact coefficient and impact grade
- Desktop graphical user interface based on PySide6

## Requirements

- Python 3.11
- PySide6
- pandas
- joblib
- lightgbm
- scikit-learn

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run

Run the application with:

```bash
python impact_coefficient_predictor_lightgbm.py
```

## Model file

The trained LightGBM model is not distributed in this repository.

Users should prepare the model file separately and place it at the following path before running the application:

```text
C:\MODEL\BO-LGB.joblib
```

If needed, the default model path can also be modified in the source code.

## Input features

The required input variables are:

- m
- theta
- phi
- Vx
- Vz
- xi
- IRI
- Hp
- L

## Batch prediction

The batch mode requires a CSV file containing the 9 input columns listed above.

The output CSV will include:

- I_pred
- Grade

A sample input template is provided in:

```text
template.csv
```

## Impact-grade thresholds

- I < 0.3: Low-impact
- 0.3 ≤ I ≤ 0.6: Medium-impact
- I > 0.6: High-impact

## Notes

This repository provides the source code, dependency list, and input template for the desktop predictor.

The trained model file is intentionally excluded from the repository. Users who need to run the application should prepare the corresponding LightGBM model file separately.
