# dataset-size-impact-on-ml-classifiers

A comparative study on how training dataset size affects the performance of four Machine Learning classifiers applied to a 7-class obesity classification task.

---

## Goal

Evaluate how progressively reducing the training dataset (20%, 40%, 60%, 80%, and 100%) impacts **accuracy**, **training time**, and **memory usage** across four models:

- Support Vector Machine (SVM)
- Random Forest (RF)
- XGBoost
- Deep Neural Network (DNN)

---

## Dataset

- **Source:** [Obesity or CVD risk (Kaggle)](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster)
- **Target:** `NObeyesdad` — 7 classes: `Insufficient_Weight`, `Normal_Weight`, `Overweight_Level_I`, `Overweight_Level_II`, `Obesity_Type_I`, `Obesity_Type_II`, `Obesity_Type_III`
- `Height` and `Weight` removed to prevent data leakage
- Custom ordinal encoding for categorical features (`CAEC`, `CALC`, binary columns)
- Numerical features rounded to remove synthetic noise
- Stratified train/test split (70/30)

---

## Project Structure

```
├── ObesityDataSet_raw_and_data_sinthetic.csv   # Raw dataset
├── Dataset_Cleaning.py                          # Preprocessing pipeline → outputs df_100.csv
├── Models_functions.py                          # Model definitions and plotting utilities
├── Training_Models.py                           # Main training script across all dataset subsets
└── df_100.csv                                   # Preprocessed dataset (output of cleaning step)
```

---

## Models

| Model | Library | Configuration |
|-------|---------|---------------|
| SVM | scikit-learn | RBF kernel, C=1.0, gamma=scale |
| Random Forest | scikit-learn | 100 estimators, random_state=42 |
| XGBoost | xgboost | 100 estimators, mlogloss |
| DNN | TensorFlow/Keras | 128→64→32→7, Dropout 0.3, 50 epochs |

---

## Metrics

- **Accuracy** per dataset size
- **Training time** (seconds)
- **Training memory** (MB, via `memory_profiler` with delta-based calculation)
- **Confusion matrix** and classification report per model × subset

---

## Requirements

```bash
pip install scikit-learn xgboost tensorflow memory-profiler matplotlib seaborn pandas
```

---

## Usage

**Step 1 — Preprocess the dataset:**
```bash
python Dataset_Cleaning.py
```
This reads `ObesityDataSet_raw_and_data_sinthetic.csv` and outputs `df_100.csv`.

**Step 2 — Train and evaluate all models:**
```bash
python Training_Models.py
```

> Update the file paths in both scripts to match your local setup before running.
