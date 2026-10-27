# diabetes-readmission-prediction
A machine learning capstone project predicting 30-day hospital readmission rates for diabetic patients using the Diabetes 130-US Hospitals dataset.

## Project Overview
This repository contains my data science capstone project focused on predicting patient readmissions. Hospital readmissions are a high-stakes challenge in healthcare, impacting both patient outcomes and hospital penalties under the Affordable Care Act. 

The primary objective of this project is to build an end-to-end Machine Learning pipeline that predicts whether a diabetic patient will be **readmitted to the hospital within 30 days** of their discharge.

## About the Dataset
The project utilizes the **Diabetes 130-US Hospitals Dataset (1999–2008)** from the UCI Machine Learning Repository. It comprises over 100,000 hospital admissions and 47 features, including:
* **Patient Demographics:** Age, gender, and race.
* **Admission/Discharge Details:** Admission source, time spent in the hospital, and medical specialty of the admitting doctor.
* **Clinical Metrics:** Number of lab procedures, medications administered, and primary/secondary/tertiary diagnoses (ICD-9 codes).
* **Medication Changes:** Changes in dosage for 23 specific diabetes medications (e.g., insulin, metformin) and HbA1c test results.

## Key Challenges Addressed
* **Handling Clinical Missing Data:** Structuring systemic missingness in features like `medical_specialty`, `payer_code`, `max_glu_serum`, and `A1Cresult` based on domain knowledge rather than simple imputation.
* **High-Cardinality Features:** Encoding and clustering extensive categorical identifiers like medical specialties and ICD-9 diagnosis codes.
* **Class Imbalance:** Managing the target variable balance to ensure the model accurately captures the patients at highest risk of returning within 30 days.

## Target Variable
The prediction task is framed as a classification problem using the modified readmission status:
* `1`: Patient was readmitted within 30 days.
* `0`: Patient was not readmitted, or readmitted after 30 days.

## Repository Structure
```
├── data/
│   ├── raw/            # Original CSVs from UCI (git-ignored, never edit these)
│   └── processed/      # Train/test Parquet files from notebook 02
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_imputation.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_pca.ipynb
│   ├── 06_model_selection.ipynb
│   └── 07_model_evaluation.ipynb
├── models/             # Final model from notebook 06 (git-ignored)
├── reports/            # Results JSON per stage
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # load_raw_data(), load_id_mappings()
│   └── preprocessing.py    # encoding functions and build_preprocessor()
├── requirements.txt
├── .gitignore
└── README.md
```

## Results
| Model | CV PR-AUC | Test PR-AUC |
|---|---|---|
| Random model | 0.114 | 0.114 |
| Logistic Regression (baseline) | 0.213 | - |
| PCA + Logistic Regression | 0.211 | - |
| Random Forest (tuned, final) | 0.226 | 0.234 |

## Setup
1. Download the dataset from [UCI](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) and put `diabetic_data.csv` and `IDS_mapping.csv` in `data/raw/`.
2. Create an environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the notebooks in order from `01_eda.ipynb` to `07_model_evaluation.ipynb` with the `.venv` kernel.
