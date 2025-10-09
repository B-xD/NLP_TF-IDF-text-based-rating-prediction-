# NLP Project – Product Rating Prediction (Belton Manhica)

## Overview
This repository contains the **H1 NLP project** where we predict product ratings based on customer comments using a **scikit-learn pipeline** that integrates:
- TF-IDF vectorization for text features,  
- Linear/Lasso regression for modeling,  
- GridSearchCV for hyperparameter tuning, and  
- PCA for dimensionality reduction.

---

## Repository Contents
- `NLP_TF_IDF_text_based_rating_pred` — Exported notebook with results and explanations.  
- (Optional) `NLP_TF_IDF_text_based_rating_pred.ipynb` — Source Jupyter notebook (if added).  
- `0_Belton_Manhica.csv` — Example dataset name format for your data.  
- `1_Belton_Manhica.csv` — Predicted output file format (contains `predict_rating`).

---

## Objectives
1. Explore and clean customer comment data.  
2. Apply TF-IDF to convert text into numerical features.  
3. Use PCA for feature dimensionality reduction.  
4. Train and optimize a Linear/Lasso regression model using GridSearchCV.  
5. Evaluate results and export predictions.  

---

## Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
