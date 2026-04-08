# Multi-Modal Prediction of Major Depressive Disorder

This project explores the use of wearable sensor data and machine learning to predict **Major Depressive Disorder (MDD)** and related affective states.

## Overview

We leverage multi-modal physiological signals (e.g., BVP, EDA, temperature) from wearable devices such as the Empatica E4 to model emotional states like **baseline, stress, and amusement**.

## Approach

* Preprocessing and synchronization of multi-frequency sensor data
* Feature extraction from physiological signals
* Models used:

  * Decision Tree
  * XGBoost (with Optuna hyperparameter tuning)
  * LSTM (for temporal modeling)
* Evaluation via **Leave-One-Subject-Out Cross-Validation (LOSOCV)**

## Key Results

* Binary classification (baseline vs stress/amusement) significantly improved performance
* Best accuracy: **~75% (LSTM, chest data)**
* Dataset reduction (25%) maintained performance while reducing training time
* Late fusion of wrist + chest data improved robustness

## Insights

* Physiological signals outperform motion-based features
* Models struggle to distinguish **stress vs amusement**
* HRV is the most informative but highly variable feature
