# Market Sentiment Strategy

A machine learning project to predict stock price movements by combining financial news sentiment analysis with historical price data.

## Features
- Integrates sentiment analysis from news headlines with technical indicators
- Implements multiple ML models (Gradient Boosting, Random Forest, Logistic Regression)
- Automated data processing and feature engineering
- Model evaluation and performance visualization

## Requirements
- Python 3.x
- scikit-learn
- pandas
- matplotlib
- seaborn

## Usage
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main scripts in `notebooks/` or `src/` to process data and train models

## Data
- Financial news and stock price CSVs are located in the `data/` folder

## Outputs
- Model predictions and performance summaries are saved in the `outputs/` folder

## Models
- Trained machine learning models and related files are saved in the `models/` folder, including:
  - `all_models.pkl`: Serialized collection of all trained models
  - `feature_names.pkl`: List of feature names used for training
  - `feature_scaler.pkl`: Scaler object for feature normalization
  - `gradient_boosting_model.pkl`: Trained Gradient Boosting model

## Source Code
- The main source code for data processing, feature engineering, and model training is located in the `src/` directory.
 
