import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Paths
DATA_DIR = '../data' if not os.path.exists('data') else 'data'
MODELS_DIR = '../models' if not os.path.exists('models') else 'models'
OUTPUTS_DIR = '../outputs' if not os.path.exists('outputs') else 'outputs'

# Load data
price_data = pd.read_csv(os.path.join(DATA_DIR, 'final_feature_dataset.csv'))

# Assume the target column is named 'target' and features are all other columns except 'date' and 'ticker'
feature_cols = [col for col in price_data.columns if col not in ['date', 'ticker', 'target']]
X = price_data[feature_cols]
y = price_data['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    predictions[name] = y_pred
    # Save each model
    joblib.dump(model, os.path.join(MODELS_DIR, f'{name.lower()}_model.pkl'))

# Save scaler and feature names
joblib.dump(scaler, os.path.join(MODELS_DIR, 'feature_scaler.pkl'))
pd.Series(feature_cols).to_pickle(os.path.join(MODELS_DIR, 'feature_names.pkl'))

# Save all models in one file
joblib.dump(models, os.path.join(MODELS_DIR, 'all_models.pkl'))

# Output predictions
if not isinstance(X_test, pd.DataFrame):
    output_df = pd.DataFrame(X_test)
else:
    output_df = X_test.copy()
output_df['actual'] = pd.Series(y_test).reset_index(drop=True)
for name in models:
    output_df[f'pred_{name.lower()}'] = pd.Series(predictions[name]).reset_index(drop=True)
output_df.to_csv(os.path.join(OUTPUTS_DIR, 'model_predictions.csv'), index=False)

# Print model performance
print('Model Accuracy Scores:')
for name, acc in results.items():
    print(f'{name}: {acc:.4f}') 