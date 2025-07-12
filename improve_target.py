import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/final_feature_dataset.csv')

print("=== CURRENT TARGET ANALYSIS ===")
print(f"Mean return: {df['target_1d'].mean():.6f}")
print(f"Std return: {df['target_1d'].std():.6f}")
print(f"Binary distribution: {df['target_1d_binary'].value_counts(normalize=True).to_dict()}")

# Create better target variables
print("\n=== IMPROVED TARGET VARIABLES ===")

# 1. Predict significant moves (>1% return)
df['target_significant'] = (df['target_1d'] > 0.01).astype(int)
print(f"Significant moves (>1%): {df['target_significant'].value_counts(normalize=True).to_dict()}")

# 2. Predict top/bottom quartile returns
return_75 = df['target_1d'].quantile(0.75)
return_25 = df['target_1d'].quantile(0.25)
df['target_top_quartile'] = (df['target_1d'] > return_75).astype(int)
df['target_bottom_quartile'] = (df['target_1d'] < return_25).astype(int)
print(f"Top quartile moves: {df['target_top_quartile'].value_counts(normalize=True).to_dict()}")
print(f"Bottom quartile moves: {df['target_bottom_quartile'].value_counts(normalize=True).to_dict()}")

# 3. Predict vs market (relative performance)
df['market_return'] = df.groupby('date')['target_1d'].transform('mean')
df['target_vs_market'] = (df['target_1d'] > df['market_return']).astype(int)
print(f"Beat market: {df['target_vs_market'].value_counts(normalize=True).to_dict()}")

# 4. Volatility-adjusted returns
df['volatility_5d'] = df.groupby('ticker')['target_1d'].rolling(5).std().reset_index(0, drop=True)
df['target_vol_adj'] = (df['target_1d'] / df['volatility_5d'] > 0.5).astype(int)
print(f"High vol-adjusted returns: {df['target_vol_adj'].value_counts(normalize=True).to_dict()}")

# Save improved dataset
df.to_csv('data/improved_feature_dataset.csv', index=False)
print("\n✅ Improved dataset saved as 'data/improved_feature_dataset.csv'")

# Feature engineering suggestions
print("\n=== FEATURE ENGINEERING SUGGESTIONS ===")
print("1. Add lagged features (1-day, 3-day, 5-day)")
print("2. Add cross-stock features (market sentiment)")
print("3. Add volatility regime features")
print("4. Add momentum features (price momentum, volume momentum)")
print("5. Add mean reversion features (RSI extremes, price vs MA)")
print("6. Add sentiment momentum (change in sentiment over time)")
print("7. Add market regime features (bull/bear market indicators)") 