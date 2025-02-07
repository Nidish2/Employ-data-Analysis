import pandas as pd
import os
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Path to the transformed dataset
DATA_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv'

# Define the output directory for feature importance results
FEATURE_IMPORTANCE_OUTPUT_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/feature_importance'

# Load the data
data = pd.read_csv(DATA_PATH)
print("Data Loaded Successfully. Shape:", data.shape)

# Create output folder if it doesn't exist
os.makedirs(FEATURE_IMPORTANCE_OUTPUT_PATH, exist_ok=True)

# Define target variable
target_variable = 'performance_score'  # Target column based on your description

if target_variable in data.columns and len(data.columns) > 1:
    # Filter numeric columns and exclude non-relevant types
    numeric_data = data.select_dtypes(include=['float64', 'int64', 'bool']).copy()
    numeric_data = numeric_data.applymap(lambda x: int(x) if isinstance(x, bool) else x)
    
    # Define features and target
    X = numeric_data.drop(columns=[target_variable])
    y = numeric_data[target_variable]

    # Handle missing values
    X.fillna(method='ffill', inplace=True)
    X.fillna(method='bfill', inplace=True)

    # RandomForest Feature Importance
    print("Calculating feature importance using RandomForest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    rf_importances = rf_model.feature_importances_
    rf_feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_importances
    }).sort_values(by='Importance', ascending=False)

    # Save and plot RandomForest results
    rf_importance_file = os.path.join(FEATURE_IMPORTANCE_OUTPUT_PATH, 'random_forest_feature_importance.csv')
    rf_importance_plot = os.path.join(FEATURE_IMPORTANCE_OUTPUT_PATH, 'random_forest_feature_importance.png')
    rf_feature_importances.to_csv(rf_importance_file, index=False)

    plt.figure(figsize=(12, 8))
    plt.barh(rf_feature_importances['Feature'], rf_feature_importances['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance - RandomForest')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(rf_importance_plot)
    plt.close()

    print(f"RandomForest feature importance saved: {rf_importance_file}")
    print(f"RandomForest feature importance plot saved: {rf_importance_plot}")

    # XGBoost Feature Importance
    print("Calculating feature importance using XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X, y)

    xgb_importances = xgb_model.feature_importances_
    xgb_feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb_importances
    }).sort_values(by='Importance', ascending=False)

    # Save and plot XGBoost results
    xgb_importance_file = os.path.join(FEATURE_IMPORTANCE_OUTPUT_PATH, 'xgboost_feature_importance.csv')
    xgb_importance_plot = os.path.join(FEATURE_IMPORTANCE_OUTPUT_PATH, 'xgboost_feature_importance.png')
    xgb_feature_importances.to_csv(xgb_importance_file, index=False)

    plt.figure(figsize=(12, 8))
    plt.barh(xgb_feature_importances['Feature'], xgb_feature_importances['Importance'], color='lightgreen')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance - XGBoost')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(xgb_importance_plot)
    plt.close()

    print(f"XGBoost feature importance saved: {xgb_importance_file}")
    print(f"XGBoost feature importance plot saved: {xgb_importance_plot}")

else:
    print(f"Target variable '{target_variable}' not found or insufficient features available.")

print("\nFeature importance analysis completed. Outputs saved.")
