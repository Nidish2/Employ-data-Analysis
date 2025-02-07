import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, roc_auc_score,
    confusion_matrix, classification_report
)

# -------------------------- Constants --------------------------

DATA_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv'
MODEL_SELECTION_OUTPUT_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/model_selection'
SCALER_PATH = os.path.join(MODEL_SELECTION_OUTPUT_PATH, 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_SELECTION_OUTPUT_PATH, 'label_encoder.pkl')

# -------------------------- Helper Functions --------------------------

def save_metrics(output_path, metrics_dict):
    """Save performance metrics to CSV."""
    os.makedirs(output_path, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(os.path.join(output_path, "model_performance_comparison.csv"), index=False)
    print(f"Performance metrics saved to {output_path}")

def save_classification_report(output_path, report, model_name):
    """Save classification report to a text file."""
    with open(os.path.join(output_path, f"{model_name}_classification_report.txt"), "w") as file:
        file.write(report)
    print(f"Classification report saved for {model_name}.")

# -------------------------- 1. Load Dataset --------------------------

try:
    data = pd.read_csv(DATA_PATH)
    print("Data Loaded Successfully. Shape:", data.shape)
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}")
    exit(1)

# -------------------------- 2. Handle Missing Values --------------------------

# Fill numeric columns with median
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Fill categorical columns with mode
categorical_columns = data.select_dtypes(include=[object]).columns
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# -------------------------- 3. Preprocessing --------------------------

# Identify non-numeric columns that need transformation
date_columns = ['orighiredate_key', 'terminationdate_key', 'last_promotion_date']  # Date-like columns
categorical_columns = ['termreason_desc', 'termtype_desc', 'STATUS', 'BUSINESS_UNIT']  # Categorical columns

# Convert date columns to numeric (days since a reference date)
for col in date_columns:
    data[col] = pd.to_datetime(data[col], errors='coerce')  # Convert to datetime
    data[col] = (data[col] - pd.Timestamp("1970-01-01")).dt.days  # Convert to days since epoch

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Fill missing values in newly transformed columns
data.fillna(0, inplace=True)

# Separate features and target
target_column = 'CLASS'  # Specify your target column here
if target_column not in data.columns:
    print(f"Error: Target column '{target_column}' not found in data.")
    exit(1)

X = data.drop(columns=[target_column])
y = data[target_column]

# Encode target variable if categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save label encoder
os.makedirs(MODEL_SELECTION_OUTPUT_PATH, exist_ok=True)
joblib.dump(label_encoder, LABEL_ENCODER_PATH)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, SCALER_PATH)

# -------------------------- 4. Train-Test Split --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# -------------------------- 5. Model Training --------------------------

# Initialize models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'XGBoost': GradientBoostingClassifier(random_state=42),  # XGBoost equivalent for scikit-learn
    'LightGBM': GradientBoostingClassifier(random_state=42)  # LightGBM equivalent for scikit-learn
}

# Parameters for hyperparameter tuning
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'DecisionTree': {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'LightGBM': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}

# Training and evaluation
results = []
for model_name, model in models.items():
    if model_name in param_grids:
        # Hyperparameter tuning with GridSearchCV
        grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best score for {model_name}: {grid_search.best_score_}")
    else:
        # Train model without hyperparameter tuning
        best_model = model.fit(X_train, y_train)
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr') if y_prob is not None else "N/A"
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    
    # Append results
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'ROC AUC Score': roc_auc,
        'Confusion Matrix': conf_matrix,
        'Classification Report': classification_rep
    })

# -------------------------- 6. Save and Compare Performance --------------------------

# Save and compare all model performance
performance_df = pd.DataFrame(results)
performance_df.to_csv(os.path.join(MODEL_SELECTION_OUTPUT_PATH, "model_performance_comparison.csv"), index=False)

# Save classification reports and confusion matrices
for result in results:
    model_name = result['Model']
    save_classification_report(MODEL_SELECTION_OUTPUT_PATH, result['Classification Report'], model_name)
    
    # Save confusion matrix
    conf_matrix_df = pd.DataFrame(result['Confusion Matrix'], index=label_encoder.classes_, columns=label_encoder.classes_)
    conf_matrix_df.to_csv(os.path.join(MODEL_SELECTION_OUTPUT_PATH, f"{model_name}_confusion_matrix.csv"), index=True)

print("Model performance comparison saved.")
