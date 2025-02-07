import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# -------------------------- Constants --------------------------

DATA_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/feature_engineered_data.csv'
TRANSFORMED_DATA_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv'
LABEL_ENCODER_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/models/label_encoder.pkl'

# -------------------------- 1. Load Dataset --------------------------

# Load the dataset
try:
    data = pd.read_csv(DATA_PATH)
    print("File loaded successfully!")
    print("Shape of data:", data.shape)
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}")
    exit(1)

# -------------------------- 2. Derive Attrition --------------------------

# Define logic for attrition
def derive_attrition(row):
    """
    Calculate Attrition based on terminationdate_key, STATUS, and termreason_desc.
    Attrition = 1 if the employee is terminated for specific reasons, else 0.
    """
    # Check if the terminationdate_key is valid (not default '1900-01-01') and the status is 'Terminated'
    if row['STATUS'].lower() == 'terminated':
        # Attrition occurs for specific termination reasons
        if row['termreason_desc'].lower() in ['layoff', 'resignation', 'retirement']:
            return 1
    return 0


# Apply logic to derive attrition
data['Attrition'] = data.apply(derive_attrition, axis=1)

# Validate Attrition values
attrition_counts = data['Attrition'].value_counts()
print("Attrition Value Counts:\n", attrition_counts)

# -------------------------- 3. Handle Missing Values --------------------------

# Separate numeric and categorical columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
categorical_columns = data.select_dtypes(include=[object]).columns

# Impute missing numeric values with KNNImputer
imputer = KNNImputer(n_neighbors=5)
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# Impute missing categorical values with mode
for col in categorical_columns:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].mode()[0])

# -------------------------- 4. Preserve Data Integrity --------------------------

# Ensure critical numeric columns retain their original ranges and types
integer_columns = [
    'EmployeeID', 'age', 'length_of_service', 'store_name', 'salary',
    'overtime_hours', 'working_hours', 'employee_satisfaction_score',
    'salary_hike_percent', 'months_since_last_promotion', 'absenteeism_rate',
    'performance_improvement'
]
data[integer_columns] = data[integer_columns].astype(int)

# -------------------------- 5. Encode Categorical Features --------------------------

# Specify columns to encode
categorical_features_to_encode = ['department_name', 'job_title', 'gender_full', 'tenure_bucket', 'absenteeism_category']

label_encoder = LabelEncoder()
for col in categorical_features_to_encode:
    if col in data.columns:
        data[col] = label_encoder.fit_transform(data[col])

# -------------------------- 6. Validate and Save Data --------------------------

# Ensure output directories exist
os.makedirs(os.path.dirname(TRANSFORMED_DATA_PATH), exist_ok=True)

# Save transformed data
data.to_csv(TRANSFORMED_DATA_PATH, index=False)

# Save label encoder for reuse
joblib.dump(label_encoder, LABEL_ENCODER_PATH)

print("Data transformation completed and saved successfully!")
