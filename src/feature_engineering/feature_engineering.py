# ----------------------------- IMPORTING LIBRARIES -----------------------------
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for clean output

# ----------------------------- LOAD DATA ---------------------------------------
# Adjust path based on your project structure
data_path = 'C:/Users/nidis/Documents/Employment_Analysis/data/cleaned_data.csv'
data = pd.read_csv(data_path)

# Clean column names to remove unnecessary characters
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces
data.columns = data.columns.str.replace('ï»¿', '', regex=False)  # Remove special characters

print(f"Data Loaded Successfully. Shape: {data.shape}")
print("Columns:", list(data.columns))

# ----------------------------- FEATURE ENGINEERING -----------------------------

# ----------------- STEP 1: CONVERT DATES TO DATETIME FORMAT ---------------------
if 'last_promotion_date' in data.columns:
    data['last_promotion_date'] = pd.to_datetime(data['last_promotion_date'], errors='coerce')

# ----------------- STEP 2: HANDLE MISSING VALUES -------------------------------
# Separate numeric and categorical columns
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

# Numeric data imputation (median)
numeric_imputer = SimpleImputer(strategy='median')
data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])

# Ensure integer columns retain their original dtype
int_columns = ['EmployeeID', 'age', 'length_of_service', 'store_name', 'STATUS_YEAR',
               'GENERAL APPEARANCE', 'MANNER OF SPEAKING', 'PHYSICAL CONDITION',
               'MENTAL ALERTNESS', 'SELF-CONFIDENCE', 'ABILITY TO PRESENT IDEAS',
               'COMMUNICATION SKILLS', 'Student Performance Rating', 'salary',
               'performance_score', 'overtime_hours', 'working_hours', 
               'employee_satisfaction_score', 'salary_hike_percent', 
               'post_promotion_performance', 'absenteeism_rate']

data[int_columns] = data[int_columns].astype(int)

# Categorical data imputation (most frequent)
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = pd.DataFrame(
    categorical_imputer.fit_transform(data[categorical_columns]), 
    columns=categorical_columns
)

# ----------------- STEP 3: TIME-BASED FEATURES ---------------------------------
# Tenure buckets
if 'length_of_service' in data.columns:
    tenure_bins = [0, 2, 5, 10, 20, np.inf]
    tenure_labels = ['<2 years', '2-5 years', '5-10 years', '10-20 years', '>20 years']
    data['tenure_bucket'] = pd.cut(data['length_of_service'], bins=tenure_bins, labels=tenure_labels)

# Months since last promotion
if 'last_promotion_date' in data.columns:
    data['months_since_last_promotion'] = (
        (pd.to_datetime('today') - data['last_promotion_date']).dt.days / 30
    ).fillna(0).astype(int)

# ----------------- STEP 4: DERIVED FEATURES ------------------------------------
# Turnover Risk Index
if 'age' in data.columns and 'salary_hike_percent' in data.columns and 'length_of_service' in data.columns:
    data['turnover_risk_index'] = (
        (data['age'] * data['salary_hike_percent']) / (data['length_of_service'] + 1)
    ).round(2)

# Engagement Index
required_columns = {'employee_satisfaction_score', 'work_life_balance_score', 'peer_feedback_score'}
if required_columns.issubset(data.columns):
    data['engagement_index'] = (
        data['employee_satisfaction_score'] +
        data['work_life_balance_score'] +
        data['peer_feedback_score']
    ) / 3
    data['engagement_index'] = data['engagement_index'].round(2)

# Absenteeism Category
if 'absenteeism_rate' in data.columns:
    absenteeism_bins = [0, 5, 10, 15, np.inf]
    absenteeism_labels = ['Low', 'Moderate', 'High', 'Critical']
    data['absenteeism_category'] = pd.cut(data['absenteeism_rate'], bins=absenteeism_bins, labels=absenteeism_labels)

# Post-Promotion Improvement
if 'performance_score' in data.columns and 'post_promotion_performance' in data.columns:
    data['performance_improvement'] = (
        data['performance_score'] - data['post_promotion_performance']
    ).clip(lower=0)  # Ensure no negative values


# ----------------- STEP 5: SCALING NUMERICAL FEATURES --------------------------
# Only scale specific columns if needed
columns_to_scale = [
    col for col in numeric_columns
    if col not in int_columns + ['age', 'length_of_service', 'salary', 
                                 'performance_score', 'manager_rating', 
                                 'self_rating', 'work_life_balance_score']
]

scaler = MinMaxScaler()
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

# Replace zeros in float columns with the column mean (if applicable)
float_columns = data.select_dtypes(include=[np.float64]).columns.tolist()
for col in float_columns:
    if (data[col] == 0).sum() > 0:  # Check if the column has zeros
        data[col] = data[col].replace(0, np.nan)  # Treat zeros as missing
        data[col].fillna(data[col].mean(), inplace=True)  # Impute with the mean


# ----------------- STEP 6: VISUALIZE CORRELATIONS ------------------------------

# Load the feature engineered dataset from the specified path
feature_engineered_data_path = 'C:/Users/nidis/Documents/Employment_Analysis/data/feature_engineered_data.csv'
data = pd.read_csv(feature_engineered_data_path)

# Compute the correlation matrix for the new dataset
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()  # Recompute numeric columns
correlation_matrix = data[numeric_columns].corr()

# Re-plot the heatmap with the updated dataset
plt.figure(figsize=(18, 18))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title('Feature Correlation Matrix (Updated from Feature Engineered Data)')
plt.tight_layout()

# Save the updated heatmap
output_dir = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/feature_engineering/'
os.makedirs(output_dir, exist_ok=True)
heatmap_path = os.path.join(output_dir, 'correlation_matrix.png')
plt.savefig(heatmap_path)
plt.show()

# ----------------- STEP 7: EXPORT PROCESSED DATA -------------------------------
# Save the feature engineered data to a CSV file
output_path = 'C:/Users/nidis/Documents/Employment_Analysis/data/feature_engineered_data.csv'
data.to_csv(output_path, index=False)

print(f"\nFeature Engineered Data Saved Successfully at: {output_path}")
print("New Data Shape:", data.shape)
print(data.head())
