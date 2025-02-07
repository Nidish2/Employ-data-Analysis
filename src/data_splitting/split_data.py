import pandas as pd
from sklearn.model_selection import train_test_split
import os

# -------------------------- Constants --------------------------

DATA_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv'
TRAIN_FEATURES_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/train_features.csv'
TEST_FEATURES_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/test_features.csv'
TRAIN_TARGET_STATUS_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/train_target_status.csv'
TEST_TARGET_STATUS_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/test_target_status.csv'
TRAIN_TARGET_ATTRITION_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/train_target_attrition.csv'
TEST_TARGET_ATTRITION_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/test_target_attrition.csv'

# -------------------------- 1. Load Transformed Dataset --------------------------

try:
    data = pd.read_csv(DATA_PATH)
    print("Loaded transformed data:")
    print(data.head())
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}")
    exit(1)

# -------------------------- 2. Prepare Features and Targets --------------------------

# Ensure required columns are present
required_columns = ['STATUS', 'Attrition']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"One or more required columns {required_columns} are missing from the dataset!")

# Map STATUS to binary values: 1 for 'TERMINATED', 0 for 'ACTIVE'
data['STATUS'] = data['STATUS'].map({'TERMINATED': 1, 'ACTIVE': 0})

# Define features (X) and target variables
X = data.drop(columns=['STATUS', 'Attrition'])
y_status = data['STATUS']
y_attrition = data['Attrition']

print("Features shape:", X.shape)
print("Target shapes:")
print("STATUS:", y_status.shape, "| Attrition:", y_attrition.shape)
print("STATUS distribution:\n", y_status.value_counts())
print("Attrition distribution:\n", y_attrition.value_counts())

# -------------------------- 3. Split Data into Training and Testing Sets --------------------------

# Split data for STATUS target
X_train_status, X_test_status, y_train_status, y_test_status = train_test_split(
    X, y_status, test_size=0.2, random_state=42, stratify=y_status
)

# Split data for Attrition target
X_train_attrition, X_test_attrition, y_train_attrition, y_test_attrition = train_test_split(
    X, y_attrition, test_size=0.2, random_state=42, stratify=y_attrition
)

print("Training features shape (STATUS):", X_train_status.shape)
print("Test features shape (STATUS):", X_test_status.shape)
print("Training target shape (STATUS):", y_train_status.shape)
print("Test target shape (STATUS):", y_test_status.shape)

print("Training features shape (Attrition):", X_train_attrition.shape)
print("Test features shape (Attrition):", X_test_attrition.shape)
print("Training target shape (Attrition):", y_train_attrition.shape)
print("Test target shape (Attrition):", y_test_attrition.shape)

# -------------------------- 4. Save Split Data --------------------------

# Ensure output directories exist
os.makedirs(os.path.dirname(TRAIN_FEATURES_PATH), exist_ok=True)

# Save the split data for STATUS target
X_train_status.to_csv(TRAIN_FEATURES_PATH, index=False)
X_test_status.to_csv(TEST_FEATURES_PATH, index=False)
y_train_status.to_csv(TRAIN_TARGET_STATUS_PATH, index=False)
y_test_status.to_csv(TEST_TARGET_STATUS_PATH, index=False)

# Save the split data for Attrition target
y_train_attrition.to_csv(TRAIN_TARGET_ATTRITION_PATH, index=False)
y_test_attrition.to_csv(TEST_TARGET_ATTRITION_PATH, index=False)

print("Data split into training and testing sets and saved successfully.")
