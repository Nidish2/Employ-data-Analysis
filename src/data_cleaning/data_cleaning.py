import pandas as pd
import numpy as np
from openpyxl import Workbook

# File paths
RAW_DATA_PATH = "C:/Users/nidis/Documents/Employment_Analysis/data/Employee_data_raw.csv"
CLEANED_DATA_PATH = "C:/Users/nidis/Documents/Employment_Analysis/data/cleaned_data.csv"
EXCEL_VALIDATION_PATH = "C:/Users/nidis/Documents/Employment_Analysis/data/cleaned_data_validation.xlsx"

def load_data(file_path):
    """
    Load raw data into a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    """
    # Drop columns with all missing values
    data = data.dropna(axis=1, how='all')

    # Fill numeric columns with mean
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].apply(lambda col: col.fillna(col.mean()))

    # Fill categorical columns with mode
    object_cols = data.select_dtypes(include=['object']).columns
    data[object_cols] = data[object_cols].apply(lambda col: col.fillna(col.mode()[0]))

    print("Missing values handled.")
    return data

def validate_and_correct_dtypes(data):
    """
    Validate and correct column data types.
    """
    # Convert date columns to datetime
    date_cols = ['orighiredate_key', 'terminationdate_key', 'last_promotion_date']
    for col in date_cols:
        data[col] = pd.to_datetime(data[col], errors='coerce')

    # Ensure numerical columns are numeric
    numeric_cols = [
        'age', 'length_of_service', 'salary', 'performance_score',
        'manager_rating', 'self_rating', 'work_life_balance_score',
        'overtime_hours', 'working_hours', 'employee_satisfaction_score',
        'salary_hike_percent', 'post_promotion_performance',
        'peer_feedback_score', 'absenteeism_rate'
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    print("Data types validated and corrected.")
    return data

def standardize_text(data):
    """
    Standardize text columns for consistency.
    """
    text_cols = data.select_dtypes(include=['object']).columns
    for col in text_cols:
        data[col] = data[col].str.strip().str.lower()

    print("Text columns standardized.")
    return data

def validate_data_consistency(data):
    """
    Perform data validation checks.
    """
    # Salary should be non-negative
    data = data[data['salary'] >= 0]

    # Working hours should be realistic (<= 24 hours per day)
    data = data[data['working_hours'] <= 24]

    # Length of service should not exceed age
    data = data[data['length_of_service'] <= data['age']]

    print("Data consistency validated.")
    return data

def save_cleaned_data(data, file_path, excel_path):
    """
    Save cleaned data to CSV and Excel formats for validation.
    """
    try:
        # Save as CSV
        data.to_csv(file_path, index=False)
        print(f"Cleaned data saved to {file_path}")

        # Save as Excel for validation
        data.to_excel(excel_path, index=False)
        print(f"Validation file saved to {excel_path}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")

def main():
    # Step 1: Load the raw data
    data = load_data(RAW_DATA_PATH)
    if data is None:
        return
    
    # Step 2: Handle missing values
    data = handle_missing_values(data)

    # Step 3: Validate and correct data types
    data = validate_and_correct_dtypes(data)

    # Step 4: Standardize text columns
    data = standardize_text(data)

    # Step 5: Validate data consistency
    data = validate_data_consistency(data)

    # Step 6: Save cleaned data
    save_cleaned_data(data, CLEANED_DATA_PATH, EXCEL_VALIDATION_PATH)

if __name__ == "__main__":
    main()
