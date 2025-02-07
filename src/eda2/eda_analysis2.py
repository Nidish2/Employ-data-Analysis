import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------- Constants --------------------------

# Path to the transformed dataset
DATA_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv'

# Define the output directory for EDA results
EDA_OUTPUT_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/eda2'

# -------------------------- 1. Load Dataset --------------------------

try:
    data = pd.read_csv(DATA_PATH)
    print("Data Loaded Successfully. Shape:", data.shape)
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}")
    exit(1)

# -------------------------- 2. Create Output Directory --------------------------

os.makedirs(EDA_OUTPUT_PATH, exist_ok=True)

# -------------------------- 3. Descriptive Statistics --------------------------

# Numerical statistics
numerical_columns = data.select_dtypes(include=[np.number]).columns
numerical_stats = data[numerical_columns].describe().transpose()
numerical_stats.to_csv(os.path.join(EDA_OUTPUT_PATH, 'numerical_stats.csv'))
print("Numerical statistics saved.")

# Categorical statistics
categorical_columns = data.select_dtypes(include=[object]).columns
categorical_stats = data[categorical_columns].nunique().reset_index()
categorical_stats.columns = ['Categorical Feature', 'Unique Values']
categorical_stats.to_csv(os.path.join(EDA_OUTPUT_PATH, 'categorical_stats.csv'))
print("Categorical statistics saved.")

# -------------------------- 4. Correlation Analysis --------------------------

correlation_matrix = data[numerical_columns].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(EDA_OUTPUT_PATH, 'correlation_heatmap.png'))
plt.close()
print("Correlation heatmap saved.")

# -------------------------- 5. Distribution Analysis --------------------------

for col in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[col], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_PATH, f'distribution_{col}.png'))
    plt.close()

print("Distribution plots saved.")

# -------------------------- 6. Categorical Analysis --------------------------

for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=col, data=data, order=data[col].value_counts().index, palette='viridis')
    plt.title(f'Value Counts for {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_PATH, f'value_counts_{col}.png'))
    plt.close()

print("Value count plots saved.")

# -------------------------- 7. Bivariate Analysis --------------------------

# Work-life balance vs employee satisfaction
plt.figure(figsize=(10, 6))
sns.boxplot(x='work_life_balance_score', y='employee_satisfaction_score', data=data, palette='coolwarm')
plt.title('Work Life Balance vs Employee Satisfaction')
plt.xlabel('Work Life Balance Score')
plt.ylabel('Employee Satisfaction Score')
plt.tight_layout()
plt.savefig(os.path.join(EDA_OUTPUT_PATH, 'work_life_balance_vs_satisfaction.png'))
plt.close()

# Overtime vs absenteeism
plt.figure(figsize=(10, 6))
sns.scatterplot(x='overtime_hours', y='absenteeism_rate', data=data, color='red', alpha=0.6)
plt.title('Overtime Hours vs Absenteeism Rate')
plt.xlabel('Overtime Hours')
plt.ylabel('Absenteeism Rate')
plt.tight_layout()
plt.savefig(os.path.join(EDA_OUTPUT_PATH, 'overtime_vs_absenteeism.png'))
plt.close()

print("Bivariate analysis plots saved.")

# -------------------------- 8. Grouped Analysis --------------------------

# Department vs work-life balance
plt.figure(figsize=(12, 8))
sns.boxplot(x='department_name', y='work_life_balance_score', data=data, palette='Set2')
plt.title('Work Life Balance by Department')
plt.xlabel('Department')
plt.ylabel('Work Life Balance Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(EDA_OUTPUT_PATH, 'work_life_balance_by_department.png'))
plt.close()

# Gender vs performance score
plt.figure(figsize=(10, 6))
sns.boxplot(x='gender_full', y='performance_score', data=data, palette='muted')
plt.title('Performance Score by Gender')
plt.xlabel('Gender')
plt.ylabel('Performance Score')
plt.tight_layout()
plt.savefig(os.path.join(EDA_OUTPUT_PATH, 'performance_by_gender.png'))
plt.close()

print("Grouped analysis plots saved.")

# -------------------------- 9. Pairwise Relationships --------------------------

sampled_data = data.sample(min(1000, len(data)), random_state=42)
sns.pairplot(sampled_data[numerical_columns], diag_kind='kde', corner=True)
plt.savefig(os.path.join(EDA_OUTPUT_PATH, 'pairwise_relationships.png'))
plt.close()
print("Pairwise relationships plot saved.")

# -------------------------- 10. Output Completion --------------------------

print(f"EDA completed. Outputs saved to {EDA_OUTPUT_PATH}.")
