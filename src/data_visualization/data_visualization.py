import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import os

# -------------------------- Constants --------------------------

# Path to the transformed dataset
DATA_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv'

# Define the output directory for visualization results
VISUALIZATION_OUTPUT_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/visualization'

# -------------------------- 1. Load Dataset --------------------------

# Load the data
data = pd.read_csv(DATA_PATH)
print("Data Loaded Successfully. Shape:", data.shape)

# -------------------------- 2. Create Output Directory --------------------------

# Create output folder if it doesn't exist
os.makedirs(VISUALIZATION_OUTPUT_PATH, exist_ok=True)

# -------------------------- 3. Visualizing Key Variables --------------------------

# Visualize the distributions of key variables
key_variables = ['age', 'length_of_service', 'salary']

for var in key_variables:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[var], kde=True, bins=30, color='blue', stat="density", linewidth=0)
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_OUTPUT_PATH, f'distribution_{var}.png'))
    plt.close()
    print(f"Distribution plot for {var} saved.")

# -------------------------- 4. Exploring Relationships Between Variables --------------------------

# Dynamically plot salary distribution for a few job titles (if job titles are available)
# Choose some of the common columns that represent job titles in the dataset
job_title_columns = ['job_title_accounts payable clerk', 'job_title_baker', 'job_title_cashier']

# Visualizing salary distribution across different job titles dynamically
for job_title in job_title_columns:
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='salary', y=job_title, data=data, palette='coolwarm')
    plt.title(f'Salary Distribution for {job_title.replace("job_title_", "").replace("_", " ").title()}')
    plt.xlabel('Salary')
    plt.ylabel('Job Title')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_OUTPUT_PATH, f'salary_by_{job_title}.png'))
    plt.close()
    print(f"Salary by {job_title} plot saved.")

# -------------------------- 5. Salary Distribution by Department --------------------------

# Explore salary vs department (You can use different department columns)
department_columns = [
    'department_name_accounts payable', 'department_name_bakery', 'department_name_customer service'
]

# Visualizing salary distribution across different departments
for department in department_columns:
    plt.figure(figsize=(14, 10))
    sns.boxplot(x='salary', y=department, data=data, palette='Set2')
    plt.title(f'Salary Distribution by {department.replace("department_name_", "").replace("_", " ").title()}')
    plt.xlabel('Salary')
    plt.ylabel('Department')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_OUTPUT_PATH, f'salary_by_{department}.png'))
    plt.close()
    print(f"Salary by {department} plot saved.")

# -------------------------- 6. Salary Distribution by City --------------------------

# Explore salary vs city (You can use different city columns)
city_columns = ['city_name_vancouver', 'city_name_surrey', 'city_name_abbotsford']

# Visualizing salary distribution across different cities
for city in city_columns:
    plt.figure(figsize=(14, 10))
    sns.boxplot(x='salary', y=city, data=data, palette='Set1')
    plt.title(f'Salary Distribution by {city.replace("city_name_", "").replace("_", " ").title()}')
    plt.xlabel('Salary')
    plt.ylabel('City')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_OUTPUT_PATH, f'salary_by_{city}.png'))
    plt.close()
    print(f"Salary by {city} plot saved.")

# -------------------------- 7. Identifying Outliers and Skewness --------------------------

# Boxplots for detecting outliers in key variables
for var in key_variables:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[var], color='orange')
    plt.title(f'Outliers in {var}')
    plt.xlabel(var)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_OUTPUT_PATH, f'outliers_{var}.png'))
    plt.close()
    print(f"Outlier plot for {var} saved.")

# -------------------------- 8. Correlation Heatmap for Numerical Variables --------------------------

# Correlation matrix and heatmap
numerical_columns = data.select_dtypes(include=[np.number]).columns
correlation_matrix = data[numerical_columns].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap of Numerical Variables')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_OUTPUT_PATH, 'correlation_heatmap.png'))
plt.close()
print("Correlation heatmap saved.")

# -------------------------- 9. Advanced Plotting with Plotly --------------------------

# Create an interactive scatter plot of salary vs. department using Plotly
fig = px.box(data_frame=data, x="salary", y="department_name_accounts payable", 
             title="Salary by Department (Interactive)", 
             color="department_name_accounts payable", 
             color_continuous_scale='Viridis')
fig.write_html(os.path.join(VISUALIZATION_OUTPUT_PATH, 'salary_by_department_plotly.html'))
print("Interactive salary by department plot saved.")

# -------------------------- 10. Conclusion --------------------------

print(f"\nData visualization phase completed. Outputs saved to {VISUALIZATION_OUTPUT_PATH}.")
