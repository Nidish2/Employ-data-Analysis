import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import os
import geopandas as gpd
from matplotlib.colors import LogNorm

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

# -------------------------- 3. Distributions of Key Variables --------------------------

# Visualize the distributions of key variables
key_variables = ['age', 'length_of_service', 'salary', 'employee_satisfaction_score', 'performance_score']

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

# -------------------------- 4. Exploring Relationships --------------------------

# Salary vs Job Title (Boxplot)
plt.figure(figsize=(14, 10))
sns.boxplot(x='salary', y='job_title_accounts payable clerk', data=data, palette='coolwarm')
plt.title('Salary Distribution by Job Title')
plt.xlabel('Salary')
plt.ylabel('Job Title')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_OUTPUT_PATH, 'salary_by_job_title.png'))
plt.close()
print("Salary by Job Title plot saved.")

# Salary vs Department (Boxplot)
plt.figure(figsize=(14, 10))
sns.boxplot(x='salary', y='department_name_accounts payable', data=data, palette='Set2')
plt.title('Salary Distribution by Department')
plt.xlabel('Salary')
plt.ylabel('Department')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_OUTPUT_PATH, 'salary_by_department.png'))
plt.close()
print("Salary by Department plot saved.")

# Salary vs City (Boxplot)
plt.figure(figsize=(14, 10))
sns.boxplot(x='salary', y='city_name_aldergrove', data=data, palette='Set1')
plt.title('Salary Distribution by City')
plt.xlabel('Salary')
plt.ylabel('City')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_OUTPUT_PATH, 'salary_by_city.png'))
plt.close()
print("Salary by City plot saved.")

# -------------------------- 5. Employee Satisfaction & Performance --------------------------

# Relationship between Work Life Balance and Employee Satisfaction (Scatterplot)
plt.figure(figsize=(12, 8))
sns.scatterplot(x='work_life_balance_score', y='employee_satisfaction_score', data=data, hue='gender_full_male', palette='coolwarm')
plt.title('Work Life Balance vs Employee Satisfaction')
plt.xlabel('Work Life Balance')
plt.ylabel('Employee Satisfaction')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_OUTPUT_PATH, 'work_life_balance_vs_satisfaction.png'))
plt.close()
print("Work Life Balance vs Satisfaction plot saved.")

# Relationship between Performance Score and Manager Rating (Scatterplot)
plt.figure(figsize=(12, 8))
sns.scatterplot(x='performance_score', y='manager_rating', data=data, hue='department_name_accounts payable', palette='Set2')
plt.title('Performance Score vs Manager Rating')
plt.xlabel('Performance Score')
plt.ylabel('Manager Rating')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_OUTPUT_PATH, 'performance_vs_manager_rating.png'))
plt.close()
print("Performance vs Manager Rating plot saved.")

# -------------------------- 6. Outliers Detection --------------------------

# Outliers in Salary (Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['salary'], color='orange')
plt.title('Outliers in Salary')
plt.xlabel('Salary')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_OUTPUT_PATH, 'outliers_salary.png'))
plt.close()
print("Outlier plot for salary saved.")

# -------------------------- 7. Correlation Heatmap for Numerical Variables --------------------------

# Correlation matrix and heatmap
numerical_columns = data.select_dtypes(include=[np.number]).columns
correlation_matrix = data[numerical_columns].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Variables')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_OUTPUT_PATH, 'correlation_heatmap.png'))
plt.close()
print("Correlation heatmap saved.")

# -------------------------- 8. Geospatial Analysis --------------------------

# For Geospatial plotting, we'll use Plotly and GeoPandas (for city-level analysis)

# Create a new column to map city to latitude and longitude (for example, using `city_name`).
# Example: Assuming we have latitude and longitude data for cities (you can use a geocoding API for real-world data)

city_coords = {
    'aldergrove': (49.0426, -122.4413),
    'bella bella': (52.1650, -128.1667),
    'blue river': (52.0150, -118.9367),
    'burnaby': (49.2488, -123.0014),
    'chilliwack': (49.1647, -121.9783),
    # Add all other cities here...
}

# Add latitude and longitude to the dataset (for demonstration purposes)
data['latitude'] = data['city_name_aldergrove'].map(lambda x: city_coords.get(x.lower(), (np.nan, np.nan))[0])
data['longitude'] = data['city_name_aldergrove'].map(lambda x: city_coords.get(x.lower(), (np.nan, np.nan))[1])

# Create an interactive plot for Salary vs City (Geospatial Visualization)
fig = px.scatter_geo(data, lat='latitude', lon='longitude', color='salary', hover_name='city_name_aldergrove', 
                     size='salary', title="Geospatial Distribution of Salaries by City", 
                     color_continuous_scale='Viridis', size_max=15, template="plotly_dark")
fig.write_html(os.path.join(VISUALIZATION_OUTPUT_PATH, 'geospatial_salary_by_city.html'))
print("Geospatial Salary by City plot saved.")

# -------------------------- 9. Conclusion --------------------------

print(f"\nData visualization phase completed. Outputs saved to {VISUALIZATION_OUTPUT_PATH}.")
