import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Paths
DATA_PATH = "C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv"
OUTPUT_DIR = "C:/Users/nidis/Documents/Employment_Analysis/outputs/eda2"

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Data
data = pd.read_csv(DATA_PATH)
print(f"Data loaded successfully. Shape: {data.shape}")

# Distribution Analysis
def analyze_distributions(data):
    """Analyze distributions of key numerical columns."""
    numeric_cols = [
        'age', 'salary', 'performance_score', 
        'manager_rating', 'self_rating', 
        'employee_satisfaction_score', 'work_life_balance_score',
        'overtime_hours', 'absenteeism_rate', 'length_of_service', 
        'GENERAL APPEARANCE', 'MANNER OF SPEAKING', 'PHYSICAL CONDITION',
        'MENTAL ALERTNESS', 'SELF-CONFIDENCE', 'ABILITY TO PRESENT IDEAS',
        'COMMUNICATION SKILLS', 'Student Performance Rating', 'STATUS_YEAR', 
        'salary_hike_percent', 'post_promotion_performance', 'peer_feedback_score',
        'turnover_risk_index', 'engagement_index', 'absenteeism_category', 
        'performance_improvement', 'months_since_last_promotion'
    ]
    
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[col], kde=True, bins=30, color='blue')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f"{OUTPUT_DIR}/{col}_distribution.png")
        plt.close()

# Categorical Analysis
def analyze_categoricals(data):
    """Analyze categorical columns."""
    categorical_cols = ['gender_full', 'department_name', 'job_title', 'STATUS', 'Attrition', 'tenure_bucket', 'absenteeism_category', 'CLASS']
    
    for col in categorical_cols:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=col, data=data, hue=col, palette='viridis', legend=False)
        plt.title(f"Count of {col}")
        plt.xticks(rotation=45)
        plt.savefig(f"{OUTPUT_DIR}/{col}_counts.png")
        plt.close()

# Correlation Analysis
def analyze_correlations(data):
    """Analyze correlations among numerical columns."""
    numeric_cols = data.select_dtypes(include=['number']).columns
    corr_matrix = data[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title("Correlation Heatmap")
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.close()

# Trend Analysis (based on promotion year and salary)
def analyze_trends(data):
    """Analyze historical trends."""
    # Convert dates
    data['orighiredate_key'] = pd.to_datetime(data['orighiredate_key'])
    data['terminationdate_key'] = pd.to_datetime(data['terminationdate_key'])
    data['last_promotion_date'] = pd.to_datetime(data['last_promotion_date'], errors='coerce')

    # Promotions by year
    data['promotion_year'] = data['last_promotion_date'].dt.year
    promotion_trends = data.groupby('promotion_year')['salary_hike_percent'].mean()
    
    plt.figure(figsize=(10, 6))
    promotion_trends.plot(kind='bar', color='green')
    plt.title("Average Salary Hike by Promotion Year")
    plt.xlabel("Year")
    plt.ylabel("Average Salary Hike (%)")
    plt.savefig(f"{OUTPUT_DIR}/promotion_trends.png")
    plt.close()

    # Attrition Rate by Year
    attrition_by_year = data.groupby('promotion_year')['Attrition'].mean()
    
    plt.figure(figsize=(10, 6))
    attrition_by_year.plot(kind='line', color='red', marker='o')
    plt.title("Attrition Rate by Promotion Year")
    plt.xlabel("Year")
    plt.ylabel("Attrition Rate (%)")
    plt.savefig(f"{OUTPUT_DIR}/attrition_trends.png")
    plt.close()

# Geospatial Analysis of Salaries (based on city_name)
def analyze_geospatial_salaries(data):
    """Analyze geospatial distribution of salaries."""
    # Ensure that the city name columns are treated as strings
    data['city_name'] = data['city_name'].astype(str)
    
    # Define city coordinates (you can use a dictionary as before or a geospatial dataset)
    city_coords = {
        'aldergrove': (49.0426, -122.4413),
        'bella bella': (52.1650, -128.1667),
        'blue river': (51.6530, -118.3567),
        'burnaby': (49.2631, -123.1304),
        'chilliwack': (49.1682, -121.9564),
        'cortes island': (50.0120, -123.7797),
        'cranbrook': (49.5035, -115.7823),
        'dawson creek': (55.7479, -120.2316),
        'dease lake': (55.5633, -130.4353),
        'fort nelson': (58.8056, -122.6981),
        'fort st john': (56.2510, -120.8457),
        'grand forks': (49.0595, -118.6042),
        'kamloops': (50.6760, -120.3300),
        'kelowna': (49.8865, -119.4983),
        'langley': (49.0464, -122.6387),
        'nanaimo': (49.1653, -123.9354),
        'richmond': (49.1666, -123.1339),
        'vancouver': (49.2827, -123.1207),
        'victoria': (48.4284, -123.3656),
    }
    
    # Add latitude and longitude columns
    data['latitude'] = data['city_name'].map(lambda x: city_coords.get(x.lower(), (np.nan, np.nan))[0])
    data['longitude'] = data['city_name'].map(lambda x: city_coords.get(x.lower(), (np.nan, np.nan))[1])

    # Create a scatter plot with Plotly to visualize salaries by city
    fig = px.scatter_geo(data,
                         lat='latitude', lon='longitude',
                         color='salary', hover_name='city_name',
                         color_continuous_scale='Viridis', size_max=15,
                         template="plotly_dark",
                         title="Geospatial Analysis of Salaries")
    
    output_geospatial_path = os.path.join(OUTPUT_DIR, 'geospatial_salaries.png')
    fig.write_image(output_geospatial_path)
    print(f"Geospatial analysis saved at {output_geospatial_path}")

# Run EDA
print("Starting EDA...")

# Distribution and other analyses
analyze_distributions(data)
analyze_categoricals(data)
analyze_correlations(data)
analyze_trends(data)

# Geospatial analysis
analyze_geospatial_salaries(data)

print(f"EDA completed. Outputs saved to {OUTPUT_DIR}.")
