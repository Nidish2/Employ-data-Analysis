import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -------------------------- Set Paths --------------------------
EDA_OUTPUT_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/eda'
DATA_VISUALIZATION_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/data_visualization'
DATA_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv'

CATEGORICAL_STATS_PATH = os.path.join(EDA_OUTPUT_PATH, 'categorical_stats.csv')
NUMERICAL_STATS_PATH = os.path.join(EDA_OUTPUT_PATH, 'numerical_stats.csv')

# -------------------------- Load Data Functions --------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"File not found at {DATA_PATH}")
        return None

@st.cache_data
def load_categorical_stats():
    try:
        return pd.read_csv(CATEGORICAL_STATS_PATH)
    except FileNotFoundError:
        st.error(f"File not found at {CATEGORICAL_STATS_PATH}")
        return None

@st.cache_data
def load_numerical_stats():
    try:
        return pd.read_csv(NUMERICAL_STATS_PATH)
    except FileNotFoundError:
        st.error(f"File not found at {NUMERICAL_STATS_PATH}")
        return None

data = load_data()

# -------------------------- Streamlit UI --------------------------
st.title("Employment Analysis - Phase 2")

st.markdown("""
    <footer style="
        position: fixed; 
        bottom: 0; 
        width: 100%; 
        background-color: #1E1E1E; 
        padding: 3px; 
        text-align: left;
    ">
        <p style="
            color: #B0B0B0; 
            font-family: 'Arial', sans-serif; 
            font-size: 14px; 
            margin: 0;
        ">
            Created By Nidish (1BG22CS095) from CSE Undergrad @ BNMIT
        </p>
    </footer>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose Section", ["Introduction", "EDA", "Data Visualization","Deep EDA", "Conclusion"])

# -------------------------- Introduction Section --------------------------
if section == "Introduction":
    st.header("Introduction")
    st.markdown("""
        ### What we will accomplish in this phase:
        - **1. Exploratory Data Analysis (EDA)**: Analyze the dataset's structure, identify trends, and summarize key statistics.
        - **2. Categorical and Numerical Statistics**: Generate detailed overviews of categorical and numerical features.
        - **3. Data Visualizations**: Create and analyze various visualizations to uncover deeper insights.
        - **4. Insights Discovery**: Highlight patterns in employee performance, satisfaction, and other key metrics.
        - **5. Prepare for Predictive Modeling**: Use the insights gained to guide subsequent modeling efforts.
    """)
    st.markdown("Navigate through the sections to explore detailed analysis, visualizations, and findings.")

# -------------------------- EDA Section --------------------------
elif section == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    
    # Introduction to EDA
    st.subheader("Introduction to EDA")
    st.markdown("""
    Exploratory Data Analysis (EDA) is a critical step in understanding the dataset. It involves:
    
    1. **Analyzing Descriptive Statistics**:
        - **Categorical Features**: Assessing the distribution and unique values of categorical variables to understand trends and diversity.
        - **Numerical Features**: Summarizing statistical measures (mean, median, min, max, and standard deviation) to identify variability, skewness, and potential outliers in numerical data.
    
    2. **Visualization of Distributions**:
        - Creating visual representations for key features to understand their distributions, central tendencies, and variability.
        - Identifying trends and anomalies in employee-related metrics such as absenteeism, performance scores, and work-life balance.
    
    3. **Correlation Analysis**:
        - Generating a heatmap to explore relationships between numerical variables.
        - Understanding the strength of associations to identify significant patterns or dependencies.
    
    4. **Categorical Analysis**:
        - Visualizing value counts of categorical features to understand data composition (e.g., gender, department, termination reasons).
    
    5. **Bivariate and Grouped Analysis**:
        - Exploring interactions between two variables to uncover potential relationships (e.g., overtime hours vs absenteeism, work-life balance by department).
        - Group-level analysis to evaluate department-specific or gender-specific trends.
    
    6. **Pairwise Relationships**:
        - Visualizing relationships across multiple features using pairplots to uncover broader patterns and dependencies in the dataset.
    
    This phase provides valuable insights into the dataset, forming the foundation for feature engineering and predictive modeling.
    """)

    # Display statistics overview
    st.subheader("Statistics Overview")

    # Display categorical stats
    categorical_stats = load_categorical_stats()
    if categorical_stats is not None:
        st.write("### Categorical Features")
        st.dataframe(categorical_stats)
        st.write("**Categorical Features Analysis**:")
        st.markdown("""
        - We analyzed the unique values for each categorical feature.
        - Key features like `termreason_desc` (termination reasons) and `STATUS` (employee status) highlight essential workforce trends.
        - Department-level segregation (`department_name`) reveals organizational structure and staffing patterns.
        """)

    # Display numerical stats
    numerical_stats = load_numerical_stats()
    if numerical_stats is not None:
        st.write("### Numerical Features")
        st.dataframe(numerical_stats)
        st.write("**Numerical Features Analysis**:")
        st.markdown("""
        - Statistical summaries for numerical features reveal key metrics such as mean, median, and variability.
        - Features like `salary` and `performance_score` show significant variability, highlighting the need for normalization in later steps.
        - Outliers were identified in variables like `overtime_hours` and `absenteeism_rate`.
        """)


    # Display saved EDA visualizations with explanations
    st.header("Key EDA Visualizations")
    image_files = [
    ("absenteeism_rate_distribution.png", "Absenteeism Rate Distribution",
     "- **Why this graph?**: To identify trends in employee absenteeism.\n- **Insight**: Most employees show low absenteeism rates, indicating a dedicated and committed workforce."),
    ("age_distribution.png", "Age Distribution",
     "- **Why this graph?**: To understand the age demographics of the workforce.\n- **Insight**: The dataset captures a diverse age range, with the majority of employees in their mid-30s to early-40s."),
    ("department_name_counts.png", "Department-wise Employee Count",
     "- **Why this graph?**: To analyze the distribution of employees across various departments.\n- **Insight**: Certain departments, like operations, have higher staff counts, reflecting their operational needs."),
    ("employee_satisfaction_score_distribution.png", "Employee Satisfaction Score Distribution",
     "- **Why this graph?**: To evaluate the overall employee satisfaction level.\n- **Insight**: Employee satisfaction skews towards high scores, suggesting a generally satisfied workforce."),
    ("gender_full_counts.png", "Gender Distribution",
     "- **Why this graph?**: To understand the gender composition of the workforce.\n- **Insight**: Male employees slightly outnumber females, highlighting a minor gender disparity."),
    ("job_title_counts.png", "Job Titles Count",
     "- **Why this graph?**: To explore the distribution of various job titles within the organization.\n- **Insight**: Job titles like 'Cashier' and 'Manager' are more prevalent, reflecting their commonality in the workforce."),
    ("manager_rating_distribution.png", "Manager Ratings Distribution",
     "- **Why this graph?**: To analyze how employees rate their managers.\n- **Insight**: Manager ratings are mostly average to high, indicating reasonable managerial effectiveness."),
    ("overtime_hours_distribution.png", "Overtime Hours Distribution",
     "- **Why this graph?**: To evaluate employee work-life balance based on overtime.\n- **Insight**: A significant portion of employees log no overtime, possibly reflecting a good work-life balance."),
    ("performance_score_distribution.png", "Performance Score Distribution",
     "- **Why this graph?**: To assess employee performance levels.\n- **Insight**: Performance scores are mostly high, indicating a workforce that generally performs well."),
    ("promotion_trends.png", "Promotion Trends",
     "- **Why this graph?**: To understand the correlation between performance and promotions.\n- **Insight**: Promotions are largely tied to high performance but occur infrequently across the workforce."),
    ("salary_distribution.png", "Salary Distribution",
     "- **Why this graph?**: To understand the distribution of salaries within the organization.\n- **Insight**: Salaries are positively skewed, with most employees earning in the lower percentiles."),
    ("self_rating_distribution.png", "Self-Rating Distribution",
     "- **Why this graph?**: To examine how employees rate their own performance.\n- **Insight**: Employees tend to rate themselves highly, reflecting confidence in their abilities."),
    ("STATUS_counts.png", "Employee Status Distribution",
     "- **Why this graph?**: To analyze the distribution of active vs inactive employees.\n- **Insight**: Active employees dominate the dataset, with very few inactive or terminated employees."),
    ("work_life_balance_score_distribution.png", "Work-Life Balance Score Distribution",
     "- **Why this graph?**: To assess employee satisfaction with work-life balance.\n- **Insight**: Work-life balance scores suggest employees are generally satisfied with their work-life balance."),
    ("distribution_manager_rating.png", "Manager Rating Distribution",
     "- **Why this graph?**: To understand the distribution of ratings given to managers.\n- **Insight**: Manager ratings show a mixture of evaluations, with peaks at average and high ratings."),
    ("distribution_overtime_hours.png", "Overtime Hours Distribution",
     "- **Why this graph?**: To examine overtime distribution among employees.\n- **Insight**: The graph shows that many employees do not log significant overtime, which may indicate a healthy work-life balance."),
    ("distribution_self_rating.png", "Self-Rating Distribution",
     "- **Why this graph?**: To analyze the self-assessment scores of employees.\n- **Insight**: Self-ratings show a strong confidence among employees, with a peak at higher scores."),
    ("distribution_work_life_balance_score.png", "Work-Life Balance Score Distribution",
     "- **Why this graph?**: To analyze employees' satisfaction with their work-life balance.\n- **Insight**: Work-life balance scores indicate moderate to high satisfaction levels among employees."),
    ("distribution_overall_performance_rating.png", "Overall Performance Rating Distribution",
     "- **Why this graph?**: To evaluate the overall performance rating distribution across the workforce.\n- **Insight**: Most employees perform well, with only a small proportion showing poor performance."),
    ("distribution_peer_feedback_score.png", "Peer Feedback Score Distribution",
     "- **Why this graph?**: To explore how employees rate their colleagues.\n- **Insight**: Peer feedback is generally positive, reflecting a collaborative and supportive environment."),
    ("distribution_performance_score.png", "Performance Score Distribution",
     "- **Why this graph?**: To understand the overall distribution of performance scores across employees.\n- **Insight**: Performance scores are clustered around average and above-average levels."),
    ("overtime_vs_absenteeism.png", "Overtime vs Absenteeism",
     "- **Why this graph?**: To examine the relationship between overtime hours and absenteeism.\n- **Insight**: Employees who log more overtime tend to have higher absenteeism rates, possibly due to burnout."),
    ("pairwise_relationships.png", "Pairwise Relationships",
     "- **Why this graph?**: To visualize relationships between key features.\n- **Insight**: Significant positive relationships exist between salary, performance, and satisfaction."),
    ("performance_by_gender.png", "Performance by Gender",
     "- **Why this graph?**: To explore gender-based performance differences.\n- **Insight**: There are slight variations in performance scores between genders, with no significant disparities."),
    ("value_counts_CLASS.png", "Class Distribution",
     "- **Why this graph?**: To analyze the distribution of employee classes.\n- **Insight**: Employee classes are spread across various levels, with some categories dominating the workforce."),
    ("value_counts_STATUS.png", "Status Distribution",
     "- **Why this graph?**: To evaluate the distribution of employee status (active/inactive).\n- **Insight**: Active employees represent the majority of the workforce."),
    ("value_counts_termreason_desc.png", "Termination Reasons",
     "- **Why this graph?**: To understand the reasons behind employee terminations.\n- **Insight**: Terminations are mostly driven by performance issues or external factors."),
    ("value_counts_termtype_desc.png", "Termination Types",
     "- **Why this graph?**: To evaluate the types of terminations.\n- **Insight**: Most terminations are voluntary, suggesting employees often resign."),
    ("work_life_balance_by_department.png", "Work-Life Balance by Department",
     "- **Why this graph?**: To analyze work-life balance scores across departments.\n- **Insight**: Work-life balance scores vary significantly between departments, reflecting differences in job demands."),
    ("work_life_balance_vs_satisfaction.png", "Work-Life Balance vs Satisfaction",
     "- **Why this graph?**: To examine how work-life balance affects employee satisfaction.\n- **Insight**: Employees with better work-life balance tend to have higher satisfaction scores.")
]


    for file_name, title, insight in image_files:
        st.subheader(title)
        st.image(os.path.join(EDA_OUTPUT_PATH, file_name))
        st.markdown(insight)

    # Insights summary
    st.subheader("Insights Summary")
    st.markdown("""
    The following insights were gathered from the EDA phase:
    - **Categorical Features**:
        - Termination reasons (`termreason_desc`) provide insight into employee turnover.
        - Gender distribution shows a minor disparity, with slightly more male employees.
        - Department-wise analysis reveals operational departments have higher staff counts.
    - **Numerical Features**:
        - Salary and performance score show considerable variability, indicating potential predictors for satisfaction or attrition.
        - Outliers in overtime hours and absenteeism rates highlight areas for targeted intervention.
    - **Visualizations**:
        - Work-life balance and satisfaction are positively correlated.
        - Promotions are strongly tied to high performance scores but are infrequent.
        - Absenteeism rates are higher for employees with excessive overtime hours.
    These insights provide a foundation for building predictive models and improving workforce management.
    """)

# -------------------------- Data Visualization Section --------------------------
elif section == "Data Visualization":
    st.header("Advanced Data Visualizations")
    st.subheader("Dataset Preview")
    if data is not None:
        st.dataframe(data.head())

    st.markdown("### Visualization Insights")
    visualization_files = [
    ("distribution_age.png", "Age Distribution",
     "- **Why this graph?**: To analyze the age distribution of employees.\n- **Insight**: The dataset showcases a wide age range, with most employees in their prime working age."),
    ("distribution_salary.png", "Salary Distribution",
     "- **Why this graph?**: To understand the salary distribution across employees.\n- **Insight**: The distribution of salaries shows a higher frequency of lower income levels."),
    ("salary_by_job_title.png", "Salary by Job Title",
     "- **Why this graph?**: To compare salaries across different job titles.\n- **Insight**: Certain job titles, such as managers, have significantly higher salaries compared to others."),
    ("salary_vs_age.png", "Salary vs Age",
     "- **Why this graph?**: To examine the relationship between salary and age.\n- **Insight**: Salaries tend to increase with age, reflecting greater experience and seniority."),
    ("satisfaction_vs_salary.png", "Employee Satisfaction vs Salary",
     "- **Why this graph?**: To study how salary influences employee satisfaction.\n- **Insight**: Higher salaries generally correlate with increased employee satisfaction."),
    ("performance_vs_salary_hike.png", "Performance vs Salary Hike",
     "- **Why this graph?**: To analyze how performance impacts salary hikes.\n- **Insight**: Better performance correlates strongly with higher salary hikes."),
    ("distribution_employee_satisfaction_score.png", "Employee Satisfaction Score Distribution",
     "- **Why this graph?**: To assess the distribution of employee satisfaction scores.\n- **Insight**: Employees rate their satisfaction towards the higher end, indicating overall contentment."),
    ("distribution_length_of_service.png", "Length of Service Distribution",
     "- **Why this graph?**: To understand the distribution of employee tenure.\n- **Insight**: Most employees have moderate lengths of service, with fewer long-term employees."),
    ("distribution_performance_score.png", "Performance Score Distribution",
     "- **Why this graph?**: To evaluate the distribution of performance scores.\n- **Insight**: Performance scores cluster around above-average ratings, with fewer employees at the lower end."),
    ("outliers_salary.png", "Outliers in Salary",
     "- **Why this graph?**: To identify salary outliers within the workforce.\n- **Insight**: Some employees earn significantly higher than the average, highlighting outliers in compensation."),
    ("work_life_balance_vs_satisfaction.png", "Work-Life Balance vs Satisfaction",
     "- **Why this graph?**: To study the impact of work-life balance on employee satisfaction.\n- **Insight**: Work-life balance directly impacts employee satisfaction positively, with better balance leading to higher satisfaction."),
    ("distribution_salary.png", "Salary Distribution",
     "- **Why this graph?**: To understand the distribution of salaries within the organization.\n- **Insight**: Salaries are positively skewed, with most employees earning in the lower percentiles."),
    ("geospatial_salaries_job_titles.png", "Geospatial Salaries by Job Titles",
     "- **Why this graph?**: To visualize salary distributions across different job titles.\n- **Insight**: Salaries vary by job title and location, with managers earning the highest salaries."),
]

    for file_name, title, insight in visualization_files:
        st.subheader(title)
        st.image(os.path.join(DATA_VISUALIZATION_PATH, file_name))
        st.markdown(insight)

# -------------------------- Deep Exploratory Data Analysis (EDA) --------------------------
# -------------------------- Deep Exploratory Data Analysis (EDA) --------------------------
elif section == "Deep EDA":
    st.header("Deep Exploratory Data Analysis (EDA)")
    
    # Introduction to Deep EDA
    st.subheader("Introduction to Deep EDA")
    st.markdown("""
    In this advanced EDA step, we go beyond basic analysis to explore deeper patterns in the dataset. This phase includes:
    
    1. **Correlations and Heatmaps**:
       - Analyze relationships between numerical variables using correlation matrices and heatmaps.
       - Identify strong predictors for outcomes such as attrition and performance.
    
    2. **Cluster Analysis**:
       - Group employees based on attributes like satisfaction, absenteeism, and performance.
       - Employ clustering techniques like K-Means to uncover employee behavior segments.
    
    3. **Anomaly Detection**:
       - Detect outliers in the dataset using techniques like Isolation Forests and Z-score analysis.
       - Highlight unusual cases such as extreme absenteeism or low engagement levels.
    
    4. **Time-Series Analysis**:
       - Explore temporal trends in attrition, promotions, and performance.
       - Utilize interactive visualizations for better trend comprehension.
    """)

    # Correlations and Heatmaps
    st.subheader("Correlations and Heatmaps")
    st.markdown("We explore numerical relationships in the dataset to identify patterns and predictors.")

    # Load numerical columns
    numerical_cols = [
        'age', 'length_of_service', 'STATUS_YEAR', 'GENERAL APPEARANCE', 
        'MANNER OF SPEAKING', 'PHYSICAL CONDITION', 'MENTAL ALERTNESS', 
        'SELF-CONFIDENCE', 'ABILITY TO PRESENT IDEAS', 'COMMUNICATION SKILLS', 
        'Student Performance Rating', 'salary', 'performance_score', 
        'manager_rating', 'self_rating', 'work_life_balance_score', 
        'overtime_hours', 'employee_satisfaction_score', 'salary_hike_percent', 
        'post_promotion_performance', 'peer_feedback_score', 'absenteeism_rate', 
        'turnover_risk_index', 'engagement_index', 'Attrition'
    ]
    numerical_data = data[numerical_cols]

    # Correlation Matrix
    corr_matrix = numerical_data.corr()
    st.write("### Correlation Matrix")
    st.dataframe(corr_matrix)

    # Heatmap of Correlations
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    st.markdown("""
    - **Insights**:
        - Strong positive correlation between `work_life_balance_score` and `employee_satisfaction_score`.
        - Negative correlation between `absenteeism_rate` and `performance_score` indicates absenteeism affects performance.
        - Weak correlation between `salary` and `attrition`, suggesting salary isn't the sole determinant of turnover.
    """)

    # Cluster Analysis
    st.subheader("Cluster Analysis")
    st.markdown("Clustering reveals groups of employees with similar characteristics.")

    # Preparing data for clustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    cluster_features = ['employee_satisfaction_score', 'absenteeism_rate', 'performance_score']
    cluster_data = numerical_data[cluster_features].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    cluster_data['Cluster'] = cluster_labels

    st.write("### Clustering Results")
    st.dataframe(cluster_data)

    st.write("### Cluster Visualization")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=cluster_data, x='employee_satisfaction_score', y='performance_score', hue='Cluster', palette='viridis', ax=ax)
    st.pyplot(fig)
    st.markdown("""
    - **Insights**:
        - Cluster 0: High satisfaction, high performance, and low absenteeism employees.
        - Cluster 1: Moderate satisfaction and performance employees.
        - Cluster 2: Employees with low satisfaction and performance, higher absenteeism.
    """)

    # Anomaly Detection
    st.subheader("Anomaly Detection")
    st.markdown("We use advanced methods to detect outliers or unusual employee behaviors.")

    # Using Isolation Forest
    from sklearn.ensemble import IsolationForest

    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    anomaly_scores = isolation_forest.fit_predict(numerical_data)
    numerical_data['Anomaly'] = anomaly_scores

    st.write("### Anomalies Detected")
    anomalies = numerical_data[numerical_data['Anomaly'] == -1]
    st.dataframe(anomalies)
    st.markdown("""
    - **Insights**:
        - Employees with unusually high absenteeism or low satisfaction scores are flagged as anomalies.
        - These employees may need targeted interventions or further investigation.
    """)

    # Time-Series Analysis
    st.subheader("Time-Series Analysis")
    st.markdown("Analyzing trends over time for metrics like attrition and performance.")

    # Example: Attrition Trends
    time_series_cols = ['STATUS_YEAR', 'Attrition']
    time_series_data = data[time_series_cols].groupby('STATUS_YEAR').mean().reset_index()

    st.write("### Attrition Trends Over Time")
    fig = px.line(time_series_data, x='STATUS_YEAR', y='Attrition', title="Attrition Over Time", markers=True)
    st.plotly_chart(fig)
    st.markdown("""
    - **Insights**:
        - Attrition shows a declining trend over the years, indicating successful retention strategies.
    """)

    # Example: Performance Trends
    st.write("### Performance Trends Over Time")
    performance_trends = data[['STATUS_YEAR', 'performance_score']].groupby('STATUS_YEAR').mean().reset_index()
    fig = px.line(performance_trends, x='STATUS_YEAR', y='performance_score', title="Performance Trends Over Time", markers=True)
    st.plotly_chart(fig)
    st.markdown("""
    - **Insights**:
        - Employee performance scores have generally improved over the years, reflecting organizational growth.
    """)

    # Insights Summary
    st.subheader("Insights Summary")
    st.markdown("""
    From this advanced EDA phase, the following insights were uncovered:
    
    1. **Correlations**:
        - High satisfaction correlates with better performance and lower absenteeism.
        - Weak correlation between salary and attrition suggests other factors influence turnover.
    
    2. **Clusters**:
        - Three employee groups identified: high performers, average performers, and low performers.
        - Low-performing employees often have higher absenteeism and lower satisfaction.
    
    3. **Anomalies**:
        - Detected employees with extreme absenteeism and low engagement as anomalies.
        - These cases may require personalized HR strategies.
    
    4. **Time-Series Trends**:
        - Attrition rates have declined over time, indicating improved retention.
        - Employee performance has shown steady growth, reflecting successful initiatives.
    """)

# -------------------------- Conclusion Section --------------------------
elif section == "Conclusion":
    st.header("Conclusion")
    st.markdown("""
        ### Complete Overview:
        - Conducted comprehensive EDA to understand dataset structure and key metrics.
        - Generated statistical summaries for categorical and numerical features.
        - Created advanced visualizations to uncover trends and patterns.
        - Identified key relationships between features, such as performance and salary.
        - Prepared data insights for predictive modeling and decision-making.
    """)

    # Summary Table
    summary_data = {
        "Step": ["EDA", "Categorical Stats", "Numerical Stats", "Data Visualization"],
        "Process": [
            "Explored dataset structure and metrics.",
            "Analyzed key categorical features for distribution and uniqueness.",
            "Summarized numerical features to detect variability and outliers.",
            "Created visualizations to uncover relationships and trends."
        ],
        "Insights": [
            "Identified key patterns in absenteeism, satisfaction, and performance.",
            "Revealed key segmentation in employee statuses and termination reasons.",
            "Detected salary skewness and relationships with performance metrics.",
            "Highlighted trends like higher salaries leading to better satisfaction."
        ],
        "Outputs": [
            "Key visualizations and insights summary.",
            "Categorical stats report.",
            "Numerical stats report.",
            "Advanced visualizations for analysis."
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)
