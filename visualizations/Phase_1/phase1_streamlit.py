import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths for data
RAW_DATA_PATH = "C:/Users/nidis/Documents/Employment_Analysis/data/Employee_data_raw.csv"
CLEANED_DATA_PATH = "C:/Users/nidis/Documents/Employment_Analysis/data/cleaned_data.csv"
EXCEL_VALIDATION_PATH = "C:/Users/nidis/Documents/Employment_Analysis/data/cleaned_data_validation.xlsx"
FEATURE_ENGINEERED_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/feature_engineered_data.csv'
CORRELATION_MATRIX_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/feature_engineering/correlation_matrix.png'
TRANSFORMED_DATA_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv'
TRAIN_FEATURES_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/train_features.csv'
TEST_FEATURES_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/test_features.csv'
TRAIN_TARGET_STATUS_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/train_target_status.csv'
TEST_TARGET_STATUS_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/test_target_status.csv'
TRAIN_TARGET_ATTRITION_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/train_target_attrition.csv'
TEST_TARGET_ATTRITION_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/test_target_attrition.csv'

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

# App Title
st.title("Phase 1: Employment Analysis - Data Preparation and Insights")

# Sidebar Navigation
st.sidebar.title("Navigation")
options = [
    "Introduction", 
    "Data Collection", 
    "Data Cleaning", 
    "Feature Engineering", 
    "Data Transformation",
    "Data Splitting",
    "Conclusion"
]
choice = st.sidebar.radio("Go to", options)

# Introduction
if choice == "Introduction":
    st.subheader("Project Overview")
    st.markdown("""
        - **Goal**: Analyze employee data to provide actionable insights on work-life balance, salary trends, and attrition trends.
        - **Phase 1**: Focuses on data preparation, cleaning, feature engineering, transformation, and splitting for deeper analysis.
        - **Data Splitting**:
          - Splits data into training and testing datasets for predictive modeling.
          - Ensures balanced class distributions using stratified sampling.
        - This project ensures data quality and insights to support decision-making.
        - Outputs from each step are saved for reproducibility and interactive analysis.
        - Navigate through the sections to explore detailed steps and outcomes.
    """)


# Data Collection
elif choice == "Data Collection":
    st.subheader("Step 1: Data Collection")
    st.markdown(f"**Raw data is loaded from:** `{RAW_DATA_PATH}`")
    raw_data = pd.read_csv(RAW_DATA_PATH)
    st.write("### Raw Data Preview", raw_data.head())

    st.markdown("#### Process Summary:")
    st.markdown("""
        - Collected data from multiple sources, including HR systems and employee self-assessments.
        - Features include demographics, salary details, performance ratings, and promotion history.
        - The dataset has missing values and inconsistent formats, requiring cleaning and standardization.
    """)

    st.markdown("#### Key Achievements:")
    st.markdown("""
        - Consolidated data into a single dataset for further processing.
        - Ensured all necessary features for analysis are included.
        - Identified issues like missing data for subsequent resolution.
    """)

# Data Cleaning
elif choice == "Data Cleaning":
    st.subheader("Step 2: Data Cleaning")
    st.markdown(f"**Cleaned data is loaded from:** `{CLEANED_DATA_PATH}`")
    cleaned_data = pd.read_csv(CLEANED_DATA_PATH)
    st.write("### Cleaned Data Preview", cleaned_data.head())

    st.markdown(f"**Validation file for cleaning saved at:** `{EXCEL_VALIDATION_PATH}`")

    st.markdown("#### Process Summary:")
    st.markdown("""
        - Removed duplicates and standardized column formats.
        - Imputed missing values where possible or removed irrelevant columns.
        - Validated the cleaning process with a saved Excel report.
    """)

    st.markdown("#### Key Achievements:")
    st.markdown("""
        - Enhanced dataset quality for reliable analysis.
        - Ensured consistency across columns and resolved missing values.
        - Created a reproducible cleaning workflow with documented outputs.
    """)

    st.write("### Missing Values Summary")
    missing_values = cleaned_data.isnull().sum()
    st.write(missing_values[missing_values > 0])

elif choice == "Feature Engineering":
    st.subheader("Step 3: Feature Engineering")
    st.markdown(f"**Feature-engineered data is loaded from:** `{FEATURE_ENGINEERED_PATH}`")
    
    # Load the updated feature-engineered dataset
    feature_data = pd.read_csv(FEATURE_ENGINEERED_PATH)
    st.write("### Feature-Engineered Data Preview", feature_data.head())
    
    st.markdown("#### Process Summary:")
    st.markdown("""
        - **Derived New Features for Enhanced Analysis:**
            - **Tenure Bucket**: Categorized employees based on their length of service into buckets like '<2 years', '2-5 years', etc.
            - **Months Since Last Promotion**: Quantified time since the last promotion in months to identify career growth stagnation.
            - **Turnover Risk Index**: Calculated to identify employees at higher risk of turnover based on age, salary hikes, and length of service.
            - **Engagement Index**: A composite metric reflecting overall engagement derived from employee satisfaction, work-life balance, and peer feedback.
            - **Performance Improvement**: Difference between post-promotion performance and prior performance, highlighting development over time.
            - **Absenteeism Category**: Categorized absenteeism rates into buckets like 'Low', 'Moderate', 'High', and 'Critical'.
        - Analyzed relationships between features using a correlation matrix.
        - Saved outputs for reproducibility and visualization.
    """)

    st.markdown("#### Correlation Matrix and Insights:")
    
    # Display the correlation matrix
    if os.path.exists(CORRELATION_MATRIX_PATH):
        st.image(CORRELATION_MATRIX_PATH, caption="Correlation Matrix of Key Features")
        st.markdown("""
            - **Purpose of Correlation Matrix**: To visualize the relationships among variables, helping to identify dependencies and impactful patterns.
            - **Insights from the New Dataset**:
                1. **Tenure and Performance Correlation**:
                   - A positive correlation suggests that employees with longer tenure are generally more experienced and perform better.
                2. **Months Since Last Promotion**:
                   - Negative correlation with employee satisfaction and engagement indices indicates that delayed promotions lower morale.
                3. **Turnover Risk Index**:
                   - High-risk employees often show lower satisfaction scores and engagement metrics, making them critical focus areas for retention strategies.
                4. **Engagement Index**:
                   - Strongly correlated with satisfaction and work-life balance, validating it as a holistic measure of employee well-being.
                5. **Salary Hike and Performance Improvement**:
                   - Positive correlation indicates that salary increments often reflect and result in improved performance.
                6. **Absenteeism**:
                   - Weak correlation with performance measures highlights independence, making it a key feature for absenteeism-specific modeling.
                7. **Weakly Correlated Features**:
                   - Features like gender or department showed limited correlation with performance, indicating independence from certain demographic factors.
        """)
    else:
        st.warning("Correlation matrix image not found!")

    st.markdown("#### Range of Values for Derived Features:")
    st.markdown("""
        - **Tenure Bucket**: Categories like '<2 years', '2-5 years', '5-10 years', etc.
        - **Months Since Last Promotion**: Ranges from 0 months to 182 months.
        - **Turnover Risk Index**: Values range from 10.2 to 270.0, calculated as a combination of age, salary hikes, and length of service.
        - **Engagement Index**: Scores range from 11.4 to 36.57, with higher values indicating stronger engagement.
        - **Performance Improvement**: Ranges from -10 to 110, highlighting the degree of improvement or decline post-promotion.
        - **Absenteeism Category**: Categorized into 'Low', 'Moderate', 'High', and 'Critical'.
    """)

# Data Transformation
elif choice == "Data Transformation":
    st.subheader("Step 4: Data Transformation")
    st.markdown(f"**Transformed data is loaded from:** `{TRANSFORMED_DATA_PATH}`")
    transformed_data = pd.read_csv(TRANSFORMED_DATA_PATH)
    st.write("### Transformed Data Preview", transformed_data.head())

    st.markdown("#### Process Summary:")
    st.markdown("""
        - **Derived Attrition**: Calculated based on `terminationdate_key`, `STATUS`, and `termreason_desc`. Attrition is set to 1 if the employee has been terminated due to layoff, resignation, or retirement, otherwise 0.
        - **Handled Missing Values**:
            - Numeric values imputed using KNNImputer to consider patterns in nearby data points.
            - Categorical values imputed with the most frequent value (mode).
        - **Preserved Data Integrity**:
            - Critical numerical columns retain original ranges and integer data types.
        - **Encoded Categorical Features**:
            - Converted selected categorical columns into numeric codes using Label Encoding for model compatibility.
        - **Saved Transformed Data**: Final dataset saved to `{TRANSFORMED_DATA_PATH}` for further analysis.
    """)

    st.write("### Insights from Numerical Feature Distributions:")
    numeric_columns = transformed_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        st.write(f"**{col} Distribution**")
        fig, ax = plt.subplots()
        sns.histplot(transformed_data[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}", fontsize=14, color='darkblue')
        st.pyplot(fig)
        st.markdown(f"""
            - **Why this graph?**: Visualizes the distribution of `{col}` to analyze normalization and detect outliers.
            - **Insights**:
              - Determines if `{col}` follows a normal distribution.
              - Highlights potential skewness or outliers for corrective actions.
        """)

    # Additional Insights
    st.markdown("### Key Observations and Insights:")
    st.markdown("""
        - **Attrition Calculation**:
            - Attrition values are correctly derived as a mix of 0s and 1s based on the termination criteria.
            - Employees with valid termination reasons (layoff, resignation, retirement) are marked as Attrition = 1, ensuring accurate representation of workforce turnover.
        - **Numerical Features**:
            - Critical ranges such as EmployeeID, age, salary, and performance scores remain intact after imputation and scaling.
        - **Categorical Features**:
            - Features like department_name, job_title, and gender_full are now machine-readable without altering their unique categories.
        - **Data Integrity**:
            - Integer columns like overtime_hours, working_hours, and absenteeism_rate maintain their original data types, preventing data distortion.
        - **Overall**:
            - Data transformation ensures compatibility with machine learning models while preserving original data distributions and integrity.
    """)

elif choice == "Data Splitting":
    st.subheader("Step 5: Data Splitting")

    # ---------------------- Process Summary ----------------------
    st.markdown("### Process Summary")
    st.write("""
    In this step, the transformed dataset was split into training and testing sets for two target variables: **STATUS** and **Attrition**. 
    Here's an overview of the process:

    - **Target Variables**:
      - **STATUS**: Mapped to binary values (1 for TERMINATED, 0 for ACTIVE).
      - **Attrition**: Already binary (1 for Attrition, 0 otherwise).
    - **Feature Variables**: All columns except STATUS and Attrition were used as features.
    - **Splitting Methodology**:
      - **Stratified Splitting** was used for both STATUS and Attrition targets to maintain class balance in the training and testing sets.
    - **Outputs**:
      - Separate training and testing datasets for features and targets:
        - Training and testing feature datasets.
        - Training and testing target datasets for **STATUS** and **Attrition**.
    """)

    # ---------------------- Load Datasets ----------------------
    try:
        # Load the datasets created during the data splitting step
        X_train = pd.read_csv(TRAIN_FEATURES_PATH)
        X_test = pd.read_csv(TEST_FEATURES_PATH)
        y_train_status = pd.read_csv(TRAIN_TARGET_STATUS_PATH)
        y_test_status = pd.read_csv(TEST_TARGET_STATUS_PATH)
        y_train_attrition = pd.read_csv(TRAIN_TARGET_ATTRITION_PATH)
        y_test_attrition = pd.read_csv(TEST_TARGET_ATTRITION_PATH)
    except FileNotFoundError as e:
        st.error("Error loading the datasets. Ensure the data splitting step has been executed correctly.")
        st.error(e)

    # ---------------------- Derived Datasets ----------------------
    st.markdown("### Derived Datasets")
    st.write("Below are the datasets derived during the data splitting step:")

    # Features (X_train and X_test)
    st.markdown("#### Feature Datasets:")
    st.write("""
    - **Training Features (X_train)**:
      - Shape: {}
      - Used for training models for both **STATUS** and **Attrition** targets.
    - **Testing Features (X_test)**:
      - Shape: {}
      - Used for evaluating models on unseen data.
    """.format(X_train.shape, X_test.shape))

    # Display feature datasets
    if st.checkbox("Show Training Features (X_train)"):
        st.dataframe(X_train.head())
    if st.checkbox("Show Testing Features (X_test)"):
        st.dataframe(X_test.head())

    # Targets for STATUS
    st.markdown("#### Target Datasets for STATUS:")
    st.write("""
    - **Training Targets (y_train_status)**:
      - Shape: {}
      - Contains the binary values for STATUS: 1 (TERMINATED) and 0 (ACTIVE).
    - **Testing Targets (y_test_status)**:
      - Shape: {}
      - Contains the binary STATUS values for evaluation.
    """.format(y_train_status.shape, y_test_status.shape))

    # Display STATUS target datasets
    if st.checkbox("Show Training Targets (y_train_status)"):
        st.dataframe(y_train_status.head())
    if st.checkbox("Show Testing Targets (y_test_status)"):
        st.dataframe(y_test_status.head())

    # Targets for Attrition
    st.markdown("#### Target Datasets for Attrition:")
    st.write("""
    - **Training Targets (y_train_attrition)**:
      - Shape: {}
      - Contains the binary values for Attrition: 1 for Attrition and 0 otherwise.
    - **Testing Targets (y_test_attrition)**:
      - Shape: {}
      - Contains the binary Attrition values for evaluation.
    """.format(y_train_attrition.shape, y_test_attrition.shape))

    # Display Attrition target datasets
    if st.checkbox("Show Training Targets (y_train_attrition)"):
        st.dataframe(y_train_attrition.head())
    if st.checkbox("Show Testing Targets (y_test_attrition)"):
        st.dataframe(y_test_attrition.head())

    # ---------------------- Key Observations and Insights ----------------------
    st.markdown("### Key Observations and Insights")
    st.write("""
    - **Feature Dataset Observations**:
      - Both training and testing feature datasets retain the same structure across the splits, ensuring consistent feature usage.
      - Features are independent of both STATUS and Attrition targets.
    
    - **STATUS Target Observations**:
      - The STATUS target was successfully mapped to binary values:
        - 1 represents TERMINATED.
        - 0 represents ACTIVE.
      - The training and testing datasets for STATUS maintain the original class distribution, ensuring balanced model training and evaluation.

    - **Attrition Target Observations**:
      - The Attrition target is binary, with 1 indicating Attrition and 0 otherwise.
      - Training and testing datasets for Attrition also maintain balanced class distributions.

    - **Data Integrity**:
      - No missing or erroneous values were found in the STATUS and Attrition columns.
      - Stratified sampling was used for both targets, preserving their class proportions in the splits.

    - **Flexibility**:
      - The independent splitting for STATUS and Attrition allows separate model training for each target, tailored to different business objectives.
    """)


# Conclusion
elif choice == "Conclusion":
    st.subheader("Conclusion")
    st.markdown("""
        ### Complete Overview:
        - The project focused on preparing, analyzing, and splitting employee data for actionable insights.
        - Achieved high-quality data through cleaning, transformation, and feature engineering.
        - Data splitting ensured readiness for predictive modeling with balanced and stratified datasets.
        - Derived new features and maintained class balance to enhance understanding and model performance.
    """)

    # Summary Table
    summary_data = {
        "Step": [
            "Data Collection",
            "Data Cleaning",
            "Feature Engineering",
            "Data Transformation",
            "Data Splitting"
        ],
        "Process": [
            "Consolidated raw data from multiple sources.",
            "Resolved missing values and standardized formats.",
            "Created new features and analyzed relationships.",
            "Normalized and encoded data for analysis.",
            "Split data into training and testing sets for STATUS and Attrition targets."
        ],
        "Insights": [
            "Identified key issues like missing values.",
            "Improved data consistency and readiness.",
            "Highlighted key trends and relationships.",
            "Prepared data for predictive modeling.",
            "Ensured class balance and model-ready datasets."
        ],
        "Outputs": [
            "Raw dataset.",
            "Cleaned dataset with validation report.",
            "Feature-engineered dataset and correlation matrix.",
            "Transformed dataset with normalized features.",
            "Training and testing datasets for STATUS and Attrition."
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)
