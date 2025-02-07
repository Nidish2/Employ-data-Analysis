import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind, levene, shapiro
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# -------------------------- Set Paths --------------------------
OUTPUT_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/hypothesis_testing'
ANOVA_FILE = os.path.join(OUTPUT_PATH, 'anova_performance_department.csv')
CHI_SQUARE_FILE = os.path.join(OUTPUT_PATH, 'chi_squared_work_life_balance_employee_satisfaction_score.txt')
T_TEST_FILE = os.path.join(OUTPUT_PATH, 't_test_salary_gender.txt')
HOMOGENEITY_TEST_FILE = os.path.join(OUTPUT_PATH, 'homogeneity_tests.txt')
NORMALITY_TEST_FILE = os.path.join(OUTPUT_PATH, 'normality_tests.txt')
DATA_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv'

# -------------------------- Load Data --------------------------
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"File not found at {file_path}")
        return None

@st.cache_data
def load_normality_results(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        results = {}
        for line in lines:
            if "Shapiro-Wilk test" in line:
                key, value = line.strip().split(": p_value=")
                feature = key.split(" for ")[1].strip()
                p_value = float(value.strip())
                results[feature] = p_value
        return results
    except FileNotFoundError:
        st.error(f"Normality test results file not found at {file_path}")
        return None

@st.cache_data
def load_anova_results(file_path):
    try:
        anova_results = pd.read_csv(file_path)
        # Replace NaN values with "Not Computed" for display
        anova_results.fillna("Not Computed", inplace=True)
        return anova_results
    except FileNotFoundError:
        st.error(f"ANOVA results file not found at {file_path}")
        return None

# Load transformed data
data = load_data(DATA_PATH)
normality_results = load_normality_results(NORMALITY_TEST_FILE)
anova_results = load_anova_results(ANOVA_FILE)

if data is None:
    st.stop()

# -------------------------- Streamlit UI --------------------------
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
option = ["Introduction", "Hypothesis Tests", "Validation of Assumptions", "Summary", "Conclusion"]
choice = st.sidebar.radio("Go to", option)

if choice == "Introduction":
    st.title("Employment Analysis - Hypothesis Testing Insights")
    st.header("Introduction")
    st.write("""
    - **Objective**: To perform hypothesis testing and validate assumptions on employment data.
    - **Load Data**: Import and load the transformed dataset for analysis.
    - **Perform Hypothesis Testing**: Conduct statistical tests like Chi-Square, T-Test, and ANOVA.
    - **Validate Assumptions**: Check for homogeneity of variances, normality, multicollinearity, and autocorrelation.
    - **Conclusions**: Derive actionable insights and summarize findings.
    """)
    
    st.write("### Phase Objective")
    st.write("""
    In this phase, we aim to:
    - Analyze the employment dataset to uncover patterns and insights.
    - Test hypotheses related to key HR metrics such as promotions, salary differences, and performance scores.
    - Validate assumptions to ensure the reliability and accuracy of statistical tests.
    - Draw actionable conclusions to support decision-making in HR and organizational strategies.
    """)


elif choice == "Hypothesis Tests":
    st.header("Hypothesis Insights and Results")

    # Hypothesis 1: Work-Life Balance and Employee Satisfaction
    st.subheader("Hypothesis 1: Is work-life balance score independent of employee satisfaction score?")
    try:
        with open(os.path.join(OUTPUT_PATH, 'work_life_balance_satisfaction.txt'), 'r') as file:
            work_life_results = file.read()
        st.text(work_life_results)
        st.write("""
        **Features Used**: Work-life balance score, Employee satisfaction score  
        **Test Used**: Pearson Correlation  
        **Output Path**: `work_life_balance_satisfaction.txt`  
        **Insight**:  
        - A strong correlation indicates that work-life balance impacts employee satisfaction.  
        - Helps HR design policies for improving satisfaction through better work-life balance.  
        """)
    except FileNotFoundError:
        st.error(f"File not found: work_life_balance_satisfaction.txt")

    # Hypothesis 2: Salary Differences by Gender
    st.subheader("Hypothesis 2: Are mean salaries for male and female employees equal?")
    try:
        with open(os.path.join(OUTPUT_PATH, 'salary_by_gender.txt'), 'r') as file:
            gender_salary_results = file.read()
        st.text(gender_salary_results)
        st.write("""
        **Features Used**: Gender, Salary  
        **Test Used**: T-Test (Independent Samples)  
        **Output Path**: `salary_by_gender.txt`  
        **Insight**:  
        - Identifies potential gender pay gaps.  
        - Supports HR initiatives to ensure pay equity across genders.  
        """)
    except FileNotFoundError:
        st.error(f"File not found: salary_by_gender.txt")

    # Hypothesis 3: Attrition and Overtime Hours
    st.subheader("Hypothesis 3: Is attrition dependent on overtime hours?")
    try:
        with open(os.path.join(OUTPUT_PATH, 'attrition_overtime.txt'), 'r') as file:
            attrition_overtime_results = file.read()
        st.text(attrition_overtime_results)
        st.write("""
        **Features Used**: Attrition, Overtime Hours  
        **Test Used**: ANOVA  
        **Output Path**: `attrition_overtime.txt`  
        **Insight**:  
        - Highlights the role of overtime hours in influencing attrition rates.  
        - Provides guidance on managing workloads to reduce attrition.  
        """)
    except FileNotFoundError:
        st.error(f"File not found: attrition_overtime.txt")

    # Hypothesis 4: Department and Performance Scores
    st.subheader("Hypothesis 4: Do performance scores differ across departments?")
    if anova_results is not None:
        st.dataframe(anova_results)
        st.write("""
        **Features Used**: Department, Performance Score  
        **Test Used**: ANOVA  
        **Output Path**: `performance_department_anova.csv`  
        **Insight**:  
        - Identifies performance variations between departments.  
        - Helps in resource allocation and addressing department-specific challenges.  
        """)
    else:
        st.error(f"Unable to load ANOVA results from {ANOVA_FILE}")

    # Hypothesis 5: Salary Hike and Turnover Risk
    st.subheader("Hypothesis 5: Is salary hike percentage independent of turnover risk?")
    try:
        with open(os.path.join(OUTPUT_PATH, 'salary_hike_turnover.txt'), 'r') as file:
            salary_hike_results = file.read()
        st.text(salary_hike_results)
        st.write("""
        **Features Used**: Salary Hike Percentage, Turnover Risk Index  
        **Test Used**: Pearson Correlation  
        **Output Path**: `salary_hike_turnover.txt`  
        **Insight**:  
        - Evaluates the impact of salary hikes on employee retention.  
        - Informs strategies to reduce turnover risk.  
        """)
    except FileNotFoundError:
        st.error(f"File not found: salary_hike_turnover.txt")

    # Hypothesis 6: Manager Ratings and Attrition
    st.subheader("Hypothesis 6: Are manager ratings independent of attrition?")
    try:
        with open(os.path.join(OUTPUT_PATH, 'manager_rating_attrition.txt'), 'r') as file:
            manager_rating_results = file.read()
        st.text(manager_rating_results)
        st.write("""
        **Features Used**: Manager Ratings, Attrition  
        **Test Used**: T-Test (Independent Samples)  
        **Output Path**: `manager_rating_attrition.txt`  
        **Insight**:  
        - Analyzes the influence of managerial feedback on attrition rates.  
        - Provides insights for improving managerial practices to retain employees.  
        """)
    except FileNotFoundError:
        st.error(f"File not found: manager_rating_attrition.txt")

    # Hypothesis 7: Absenteeism and Performance Scores
    st.subheader("Hypothesis 7: Is absenteeism rate independent of performance scores?")
    try:
        with open(os.path.join(OUTPUT_PATH, 'absenteeism_performance.txt'), 'r') as file:
            absenteeism_results = file.read()
        st.text(absenteeism_results)
        st.write("""
        **Features Used**: Absenteeism Rate, Performance Score  
        **Test Used**: Pearson Correlation  
        **Output Path**: `absenteeism_performance.txt`  
        **Insight**:  
        - Identifies the impact of absenteeism on performance.  
        - Encourages policies to manage absenteeism effectively.  
        """)
    except FileNotFoundError:
        st.error(f"File not found: absenteeism_performance.txt")

    # Hypothesis 8: Peer Feedback and Post-Promotion Performance
    st.subheader("Hypothesis 8: Are peer feedback scores independent of post-promotion performance?")
    try:
        with open(os.path.join(OUTPUT_PATH, 'peer_feedback_promotion.txt'), 'r') as file:
            peer_feedback_results = file.read()
        st.text(peer_feedback_results)
        st.write("""
        **Features Used**: Peer Feedback Score, Post-Promotion Performance  
        **Test Used**: Pearson Correlation  
        **Output Path**: `peer_feedback_promotion.txt`  
        **Insight**:  
        - Evaluates the effectiveness of peer feedback in predicting post-promotion performance.  
        - Provides guidance on using peer reviews for promotion decisions.  
        """)
    except FileNotFoundError:
        st.error(f"File not found: peer_feedback_promotion.txt")

    # Hypothesis Tests Table
    st.write("### Hypothesis Tests Overview")
    hypothesis_tests_data = {
    "Test Name": [
        "Pearson Correlation (Work-Life Balance & Employee Satisfaction)",
        "T-Test (Salary Differences by Gender)",
        "ANOVA (Attrition & Overtime Hours)",
        "ANOVA (Performance Scores by Department)",
        "Pearson Correlation (Salary Hike & Turnover Risk)",
        "T-Test (Manager Ratings & Attrition)",
        "Pearson Correlation (Absenteeism & Performance Scores)",
        "Pearson Correlation (Peer Feedback & Post-Promotion Performance)"
    ],
    "Description": [
        "Tests for the relationship between work-life balance and employee satisfaction.",
        "Compares mean salaries between male and female employees.",
        "Evaluates whether attrition is related to overtime hours using ANOVA.",
        "Compares performance scores across different departments using ANOVA.",
        "Examines the relationship between salary hike percentage and turnover risk using Pearson correlation.",
        "Compares manager ratings between employees who stay vs. those who leave (attrition).",
        "Assesses the relationship between absenteeism rate and performance score.",
        "Tests the relationship between peer feedback scores and post-promotion performance."
    ],
    "Usage in This Step": [
        "Examined the relationship between work-life balance and employee satisfaction.",
        "Examined salary differences between genders.",
        "Assessed attrition in relation to overtime hours worked.",
        "Analyzed department-based performance variations.",
        "Explored how salary hikes affect turnover risk.",
        "Analyzed how manager ratings influence employee attrition.",
        "Evaluated the impact of absenteeism on performance scores.",
        "Analyzed the relationship between peer feedback and post-promotion performance."
    ],
    "Output": [
        "Work-life balance significantly impacts employee satisfaction.",
        "No significant salary differences found between male and female employees.",
        "Overtime hours have a significant impact on attrition.",
        "Performance scores differ significantly between departments.",
        "No significant relationship between salary hike percentage and turnover risk.",
        "Manager ratings show a significant impact on employee attrition.",
        "Absenteeism is significantly related to performance scores.",
        "Peer feedback scores have a significant relationship with post-promotion performance."
    ],
    "Insight": [
        "Improving work-life balance can lead to higher employee satisfaction.",
        "Salary equity exists between genders, indicating fair pay practices.",
        "Reducing overtime hours may help lower attrition rates.",
        "Department-specific performance strategies can help address performance issues.",
        "Salary hikes alone do not significantly reduce turnover risk.",
        "Improving manager ratings could help reduce employee attrition.",
        "Managing absenteeism could improve employee performance outcomes.",
        "Effective peer feedback can improve post-promotion performance."
    ]
}

    hypothesis_tests_df = pd.DataFrame(hypothesis_tests_data)
    st.table(hypothesis_tests_df)



elif choice == "Validation of Assumptions":
    st.header("Validation of Assumptions")

    # Validation 1: Normality Check for Salary
    st.subheader("Validation 1: Normality Check for Salary (Shapiro-Wilk Test)")
    try:
        with open(os.path.join(OUTPUT_PATH, 'normality_salary.txt'), 'r') as file:
            salary_normality_results = file.read()
        st.text(salary_normality_results)
        st.write("""
        **Insight**:
        - The Shapiro-Wilk test checks if the salary data follows a normal distribution.
        - A significant p-value indicates the data deviates from normality.
        - Ensures the validity of parametric tests using salary data.
        - Provides a basis for selecting the correct statistical methods.
        - Enhances the reliability of conclusions drawn from salary analyses.
        """)
    except FileNotFoundError:
        st.error(f"Normality test results file not found at {os.path.join(OUTPUT_PATH, 'normality_salary')}")

    # Validation 2: Homogeneity of Variance for Performance Scores
    st.subheader("Validation 2: Homogeneity of Variance (Levene's Test) for Performance Scores")
    try:
        with open(os.path.join(OUTPUT_PATH, 'homogeneity_tests.txt'), 'r') as file:
            homogeneity_performance_results = file.read()
        st.text(homogeneity_performance_results)
        st.write("""
        **Insight**:
        - Levene's test checks if variances across department performance scores are equal.
        - A significant result indicates unequal variances, violating ANOVA assumptions.
        - Ensures fairness and reliability of inter-department performance comparisons.
        - Supports accurate allocation of resources based on departmental performance.
        - Helps identify departments with high performance variability.
        """)
    except FileNotFoundError:
        st.error(f"Homogeneity test results file not found at {os.path.join(OUTPUT_PATH, 'homogeneity_tests.txt')}")

    # Validation 3: Normality Check for Performance Improvement
    st.subheader("Validation 3: Normality Check for Performance Improvement (Shapiro-Wilk Test)")
    try:
        with open(os.path.join(OUTPUT_PATH, 'normality_performance_improvement.txt'), 'r') as file:
            performance_improvement_results = file.read()
        st.text(performance_improvement_results)
        st.write("""
        **Insight**:
        - The Shapiro-Wilk test ensures the performance improvement data is normally distributed.
        - A significant p-value indicates deviations from normality.
        - Validates the use of parametric tests for performance improvement analysis.
        - Helps HR design programs to improve employee performance effectively.
        - Provides confidence in performance improvement conclusions.
        """)
    except FileNotFoundError:
        st.error(f"Normality test results file not found at {os.path.join(OUTPUT_PATH, 'normality_performance_improvement.txt')}")

    # Validation 4: Homogeneity of Variances for Salary across Tenure Buckets
    st.subheader("Validation 4: Homogeneity of Variance for Salary Across Tenure Buckets (Levene's Test)")
    try:
        with open(os.path.join(OUTPUT_PATH, 'homogeneity_salary_tenure.txt'), 'r') as file:
            homogeneity_salary_tenure_results = file.read()
        st.text(homogeneity_salary_tenure_results)
        st.write("""
        **Insight**:
        - Levene's test examines if salary variances are equal across tenure groups.
        - A significant result suggests unequal salary distributions among tenure buckets.
        - Identifies discrepancies in salary distribution across employee tenures.
        - Provides insights into fair compensation practices.
        - Aids in making tenure-based salary policies more equitable.
        """)
    except FileNotFoundError:
        st.error(f"Homogeneity test results file not found at {os.path.join(OUTPUT_PATH, 'homogeneity_salary_tenure.txt')}")

    # Validation 5: Multicollinearity Check using VIF
    st.subheader("Validation 5: Multicollinearity Check (Variance Inflation Factor - VIF)")
    try:
        vif_data = pd.read_csv(os.path.join(OUTPUT_PATH, 'vif_multicollinearity_check.csv'))
        st.dataframe(vif_data)
        st.write("""
        **Insight**:
        - VIF identifies multicollinearity among features like salary, performance, and engagement indices.
        - High VIF values (>10) indicate significant multicollinearity, requiring feature reduction.
        - Ensures robust predictive models by avoiding redundant variables.
        - Improves interpretability of relationships between key metrics.
        - Supports better decision-making in HR analytics and predictive modeling.
        """)
    except FileNotFoundError:
        st.error(f"VIF results file not found at {os.path.join(OUTPUT_PATH, 'vif_multicollinearity_check.csv')}")

    # Validation 6: Autocorrelation in Turnover Risk (Durbin-Watson Test)
    st.subheader("Validation 6: Autocorrelation in Turnover Risk (Durbin-Watson Test)")
    try:
        with open(os.path.join(OUTPUT_PATH, 'autocorrelation_attrition.txt'), 'r') as file:
            autocorrelation_results = file.read()
        st.text(autocorrelation_results)
        st.write("""
        **Insight**:
        - Durbin-Watson test checks for autocorrelation in turnover risk index.
        - A statistic near 2 indicates no autocorrelation.
        - Validates the independence of turnover risk data points.
        - Ensures the reliability of turnover predictions.
        - Helps design turnover mitigation strategies based on reliable patterns.
        """)
    except FileNotFoundError:
        st.error(f"Autocorrelation test results file not found at {os.path.join(OUTPUT_PATH, 'autocorrelation_attrition.txt')}")

    # General Conclusion for Validation
    st.write("""
    **General Insights**:
    - Validating assumptions ensures the robustness and reliability of statistical analyses.
    - The tests conducted confirm or highlight deviations from key statistical assumptions.
    - Provides a strong foundation for hypothesis testing and predictive modeling.
    - Improves the interpretability of results and the accuracy of insights.
    - Aids in making informed decisions on policies, compensation, and resource allocation.
    """)

    st.write("### Validation of Assumptions Overview")
    validation_tests_data = {
    "Test Name": [
        "Shapiro-Wilk Test for Salary",
        "Levene's Test for Performance Scores",
        "Shapiro-Wilk Test for Performance Improvement",
        "Levene's Test for Salary Across Tenure Buckets",
        "Variance Inflation Factor (VIF)",
        "Durbin-Watson Test for Autocorrelation"
    ],
    "Description": [
        "Tests if the salary data follows a normal distribution.",
        "Checks if performance score variances are equal across departments.",
        "Tests if performance improvement data is normally distributed.",
        "Checks if salary variances are equal across tenure groups.",
        "Measures multicollinearity among key numerical features.",
        "Tests for autocorrelation in the turnover risk index."
    ],
    "Usage in This Step": [
        "Validate the suitability of salary data for parametric tests.",
        "Ensure fairness in inter-department performance score comparisons.",
        "Validate the use of parametric tests for performance improvement data.",
        "Ensure equitable salary distribution across tenure groups.",
        "Identify and address multicollinearity in predictive models.",
        "Validate the independence of turnover risk data points."
    ],
    "Output": [
        "Salary data deviates from normality.",
        "Significant differences in department performance score variances.",
        "Performance improvement data is normally distributed.",
        "Significant variance differences in salary across tenure buckets.",
        "Low multicollinearity among numerical features.",
        "No significant autocorrelation in turnover risk index."
    ],
    "Insight": [
        "Non-normality of salary data requires non-parametric tests or transformations.",
        "Departmental performance score analysis needs adjusted methods (e.g., Welch's ANOVA).",
        "Performance improvement data is suitable for parametric analysis.",
        "Address disparities in salary distribution across tenure groups.",
        "Multicollinearity is under control, ensuring robust modeling and interpretability.",
        "Turnover risk index analysis is reliable and robust with no autocorrelation."
    ]
}

    validation_tests_df = pd.DataFrame(validation_tests_data)
    st.table(validation_tests_df)


elif choice == "Summary":
    st.header("Summary of Analysis")

    summary_data = {
        "Step": ["Load Data", "Perform Hypothesis Testing", "Validate Assumptions", "Visualize Results", "Conclusions"],
        "Description": [
            "Imported and loaded the transformed dataset for analysis.",
            "Conducted Chi-Square, T-Test, and ANOVA to test key hypotheses.",
            "Validated statistical assumptions using tests for homogeneity, normality, multicollinearity, and autocorrelation.",
            "Created visualizations to highlight key results and insights.",
            "Summarized findings to draw actionable insights."
        ],
        "Insights": [
            "Dataset is ready for comprehensive analysis.",
            "Identified significant relationships and trends in employment data.",
            "Ensured reliability and validity of statistical tests.",
            "Revealed meaningful patterns and relationships in the data.",
            "Provided a strong foundation for HR strategies and decision-making."
        ],
        "Output": [
            "Dataset loaded successfully.",
            "Statistical test results obtained.",
            "Assumptions validated for accurate analysis.",
            "Visualizations created to enhance understanding.",
            "Comprehensive insights generated for strategic decisions."
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

elif choice == "Conclusion":
    st.header("Conclusion")
    st.write("""
    In this phase, we conducted a comprehensive analysis of employment data to uncover critical insights.
    
    ### Key Steps Undertaken:
    - **Loading Data**: Successfully imported and loaded the transformed dataset for hypothesis testing and validation.
    - **Hypothesis Testing**: Performed Chi-Square, T-Test, and ANOVA to explore relationships and differences in employment data.
    - **Assumption Validation**: Conducted detailed checks for homogeneity, normality, multicollinearity, and autocorrelation to ensure test reliability.
    - **Visualizations**: Created clear and effective visual representations of results to facilitate better understanding.
    - **Summary and Insights**: Derived actionable conclusions to support HR strategies and decision-making.

    ### Overall Insights:
    - The analyses revealed significant relationships and differences that are vital for understanding organizational dynamics.
    - Validation tests ensured the robustness and accuracy of all statistical conclusions.
    - Findings provide actionable guidance for fair promotions, equitable salary policies, and effective performance management.
    - This analysis forms a strong foundation for future predictive modeling and advanced HR analytics.
    """)
# -------------------------- End of Code --------------------------