import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.express as px

# Paths
FEATURE_IMPORTANCE_OUTPUT_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/feature_importance'
DATA_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv'

# Load data
data = pd.read_csv(DATA_PATH)
rf_importance_status_file = os.path.join(FEATURE_IMPORTANCE_OUTPUT_PATH, 'rf_feature_importance_status.csv')
xgb_importance_status_file = os.path.join(FEATURE_IMPORTANCE_OUTPUT_PATH, 'xgb_feature_importance_status.csv')
rf_importance_attrition_file = os.path.join(FEATURE_IMPORTANCE_OUTPUT_PATH, 'rf_feature_importance_attrition.csv')
xgb_importance_attrition_file = os.path.join(FEATURE_IMPORTANCE_OUTPUT_PATH, 'xgb_feature_importance_attrition.csv')

rf_status_plot_file = os.path.join(FEATURE_IMPORTANCE_OUTPUT_PATH, 'Top_20_Features_RandomForest_(STATUS).png')
xgb_status_plot_file = os.path.join(FEATURE_IMPORTANCE_OUTPUT_PATH, 'Top_20_Features_XGBoost_(STATUS).png')
rf_attrition_plot_file = os.path.join(FEATURE_IMPORTANCE_OUTPUT_PATH, 'Top_20_Features_RandomForest_(Attrition).png')
xgb_attrition_plot_file = os.path.join(FEATURE_IMPORTANCE_OUTPUT_PATH, 'Top_20_Features_XGBoost_(Attrtion).png')

rf_importances_status = pd.read_csv(rf_importance_status_file)
xgb_importances_status = pd.read_csv(xgb_importance_status_file)
rf_importances_attrition = pd.read_csv(rf_importance_attrition_file)
xgb_importances_attrition = pd.read_csv(xgb_importance_attrition_file)

# Streamlit App Configuration
st.set_page_config(page_title="Employment Analysis - Feature Importance", layout="wide")

# Footer for the App
st.markdown("""
    <footer style="
        position: fixed; 
        bottom: 0; 
        width: 100%; 
        background-color: #1E1E1E; 
        padding: 3px; 
        text-align: left;">
        <p style="color: #B0B0B0; font-family: 'Arial', sans-serif; font-size: 14px; margin: 0;">
            Created By Nidish (1BG22CS095) from CSE Undergrad @ BNMIT
        </p>
    </footer>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Select a section",
    ["Introduction", "Status", "Attrition", "Conclusion"]
)

# Introduction Section
if option == "Introduction":
    st.title("Employment Analysis - Feature Importance Insights")
    st.header("Introduction")
    st.write("""
    **Objective**: To analyze employment data for identifying key factors influencing **STATUS** and **Attrition** using advanced feature importance techniques.
    
    - **Load Data**: Import and explore the dataset containing employee information.
    - **Target Variables**: STATUS (performance indicators) and Attrition (employee turnover).
    - **Analyze Features**: Use RandomForest and XGBoost to evaluate feature importance.
    - **Comparative Analysis**: Compare feature importances for STATUS and Attrition.
    - **Visualization**: Display results with visually appealing graphs and tables.
    """)

    st.write("### Dataset Overview")
    st.write("""
    The dataset contains features like salary, tenure, work-life balance, performance feedback, and more.
    These factors play a vital role in understanding employee behavior and performance.
    """)
    st.dataframe(data.head(), use_container_width=True)

# Status Section
elif option == "Status":
    st.header("Feature Importance Analysis - STATUS")
    st.write("""
    **About the Target Variable: STATUS**
    - STATUS represents performance indicators such as employee ratings and scores.
    - In Phase 1, Step 5 (Data Splitting), we defined STATUS as a key target variable for predicting employee performance.
    - This analysis evaluates the influence of various features on STATUS using two models: RandomForest and XGBoost.
    """)

    # Test 1: RandomForest
    st.write("### Test 1: RandomForest Analysis for STATUS")
    st.write("""
    **Why RandomForest?**
    - RandomForest is a robust ensemble learning method.
    - It effectively handles non-linear relationships in the data.
    - Provides reliable feature importance scores.
    - Reduces overfitting through multiple decision trees.
    - Suitable for datasets with diverse features.
    """)

    st.write("**Target Variable Considerations:**")
    st.write("""
    - For RandomForest analysis, we focused on identifying top predictors of STATUS.
    - Variables like salary, tenure, and work-life balance were prioritized.
    - Feature importance was calculated for each predictor.
    - Insights were visualized to aid understanding.
    - Results were saved as `.csv` and `.png` files for further exploration.
    """)
    st.dataframe(rf_importances_status, use_container_width=True)
    st.image(rf_status_plot_file, caption="Top 20 Features - RandomForest (STATUS)", use_container_width=True)

    st.write("**Insights from RandomForest Analysis:**")
    st.write("""
    1. Financial factors like salary are the most influential predictors.
    2. Tenure highlights the role of organizational experience in driving STATUS.
    3. Work-life balance metrics show moderate impact.
    4. Feedback mechanisms such as peer reviews contribute significantly.
    5. RandomForest provides a balanced perspective on STATUS determinants.
    """)

    # Test 2: XGBoost
    st.write("### Test 2: XGBoost Analysis for STATUS")
    st.write("""
    **Why XGBoost?**
    - XGBoost leverages gradient boosting for superior accuracy.
    - Handles feature interactions effectively.
    - Suitable for uncovering complex relationships in the data.
    - Offers faster computations and better optimization.
    - Provides detailed insights into key predictors of STATUS.
    """)

    st.write("**Target Variable Considerations:**")
    st.write("""
    - XGBoost analysis focuses on the same target variable (STATUS) as RandomForest.
    - Key predictors were evaluated with gradient boosting algorithms.
    - Insights were visualized and saved for comparison.
    - The `.csv` and `.png` files capture detailed feature importance rankings.
    - This approach highlights nuanced contributions from diverse predictors.
    """)
    st.dataframe(xgb_importances_status, use_container_width=True)
    st.image(xgb_status_plot_file, caption="Top 20 Features - XGBoost (STATUS)", use_container_width=True)

    st.write("**Insights from XGBoost Analysis:**")
    st.write("""
    1. Self-assessments and peer feedback emerge as critical predictors.
    2. Work-life balance plays a pivotal role in determining STATUS.
    3. Experience-related metrics like tenure hold significant weight.
    4. XGBoost uncovers department-specific patterns affecting STATUS.
    5. Results validate the multifaceted nature of employee performance.
    """)

    # Comparison of Tests
    st.write("### Comparison of STATUS Tests")
    st.write("""
    **Analysis:**
    - Comparing RandomForest and XGBoost offers a comprehensive understanding of STATUS predictors.
    - Both models agree on key factors like salary and tenure.
    - XGBoost provides additional insights into lifestyle and departmental trends.
    - Differences between the models highlight their complementary strengths.
    """)

    comparison_status_data = {
        "Feature": rf_importances_status["Feature"],
        "RandomForest Importance": rf_importances_status["Importance"],
        "XGBoost Importance": xgb_importances_status["Importance"]
    }
    comparison_status_df = pd.DataFrame(comparison_status_data)
    st.table(comparison_status_df)

    st.write("**Insights from Test Comparison:**")
    st.write("""
    1. Both models emphasize salary and tenure as primary STATUS drivers.
    2. XGBoost identifies additional lifestyle-related insights not captured by RandomForest.
    3. RandomForest offers more consistent feature rankings, while XGBoost explores complex interactions.
    4. Combining insights ensures a well-rounded understanding of STATUS determinants.
    5. Dual-model analysis enhances the reliability of actionable recommendations.
    """)

# Attrition Section
elif option == "Attrition":
    st.header("Feature Importance Analysis - ATTRITION")
    st.write("""
    **About the Target Variable: ATTRITION**
    - ATTRITION reflects employee turnover and retention within the organization.
    - In Phase 1, Step 5 (Data Splitting), we identified ATTRITION as a critical target variable for workforce planning.
    - Understanding its predictors helps in designing retention strategies and improving employee satisfaction.
    """)

    # Test 1: RandomForest
    st.write("### Test 1: RandomForest Analysis for ATTRITION")
    st.write("""
    **Why RandomForest?**
    - RandomForest is a reliable ensemble method for classification tasks like attrition prediction.
    - It captures complex patterns in the data.
    - Offers a robust way to rank features by their contribution to the target variable.
    - Handles class imbalance effectively.
    - Reduces overfitting through multiple decision trees.
    """)

    st.write("**Target Variable Considerations:**")
    st.write("""
    - For RandomForest, we aimed to identify the top predictors of ATTRITION.
    - Focused on features such as job satisfaction, workload, and career progression.
    - Evaluated feature importance scores to uncover key drivers of employee turnover.
    - Insights were saved as `.csv` and `.png` files for visualization and analysis.
    """)
    st.dataframe(rf_importances_attrition, use_container_width=True)
    st.image(rf_attrition_plot_file, caption="Top 20 Features - RandomForest (ATTRITION)", use_container_width=True)

    st.write("**Insights from RandomForest Analysis:**")
    st.write("""
    1. Job satisfaction and career progression are the strongest predictors of ATTRITION.
    2. Work-life balance plays a critical role in employee retention.
    3. Financial factors like salary influence turnover rates moderately.
    4. Peer feedback and manager ratings also contribute to attrition predictions.
    5. RandomForest highlights both organizational and personal factors affecting ATTRITION.
    """)

    # Test 2: XGBoost
    st.write("### Test 2: XGBoost Analysis for ATTRITION")
    st.write("""
    **Why XGBoost?**
    - XGBoost is a powerful gradient boosting model for classification tasks.
    - Offers enhanced predictive accuracy through gradient boosting.
    - Captures subtle interactions between features.
    - Ideal for datasets with mixed feature types and complex relationships.
    - Provides detailed feature importance rankings.
    """)

    st.write("**Target Variable Considerations:**")
    st.write("""
    - XGBoost was applied to analyze the same target variable (ATTRITION).
    - Focused on uncovering nuanced predictors of turnover.
    - Saved results in `.csv` and `.png` formats for deeper exploration.
    - Generated detailed visualizations for enhanced interpretation.
    - This analysis complements RandomForest results with additional insights.
    """)
    st.dataframe(xgb_importances_attrition, use_container_width=True)
    st.image(xgb_attrition_plot_file, caption="Top 20 Features - XGBoost (ATTRITION)", use_container_width=True)

    st.write("**Insights from XGBoost Analysis:**")
    st.write("""
    1. Job satisfaction and work-life balance emerge as critical predictors.
    2. XGBoost emphasizes department-specific patterns influencing turnover.
    3. Managerial feedback and performance reviews hold substantial weight.
    4. Lifestyle factors like commute time are highlighted as potential contributors.
    5. Insights validate the complex interplay of personal and organizational factors in ATTRITION.
    """)

    # Comparison of Tests
    st.write("### Comparison of ATTRITION Tests")
    st.write("""
    **Analysis:**
    - Comparing RandomForest and XGBoost results provides a comprehensive view of attrition drivers.
    - Both models emphasize job satisfaction and work-life balance as top predictors.
    - RandomForest focuses on broader organizational trends, while XGBoost captures finer feature interactions.
    - The dual-model approach offers robust insights for designing retention strategies.
    """)

    comparison_attrition_data = {
        "Feature": rf_importances_attrition["Feature"],
        "RandomForest Importance": rf_importances_attrition["Importance"],
        "XGBoost Importance": xgb_importances_attrition["Importance"]
    }
    comparison_attrition_df = pd.DataFrame(comparison_attrition_data)
    st.table(comparison_attrition_df)

    st.write("**Insights from Test Comparison:**")
    st.write("""
    1. Both models consistently highlight job satisfaction and work-life balance as key attrition drivers.
    2. RandomForest provides broader trends, while XGBoost identifies nuanced patterns.
    3. Financial and lifestyle factors are secondary but significant contributors.
    4. Combining insights ensures a comprehensive understanding of attrition.
    5. A dual-model analysis is critical for designing effective workforce retention strategies.
    """)

# Conclusion Section
elif option == "Conclusion":
    st.header("Conclusion")
    st.write("""
    **Summary**:
    - Successfully analyzed feature importance for STATUS and Attrition.
    - Used RandomForest and XGBoost to derive actionable insights.
    - Visualized results with comprehensive tables and graphs.
    - Comparative analysis highlighted unique contributions from both models.
    - Findings guide strategies for employee engagement and performance improvement.
    """)
    st.write("### Summary Table")
    summary_data = {
        "Step": ["Load Data", "Feature Importance Analysis", "Visualization", "Comparison", "Insights"],
        "Description": [
            "Loaded and explored employee dataset.",
            "Evaluated STATUS and Attrition predictors.",
            "Visualized feature importance results.",
            "Compared RandomForest and XGBoost outputs.",
            "Derived actionable insights for decision-making."
        ],
        "Output": [
            "Dataset ready for analysis.",
            "Key features identified for both targets.",
            "Graphs and tables generated.",
            "Holistic understanding of models.",
            "Strategies informed by findings."
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)
