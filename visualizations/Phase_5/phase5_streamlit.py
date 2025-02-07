import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Constants
MODEL_SELECTION_OUTPUT_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/model_selection'
MODEL_FOLDERS = ['Attrition', 'STATUS']

# Page setup
st.set_page_config(
    page_title="Model Development and Evaluation Dashboard",
    layout="wide"
)

st.title("🚀 Employment Analysis: Model Development and Evaluation")

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

# Sidebar for navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio(
    "Select a view:",
    ["Introduction", "Attrition", "Status", "Conclusion"]
)

# Utility functions
def load_classification_report(file_path):
    """Load classification report from a text file."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "Classification report not found."

def load_confusion_matrix(file_path):
    """Load confusion matrix from a CSV file."""
    try:
        return pd.read_csv(file_path, index_col=0).values
    except FileNotFoundError:
        st.error(f"Confusion matrix file not found: {file_path}")
        return None

def plot_confusion_matrix(matrix, model_name, labels, y_true=None, y_pred=None):
    """Plot the confusion matrix and analyze actual vs. predicted with advanced graphs."""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import streamlit as st

    # Ensure y_true and y_pred are numeric
    if y_true is not None and y_pred is not None:
        y_true = pd.to_numeric(y_true, errors='coerce')
        y_pred = pd.to_numeric(y_pred, errors='coerce')

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pd.DataFrame(matrix, index=labels, columns=labels), annot=True, fmt='g', cmap='Blues', ax=ax)
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    st.pyplot(fig)

    if y_true is not None and y_pred is not None:
        differences = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
        differences['Difference'] = differences['Actual'] - differences['Predicted']
        differences['Difference'] = differences['Difference'].clip(-1, 1)  # Restrict to -1, 0, 1

        # 1. Line Plot: Actual vs Predicted (Two Colors)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(differences.index, differences['Actual'], color='green', alpha=0.7, linestyle='-', marker='o', label='Actual')
        ax.plot(differences.index, differences['Predicted'], color='red', alpha=0.7, linestyle='-', marker='x', label='Predicted')
        ax.set_title(f'{model_name} - Line Plot: Actual (Green) vs Predicted (Red)', fontsize=14)
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.legend()
        st.pyplot(fig)

        # 2. Bar Plot: Frequency of Deviations (-1, 0, 1)
        fig, ax = plt.subplots(figsize=(10, 7))
        deviation_counts = differences['Difference'].value_counts().sort_index()
        sns.barplot(x=deviation_counts.index, y=deviation_counts.values, palette='coolwarm', ax=ax)
        ax.set_title(f'{model_name} - Frequency of Deviations (-1, 0, 1)', fontsize=14)
        ax.set_xlabel('Difference (Actual - Predicted)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        st.pyplot(fig)

        # 3. Heatmap: Deviations (-1, 0, 1)
        fig, ax = plt.subplots(figsize=(10, 7))
        deviation_matrix = pd.crosstab(differences['Actual'], differences['Predicted'])
        sns.heatmap(deviation_matrix, annot=True, fmt='g', cmap='coolwarm', ax=ax)
        ax.set_title(f'{model_name} - Heatmap of Deviations', fontsize=14)
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Actual Values', fontsize=12)
        st.pyplot(fig)

def display_test_results(model_name, target_variable, labels, folder):
    """Display results for a specific test (model)."""
    st.subheader(f"Results for {model_name}")
    
    # File paths
    classification_report_path = os.path.join(
        MODEL_SELECTION_OUTPUT_PATH, folder, f"{model_name}_{target_variable}_classification_report.txt"
    )
    confusion_matrix_path = os.path.join(
        MODEL_SELECTION_OUTPUT_PATH, folder, f"{model_name}_{target_variable}_confusion_matrix.csv"
    )
    actual_values_path = os.path.join(
        MODEL_SELECTION_OUTPUT_PATH, folder, f"{model_name}_{target_variable}_y_true.csv"
    )
    predicted_values_path = os.path.join(
        MODEL_SELECTION_OUTPUT_PATH, folder, f"{model_name}_{target_variable}_y_pred.csv"
    )
    
    # Load and display classification report
    classification_report = load_classification_report(classification_report_path)
    st.markdown("#### Classification Report")
    st.text(classification_report)
    
    # Load actual and predicted values, ensuring they are 1D arrays
    actual_values = pd.read_csv(actual_values_path, header=None).squeeze()  # Ensures 1D Series
    predicted_values = pd.read_csv(predicted_values_path, header=None).squeeze()  # Ensures 1D Series
    
    # Load and display confusion matrix
    confusion_matrix = load_confusion_matrix(confusion_matrix_path)
    st.markdown("#### Confusion Matrix")
    if confusion_matrix is not None:
        plot_confusion_matrix(confusion_matrix, model_name, labels, y_true=actual_values, y_pred=predicted_values)

    
    # Insights
    st.markdown("#### Insights")
    if model_name == "DecisionTree":
        st.write("""
        - **DecisionTree Observations**:
            - Easy to interpret and visualize results.
            - **Accuracy**: 86.72%, suitable for simple use cases.
            - Recall for less-employable: 82%, demonstrating moderate sensitivity.
            - Precision for employable: 87%, effectively reducing false positives.
            - Provides quick insights but may lack generalization.
        """)
    elif model_name == "LightGBM":
        st.write("""
        - **LightGBM Observations**:
            - Extremely efficient and scalable for large datasets.
            - **Accuracy**: 87.93%, highest among all models.
            - ROC AUC: 96.02%, showcasing superior classification capability.
            - Recall and precision are well-balanced for both classes.
            - Recommended for applications requiring high performance.
        """)
    elif model_name == "RandomForest":
        st.write("""
        - **RandomForest Observations**:
            - Robust against overfitting due to ensemble nature.
            - **Accuracy**: 85.85%, consistent performance across metrics.
            - High precision for less-employable: 90%.
            - Recall for less-employable: 75%, indicating room for improvement.
            - Best suited for reliable feature importance evaluation.
        """)
    elif model_name == "XGBoost":
        st.write("""
        - **XGBoost Observations**:
            - Advanced boosting techniques ensure top-notch performance.
            - **Accuracy**: 87.93%, matching LightGBM.
            - ROC AUC: 96.02%, indicating excellent predictive capability.
            - Handles complex patterns effectively.
            - Balanced metrics across all categories.
        """)

if options == "Introduction":
    st.header("Introduction")
    st.write("""
    **About this phase**:
    - **Objective**: Develop and evaluate machine learning models for employment analysis.
    - **Steps**:
      1. Model Selection
      2. Model Training
      3. Model Evaluation
      4. Model Improvement
      5. Insights and Comparison
      
      The steps undertaken in this phase include:
      1. **Model Selection**: Identified suitable machine learning models for the dataset.
      2. **Model Training**: Trained models to capture patterns in the data.
      3. **Model Evaluation**: Evaluated performance metrics like accuracy, precision, and recall.
      4. **Insights and Comparison**: Analyzed results to provide actionable insights.
      5. **Final Recommendations**: Selected the best-performing models for deployment.      
    - **Phase Objectives**:
      1. Identify suitable machine learning models for employment data.
      2. Train and fine-tune selected models to improve performance.
      3. Evaluate models based on accuracy, precision, and recall metrics.
      4. Enhance model performance through iterative improvements.
      5. Compare model results to derive actionable insights for deployment.
    ### Dataset Overview
    - Contains employee data with attributes such as salary, tenure, and work-life balance.
    - Two primary target variables:
      1. **Attrition** : (0 or 1):
        0 (No Attrition): Indicates that the employee is likely to stay with the organization.
        1 (Attrition): Indicates that the employee is likely to leave the organization.
      2. **Status**: (active or terminated) 
        Employment status of an individual.
        Mapped into binary values: 1 for 'terminated', 0 for 'active'
    - Aim: Identify key factors influencing employment outcomes.
    """)
    st.dataframe(pd.read_csv('C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv').head(), use_container_width=True)

elif options == "Attrition":
    st.header("Attrition Analysis")
    st.write("""
    **About Attrition Target Variable**:
    - Represents the likelihood of an employee leaving the organization.
    - Essential for understanding and implementing retention strategies.
    - Focused on predicting employee attrition using various machine learning models.
    """)

    # Test: DecisionTree
    st.markdown("### Test 1: DecisionTree")
    st.write("""
    - DecisionTree is a simple yet effective classification model.
    - Suitable for understanding feature interactions and importance.
    """)
    display_test_results("DecisionTree", "Attrition", ["Staying", "Leaving"], MODEL_FOLDERS[0])

    # Test: LightGBM
    st.markdown("### Test 2: LightGBM")
    st.write("""
    - LightGBM excels in efficiency and scalability.
    - Handles large datasets and provides superior classification performance.
    """)
    display_test_results("LightGBM", "Attrition", ["Staying", "Leaving"], MODEL_FOLDERS[0])

    # Test: RandomForest
    st.markdown("### Test 3: RandomForest")
    st.write("""
    - RandomForest is an ensemble-based learning model.
    - Combines multiple decision trees to enhance performance and reduce overfitting.
    """)
    display_test_results("RandomForest", "Attrition", ["Staying", "Leaving"], MODEL_FOLDERS[0])

    # Test: XGBoost
    st.markdown("### Test 4: XGBoost")
    st.write("""
    - XGBoost stands for Extreme Gradient Boosting.
    - Known for its predictive power and ability to handle complex patterns.
    """)
    display_test_results("XGBoost", "Attrition", ["Staying", "Leaving"], MODEL_FOLDERS[0])

    # Performance Comparison
    st.markdown("### Model Performance Comparison")
    performance_file = os.path.join(MODEL_SELECTION_OUTPUT_PATH, MODEL_FOLDERS[0], "attrition_model_performance_comparison.csv")
    performance_df = pd.read_csv(performance_file)

    st.dataframe(performance_df)
    st.markdown("#### Performance Metrics Chart")
    st.bar_chart(performance_df.set_index("Model")[["Accuracy", "Precision"]])

    # Final Insights
    st.markdown("### Final Insights")
    st.write("""
    - DecisionTree: Simplicity and interpretability make it useful for initial analysis.
    - LightGBM: Best performance in accuracy and ROC AUC, highly efficient.
    - RandomForest: Robust and reliable but slightly lower recall.
    - XGBoost: Matches LightGBM in accuracy and ROC AUC, excels in complex scenarios.
    - **Recommendation**: LightGBM and XGBoost are top choices for deployment.
    """)


# Status Section
elif options == "Status":
    st.header("Status Analysis")
    st.write("""
    **About Status Target Variable**:
    - The target variable "Status" indicates the employment status of an individual.
    - Objective: Predict the employment status based on various attributes and provide actionable insights.
    - Evaluated using four machine learning models: DecisionTree, LightGBM, RandomForest, and XGBoost.
    """)

    # Test: DecisionTree
    st.markdown("### Test 1: DecisionTree")
    st.write("""
    - DecisionTree is a simple yet interpretable model.
    - Useful for understanding key feature interactions affecting employment status.
    """)
    display_test_results("DecisionTree", "Status", ["Terminated", "Active"], MODEL_FOLDERS[1])

    # Test: LightGBM
    st.markdown("### Test 2: LightGBM")
    st.write("""
    - LightGBM is known for its efficiency and high performance.
    - Suitable for large datasets and provides fast predictions.
    """)
    display_test_results("LightGBM", "Status", ["Terminated", "Active"], MODEL_FOLDERS[1])

    # Test: RandomForest
    st.markdown("### Test 3: RandomForest")
    st.write("""
    - RandomForest is an ensemble learning model that improves performance by combining multiple decision trees.
    - Robust against overfitting and effective in identifying important features.
    """)
    display_test_results("RandomForest", "Status", ["Terminated", "Active"], MODEL_FOLDERS[1])

    # Test: XGBoost
    st.markdown("### Test 4: XGBoost")
    st.write("""
    - XGBoost is an advanced gradient boosting algorithm.
    - Known for its superior predictive accuracy and ability to handle complex data patterns.
    """)
    display_test_results("XGBoost", "Status", ["Terminated", "Active"], MODEL_FOLDERS[1])

    # Performance Comparison
    st.markdown("### Model Performance Comparison")
    performance_file = os.path.join(MODEL_SELECTION_OUTPUT_PATH, MODEL_FOLDERS[1], "status_model_performance_comparison.csv")
    performance_df = pd.read_csv(performance_file)

    st.dataframe(performance_df)
    st.markdown("#### Performance Metrics Chart")
    st.bar_chart(performance_df.set_index("Model")[["Accuracy", "Precision"]])

    # Final Insights
    st.markdown("### Final Insights")
    st.write("""
    - DecisionTree: A straightforward model offering good interpretability but moderate accuracy.
    - LightGBM: Top performer in accuracy and ROC AUC, suitable for large-scale deployments.
    - RandomForest: Reliable with consistent metrics but slightly lower recall for the unemployed class.
    - XGBoost: Matches LightGBM in performance, handles complex data effectively.
    - **Recommendation**: LightGBM and XGBoost are the best candidates for production deployment.
    """)

elif options == "Conclusion":
    st.header("Conclusion")
    st.write("""
    In this phase, we developed, trained, evaluated, and compared multiple machine learning models to analyze employment data. 
    The steps involved include:
    - **Model Selection**: DecisionTree, LightGBM, RandomForest, and XGBoost were identified as suitable models.
    - **Model Training**: Models were trained using employee data to predict outcomes.
    - **Model Evaluation**: Performance was assessed using metrics such as accuracy, precision, recall, and confusion matrices.
    - **Insights and Comparison**: Models were compared to derive meaningful insights for deployment.
    """)

    # Attrition Tests Table
    st.markdown("### Attrition Test Summary")
    attrition_table = pd.DataFrame({
        "Test Name": ["DecisionTree", "LightGBM", "RandomForest", "XGBoost"],
        "Description": [
            "Simple and interpretable tree-based model.",
            "Fast, efficient, and scalable gradient boosting model.",
            "Ensemble model combining decision trees.",
            "Advanced gradient boosting model with strong predictive power."
        ],
        "Usage in This Step": [
            "Predict attrition probability and key patterns.",
            "Efficient prediction with high accuracy.",
            "Balance precision and recall for attrition analysis.",
            "Handle complex data patterns for attrition."
        ],
        "Output": [
            "Accuracy: 86.72%, Recall: 82%, Precision: 87%",
            "Accuracy: 87.93%, ROC AUC: 96.02%",
            "Accuracy: 85.85%, Precision (low attrition): 90%",
            "Accuracy: 87.93%, Recall (low attrition): 78%"
        ],
        "Insight": [
            "Good interpretability but lower recall for attrition.",
            "Top accuracy and balanced metrics make it a top choice.",
            "Reliable but slightly lower recall for less-employable.",
            "Matches LightGBM but better with complex patterns."
        ]
    })
    st.dataframe(attrition_table)

    # Status Tests Table
    st.markdown("### Status Test Summary")
    status_table = pd.DataFrame({
        "Test Name": ["DecisionTree", "LightGBM", "RandomForest", "XGBoost"],
        "Description": [
            "Simple and interpretable tree-based model.",
            "Fast, efficient, and scalable gradient boosting model.",
            "Ensemble model combining decision trees.",
            "Advanced gradient boosting model with strong predictive power."
        ],
        "Usage in This Step": [
            "Predict employment status and feature importance.",
            "Efficient prediction with high accuracy.",
            "Balance precision and recall for employment status.",
            "Handle complex data patterns for employment."
        ],
        "Output": [
            "Accuracy: 85.45%, Recall: 79%, Precision: 88%",
            "Accuracy: 88.20%, ROC AUC: 95.90%",
            "Accuracy: 86.00%, Precision (unemployed): 91%",
            "Accuracy: 88.15%, Recall (unemployed): 80%"
        ],
        "Insight": [
            "Interpretable but moderate recall for unemployed.",
            "Best accuracy and balanced metrics for large-scale usage.",
            "Reliable with robust metrics but slightly lower recall.",
            "Strong predictive capabilities for complex data."
        ]
    })
    st.dataframe(status_table)

    # Summary Table
    st.markdown("### Summary Table")
    summary_table = pd.DataFrame({
        "Step": [
            "Model Selection",
            "Model Training",
            "Model Evaluation",
            "Insights and Comparison"
        ],
        "Description": [
            "Identified suitable models for the dataset.",
            "Trained models using employee data.",
            "Evaluated performance using accuracy, precision, and recall.",
            "Derived insights from results and compared models."
        ],
        "Insights": [
            "DecisionTree, LightGBM, RandomForest, and XGBoost selected.",
            "Models captured key patterns in employment data.",
            "LightGBM and XGBoost consistently performed best.",
            "LightGBM and XGBoost recommended for deployment."
        ],
        "Output": [
            "4 models selected.",
            "Models trained successfully.",
            "Classification reports and confusion matrices generated.",
            "Final recommendations for deployment."
        ]
    })
    st.dataframe(summary_table)
