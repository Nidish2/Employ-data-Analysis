import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Configure the Streamlit page
st.set_page_config(
    page_title="Employee Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Title and description
st.title("Employee Data Analysis")
st.markdown("""
This dashboard provides real-time insights into employee satisfaction, salary hikes, work-life balance, 
promotions, and other critical factors affecting the workforce. Explore trends, analyze performance, 
and gain actionable insights with dynamic visualizations and predictive modeling.
""")

# File path to the dataset
DATA_PATH = r"C:\Users\nidis\Documents\DS\DS_Projects\Employment_Analysis\data\Employee_data_raw.csv"

# Load data function with caching
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Load the data
data = load_data(DATA_PATH)

# Sidebar options
st.sidebar.title("Options")
st.sidebar.subheader("Data Filters")

# Sidebar filters
gender_filter = st.sidebar.selectbox("Filter by Gender", ["All", "Male", "Female"])
business_unit_filter = st.sidebar.selectbox(
    "Filter by Business Unit", 
    ["All"] + data['BUSINESS_UNIT'].unique().tolist()
)
status_year_filter = st.sidebar.slider(
    "Filter by Status Year",
    int(data["STATUS_YEAR"].min()),
    int(data["STATUS_YEAR"].max()),
    (int(data["STATUS_YEAR"].min()), int(data["STATUS_YEAR"].max()))
)

# Additional filters based on sample data
performance_score_filter = st.sidebar.slider(
    "Filter by Performance Score", 0, 100, (0, 100)
)
manager_rating_filter = st.sidebar.slider(
    "Filter by Manager Rating", -5.0, 5.0, (-5.0, 5.0)
)
self_rating_filter = st.sidebar.slider(
    "Filter by Self Rating", 0.0, 5.0, (0.0, 5.0)
)
work_life_balance_score_filter = st.sidebar.slider(
    "Filter by Work-Life Balance Score", 0, 10, (0, 10)
)
overtime_hours_filter = st.sidebar.slider(
    "Filter by Overtime Hours", 0, 50, (0, 50)
)
employee_satisfaction_score_filter = st.sidebar.slider(
    "Filter by Employee Satisfaction Score", 0, 100, (0, 100)
)
salary_hike_percent_filter = st.sidebar.slider(
    "Filter by Salary Hike Percent", 0, 50, (0, 50)
)
post_promotion_performance_filter = st.sidebar.slider(
    "Filter by Post-Promotion Performance", -5, 5, (-5, 5)
)

# Apply filters
filtered_data = data.copy()

if gender_filter != "All":
    filtered_data = filtered_data[filtered_data["gender_full"] == gender_filter]

if business_unit_filter != "All":
    filtered_data = filtered_data[filtered_data["BUSINESS_UNIT"] == business_unit_filter]

filtered_data = filtered_data[
    (filtered_data["STATUS_YEAR"] >= status_year_filter[0]) & 
    (filtered_data["STATUS_YEAR"] <= status_year_filter[1]) &
    (filtered_data["performance_score"].between(performance_score_filter[0], performance_score_filter[1])) &
    (filtered_data["manager_rating"].between(manager_rating_filter[0], manager_rating_filter[1])) &
    (filtered_data["self_rating"].between(self_rating_filter[0], self_rating_filter[1])) &
    (filtered_data["work_life_balance_score"].between(work_life_balance_score_filter[0], work_life_balance_score_filter[1])) &
    (filtered_data["overtime_hours"].between(overtime_hours_filter[0], overtime_hours_filter[1])) &
    (filtered_data["employee_satisfaction_score"].between(employee_satisfaction_score_filter[0], employee_satisfaction_score_filter[1])) &
    (filtered_data["salary_hike_percent"].between(salary_hike_percent_filter[0], salary_hike_percent_filter[1])) &
    (filtered_data["post_promotion_performance"].between(post_promotion_performance_filter[0], post_promotion_performance_filter[1]))
]

# Data preview
st.write("### Data Preview")
st.dataframe(filtered_data.head())

# Key metrics
st.write("### Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Average Age", round(filtered_data["age"].mean(), 2))

with col2:
    st.metric("Average Salary", f"${round(filtered_data['salary'].mean(), 2)}")

with col3:
    st.metric("Average Tenure (Years)", round(filtered_data["length_of_service"].mean(), 2))

with col4:
    st.metric("Overall Employee Satisfaction", round(filtered_data["employee_satisfaction_score"].mean(), 2))

# Section for visualizations
st.write("### Visualizations")

# 1. Age Distribution
st.write("#### Age Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_data["age"], bins=20, kde=True, color="skyblue", ax=ax)
ax.set_title("Age Distribution of Employees")
st.pyplot(fig)

# 2. Salary vs. Performance
st.write("#### Salary vs. Performance")
fig, ax = plt.subplots()
if "gender_full" in filtered_data.columns:
    gender_palette = {"Male": "blue", "Female": "pink"}  # Corrected palette
    sns.scatterplot(
        x="salary", y="performance_score", hue="gender_full",
        data=filtered_data, palette=gender_palette, ax=ax
    )
else:
    sns.scatterplot(x="salary", y="performance_score", data=filtered_data, ax=ax)
ax.set_title("Salary vs Performance")
ax.set_xlabel("Salary")
ax.set_ylabel("Performance Score")
st.pyplot(fig)

# 3. Work-Life Balance
st.write("#### Work-Life Balance Score Distribution")
fig, ax = plt.subplots()
sns.boxplot(x="BUSINESS_UNIT", y="work_life_balance_score", data=filtered_data, ax=ax)
ax.set_title("Work-Life Balance Score by Business Unit")
ax.set_ylabel("Work-Life Balance Score")
ax.set_xlabel("Business Unit")
plt.xticks(rotation=45)
st.pyplot(fig)

# 4. Employee Satisfaction Analysis
st.write("#### Employee Satisfaction by Business Unit")
department_satisfaction = (
    filtered_data.groupby("BUSINESS_UNIT")["employee_satisfaction_score"]
    .mean()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots()
department_satisfaction.plot(kind="bar", color="orange", ax=ax)
ax.set_title("Average Employee Satisfaction by Business Unit")
ax.set_ylabel("Satisfaction Score")
ax.set_xlabel("Business Unit")
plt.xticks(rotation=45)
st.pyplot(fig)

# 5. Overtime Hours vs. Satisfaction
st.write("#### Overtime Hours vs. Employee Satisfaction")
fig, ax = plt.subplots()
if "gender_full" in filtered_data.columns:
    sns.scatterplot(
        x="overtime_hours", y="employee_satisfaction_score",
        hue="gender_full", data=filtered_data, palette=gender_palette, ax=ax
    )
else:
    sns.scatterplot(x="overtime_hours", y="employee_satisfaction_score", data=filtered_data, ax=ax)
ax.set_title("Overtime Hours vs Employee Satisfaction")
ax.set_xlabel("Overtime Hours")
ax.set_ylabel("Employee Satisfaction Score")
st.pyplot(fig)

# 6. Employee Performance Over Time
st.write("#### Employee Performance Over Time")
fig, ax = plt.subplots()
if "gender_full" in filtered_data.columns:
    sns.lineplot(
        x="STATUS_YEAR", y="performance_score", hue="gender_full",
        data=filtered_data, palette=gender_palette, ax=ax
    )
else:
    sns.lineplot(x="STATUS_YEAR", y="performance_score", data=filtered_data, ax=ax)
ax.set_title("Employee Performance Over Time")
ax.set_xlabel("Status Year")
ax.set_ylabel("Performance Score")
st.pyplot(fig)

# 7. Correlation Heatmap

# Label encoding for categorical variables
le = LabelEncoder()
for column in filtered_data.select_dtypes(include=['object']).columns:
    filtered_data[column] = le.fit_transform(filtered_data[column].astype(str))

# Dropping non-numeric columns for correlation calculation
numeric_data = filtered_data.select_dtypes(include=[np.number]).dropna()

# Check if 'attrition' column is present before performing correlation analysis
if 'attrition' in numeric_data.columns:
    corr = numeric_data.drop(columns=['attrition']).corr()  # Exclude 'attrition' for correlation analysis
else:
    corr = numeric_data.corr()  # If 'attrition' not found, use all numeric data
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
# Correlation Heatmap
st.write("#### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(20, 14))

# Check if 'attrition' column is present for correlation calculation
if 'attrition' in filtered_data.columns:
    corr = filtered_data.drop(columns=['attrition']).corr()  # Exclude 'attrition' for correlation analysis
else:
    corr = filtered_data.corr()  # If 'attrition' not found, use all numeric data

# Plot the heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt=".2f", cbar=True, linewidths=0.5, linecolor='white')
ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust tick label size

# Set title and display the plot
ax.set_title("Correlation Heatmap", fontsize=14)
st.pyplot(fig)


# Predictive Modeling
# Predictive Modeling
st.write("### Predictive Modeling")

# Check if 'attrition' column exists
if 'attrition' in filtered_data.columns:
    X = filtered_data.drop(columns=["attrition"])  # Exclude the 'attrition' column from features
    y = filtered_data["attrition"]
else:
    # If 'attrition' column does not exist, we use a different set of columns for modeling
    X = filtered_data.drop(columns=["age", "length_of_service", "city_name", "department_name", "job_title",
                                    "store_name", "gender_full", "termreason_desc", "termtype_desc", "STATUS_YEAR",
                                    "STATUS", "BUSINESS_UNIT", "GENERAL APPEARANCE", "MANNER OF SPEAKING",
                                    "PHYSICAL CONDITION", "MENTAL ALERTNESS", "SELF-CONFIDENCE", "ABILITY TO PRESENT IDEAS",
                                    "COMMUNICATION SKILLS", "Student Performance Rating", "CLASS", "salary",
                                    "last_promotion_date", "performance_score", "manager_rating", "self_rating",
                                    "work_life_balance_score", "overtime_hours", "working_hours",
                                    "employee_satisfaction_score", "salary_hike_percent", "post_promotion_performance",
                                    "peer_feedback_score", "absenteeism_rate"])  # Use these as features
    y = filtered_data["STATUS"]  # Assuming `STATUS` is a target variable to be predicted

# If X and y are valid, proceed with model training
if X is not None and y is not None:
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Model Predictions
    y_pred = rf_model.predict(X_test)

    # Model Performance
    st.write("### Model Performance")
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100
    st.write(f"Accuracy: {accuracy:.2f}%")
    
    # Additional Metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = auc(*roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])[:2])
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_fig, cm_ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=cm_ax)
    cm_ax.set_title("Confusion Matrix")
    cm_ax.set_xlabel("Predicted")
    cm_ax.set_ylabel("True")
    st.pyplot(cm_fig)
    
    # Error Rate
    error_rate = 1 - accuracy / 100
    
    # Evaluation Summary Table
    eval_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "Error Rate"],
        "Value": [accuracy, precision, recall, f1, roc_auc, error_rate]
    }
    eval_df = pd.DataFrame(eval_data)
    st.write(eval_df)

    # Insights
    st.write("""
    **Insights**:
    - The predictive model achieved an accuracy of {:.2f}%. This indicates that the model is relatively effective in predicting employee attrition.
    - Precision, recall, and F1 Score are key indicators of model quality. Precision is {:.2f}, which means that {:.2f}% of the positive predictions were actually correct. Recall is {:.2f}, indicating that {:.2f}% of the true positives were correctly identified. The F1 Score of {:.2f} balances precision and recall, reflecting the model's overall performance.
    - The ROC AUC score of {:.2f} indicates the model's ability to distinguish between classes, with a higher value suggesting better performance.
    - The confusion matrix highlights areas where the model misclassified employees, helping to identify potential improvements.
    - The error rate of {:.2f} indicates that {:.2f}% of predictions were incorrect, suggesting room for enhancement in the model's accuracy.
    """.format(accuracy, precision, precision * 100, recall, recall * 100, f1, roc_auc, error_rate, error_rate * 100))
    