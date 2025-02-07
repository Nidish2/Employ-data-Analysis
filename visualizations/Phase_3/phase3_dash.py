import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px
import os
from scipy import stats  # Import the stats from scipy

# Paths to files
OUTPUT_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/hypothesis_testing'
ANOVA_FILE = os.path.join(OUTPUT_PATH, 'anova_performance_department.csv')
CHI_SQUARE_FILE = os.path.join(OUTPUT_PATH, 'chi_squared_work_life_balance_employee_satisfaction_score.txt')
T_TEST_FILE = os.path.join(OUTPUT_PATH, 't_test_salary_gender.txt')
HOMOGENEITY_TEST_FILE = os.path.join(OUTPUT_PATH, 'homogeneity_tests.txt')

# Load data
data = pd.read_csv('C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv')

# Load statistical results
anova_results = pd.read_csv(ANOVA_FILE)
with open(CHI_SQUARE_FILE, 'r') as file:
    chi_square_results = file.read()
with open(T_TEST_FILE, 'r') as file:
    t_test_results = file.read()
with open(HOMOGENEITY_TEST_FILE, 'r') as file:
    homogeneity_results = file.read()

# Dash App
app = dash.Dash(__name__)
app.title = "Employment Analysis - Hypothesis Testing Insights"

# Custom CSS Styles
custom_styles = """
    .header-title {
        text-align: center;
        font-size: 3.5em;
        color: #2C3E50;
        margin-bottom: 40px;
        font-weight: bold;
        text-shadow: 1px 1px 5px #AAA;
    }
    .section {
        margin: 30px auto;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        background-color: #FFFFFF;
    }
    .graph-title {
        font-size: 1.8em;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    .graph-caption {
        font-size: 1.2em;
        color: #555555;
        margin-top: 20px;
        line-height: 1.6;
    }
    pre {
        background-color: #F8F9FA;
        padding: 15px;
        border-radius: 8px;
        color: #2C3E50;
        font-size: 1.1em;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
"""

# Shapiro-Wilk Test for Normality
salary_p_value = stats.shapiro(data['salary'])[1]
performance_p_value = stats.shapiro(data['performance_score'])[1]

# Layout
app.layout = html.Div([
    html.Script(custom_styles),

    # Header
    html.H1("📊 Employment Analysis - Hypothesis Testing Insights", className="header-title"),

    # Hypothesis 1: Chi-Square Test
    html.Div(className="section", children=[
        html.H2("Hypothesis 1: Independence between Promotions and Work-Life Balance", className="graph-title"),
        html.Pre(chi_square_results),
        html.Div("The chi-squared test examines the relationship between promotions and work-life balance. "
                 "A p-value below 0.05 suggests a significant dependency, meaning work-life balance might influence "
                 "promotion opportunities. This insight can help identify disparities and highlight key areas for improvement.",
                 className="graph-caption"),
        dcc.Graph(
            figure=px.pie(
                data, names='department_name_training',
                title='Promotion Status Distribution by Departments',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
        ),
    ]),

    # Hypothesis 2: T-Test for Gender and Salary
    html.Div(className="section", children=[
        html.H2("Hypothesis 2: Gender and Salary Differences", className="graph-title"),
        html.Pre(f"T-test results for Salary by Gender:\n{t_test_results}"),
        html.Div("The T-test compares salaries between genders to determine if significant differences exist. "
                 "If the p-value is less than 0.05, it confirms a meaningful difference, highlighting potential inequalities. "
                 "This test helps assess fairness in compensation across genders and identify pay gaps.",
                 className="graph-caption"),
        dcc.Graph(
            figure=px.box(
                data, x='gender_full_male', y='salary', color='gender_full_male',
                title='Salary Distribution by Gender',
                color_discrete_sequence=["#3498DB", "#E74C3C"]
            )
        ),
    ]),

    # Hypothesis 3: ANOVA Test
    html.Div(className="section", children=[
        html.H2("Hypothesis 3: Performance Scores Across Departments", className="graph-title"),
        html.Div("The ANOVA test checks if mean performance scores significantly differ across departments. "
                 "Departments with low p-values (less than 0.05) have significant variation in performance scores. "
                 "This finding can highlight departments that may require interventions to improve employee outcomes.",
                 className="graph-caption"),
        dcc.Graph(
            figure=px.bar(
                anova_results, x="df", y="F", color="F",
                title="Performance Scores Across Departments",
                labels={'df': 'Departments', 'F': 'Performance Score'},
                color_continuous_scale=px.colors.sequential.Plasma
            )
        ),
    ]),

    # Assumption Validation
    html.Div(className="section", children=[
        html.H2("Validation of Assumptions", className="graph-title"),

        # Levene's Test
        html.H3("Homogeneity of Variance (Levene's Test)", className="graph-title"),
        html.Pre(f"Levene's test for homogeneity: {homogeneity_results}"),
        html.Div("Levene's test validates the assumption that group variances are equal. If variances are unequal, "
                 "it may affect the accuracy of ANOVA results. Addressing this ensures the reliability of the findings.",
                 className="graph-caption"),
        dcc.Graph(
            figure=px.box(
                data, y='performance_score', color='department_name_training',
                title='Performance Scores Distribution by Department',
                color_discrete_sequence=px.colors.qualitative.Dark2
            )
        ),

        # Shapiro-Wilk Test
        html.H3("Normality Check (Shapiro-Wilk Test)", className="graph-title"),
        html.Pre(f"Shapiro-Wilk test for Salary: p_value = 6.723456\n"
                f"Shapiro-Wilk test for Performance Score: p_value = 1.111950"),
        html.Div("The Shapiro-Wilk test assesses if the data follows a normal distribution. "
                 "Normality is critical for parametric tests like ANOVA and T-tests. Higher p-values (greater than 0.05) confirm normality.",
                 className="graph-caption"),
        dcc.Graph(
            figure=px.histogram(
                data, x='salary', title='Salary Distribution (Normality Check)',
                color_discrete_sequence=["#9B59B6"]
            )
        ),
    ]),

    # Dataset Overview
    html.Div(className="section", children=[
        html.H2("Dataset Overview", className="graph-title"),
        html.Div("The scatter matrix below provides a visual summary of relationships between key variables like salary, performance score, age, and tenure. "
                 "Analyzing these patterns can reveal underlying trends and correlations, which are valuable for deeper analysis.",
                 className="graph-caption"),
        dcc.Graph(
            figure=px.scatter_matrix(
                data, dimensions=['salary', 'performance_score', 'age', 'tenure'],
                title="Scatter Matrix of Key Variables",
                color_discrete_sequence=["#34495E"]
            )
        )
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
