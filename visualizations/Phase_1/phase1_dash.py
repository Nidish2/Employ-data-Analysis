from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import base64

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])

# Paths for data
RAW_DATA_PATH = "C:/Users/nidis/Documents/Employment_Analysis/data/Employee_data_raw.csv"
CLEANED_DATA_PATH = "C:/Users/nidis/Documents/Employment_Analysis/data/cleaned_data.csv"
FEATURE_ENGINEERED_PATH = "C:/Users/nidis/Documents/Employment_Analysis/data/feature_engineered_data.csv"
CORRELATION_MATRIX_PATH = "C:/Users/nidis/Documents/Employment_Analysis/outputs/feature_engineering/correlation_matrix.png"
TRANSFORMED_DATA_PATH = "C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv"

# Load data
try:
    raw_data = pd.read_csv(RAW_DATA_PATH)
    cleaned_data = pd.read_csv(CLEANED_DATA_PATH)
    feature_data = pd.read_csv(FEATURE_ENGINEERED_PATH)
    transformed_data = pd.read_csv(TRANSFORMED_DATA_PATH)
except FileNotFoundError as e:
    raise Exception(f"File not found: {e}")

# Encode image
def encode_image(image_path):
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

encoded_correlation_matrix = encode_image(CORRELATION_MATRIX_PATH)

# Custom Styles
custom_styles = """
    .main-header {
        background-color: #004085;
        color: white;
        text-align: center;
        padding: 30px;
        font-size: 3em;
        margin-bottom: 50px;
        border-radius: 12px;
        box-shadow: 0px 8px 12px rgba(0, 0, 0, 0.2);
    }
    .graph-container {
        padding: 30px;
        margin: 20px 0;
        border-radius: 12px;
        box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.2);
        background: #ffffff;
    }
    .graph-title {
        font-size: 1.8em;
        font-weight: bold;
        margin-bottom: 20px;
        color: #004085;
        text-align: center;
    }
    .graph-insights {
        font-size: 1.2em;
        margin: 20px 0 10px 0;
        color: #333333;
    }
    .card {
        background-color: #f8f9fa;
        padding: 40px;
        margin: 30px 0;
        border-radius: 12px;
        box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.2);
    }
    .card-title {
        font-size: 2em;
        font-weight: bold;
        color: #004085;
        margin-bottom: 30px;
        text-align: center;
    }
    ul li {
        margin: 10px 0;
    }
"""

# Layout
app.layout = html.Div([
    html.Script(custom_styles),

    # Header
    html.H1("📊 Employment Analysis Dashboard", className="main-header"),

    # Tabs
    dcc.Tabs(id="tabs", value="tab-raw-data", children=[
        dcc.Tab(label="Raw Data", value="tab-raw-data"),
        dcc.Tab(label="Cleaned Data", value="tab-cleaned-data"),
        dcc.Tab(label="Feature Engineering", value="tab-feature-engineering"),
        dcc.Tab(label="Transformed Data", value="tab-transformed-data"),
    ], style={"fontSize": "20px", "fontWeight": "bold"}),

    # Tab Content
    html.Div(id="tab-content", className="tab-content")
])

# Callback for tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def render_content(tab):
    if tab == "tab-raw-data":
        return html.Div([
            html.Div(className="graph-container", children=[
                html.H3("Salary Distribution", className="graph-title"),
                dcc.Graph(
                    id="raw-salary-distribution",
                    figure=px.histogram(raw_data, x="salary", title="Salary Distribution", color_discrete_sequence=["#004085"])
                ),
                html.Div(className="graph-insights", children=[
                    html.Ul([
                        html.Li("Most employees have salaries concentrated in the lower range."),
                        html.Li("A few employees have significantly higher salaries, forming outliers."),
                        html.Li("The data highlights the need for salary normalization during preprocessing."),
                        html.Li("Identifying salary distribution helps evaluate compensation strategies."),
                    ])
                ]),
            ]),
            html.Div(className="card", children=[
                html.H2("Insights about Raw Data", className="card-title"),
                html.Ul([
                    html.Li("The raw data contains all unprocessed information about employees."),
                    html.Li("This data is critical for understanding initial trends and anomalies."),
                    html.Li("Salary distribution analysis helps identify disparities and ranges."),
                    html.Li("Further cleaning and preprocessing will remove inconsistencies."),
                    html.Li("This step lays the foundation for deeper analysis in later phases."),
                ]),
            ])
        ])
    elif tab == "tab-cleaned-data":
        return html.Div([
            html.Div(className="graph-container", children=[
                html.H3("Salary Distribution by Department", className="graph-title"),
                dcc.Graph(
                    id="cleaned-salary-department",
                    figure=px.box(
                        cleaned_data, x="department_name", y="salary", title="Salary Distribution by Department",
                        color_discrete_sequence=["#28a745"]
                    )
                ),
                html.Div(className="graph-insights", children=[
                    html.Ul([
                        html.Li("Some departments show higher median salaries compared to others."),
                        html.Li("Outliers in specific departments indicate exceptional cases."),
                        html.Li("Departments with low variation in salaries may reflect consistent compensation."),
                        html.Li("Identifying salary trends across departments reveals workforce patterns."),
                    ])
                ]),
            ]),
            html.Div(className="card", children=[
                html.H2("Insights about Cleaned Data", className="card-title"),
                html.Ul([
                    html.Li("Data cleaning removes duplicates, missing values, and incorrect entries."),
                    html.Li("The cleaned dataset provides a more reliable basis for analysis."),
                    html.Li("Salary distribution across departments reveals key workforce trends."),
                    html.Li("Identifying outliers is critical to understanding organizational disparities."),
                    html.Li("Visualizations at this stage highlight the effectiveness of preprocessing."),
                ]),
            ])
        ])
    elif tab == "tab-feature-engineering":
        return html.Div([
            html.Div(className="graph-container", children=[
                html.H3("Tenure vs Overall Performance Rating", className="graph-title"),
                dcc.Graph(
                    id="tenure-vs-performance",
                    figure=px.scatter(
                        feature_data, x="tenure", y="overall_performance_rating",
                        title="Tenure vs Overall Performance Rating", color_discrete_sequence=["#17a2b8"]
                    )
                ),
                html.Div(className="graph-insights", children=[
                    html.Ul([
                        html.Li("Employees with higher tenure generally achieve better performance ratings."),
                        html.Li("Variability exists among short-tenured employees' performance."),
                        html.Li("Feature analysis highlights opportunities for workforce optimization."),
                        html.Li("Tenure is a critical factor for predictive analysis models."),
                    ])
                ]),
                html.Img(src=encoded_correlation_matrix, style={"width": "100%", "marginTop": "30px", "borderRadius": "12px"})
            ]),
            html.Div(className="card", children=[
                html.H2("Insights about Feature Engineering", className="card-title"),
                html.Ul([
                    html.Li("Feature engineering extracts meaningful insights from raw data."),
                    html.Li("Relationships between tenure and performance are visualized here."),
                    html.Li("The correlation matrix highlights interdependencies between variables."),
                    html.Li("New features enhance the predictive capabilities of models."),
                    html.Li("This stage bridges data collection and model development phases."),
                ]),
            ])
        ])
    elif tab == "tab-transformed-data":
        return html.Div([
            html.Div(className="graph-container", children=[
                html.H3("Work-life Balance Score", className="graph-title"),
                dcc.Graph(
                    id="work-life-balance",
                    figure=px.histogram(
                        transformed_data, x="work_life_balance_score",
                        title="Work-life Balance Score", color_discrete_sequence=["#dc3545"]
                    )
                ),
                html.Div(className="graph-insights", children=[
                    html.Ul([
                        html.Li("Most employees report moderate work-life balance scores."),
                        html.Li("A few employees have significantly lower satisfaction levels."),
                        html.Li("Work-life balance directly impacts employee performance."),
                        html.Li("Understanding these scores is key for retention strategies."),
                    ])
                ]),
            ]),
            html.Div(className="card", children=[
                html.H2("Insights about Transformed Data", className="card-title"),
                html.Ul([
                    html.Li("Transformed data undergoes scaling and normalization."),
                    html.Li("This stage ensures uniformity across features for model compatibility."),
                    html.Li("Analysis of job titles distribution provides insights into workforce diversity."),
                    html.Li("Transformations improve the reliability of machine learning models."),
                    html.Li("Feature selection and scaling are pivotal for advanced analysis."),
                ]),
            ])
        ])
    return html.Div("Error: Tab not found.")

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
