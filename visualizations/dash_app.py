import os
import socket
import subprocess
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Function to find an available port
def get_open_port(default_port=8050):
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", default_port))
                return default_port
        except OSError:
            default_port += 1

# Function to run sub-apps on unique ports and return the iframe URL
def run_dash_app(file_path):
    try:
        # Ensure the file exists
        if not os.path.isfile(file_path):
            return f"Error: File {file_path} not found."

        # Find an open port and start the app
        port = get_open_port()
        subprocess.Popen(["python", file_path, "--port", str(port)], shell=True)
        return f"http://127.0.0.1:{port}"
    except Exception as e:
        return f"Error running file {file_path}: {e}"

# Initialize the main Dash app with Bootstrap styling
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Employment Analysis Dashboard"

# Define a reusable card component for displaying content
def create_card(title, content):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4(title, className="card-title", style={"color": "white"}),
                html.Ul(
                    [html.Li(line, style={"color": "white", "fontSize": "16px"}) for line in content],
                    style={"paddingLeft": "20px"},
                ),
            ],
            style={"backgroundColor": "#343a40", "padding": "20px", "borderRadius": "10px"},
        ),
        className="mb-3",
    )

# Define styles for smooth transitions and colors
custom_styles = """
    .tab-content {
        opacity: 0;
        transition: opacity 0.5s ease-in-out;
    }
    .tab-content.active {
        opacity: 1;
    }
    .main-header {
        background-color: #007bff;
        color: white;
        padding: 10px;
        border-radius: 10px;
    }
    .iframe-container {
        width: 100%;
        height: 600px;
        border: none;
        margin-top: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
"""

# Define the layout of the main app
app.layout = dbc.Container(
    fluid=True,
    children=[
        # Inject custom CSS styles
        html.Div(children=html.Script(custom_styles)),
        # Header Section
        dbc.Row(
            dbc.Col(
                html.H1(
                    "📊 Employment Analysis Dashboard",
                    className="text-center my-4 main-header",
                )
            )
        ),
        # Tabs for navigation
        dbc.Row(
            dbc.Col(
                dcc.Tabs(
                    id="tabs",
                    value="tab-introduction",
                    children=[
                        dcc.Tab(label="Introduction", value="tab-introduction"),
                        dcc.Tab(label="Problem Statement", value="tab-problem-statement"),
                        dcc.Tab(label="Technologies Used", value="tab-technologies-used"),
                        dcc.Tab(label="Phase 1: Data Collection and Preprocessing", value="tab-phase-1"),
                        dcc.Tab(label="Phase 2: Exploratory Data Analysis (EDA)"),
                        dcc.Tab(label="Phase 3: Hypothesis Testing", value="tab-phase-3"),
                        dcc.Tab(label="Phase 4: Feature Importance", value="tab-phase-4"),
                        dcc.Tab(label="Phase 5: Model Development and Evaluation", value="tab-phase-5"),
                    ],
                    style={
                        "fontSize": "18px",
                        "fontWeight": "bold",
                        "backgroundColor": "#f8f9fa",
                        "padding": "10px",
                        "borderRadius": "10px",
                    },
                )
            )
        ),
        # Content area for the selected tab
        dbc.Row(
            dbc.Col(html.Div(id="tab-content", className="tab-content mt-4"), width=12)
        ),
        # Footer
        dbc.Row(
            dbc.Col(
                html.Footer(
                    "Created By Nidish (1BG22CS095) from CSE Undergrad @ BNMIT",
                    className="text-center text-muted mt-4 mb-2",
                )
            )
        ),
    ],
)

# Define callback to update content dynamically
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
)
def render_content(tab):
    if tab == "tab-introduction":
        return html.Div(
            create_card(
                "Introduction",
                [
                    "Welcome to the Employment Analysis Dashboard!",
                    "This project provides insights into employee trends such as attrition, performance, and other workforce issues.",
                    "Each phase represents a critical step in the data science lifecycle.",
                    "Navigate through the phases to explore detailed analysis and visualizations.",
                    "The aim is to assist organizations in making data-driven decisions for improved workforce management.",
                ],
            ),
            className="active",
        )
    elif tab == "tab-problem-statement":
        return html.Div(
            create_card(
                "Problem Statement",
                [
                    "Organizations face challenges in understanding employee behavior.",
                    "This can lead to unexpected attrition or inefficient workforce planning.",
                    "Our project aims to analyze employee data to uncover actionable insights.",
                    "We strive to provide a better understanding of employee engagement and retention.",
                    "Data-driven decisions are essential to improving organizational efficiency and productivity.",
                ],
            ),
            className="active",
        )
    elif tab == "tab-technologies-used":
        return html.Div(
            create_card(
                "Technologies Used",
                [
                    "Python: For data preprocessing, modeling, and visualization.",
                    "Pandas & NumPy: For efficient data manipulation.",
                    "Scikit-learn: For machine learning and predictive modeling.",
                    "Dash: For interactive visualizations and dashboards.",
                    "XGBoost & LightGBM: For advanced machine learning algorithms.",
                ],
            ),
            className="active",
        )
    elif tab.startswith("tab-phase"):
        # Map phase tabs to corresponding scripts
        phase_scripts = {
            "tab-phase-1": "C:\\Users\\nidis\\Documents\\DS\\DS_Projects\\Employment_Analysis\\visualizations\\Phase_1\\phase1_dash.py",
            "tab-phase-3": "C:\\Users\\nidis\\Documents\\DS\\DS_Projects\\Employment_Analysis\\visualizations\\Phase_3\\phase3_dash.py",
            "tab-phase-4": "C:\\Users\\nidis\\Documents\\DS\\DS_Projects\\Employment_Analysis\\visualizations\\Phase_4\\phase4_dash.py",
            "tab-phase-5": "C:\\Users\\nidis\\Documents\\DS\\DS_Projects\\Employment_Analysis\\visualizations\\Phase_5\\phase5_dash.py",
        }
        script_path = phase_scripts.get(tab)
        iframe_url = run_dash_app(script_path)
        if iframe_url.startswith("Error"):
            return html.Div(
                create_card(
                    "Error",
                    [iframe_url],
                ),
                className="active",
            )
        return html.Iframe(src=iframe_url, className="iframe-container")
    else:
        return html.Div(
            create_card(
                "Error",
                ["An unknown tab was selected. Please choose a valid option."],
            ),
            className="active",
        )

# Run the main app
if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
