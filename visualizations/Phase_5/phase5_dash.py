import dash
from dash import dcc, html
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

# Constants
MODEL_SELECTION_OUTPUT_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/model_selection'
MODEL_FOLDERS = ['DecisionTree', 'LightGBM', 'RandomForest', 'XGBoost']

# Initialize Dash app
app = dash.Dash(__name__, title="Model Development and Evaluation")
server = app.server

# Load performance comparison data
performance_file = os.path.join(MODEL_SELECTION_OUTPUT_PATH, "model_performance_comparison.csv")
performance_df = pd.read_csv(performance_file) if os.path.exists(performance_file) else None

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
        return pd.read_csv(file_path, index_col=0)
    except FileNotFoundError:
        return None

# Generate content for individual model results
def get_model_result_section(model_name):
    classification_report_path = os.path.join(MODEL_SELECTION_OUTPUT_PATH, model_name, f"{model_name}_classification_report.txt")
    confusion_matrix_path = os.path.join(MODEL_SELECTION_OUTPUT_PATH, model_name, f"{model_name}_confusion_matrix.csv")
    
    classification_report = load_classification_report(classification_report_path)
    confusion_matrix = load_confusion_matrix(confusion_matrix_path)

    # Generate confusion matrix figure
    if confusion_matrix is not None:
        confusion_matrix_fig = px.imshow(
            confusion_matrix.values,
            labels=dict(x="Predicted", y="True"),
            x=confusion_matrix.columns,
            y=confusion_matrix.index,
            color_continuous_scale="Blues",
            text_auto=True,
            title=f"{model_name} - Confusion Matrix"
        )
    else:
        confusion_matrix_fig = None

    # Add insights specific to each model
    insights = []
    if model_name == "DecisionTree":
        insights = [
            "1. Simple and interpretable model, ideal for initial analysis.",
            "2. Accuracy: **86.72%**, slightly below ensemble models.",
            "3. High recall for lessemployable (Class 1): **82%**.",
            "4. Suitable for datasets where interpretability is crucial.",
            "5. Tends to overfit on smaller datasets."
        ]
    elif model_name == "LightGBM":
        insights = [
            "1. Efficient for large datasets with high accuracy: **87.93%**.",
            "2. Matches XGBoost in ROC AUC: **96.02%**.",
            "3. High precision for lessemployable (Class 1): **92%**.",
            "4. Handles missing data effectively.",
            "5. Fast training time compared to other boosting methods."
        ]
    elif model_name == "RandomForest":
        insights = [
            "1. Robust and reliable with an accuracy of **85.85%**.",
            "2. High precision for lessemployable (Class 1): **90%**.",
            "3. Handles noisy data well due to bagging techniques.",
            "4. Struggles with recall for lessemployable: **75%**.",
            "5. Suitable for datasets with high dimensionality."
        ]
    elif model_name == "XGBoost":
        insights = [
            "1. State-of-the-art boosting method with accuracy: **87.93%**.",
            "2. Excellent ROC AUC: **96.02%**, ensuring strong model generalization.",
            "3. Balanced recall (78%) and precision (92%) for lessemployable.",
            "4. Optimized for speed and performance in large datasets.",
            "5. Requires careful hyperparameter tuning for best results."
        ]

    # Return the section as a Div
    return html.Div([
        html.H3(f"{model_name} - Individual Results", style={'text-align': 'center', 'color': '#1F618D', 'font-size': '24px'}),
        html.H4("Classification Report", style={'color': '#117A65', 'font-size': '20px'}),
        html.Pre(classification_report, style={'font-size': '16px', 'background-color': '#F8F9F9', 'padding': '10px'}),
        html.H4("Confusion Matrix", style={'color': '#117A65', 'font-size': '20px'}),
        dcc.Graph(figure=confusion_matrix_fig) if confusion_matrix_fig else html.P("Confusion matrix not found."),
        html.H4("Insights", style={'color': '#D35400', 'font-size': '20px'}),
        html.Ul([html.Li(insight, style={'font-size': '16px', 'margin-bottom': '8px'}) for insight in insights])
    ], style={'margin-bottom': '50px'})

# Layout
app.layout = html.Div([
    html.Div([
        html.H1("🚀 Model Development and Evaluation Dashboard", style={'text-align': 'center', 'font-size': '32px', 'color': '#2C3E50'}),
    ]),
    dcc.Tabs([
        dcc.Tab(label="Individual Model Results", children=[
            html.Div([get_model_result_section(model_name) for model_name in MODEL_FOLDERS])
        ]),
        dcc.Tab(label="Model Performance Comparison", children=[
            html.Div([
                html.H3("📊 Model Performance Comparison", style={'text-align': 'center', 'color': '#1F618D', 'font-size': '24px'}),
                dcc.Graph(
                    id="comparison-bar-chart",
                    figure=px.bar(
                        performance_df,
                        x="Model",
                        y=["Accuracy", "Precision"],
                        barmode="group",
                        title="Model Accuracy and Precision Comparison",
                        labels={"value": "Score", "variable": "Metric"},
                        text_auto=True,
                        color_discrete_sequence=px.colors.qualitative.Set2
                    ) if performance_df is not None else {}
                ),
                html.H4("Performance Metrics Table", style={'color': '#117A65', 'font-size': '20px'}),
                html.Table([
                    html.Thead(html.Tr([html.Th(col, style={'text-align': 'center', 'padding': '10px', 'color': '#283747'}) for col in ["Model", "Accuracy", "Precision", "ROC AUC", "Key Insight"]])),
                    html.Tbody([
                        html.Tr([
                            html.Td("RandomForest"), html.Td("85.85%"), html.Td("86.27%"), html.Td("92.72%"),
                            html.Td("Balanced performance but struggled with Class 1 recall.")
                        ]),
                        html.Tr([
                            html.Td("DecisionTree"), html.Td("86.72%"), html.Td("86.70%"), html.Td("92.75%"),
                            html.Td("Improved recall but less generalization than ensembles.")
                        ]),
                        html.Tr([
                            html.Td("XGBoost"), html.Td("87.93%"), html.Td("88.40%"), html.Td("96.02%"),
                            html.Td("Best overall, excelling in precision and ranking.")
                        ]),
                        html.Tr([
                            html.Td("LightGBM"), html.Td("87.93%"), html.Td("88.40%"), html.Td("96.02%"),
                            html.Td("Matches XGBoost with high efficiency for large datasets.")
                        ]),
                    ])
                ], style={'width': '100%', 'margin-top': '20px', 'border-collapse': 'collapse', 'border': '1px solid #BDC3C7'}),
                html.Div([
                    html.H4("Insights", style={'color': '#D35400', 'font-size': '20px'}),
                    html.Ul([
                        html.Li("RandomForest achieved high accuracy and robustness.", style={'font-size': '16px'}),
                        html.Li("LightGBM excels in structured datasets with efficiency.", style={'font-size': '16px'}),
                        html.Li("DecisionTree provides interpretability but lacks generalization.", style={'font-size': '16px'}),
                        html.Li("XGBoost delivers the best overall performance.", style={'font-size': '16px'}),
                    ])
                ])
            ])
        ])
    ])
])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
