import dash
from dash import html, dcc
from dash.dash_table import DataTable
import pandas as pd
import plotly.express as px
import os

# Paths
FEATURE_IMPORTANCE_OUTPUT_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/feature_importance'
DATA_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv'

# Load data
data = pd.read_csv(DATA_PATH)
rf_importance_file = os.path.join(FEATURE_IMPORTANCE_OUTPUT_PATH, 'random_forest_feature_importance.csv')
xgb_importance_file = os.path.join(FEATURE_IMPORTANCE_OUTPUT_PATH, 'xgboost_feature_importance.csv')

rf_importances = pd.read_csv(rf_importance_file)
xgb_importances = pd.read_csv(xgb_importance_file)

# Dash App
app = dash.Dash(__name__)
app.title = "Employment Analysis - Feature Importance"

# Layout
app.layout = html.Div([
    # Title Section
    html.Div([
        html.H1("Employment Analysis - Feature Importance Insights", style={
            'textAlign': 'center',
            'color': '#2E86C1',
            'font-family': 'Arial, sans-serif',
            'fontSize': '36px',
            'margin-bottom': '30px',
            'text-shadow': '1px 1px 2px gray'
        }),
    ]),

    # Dataset Overview Section
    html.Div(style={'margin': '30px', 'padding': '20px', 'box-shadow': '2px 2px 5px #ccc', 'border-radius': '10px'}, children=[
        html.H2("Dataset Overview", style={
            'color': '#283747',
            'font-family': 'Arial, sans-serif',
            'fontSize': '28px'
        }),
        html.P("The dataset below provides an overview of key features and records. The table shows the first 10 rows of the transformed dataset used in the analysis.",
               style={'font-size': '16px', 'color': '#555', 'font-family': 'Arial'}),
        DataTable(
            data=data.head(10).to_dict('records'),
            columns=[{"name": col, "id": col} for col in data.columns],
            page_size=10,
            style_table={'overflowX': 'auto', 'border': '1px solid #ccc'},
            style_cell={'textAlign': 'left', 'font-family': 'Arial', 'fontSize': '14px'},
            style_header={'backgroundColor': '#D5DBDB', 'fontWeight': 'bold', 'fontSize': '15px'}
        ),
    ]),

    # RandomForest Feature Importance
    html.Div(style={'margin': '30px', 'padding': '20px', 'box-shadow': '2px 2px 5px #ccc', 'border-radius': '10px'}, children=[
        html.H2("RandomForest Feature Importance", style={
            'color': '#283747',
            'font-family': 'Arial, sans-serif',
            'fontSize': '28px'
        }),
        html.Ul(style={'font-size': '16px', 'color': '#555', 'font-family': 'Arial'}, children=[
            html.Li("RandomForest is an ensemble learning method that operates by constructing multiple decision trees.",
                    style={'margin-bottom': '10px'}),
            html.Li("It improves accuracy by reducing variance and overfitting, making it robust for feature selection.",
                    style={'margin-bottom': '10px'}),
            html.Li("We chose this method for its ability to handle high-dimensional data and provide interpretable importance rankings.",
                    style={'margin-bottom': '10px'}),
            html.Li("RandomForest also excels at capturing non-linear relationships between features and the target variable.",
                    style={'margin-bottom': '10px'})
        ]),
        dcc.Graph(
            figure=px.bar(
                rf_importances.head(20),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 20 Features - RandomForest",
                color="Importance",
                color_continuous_scale=px.colors.sequential.Viridis
            ).update_layout(
                yaxis={'categoryorder': 'total ascending'},
                title={'x': 0.5, 'font': {'size': 22}},
                font={'family': 'Arial', 'size': 14}
            )
        ),
        html.Ul(style={'font-size': '16px', 'color': '#555', 'font-family': 'Arial'}, children=[
            html.Li("Salary and tenure emerged as the most critical features in this model."),
            html.Li("Features like self-assessment scores and peer feedback also hold significant importance."),
            html.Li("The feature rankings align with RandomForest's strength in interpreting non-linear interactions."),
            html.Li("Lower-ranked features provide insights into lesser influential factors, which can aid in further research.")
        ]),
        html.H3("Key Insights from RandomForest", style={'margin-top': '20px'}),
        html.Ul(style={'font-size': '16px', 'color': '#555', 'font-family': 'Arial'}, children=[
            html.Li("Salary and tenure are pivotal for predicting employee success."),
            html.Li("The model's robust handling of feature complexity makes it reliable for decision-making."),
            html.Li("Peer feedback introduces a behavioral dimension into the analysis."),
            html.Li("The consistent ranking of key features strengthens the credibility of these results."),
            html.Li("RandomForest underscores the importance of balancing financial and personal performance metrics.")
        ])
    ]),

    # XGBoost Feature Importance
    html.Div(style={'margin': '30px', 'padding': '20px', 'box-shadow': '2px 2px 5px #ccc', 'border-radius': '10px'}, children=[
        html.H2("XGBoost Feature Importance", style={
            'color': '#283747',
            'font-family': 'Arial, sans-serif',
            'fontSize': '28px'
        }),
        html.Ul(style={'font-size': '16px', 'color': '#555', 'font-family': 'Arial'}, children=[
            html.Li("XGBoost is a gradient boosting framework designed for speed and accuracy."),
            html.Li("It uses a boosting approach to iteratively improve model performance.", style={'margin-bottom': '10px'}),
            html.Li("XGBoost is particularly effective at handling missing data and capturing feature interactions."),
            html.Li("We selected this model for its high predictive power and ability to optimize feature importance rankings.")
        ]),
        dcc.Graph(
            figure=px.bar(
                xgb_importances.head(20),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 20 Features - XGBoost",
                color="Importance",
                color_continuous_scale=px.colors.sequential.Plasma
            ).update_layout(
                yaxis={'categoryorder': 'total ascending'},
                title={'x': 0.5, 'font': {'size': 22}},
                font={'family': 'Arial', 'size': 14}
            )
        ),
        html.Ul(style={'font-size': '16px', 'color': '#555', 'font-family': 'Arial'}, children=[
            html.Li("Self-assessment performance scores rank highest in feature importance."),
            html.Li("Managerial feedback and absenteeism rates also play significant roles."),
            html.Li("The model identifies nuanced behavioral patterns impacting employee outcomes."),
            html.Li("Lower-ranked features provide context to refine specific HR strategies.")
        ]),
        html.H3("Key Insights from XGBoost", style={'margin-top': '20px'}),
        html.Ul(style={'font-size': '16px', 'color': '#555', 'font-family': 'Arial'}, children=[
            html.Li("Performance metrics are central to predicting success in this model."),
            html.Li("Managerial feedback reveals the importance of organizational oversight."),
            html.Li("Absenteeism rates highlight the role of consistency and presence."),
            html.Li("The alignment of features with RandomForest strengthens cross-model reliability."),
            html.Li("XGBoost provides a detailed view of behavioral and performance factors.")
        ])
    ]),

    # Comparison Table
    html.Div(style={'margin': '30px', 'padding': '20px', 'box-shadow': '2px 2px 5px #ccc', 'border-radius': '10px'}, children=[
        html.H2("Comparison of RandomForest and XGBoost", style={
            'color': '#283747',
            'font-family': 'Arial, sans-serif',
            'fontSize': '28px'
        }),
        DataTable(
            data=[
                {"Aspect": "Model Type", "RandomForest": "Ensemble of decision trees", "XGBoost": "Gradient boosting framework"},
                {"Aspect": "Top Feature", "RandomForest": "Salary", "XGBoost": "Self-assessment scores"},
                {"Aspect": "Strength", "RandomForest": "Handles non-linear relationships", "XGBoost": "Optimizes feature interaction"},
                {"Aspect": "Focus", "RandomForest": "Financial and tenure-based features", "XGBoost": "Performance and behavioral metrics"},
                {"Aspect": "Flexibility", "RandomForest": "Works well with high-dimensional data", "XGBoost": "Effective with missing data"}
            ],
            columns=[
                {"name": "Aspect", "id": "Aspect"},
                {"name": "RandomForest", "id": "RandomForest"},
                {"name": "XGBoost", "id": "XGBoost"}
            ],
            style_table={'overflowX': 'auto', 'border': '1px solid #ccc'},
            style_cell={'textAlign': 'left', 'font-family': 'Arial', 'fontSize': '14px'},
            style_header={'backgroundColor': '#D5DBDB', 'fontWeight': 'bold', 'fontSize': '15px'}
        )
    ])
])

# Run server
if __name__ == '__main__':
    app.run_server(debug=True)
