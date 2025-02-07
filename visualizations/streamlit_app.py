import streamlit as st
import subprocess
import pandas as pd

# Custom CSS for consistent styling and white font
st.markdown("""
    <style>
        body {
            background-color: #1E1E1E; /* Dark background for contrast */
        }
        .main-title {
            font-size: 38px;
            font-weight: bold;
            color: #FFFFFF; /* White font */
            text-align: center;
            margin-bottom: 20px;
            font-family: 'Arial Black', sans-serif;
        }
        .sub-title {
            font-size: 26px;
            font-weight: bold;
            color: #FFFFFF; /* White font */
            margin-top: 20px;
            margin-bottom: 15px;
            font-family: 'Trebuchet MS', sans-serif;
        }
        .section-content {
            font-size: 18px;
            color: #FFFFFF; /* White font */
            line-height: 1.8;
            text-align: justify;
            font-family: 'Georgia', serif;
        }
        .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            color: #FFFFFF; /* White font */
            margin-bottom: 10px;
            font-family: 'Verdana', sans-serif;
        }
        .sidebar-radio {
            font-size: 18px;
            color: #FFFFFF; /* White font */
            font-family: 'Courier New', monospace;
        }
    </style>
""", unsafe_allow_html=True)

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


# Sidebar Navigation
st.sidebar.markdown('<div class="sidebar-title">📌 Navigate Phases</div>', unsafe_allow_html=True)
navigation = st.sidebar.radio(
    "",
    [
        "Introduction",
        "Problem Statement",
        "Technologies Used",
        "Phase 1: Data Collection and Preprocessing",
        "Phase 2: Exploratory Data Analysis (EDA)",
        "Phase 3: Hypothesis Testing",
        "Phase 4: Feature Importance",
        "Phase 5: Model Development and Evaluation",
        "Phase 6: Predictive Analytics",
        "Realtime Analysis",
        "Conclusion",
    ],
)

# Introduction Section
if navigation == "Introduction":
    st.markdown('<div class="main-title">🔍 Employment Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="section-content">
            Welcome to the <b>Employment Analysis Dashboard</b>!  
            This project focuses on understanding employment data through a systematic approach divided into phases:  
            <ul>
                <li><b>Phase 1:</b> Data Collection and Preprocessing - Cleaning and preparing the dataset for analysis.</li>
                <li><b>Phase 2:</b> Exploratory Data Analysis (EDA) - Discovering trends and relationships in the data.</li>
                <li><b>Phase 3:</b> Hypothesis Testing - Validating assumptions with statistical evidence.</li>
                <li><b>Phase 4:</b> Feature Importance - Identifying key variables for model building.</li>
                <li><b>Phase 5:</b> Model Development and Evaluation - Building, optimizing, and assessing ML models.</li>
            </ul>
            <br>
            <b>Project Objectives:</b>
            <br><br>
            <table style="width:100%; border-collapse: collapse; font-family: Arial, sans-serif;">
                <thead>
                    <tr style="background-color: #1F1F1F; color: #FFFFFF;">
                        <th style="padding: 12px; text-align: left; font-size: 16px; border-bottom: 2px solid #4CAF50;">Objective</th>
                        <th style="padding: 12px; text-align: left; font-size: 16px; border-bottom: 2px solid #4CAF50;">Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="background-color: #2E2E2E; color: #E0E0E0;">
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Data Accuracy</td>
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Enhance preprocessing techniques to improve data accuracy and consistency.</td>
                    </tr>
                    <tr style="background-color: #383838; color: #E0E0E0;">
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Exploratory Analysis</td>
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Conduct detailed exploratory analysis to uncover meaningful patterns and trends.</td>
                    </tr>
                    <tr style="background-color: #2E2E2E; color: #E0E0E0;">
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Statistical Insights</td>
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Use hypothesis testing to validate key assumptions with evidence-based insights.</td>
                    </tr>
                    <tr style="background-color: #383838; color: #E0E0E0;">
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Key Feature Identification</td>
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Determine critical variables that contribute significantly to predictions.</td>
                    </tr>
                    <tr style="background-color: #2E2E2E; color: #E0E0E0;">
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Interactive Visualization</td>
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Develop advanced visualizations to enhance data storytelling and insights.</td>
                    </tr>
                    <tr style="background-color: #383838; color: #E0E0E0;">
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Model Optimization</td>
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Focus on building and refining machine learning models for robust predictions.</td>
                    </tr>
                    <tr style="background-color: #2E2E2E; color: #E0E0E0;">
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Real-Time Analysis</td>
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Incorporate methods for real-time data analysis to enable timely decisions.</td>
                    </tr>
                    <tr style="background-color: #2E2E2E; color: #E0E0E0;">
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">User-Friendly Interface</td>
                        <td style="padding: 10px; border-bottom: 1px solid #4CAF50;">Create an intuitive and visually appealing dashboard for end-users.</td>
                    </tr>
                </tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)


# Problem Statement Section
elif navigation == "Problem Statement":
    st.markdown('<div class="sub-title">❓ Problem Statement</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="section-content">
            Understanding employment trends is critical in today’s data-driven organizations.  
            This project focuses on analyzing employee data to uncover patterns and insights.  
            By leveraging machine learning and statistical analysis, we aim to predict outcomes
            such as attrition, performance, and well-being.  

            In addition to traditional analysis, this project emphasizes the increasing
            concern over employee well-being in high-pressure work environments.
            Issues such as excessive overtime, chronic stress,and burnout have led
            to significant declines in productivity, morale, and in extreme cases,
            severe health implications or fatalities.  

            Recent studies and organizational reports reveal a worrying trend where 
            employees face severe health risks due to unmanaged stress and excessive 
            workloads. This analysis will explore these critical aspects, aiming to 
            shed light on the patterns contributing to these outcomes, enabling
            organizations to implement proactive measures to safeguard their workforce.
        </div>
    """, unsafe_allow_html=True)
    
    # Displaying a graph
    st.markdown("### Workplace Stress & Health Impact Analysis")
    st.markdown("""
        The following visualization demonstrates the correlation between stress-related factors, overtime, and their effects on employee well-being.  
        It provides insights into how these factors impact organizational productivity and employee health outcomes.
    """)
    
    # Placeholder for Graph or Table
    import matplotlib.pyplot as plt
    import pandas as pd

    # Example data
    data = {
        'Year': [2018, 2019, 2020, 2021, 2022],
        'Workplace Fatalities Due to Overwork': [120, 150, 180, 210, 250],
        'Reported Health Deteriorations (in Thousands)': [15, 18, 22, 27, 30]
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df['Year'], df['Workplace Fatalities Due to Overwork'], color='red', alpha=0.7, label='Fatalities')
    ax.plot(df['Year'], df['Reported Health Deteriorations (in Thousands)'], marker='o', color='blue', label='Health Deteriorations')

    ax.set_title('Impact of Overwork and Stress on Employee Well-being', fontsize=14)
    ax.set_xlabel('Year')
    ax.set_ylabel('Cases')
    ax.legend()

    st.pyplot(fig)

# Technologies Used Section
elif navigation == "Technologies Used":
    st.markdown('<div class="sub-title">⚙️ Technologies Used</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="section-content">
            This project leverages a comprehensive set of tools and technologies to efficiently process, analyze, and visualize data. Below is an overview of the technologies used:
            <ul>
                <li><b>Python</b>: Core programming language for data analysis and machine learning.</li>
                <li><b>Pandas</b>: Library for data manipulation and analysis.</li>
                <li><b>NumPy</b>: Numerical computations library.</li>
                <li><b>OpenPyXL</b>: Used for handling Excel files.</li>
                <li><b>Scikit-learn</b>: Machine learning library for model training and evaluation.</li>
                <li><b>XGBoost</b>: High-performance gradient boosting framework.</li>
                <li><b>LightGBM</b>: Fast gradient boosting framework.</li>
                <li><b>Matplotlib</b>: Visualization library for creating static plots.</li>
                <li><b>Seaborn</b>: Statistical data visualization library.</li>
                <li><b>Plotly</b>: Interactive visualization library.</li>
                <li><b>Rasterio</b>: Raster data processing library.</li>
                <li><b>Statsmodels</b>: For statistical analysis and modeling.</li>
                <li><b>Joblib</b>: Library for efficient model serialization.</li>
                <li><b>Streamlit</b>: Framework for building interactive dashboards.</li>
                <li><b>Dash</b>: Web-based analytical dashboards framework.</li>
                <li><b>Dash Bootstrap Components</b>: For enhanced Dash components and styling.</li>
                <li><b>Dash Extensions</b>: Provides advanced functionality for Dash applications.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Create a DataFrame for the table
    import pandas as pd

    technologies = [
        {"Tech Name": "Pandas", "Description": "Library for data manipulation and analysis.", "Usage": "Data Handling", "Phase Used": "All"},
        {"Tech Name": "NumPy", "Description": "Numerical computations library.", "Usage": "Numerical Processing", "Phase Used": "1, 2, 3, 5"},
        {"Tech Name": "OpenPyXL", "Description": "Read/write Excel files.", "Usage": "Excel File Handling", "Phase Used": "1"},
        {"Tech Name": "Scikit-learn", "Description": "Machine learning library.", "Usage": "Model Training", "Phase Used": "1, 4, 5"},
        {"Tech Name": "XGBoost", "Description": "High-performance gradient boosting.", "Usage": "Advanced Modeling", "Phase Used": "4"},
        {"Tech Name": "LightGBM", "Description": "Fast gradient boosting framework.", "Usage": "Advanced Modeling", "Phase Used": "4"},
        {"Tech Name": "Matplotlib", "Description": "Data visualization library.", "Usage": "Visualizations", "Phase Used": "1, 2, 4"},
        {"Tech Name": "Seaborn", "Description": "Statistical data visualization.", "Usage": "Visualizations", "Phase Used": "1, 2"},
        {"Tech Name": "Plotly", "Description": "Interactive visualizations.", "Usage": "Interactive Dashboards", "Phase Used": "2"},
        {"Tech Name": "Rasterio", "Description": "Raster data processing.", "Usage": "Raster Data Handling", "Phase Used": "2"},
        {"Tech Name": "Statsmodels", "Description": "Statistical analysis and modeling.", "Usage": "Statistical Modeling", "Phase Used": "3"},
        {"Tech Name": "Joblib", "Description": "Efficient model serialization.", "Usage": "Model Persistence", "Phase Used": "1, 5"},
        {"Tech Name": "Streamlit", "Description": "Framework for interactive dashboards.", "Usage": "Dashboard Development", "Phase Used": "All"},
        {"Tech Name": "Dash", "Description": "Web-based analytical dashboards.", "Usage": "Dashboard Development", "Phase Used": "All"},
        {"Tech Name": "Dash Bootstrap Components", "Description": "Enhanced Dash components.", "Usage": "Dashboard Styling", "Phase Used": "All"},
        {"Tech Name": "Dash Extensions", "Description": "Advanced Dash functionality.", "Usage": "Dashboard Extensions", "Phase Used": "All"},
    ]

    df = pd.DataFrame(technologies)

    # Check DataFrame structure
    st.write("Technologies DataFrame Preview:", df.head())

    # Render the table
    rows = "".join([
        f"<tr><td>{row['Tech Name']}</td><td>{row['Description']}</td><td>{row['Usage']}</td><td>{row['Phase Used']}</td></tr>"
        for _, row in df.iterrows()
    ])

    st.markdown(f"""
        <div class="section-content">
            <table>
                <thead>
                    <tr>
                        <th>Tech Name</th>
                        <th>Description</th>
                        <th>Usage</th>
                        <th>Phase Used</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)


# Phase 1: Data Collection and Preprocessing
elif navigation == "Phase 1: Data Collection and Preprocessing":
    st.markdown('<div class="sub-title">📂 Phase 1: Data Collection and Preprocessing</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="section-content">
            Data preprocessing is the foundation of any data analysis project.  
            Key steps include:
            <ul>
                <li>Handling missing values to ensure data completeness.</li>
                <li>Encoding categorical variables for compatibility with machine learning models.</li>
                <li>Scaling numerical features to standardize ranges.</li>
            </ul>
            These steps prepare the data for accurate and efficient analysis in the subsequent phases.
        </div>
    """, unsafe_allow_html=True)
    subprocess.run(["streamlit", "run", "C:/Users/nidis/Documents/Employment_Analysis/visualizations/Phase_1/phase1_streamlit.py"])

# Phase 2: Exploratory Data Analysis (EDA)
elif navigation == "Phase 2: Exploratory Data Analysis (EDA)":
    st.markdown('<div class="sub-title">📊 Phase 2: Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="section-content">
            EDA involves exploring the dataset to identify trends and relationships between variables.  
            Key actions in this phase include:
            <ul>
                <li>Visualizing data distributions and identifying anomalies.</li>
                <li>Analyzing relationships between features and target variables.</li>
                <li>Deriving insights that guide feature engineering and model development.</li>
            </ul>
            This phase provides a deep understanding of the dataset’s structure and dynamics.
        </div>
    """, unsafe_allow_html=True)
    subprocess.run(["streamlit", "run", "C:/Users/nidis/Documents/Employment_Analysis/visualizations/Phase_2/phase2_streamlit.py"])

# Phase 3: Hypothesis Testing
elif navigation == "Phase 3: Hypothesis Testing":
    st.markdown('<div class="sub-title">🔬 Phase 3: Hypothesis Testing</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="section-content">
            Hypothesis testing validates assumptions about the data.  
            In this phase:
            <ul>
                <li>We use statistical methods to test the significance of observed patterns.</li>
                <li>Analyze p-values and confidence intervals to draw conclusions.</li>
                <li>Support data-driven decisions with rigorous statistical evidence.</li>
            </ul>
            This step is essential for building a robust analytical foundation.
        </div>
    """, unsafe_allow_html=True)
    subprocess.run(["streamlit", "run", "C:/Users/nidis/Documents/Employment_Analysis/visualizations/Phase_3/phase3_streamlit.py"])

# Phase 4: Feature Importance
elif navigation == "Phase 4: Feature Importance":
    st.markdown('<div class="sub-title">🌟 Phase 4: Feature Importance</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="section-content">
            Identifying impactful features is key to building efficient models.  
            This phase includes:
            <ul>
                <li>Ranking features based on their contributions to the target variable.</li>
                <li>Visualizing feature importance scores for better understanding.</li>
                <li>Focusing on critical variables to improve model performance.</li>
            </ul>
            These insights streamline the modeling process and boost accuracy.
        </div>
    """, unsafe_allow_html=True)
    subprocess.run(["streamlit", "run", "C:/Users/nidis/Documents/Employment_Analysis/visualizations/Phase_4/phase4_streamlit.py"])

# Phase 5: Model Development and Evaluation
elif navigation == "Phase 5: Model Development and Evaluation":
    st.markdown('<div class="sub-title">🤖 Phase 5: Model Development and Evaluation</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="section-content">
            Building and evaluating machine learning models is the heart of this project.  
            Key actions in this phase include:
            <ul>
                <li>Training various ML models like Decision Tree, XGBoost, and Random Forest.</li>
                <li>Evaluating performance using metrics such as accuracy, precision, and ROC-AUC.</li>
                <li>Optimizing the best-performing models for deployment.</li>
            </ul>
            This step ensures that our models are accurate, reliable, and ready for real-world application.
        </div>
    """, unsafe_allow_html=True)
    subprocess.run(["streamlit", "run", "C:/Users/nidis/Documents/Employment_Analysis/visualizations/Phase_5/phase5_streamlit.py"])

elif navigation == "Phase 6: Predictive Analytics":
    st.markdown('<div class="sub-title">🔮 Phase 6: Predictive Analytics</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="section-content">
            Predictive analytics involves forecasting future outcomes based on historical data.  
            In this phase, we focus on predicting two key aspects:
            <ul>
                <li><b>Promotion Prediction:</b> Forecasting employee promotions within the next 12 months.</li>
                <li><b>Performance Prediction:</b> Predicting an employee’s post-promotion performance rating.</li>
            </ul>
            The predictive models include:
            <ul>
                <li><b>Logistic Regression (Baseline Model)</b></li>
                <li><b>Decision Tree</b></li>
                <li><b>Random Forest</b></li>
                <li><b>XGBoost</b></li>
                <li><b>LightGBM</b></li>
            </ul>
            We evaluate these models using various performance metrics and explain the model predictions using DALEX and LIME.
        </div>
    """, unsafe_allow_html=True)
    subprocess.run(["streamlit", "run", "C:/Users/nidis/Documents/Employment_Analysis/visualizations/Phase_6/phase6_streamlit.py"])

# Realtime Analysis
elif navigation == "Realtime Analysis":
    st.markdown('<div class="sub-title">📈 Realtime Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="section-content">
            Real-time analysis enables dynamic predictions and insights.  
            This phase provides:
            <ul>
                <li>Live predictions based on user input.</li>
                <li>Interactive visualizations for immediate insights.</li>
                <li>A practical tool for on-the-go decision-making.</li>
            </ul>
            Experience the power of real-time data-driven solutions!
        </div>
    """, unsafe_allow_html=True)
    subprocess.run(["streamlit", "run", "C:/Users/nidis/Documents/Employment_Analysis/visualizations/Real_time_analysis/Real_time_analysis.py"])

# Conclusion Section
elif navigation == "Conclusion":
    st.markdown('<div class="sub-title">📚 Conclusion</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="section-content">
            <b>Summary of What We've Achieved:</b>
            <ul>
                <li><b>Phase 1:</b> Cleaned and preprocessed data to ensure it was ready for analysis.</li>
                <li><b>Phase 2:</b> Conducted EDA to uncover patterns, relationships, and data distributions.</li>
                <li><b>Phase 3:</b> Performed hypothesis testing to validate assumptions and draw statistical insights.</li>
                <li><b>Phase 4:</b> Determined key features impacting target variables using feature importance techniques.</li>
                <li><b>Phase 5:</b> Built, evaluated, and optimized machine learning models for predictive analytics.</li>
            </ul>
            <br>
            <b>Summary Table:</b>
            <table style="width:100%; border-collapse: collapse;">
                <tr style="background-color: #333333; color: #FFFFFF;">
                    <th style="padding: 8px; text-align: left;">Phase</th>
                    <th style="padding: 8px; text-align: left;">Description</th>
                    <th style="padding: 8px; text-align: left;">Insights</th>
                    <th style="padding: 8px; text-align: left;">Output</th>
                </tr>
                <tr style="background-color: #1E1E1E; color: #FFFFFF;">
                    <td style="padding: 8px;">Phase 1</td>
                    <td style="padding: 8px;">Data Collection and Preprocessing</td>
                    <td style="padding: 8px;">Prepared data by handling missing values, encoding, and scaling.</td>
                    <td style="padding: 8px;">Clean and processed dataset.</td>
                </tr>
                <tr style="background-color: #1E1E1E; color: #FFFFFF;">
                    <td style="padding: 8px;">Phase 2</td>
                    <td style="padding: 8px;">Exploratory Data Analysis (EDA)</td>
                    <td style="padding: 8px;">Identified trends, relationships, and potential anomalies.</td>
                    <td style="padding: 8px;">Data-driven insights for decision-making.</td>
                </tr>
                <tr style="background-color: #1E1E1E; color: #FFFFFF;">
                    <td style="padding: 8px;">Phase 3</td>
                    <td style="padding: 8px;">Hypothesis Testing</td>
                    <td style="padding: 8px;">Validated assumptions with p-values and confidence intervals.</td>
                    <td style="padding: 8px;">Statistical evidence supporting analysis.</td>
                </tr>
                <tr style="background-color: #1E1E1E; color: #FFFFFF;">
                    <td style="padding: 8px;">Phase 4</td>
                    <td style="padding: 8px;">Feature Importance</td>
                    <td style="padding: 8px;">Highlighted the most influential features for modeling.</td>
                    <td style="padding: 8px;">Optimized feature set for modeling.</td>
                </tr>
                <tr style="background-color: #1E1E1E; color: #FFFFFF;">
                    <td style="padding: 8px;">Phase 5</td>
                    <td style="padding: 8px;">Model Development and Evaluation</td>
                    <td style="padding: 8px;">Built and assessed predictive models using metrics like accuracy and ROC-AUC.</td>
                    <td style="padding: 8px;">Deployable machine learning models.</td>
                </tr>
            </table>
            <br>
            <div style="text-align: center;">
                <b>This marks the completion of the Employment Analysis Dashboard. Thank you for exploring!</b>
            </div>
        </div>
    """, unsafe_allow_html=True)

