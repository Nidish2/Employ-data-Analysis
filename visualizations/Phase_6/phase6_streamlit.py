import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# Set up the output path
OUTPUT_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/predictive_analytics'

# Page setup
st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    layout="wide"
)

#Phase 6: Predictive Analytics Dashboard
st.title("🚀 Employment Analysis:Predictive Analytics Dashboard")

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

# Create navigation menu in Streamlit
st.sidebar.title("Navigation")
options = ["Introduction", "Model Training", "Model Evaluation", "Hyperparameter Tuning", "Explainability", "Conclusion"] 

choice = st.sidebar.radio("Go to", options)

# Introduction Section
if choice == "Introduction":
    st.header("Introduction")
    st.write("""
    In this phase of predictive analytics, we aim to predict two important aspects related to employee management:
    - **Promotion Prediction**: Predict whether an employee will be promoted within the next 12 months.
    - **Performance Prediction**: Predict an employee’s post-promotion performance rating.
    
    The predictive models include:
    - **Logistic Regression (Baseline Model)**
    - **Decision Tree**
    - **Random Forest**
    - **XGBoost**
    - **LightGBM**
    
    We will evaluate these models using various performance metrics and also explain the model predictions using DALEX and LIME.
    """)

# Steps of Predictive Analytics
# Model Training
elif choice == "Model Training":
    st.header("Model Training")
    
    st.write("""
    In this step, we trained multiple models for predicting employee promotions and performance. The models included:
    - **Logistic Regression**: Used as a baseline model.
    - **Decision Tree**: Provides interpretability but may suffer from overfitting.
    - **Random Forest**: An ensemble of decision trees that improves predictive accuracy.
    - **XGBoost**: A powerful gradient boosting model known for high performance.
    - **LightGBM**: Another gradient boosting model optimized for speed and efficiency.
    
    All the models were trained using a set of features related to employee information, such as age, tenure, performance scores, and more. The training process involved splitting the data into training and test sets and evaluating the models on the test data.
    """)

    # Show Training Results (Models, Classification Report, and Results)
    model_files = os.listdir(OUTPUT_PATH)
    
    # Filter and show classification reports
    classification_reports = [file for file in model_files if file.endswith("_classification_report.txt")]
    st.write("### Classification Reports")
    for report in classification_reports:
        with open(os.path.join(OUTPUT_PATH, report), "r") as file:
            st.write(f"**{report}**")
            st.text(file.read())

    st.write("""
    **Final Insights**:
    - The classification reports show the overall performance of each model.
    - XGBoost and Random Forest showed higher accuracy compared to Logistic Regression, indicating their better performance on this task.
    - The Decision Tree model, while interpretable, performed less well due to overfitting.
    - These models can be used to predict employee promotions, with XGBoost being the most reliable model.
    """)

# Model Evaluation
elif choice == "Model Evaluation":
    st.header("Model Evaluation")
    
    st.write("""
    The evaluation step focuses on assessing the performance of each model using several metrics:
    - **Accuracy**: The percentage of correct predictions.
    - **Precision**: The percentage of true positives among all positive predictions.
    - **Recall**: The percentage of true positives among all actual positives.
    - **F1-score**: The harmonic mean of precision and recall.
    - **Confusion Matrix**: A matrix showing true positives, false positives, false negatives, and true negatives.
    
    The models were evaluated on their ability to predict promotions and performance accurately. The evaluation also helps in identifying potential improvements for model performance.
    """)

    # Show Training Results (Models, Classification Report, and Results)
    model_files = os.listdir(OUTPUT_PATH)
    # Show Confusion Matrices
    confusion_matrices = [file for file in model_files if file.endswith("_confusion_matrix.csv")]
    st.write("### Confusion Matrices")
    for matrix in confusion_matrices:
        matrix_df = pd.read_csv(os.path.join(OUTPUT_PATH, matrix))
        st.write(f"**{matrix}**")
        st.dataframe(matrix_df)

    st.write("""
    **Final Insights**:
    - The confusion matrices provided a clear indication of how well each model performed in terms of true positives, false positives, false negatives, and true negatives.
    - Random Forest and XGBoost had fewer false positives, making them more reliable in promotion prediction.
    - Logistic Regression had more false negatives, which could lead to missing promotions for qualified employees.
    - Overall, the models with ensemble techniques (Random Forest and XGBoost) showed better generalization performance on unseen data.
    """)

# Hyperparameter Tuning
elif choice == "Hyperparameter Tuning":
    st.header("Hyperparameter Tuning")
    
    st.write("""
    In this step, we performed hyperparameter tuning using techniques like **GridSearchCV** or **Optuna** to optimize the models for better performance.
    - Hyperparameters such as learning rate, max depth, number of estimators (trees), and others were tuned for Random Forest, XGBoost, and LightGBM models.
    - This process helps in selecting the optimal parameters that can improve the model's accuracy, reduce overfitting, and enhance the predictive power.
    """)

    st.write("""
    We tested various combinations of hyperparameters for each model to find the optimal set that resulted in the highest performance.
    - **Random Forest** and **XGBoost** benefited the most from hyperparameter tuning, with increased accuracy after tuning.
    - Hyperparameter tuning for **LightGBM** also improved its speed and efficiency, making it a good choice for large datasets.
    """)

    st.write("""
    **Final Insights**:
    - Hyperparameter tuning significantly improved the performance of **XGBoost** and **Random Forest**.
    - The **learning rate** and **n_estimators** were the most influential hyperparameters for XGBoost.
    - For **LightGBM**, tuning helped improve model speed without sacrificing too much accuracy.
    - Overall, the hyperparameter tuning ensured that the models were well-optimized for the task of predicting promotions and performance.
    """)

# Explainability
elif choice == "Explainability":
    st.header("Explainability")
    
    st.write("""
    In this step, we focused on interpreting the predictions made by the models. Using tools like **DALEX** and **LIME**, we provided insights into:
    - **Feature Importance**: Identifying which features most influenced the model's predictions.
    - **Model Interpretability**: Understanding how the model made its decisions, especially for employees that were predicted to be promoted.
    - **LIME** (Local Interpretable Model-agnostic Explanations) helped us explain individual predictions by approximating the model locally.
    - **DALEX** provided global explanations, showing how different features affected the overall model performance.
    
    These tools were used to generate plots and explain how certain employee features, like salary and performance scores, contributed to promotion decisions.
    """)

    # Show Explanations
    explanations_path = os.path.join(OUTPUT_PATH, "explanations")
    explanation_files = os.listdir(explanations_path)
    
    st.write("### Model Explanations")
    for file in explanation_files:
        if file.endswith(".txt"):
            with open(os.path.join(explanations_path, file), "r") as f:
                st.write(f"**{file}**")
                st.text(f.read())
        
        if file.endswith(".png"):
            st.image(os.path.join(explanations_path, file), caption=f"{file}", use_container_width=True)

    st.write("""
    **Final Insights**:
    - The **DALEX** and **LIME** explanations provided crucial insights into the model's decision-making process.
    - We discovered that features such as **performance score**, **salary**, and **manager rating** had the highest impact on promotion predictions.
    - The **LIME** explanation for individual predictions helped us understand why certain employees were predicted to be promoted.
    - These insights will help HR professionals trust the model's decisions and make data-driven promotion decisions.
    """)

# Conclusion Section
elif choice == "Conclusion":
    st.header("Conclusion")
    
    st.write("""
    Based on the analysis of our models for predicting employee promotions and performance:
    - **Random Forest** and **XGBoost** exhibited the highest accuracy and robustness in predicting promotions and performance.
    - **Logistic Regression**, while simple, provided valuable baseline insights for performance.
    - The **Confusion Matrices** showed that models like Random Forest had lower false positive and false negative rates compared to Decision Trees.
    - **Explanations** from DALEX and LIME proved invaluable in understanding how certain features (e.g., salary, performance scores, age, etc.) influenced promotion and performance outcomes.
    - **Hyperparameter Tuning** was crucial in improving model performance, especially for XGBoost and LightGBM.
    
    The insights from this predictive modeling phase will help the HR department in making better decisions about employee promotions and understanding key performance indicators for each employee. The final models will be useful for real-time decision-making processes.
    """)
    
    st.write("#### Summary:")

    st.write("""
    In summary, our predictive analytics process has provided valuable insights into the models' capabilities and how they can be used to predict promotions and performance.
    
    - **Best Performing Models**: The **Random Forest** and **XGBoost** models performed the best overall in predicting promotions and performance, with high accuracy and fewer false positives.
    - **Key Features**: Features such as **performance score**, **salary**, **manager rating**, and **engagement score** were found to be the most influential predictors.
    - **Model Interpretability**: Tools like **DALEX** and **LIME** played a key role in making the models interpretable and transparent.
    - **Hyperparameter Tuning**: The tuning process improved model performance, especially for **XGBoost** and **Random Forest**.
    
    The final models will help the HR department in making better, data-driven decisions regarding employee promotions and understanding the factors influencing post-promotion performance.
    """)

    st.write("#### Final Insights:")
    st.write("""
    1. **Best Performing Models**: Random Forest and XGBoost were the best-performing models for both promotion prediction and performance prediction tasks.
    2. **Key Features Influencing Predictions**: Features such as **salary**, **performance score**, and **manager ratings** were found to be the most influential in the promotion and performance prediction models.
    3. **Importance of Model Interpretability**: DALEX and LIME played a key role in explaining the model's decision-making process, enhancing trust and transparency.
    4. **Model Improvement**: Hyperparameter tuning significantly improved the performance of XGBoost and LightGBM models, helping them to generalize better on unseen data.
    5. **Next Steps**: The insights from this phase will be used to refine the models and apply them in real-world HR analytics for predicting promotions and employee performance.

    **Output Directory**: All outputs, including classification reports, confusion matrices, and explanations, have been saved in the following directory:  
    `C:/Users/nidis/Documents/Employment_Analysis/outputs/predictive_analytics`
    """)
