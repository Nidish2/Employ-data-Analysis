import streamlit as st
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, roc_curve
import seaborn as sns
import folium
from scipy.stats import kurtosis, skew

# Load data
def load_data():
    file_path = 'Employee.xlsx'
    try:
        df = pd.read_excel(file_path, sheet_name="Employee Sample Data")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data()

if data is None:
    st.stop()

# Add "Churn" column if "Exit Date" exists
if "Exit Date" in data.columns:
    data["Churn"] = data["Exit Date"].notna().astype(int)

# App layout
st.title("Employee Data Analytics")
st.sidebar.title("Options")
option = st.sidebar.selectbox("Select Analysis Type:", [
    "Overview",
    "Visualization",
    "Clustering",
    "Churn Prediction",
    "Attrition Analysis",
    "Turnover Probability",
    "Salary Prediction",
])

# Filter options
st.sidebar.title("Filters")
department = st.sidebar.multiselect("Select Department:", data["Department"].dropna().unique())
gender = st.sidebar.multiselect("Select Gender:", data["Gender"].dropna().unique())
age_range = st.sidebar.slider("Select Age Range:", int(data["Age"].min()), int(data["Age"].max()), (20, 50))

# Apply filters
filtered_data = data.copy()
if department:
    filtered_data = filtered_data[filtered_data["Department"].isin(department)]
if gender:
    filtered_data = filtered_data[filtered_data["Gender"].isin(gender)]
filtered_data = filtered_data[(filtered_data["Age"] >= age_range[0]) & (filtered_data["Age"] <= age_range[1])]

if option == "Overview":
    st.header("Employee Dataset Overview")
    st.write(filtered_data.head())
    st.write("*Dataset Summary:*")
    st.write(filtered_data.describe())

    # Advanced statistics (Skewness, Kurtosis)
    st.write("*Advanced Summary Statistics:*")
    numeric_columns = filtered_data.select_dtypes(include=["number"]).columns

    for col in numeric_columns:
        st.write(f"{col}:")
        st.write(f"Skewness: {skew(filtered_data[col].dropna()):.2f}")
        st.write(f"Kurtosis: {kurtosis(filtered_data[col].dropna()):.2f}")

    # Percentile ranges for numeric columns
    st.write("*Percentile Distribution:*")
    percentiles = filtered_data[numeric_columns].quantile([0.25, 0.5, 0.75])
    st.write(percentiles)

elif option == "Visualization":
    st.header("Data Visualization")
    
    # Select the type of visualization
    vis_option = st.selectbox("Select Visualization Type:", ["Histogram", "Scatter Plot", "Box Plot"])

    if vis_option == "Histogram":
        col = st.selectbox("Select Column for Histogram:", ["Age", "Annual Salary", "Bonus %"])
        fig, ax = plt.subplots()
        ax.hist(filtered_data[col].dropna(), bins=15, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    elif vis_option == "Scatter Plot":
        x_col = st.selectbox("Select X-axis for Scatter Plot:", ["Age", "Annual Salary", "Bonus %"])
        y_col = st.selectbox("Select Y-axis for Scatter Plot:", ["Age", "Annual Salary", "Bonus %"])
        fig, ax = plt.subplots()
        ax.scatter(filtered_data[x_col], filtered_data[y_col], color='purple', edgecolor='black')
        ax.set_title(f"Scatter Plot of {x_col} vs {y_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig)

    elif vis_option == "Box Plot":
        col = st.selectbox("Select Column for Box Plot:", ["Age", "Annual Salary", "Bonus %"])
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_data, x=col, ax=ax, color='lightgreen')
        ax.set_title(f"Box Plot of {col}")
        st.pyplot(fig)

elif option == "Clustering":
    st.header("Employee Clustering")
    features = st.multiselect("Select Features for Clustering:", ["Age", "Annual Salary", "Bonus %"], default=["Age", "Annual Salary"])

    if len(features) > 1:
        try:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(filtered_data[features].dropna())

            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            filtered_data.loc[filtered_data.index[:len(clusters)], "Cluster"] = clusters

            st.write("Clustering Results:")
            st.write(filtered_data[["Full Name"] + features + ["Cluster"]].head())

            fig, ax = plt.subplots()
            scatter = ax.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters, cmap="viridis")
            ax.set_title("Employee Clusters")
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            plt.colorbar(scatter, ax=ax, label="Cluster")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error in clustering: {e}")
    else:
        st.warning("Please select at least two features for clustering.")

elif option == "Churn Prediction":
    st.header("Employee Churn Prediction")

    # Ensure "Churn" column exists in filtered data
    if "Churn" in filtered_data.columns:
        features = ["Age", "Annual Salary", "Bonus %"]
        if all(feature in filtered_data.columns for feature in features):
            df_model = filtered_data[features + ["Churn"]].dropna()

            X = df_model[features]
            y = df_model["Churn"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Predictions and evaluation
            y_pred = model.predict(X_test)
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Feature importance
            importance = model.feature_importances_
            importance_df = pd.DataFrame({"Feature": features, "Importance": importance})
            importance_df = importance_df.sort_values(by="Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Feature"))
        else:
            st.error("Required columns are missing in the filtered data for churn prediction.")
    else:
        st.warning("The 'Exit Date' column is missing or invalid. Churn analysis cannot be performed.")

elif option == "Attrition Analysis":
    st.header("Employee Attrition Analysis")

    # Attrition rate by department
    department_counts = filtered_data["Department"].value_counts()
    attrition_by_department = (filtered_data.groupby("Department")["Exit Date"].count() / department_counts).fillna(0)
    attrition_by_department = attrition_by_department.sort_values(ascending=False)

    st.subheader("Attrition Rate by Department")
    st.bar_chart(attrition_by_department)

    # Attrition by gender
    gender_counts = filtered_data["Gender"].value_counts()
    attrition_by_gender = (filtered_data.groupby("Gender")["Exit Date"].count() / gender_counts).fillna(0)
    attrition_by_gender = attrition_by_gender.sort_values(ascending=False)

    st.subheader("Attrition Rate by Gender")
    st.bar_chart(attrition_by_gender)

    # Heatmap of correlations
    st.subheader("Correlation Heatmap")
    numeric_columns = filtered_data.select_dtypes(include=['number']).columns
    filtered_data_numeric = filtered_data[numeric_columns]
    corr = filtered_data_numeric.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif option == "Turnover Probability":
    st.header("Employee Turnover Probability")

    if "Churn" in filtered_data.columns and "Age" in filtered_data.columns and "Annual Salary" in filtered_data.columns:
        features = ["Age", "Annual Salary", "Bonus %"]
        df_model = filtered_data[features + ["Churn"]].dropna()

        X = df_model[features]
        y = df_model["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

        st.write("Logistic Regression Model")
        st.text(classification_report(y_test, y_pred))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="blue", label=f"AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve for Turnover Prediction")
        st.pyplot(fig)
    else:
        st.warning("Required columns for turnover prediction are missing.")

elif option == "Salary Prediction":
    st.header("Employee Salary Prediction")

    if "Annual Salary" in filtered_data.columns and "Age" in filtered_data.columns and "Department" in filtered_data.columns:
        features = ["Age", "Bonus %", "Department"]
        df_model = filtered_data[features + ["Annual Salary"]].dropna()

        df_model["Department"] = df_model["Department"].astype("category").cat.codes

        X = df_model[features]
        y = df_model["Annual Salary"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.write("Regression Model Metrics")
        st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R2 Score: {model.score(X_test, y_test):.2f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='purple', alpha=0.6)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.set_xlabel("Actual Salary")
        ax.set_ylabel("Predicted Salary")
        ax.set_title("Predicted vs Actual Salary")
        st.pyplot(fig)
    else: 
        st.warning("Required columns for salary prediction are missing.")