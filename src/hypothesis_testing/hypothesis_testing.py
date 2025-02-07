import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import os

# -------------------------- Constants --------------------------

DATA_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/data/transformed_data.csv'
HYPOTHESIS_TESTING_OUTPUT_PATH = 'C:/Users/nidis/Documents/Employment_Analysis/outputs/hypothesis_testing'

# -------------------------- Load Data --------------------------

data = pd.read_csv(DATA_PATH)
print("Data Loaded Successfully. Shape:", data.shape)

# Create output folder if it doesn't exist
os.makedirs(HYPOTHESIS_TESTING_OUTPUT_PATH, exist_ok=True)

# -------------------------- Hypotheses Testing --------------------------

# 1. Work-Life Balance and Employee Satisfaction
# Null Hypothesis (H0): Work-life balance score is independent of employee satisfaction score.
# Alternative Hypothesis (H1): Work-life balance score is not independent of employee satisfaction score.
if 'work_life_balance_score' in data.columns and 'employee_satisfaction_score' in data.columns:
    corr, p_value = stats.pearsonr(data['work_life_balance_score'], data['employee_satisfaction_score'])
    print(f"Pearson correlation between work-life balance and satisfaction: corr={corr}, p_value={p_value}")

    # Save results
    with open(os.path.join(HYPOTHESIS_TESTING_OUTPUT_PATH, 'work_life_balance_satisfaction.txt'), 'w') as f:
        f.write(f"Pearson correlation between work-life balance and satisfaction:\n")
        f.write(f"corr={corr}, p_value={p_value}\n")

# 2. Salary Differences by Gender
# Null Hypothesis (H0): Mean salaries for male and female employees are equal.
# Alternative Hypothesis (H1): Mean salaries for male and female employees are different.
if 'gender_full' in data.columns and 'salary' in data.columns:
    male_salaries = data[data['gender_full'] == 1]['salary']
    female_salaries = data[data['gender_full'] == 0]['salary']
    t_stat, p_value = stats.ttest_ind(male_salaries, female_salaries, equal_var=False)
    print(f"T-test for Salary by Gender: t_stat={t_stat}, p_value={p_value}")

    # Save results
    with open(os.path.join(HYPOTHESIS_TESTING_OUTPUT_PATH, 'salary_by_gender.txt'), 'w') as f:
        f.write(f"T-test for Salary by Gender:\n")
        f.write(f"t_stat={t_stat}, p_value={p_value}\n")

# 3. Attrition and Overtime Hours
# Null Hypothesis (H0): Attrition is independent of overtime hours.
# Alternative Hypothesis (H1): Attrition is dependent on overtime hours.
if 'overtime_hours' in data.columns and 'Attrition' in data.columns:
    attrition_groups = data.groupby('Attrition')['overtime_hours']
    f_stat, p_value = stats.f_oneway(*[group for _, group in attrition_groups])
    print(f"ANOVA for Attrition and Overtime Hours: f_stat={f_stat}, p_value={p_value}")

    # Save results
    with open(os.path.join(HYPOTHESIS_TESTING_OUTPUT_PATH, 'attrition_overtime.txt'), 'w') as f:
        f.write(f"ANOVA for Attrition and Overtime Hours:\n")
        f.write(f"f_stat={f_stat}, p_value={p_value}\n")

# 4. Department and Performance Scores
# Null Hypothesis (H0): Mean performance scores are equal across all departments.
# Alternative Hypothesis (H1): Mean performance scores differ across departments.
if 'department_name' in data.columns and 'performance_score' in data.columns:
    anova_model = sm.formula.ols('performance_score ~ department_name', data=data).fit()
    anova_results = sm.stats.anova_lm(anova_model, typ=2)
    print("ANOVA for Department and Performance Scores:\n", anova_results)

    # Save results
    anova_results.to_csv(os.path.join(HYPOTHESIS_TESTING_OUTPUT_PATH, 'performance_department_anova.csv'))

# 5. Salary Hike and Turnover Risk
# Null Hypothesis (H0): Salary hike percentage is independent of turnover risk.
# Alternative Hypothesis (H1): Salary hike percentage is dependent on turnover risk.
if 'salary_hike_percent' in data.columns and 'turnover_risk_index' in data.columns:
    corr, p_value = stats.pearsonr(data['salary_hike_percent'], data['turnover_risk_index'])
    print(f"Correlation between Salary Hike and Turnover Risk: corr={corr}, p_value={p_value}")

    # Save results
    with open(os.path.join(HYPOTHESIS_TESTING_OUTPUT_PATH, 'salary_hike_turnover.txt'), 'w') as f:
        f.write(f"Correlation between Salary Hike and Turnover Risk:\n")
        f.write(f"corr={corr}, p_value={p_value}\n")

# 6. Manager Ratings and Attrition
# Null Hypothesis (H0): Manager ratings are independent of attrition.
# Alternative Hypothesis (H1): Manager ratings are not independent of attrition.
if 'manager_rating' in data.columns and 'Attrition' in data.columns:
    t_stat, p_value = stats.ttest_ind(data[data['Attrition'] == 1]['manager_rating'],
                                      data[data['Attrition'] == 0]['manager_rating'])
    print(f"T-test for Manager Ratings and Attrition: t_stat={t_stat}, p_value={p_value}")

    # Save results
    with open(os.path.join(HYPOTHESIS_TESTING_OUTPUT_PATH, 'manager_rating_attrition.txt'), 'w') as f:
        f.write(f"T-test for Manager Ratings and Attrition:\n")
        f.write(f"t_stat={t_stat}, p_value={p_value}\n")

# 7. Absenteeism and Performance Scores
# Null Hypothesis (H0): Absenteeism rate is independent of performance score.
# Alternative Hypothesis (H1): Absenteeism rate is dependent on performance score.
if 'absenteeism_rate' in data.columns and 'performance_score' in data.columns:
    corr, p_value = stats.pearsonr(data['absenteeism_rate'], data['performance_score'])
    print(f"Correlation between Absenteeism Rate and Performance Score: corr={corr}, p_value={p_value}")

    # Save results
    with open(os.path.join(HYPOTHESIS_TESTING_OUTPUT_PATH, 'absenteeism_performance.txt'), 'w') as f:
        f.write(f"Correlation between Absenteeism Rate and Performance Score:\n")
        f.write(f"corr={corr}, p_value={p_value}\n")

# 8. Peer Feedback and Post-Promotion Performance
# Null Hypothesis (H0): Peer feedback scores are independent of post-promotion performance.
# Alternative Hypothesis (H1): Peer feedback scores are not independent of post-promotion performance.
if 'peer_feedback_score' in data.columns and 'post_promotion_performance' in data.columns:
    corr, p_value = stats.pearsonr(data['peer_feedback_score'], data['post_promotion_performance'])
    print(f"Correlation between Peer Feedback and Post-Promotion Performance: corr={corr}, p_value={p_value}")

    # Save results
    with open(os.path.join(HYPOTHESIS_TESTING_OUTPUT_PATH, 'peer_feedback_promotion.txt'), 'w') as f:
        f.write(f"Correlation between Peer Feedback and Post-Promotion Performance:\n")
        f.write(f"corr={corr}, p_value={p_value}\n")

# -------------------------- Validation of Assumptions --------------------------

# 1. Normality Check for Salary
if 'salary' in data.columns:
    _, p_value_salary = stats.shapiro(data['salary'])
    print(f"Shapiro-Wilk test for Salary Normality: p_value={p_value_salary}")
    with open(os.path.join(HYPOTHESIS_TESTING_OUTPUT_PATH, 'normality_salary.txt'), 'w') as f:
        f.write(f"Shapiro-Wilk test for Performance Improvement Normality: p_value={p_value_salary}\n")
# 2. Homogeneity of Variance for Performance Score
if 'department_name' in data.columns and 'performance_score' in data.columns:
    p_value_levene = stats.levene(*[data[data['department_name'] == dep]['performance_score'] for dep in data['department_name'].unique()])
    print(f"Levene's Test for Homogeneity of Variance in Performance Score: p_value={p_value_levene}")
    with open(os.path.join(HYPOTHESIS_TESTING_OUTPUT_PATH, 'homogeneity_performance_score.txt'), 'w') as f:
        f.write(f"Shapiro-Wilk test for Performance Improvement Normality: p_value={p_value_levene}\n")

# 3. Normality Check for Performance Improvement
if 'performance_improvement' in data.columns:
    _, p_value_performance_improvement = stats.shapiro(data['performance_improvement'])
    print(f"Shapiro-Wilk test for Performance Improvement Normality: p_value={p_value_performance_improvement}")
    with open(os.path.join(HYPOTHESIS_TESTING_OUTPUT_PATH, 'normality_performance_improvement.txt'), 'w') as f:
        f.write(f"Shapiro-Wilk test for Performance Improvement Normality: p_value={p_value_performance_improvement}\n")

# 4. Homogeneity of Variances for Salary across Tenure Buckets
if 'tenure_bucket' in data.columns and 'salary' in data.columns:
    p_value_levene_salary_tenure = stats.levene(*[data[data['tenure_bucket'] == bucket]['salary'] for bucket in data['tenure_bucket'].unique()])
    print(f"Levene's Test for Salary Homogeneity across Tenure Buckets: p_value={p_value_levene_salary_tenure}")
    with open(os.path.join(HYPOTHESIS_TESTING_OUTPUT_PATH, 'homogeneity_salary_tenure.txt'), 'w') as f:
        f.write(f"Levene's Test for Salary Homogeneity across Tenure Buckets: p_value={p_value_levene_salary_tenure}\n")

# 5. Multicollinearity Check using VIF
if {'salary', 'performance_score', 'engagement_index', 'turnover_risk_index', 'employee_satisfaction_score'}.issubset(data.columns):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    features = ['salary', 'performance_score', 'engagement_index', 'turnover_risk_index', 'employee_satisfaction_score']
    X = data[features].dropna()
    X = sm.add_constant(X)
    vif_data = pd.DataFrame({'Feature': features, 'VIF': [variance_inflation_factor(X.values, i+1) for i in range(len(features))]})
    print("VIF Results for Multicollinearity Check:\n", vif_data)
    vif_data.to_csv(os.path.join(HYPOTHESIS_TESTING_OUTPUT_PATH, 'vif_multicollinearity_check.csv'), index=False)

# 6. Autocorrelation Test for Attrition
if 'Attrition' in data.columns:
    from statsmodels.stats.stattools import durbin_watson
    if 'turnover_risk_index' in data.columns:
        dw_stat = durbin_watson(data['turnover_risk_index'])
        print(f"Durbin-Watson Test for Autocorrelation in Turnover Risk Index: dw_stat={dw_stat}")
        with open(os.path.join(HYPOTHESIS_TESTING_OUTPUT_PATH, 'autocorrelation_attrition.txt'), 'w') as f:
            f.write(f"Durbin-Watson Test for Autocorrelation in Turnover Risk Index: dw_stat={dw_stat}\n")

# -------------------------- Completion Message --------------------------

print("\nHypothesis testing completed. Outputs saved to", HYPOTHESIS_TESTING_OUTPUT_PATH)
