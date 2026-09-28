# HR Analytics — Employee Workload, Satisfaction & Attrition Analysis

## 1. Project Description
This project is an end-to-end data science and machine learning pipeline that analyzes employee data to understand workload distribution, job satisfaction, and the key drivers of employee attrition. It includes an interactive, fully dynamic Streamlit dashboard for real-time analysis of HR datasets.

## 2. Problem Statement
High employee turnover is incredibly costly to organizations. HR departments often struggle to identify whether chronic overload, serial job transitions, or low job satisfaction is primarily driving turnover. This project seeks to bridge the gap by building a predictive and analytical pipeline that automatically uncovers hidden workforce risk factors.

## 3. Objectives
- Dynamically ingest and validate any tabular HR dataset.
- Clean data and engineer targeted features (e.g., Workload Categories, Career Velocity).
- Perform Exploratory Data Analysis (EDA) on workloads and satisfaction.
- Train a Random Forest Machine Learning model to predict employee attrition.
- Extract actionable, data-driven HR insights to inform management strategies.

## 4. Dataset Description
The system is built to ingest standard HR datasets containing variables such as:
- Employee Job Satisfaction (1-4 scale)
- Department and Job Role
- Overtime Status
- Standard Working Hours
- Number of Previous Companies Worked (Proxy for career velocity)
- Employee Attrition (Yes/No)

*(Note: Raw data is purposely excluded from this repository for privacy and security.)*

## 5. Technologies Used
- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Web Dashboard:** Streamlit
- **Model Serialization:** Joblib

## 6. Project Workflow
1. **Data Upload & Validation:** Ensures file structure and detects key HR variables.
2. **Data Cleaning:** Removes duplicates and handles missing values.
3. **Feature Engineering:** Creates composite indicators for high workload and career velocity.
4. **Exploratory Data Analysis (EDA):** Identifies macro trends.
5. **Machine Learning:** Trains an imbalanced-class Random Forest classifier.
6. **Dashboard Generation:** Synthesizes visualizations, model metrics, and insights.

## 7. Machine Learning Algorithm
The core predictive engine uses a **Random Forest Classifier**. Given that employee attrition is inherently imbalanced (e.g., ~84% of employees typically stay), the algorithm leverages balanced class weights and is evaluated strictly on Precision, Recall, F1-Score, and ROC-AUC rather than misleading baseline accuracy.

## 8. Dashboard Features
The Streamlit application features a fully dynamic, reactive interface:
- **Responsive KPI Cards:** Automatically calculate aggregate statistics based on detected columns.
- **Dynamic Charting:** Generates Overtime vs. Satisfaction vs. Attrition plots.
- **On-the-Fly ML:** Trains the Random Forest model directly in the browser.
- **Action Plan Generation:** Presents strategic HR recommendations.

## 9. How to Install Dependencies
To run this project locally, clone the repository and install the dependencies:
```bash
git clone https://github.com/srinath-450718/HR-analytics.git
cd HR-analytics
pip install -r requirements.txt
```

## 10. How to Run the Streamlit Application
Execute the following command in your terminal from the project root:
```bash
streamlit run app.py
```
Then navigate to `http://localhost:8501` in your web browser.

## 11. Dataset Download Instructions
You can upload any CSV dataset containing standard HR variables into the running Streamlit dashboard. 
Alternatively, IBM's public "HR Employee Attrition" dataset is recommended as a benchmark.

## 12. Project Limitations
- **Cross-Sectional vs. Time-Series Data:** The system currently analyzes point-in-time cross-sectional data. It cannot perfectly separate a *temporary* workload spike from *chronic* historical overload unless historical timestamps are provided in the CSV.
- **Work-Hour Variance:** If the dataset lacks variance in working hours (e.g., static 80 hours recorded for all), the engine relies heavily on Overtime and Work-Life Balance proxies.

## 13. Author
**Srinath**
College Project / GitHub Portfolio
