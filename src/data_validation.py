import pandas as pd
import os

def validate_data(file_path):
    print("Starting data validation...")
    
    # 1. Dataset file name
    file_name = os.path.basename(file_path)
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return
        
    # 2 & 3. Shape
    rows, cols = df.shape
    
    # 4 & 5. Column names and Data types
    dtypes = df.dtypes
    
    # 6. Missing values
    missing = df.isnull().sum()
    
    # 7. Duplicate records
    duplicates = df.duplicated().sum()
    
    # 8. Unique values
    uniques = df.nunique()
    
    # 9 & 10. Numerical and Categorical columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 11. Basic statistical summary
    stats_summary = df.describe().to_string()
    
    # 12. Target/attrition column
    target_col = 'Attrition' if 'Attrition' in df.columns else 'Not Found'
    
    # Specific variables check
    expected_vars = {
        'Employee satisfaction': ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction'],
        'Number of projects': ['NumCompaniesWorked'], # IBM dataset typically doesn't have 'number of projects' like some other datasets, checking alternatives
        'Average monthly hours': ['StandardHours'], # Or maybe DailyRate/HourlyRate depending on dataset
        'Overtime': ['OverTime'],
        'Years at company': ['YearsAtCompany'],
        'Department': ['Department'],
        'Salary': ['MonthlyIncome', 'HourlyRate', 'DailyRate', 'MonthlyRate'],
        'Employee attrition': ['Attrition']
    }
    
    found_vars = {}
    for conceptual_var, possible_names in expected_vars.items():
        found = [col for col in possible_names if col in df.columns]
        found_vars[conceptual_var] = found if found else "Not found in dataset"
        
    report = []
    report.append(f"--- DATA VALIDATION REPORT ---")
    report.append(f"1. Dataset file name: {file_name}")
    report.append(f"2. Number of rows: {rows}")
    report.append(f"3. Number of columns: {cols}")
    report.append(f"\n4 & 5. Column names and Data types:\n{dtypes.to_string()}")
    report.append(f"\n6. Missing values:\n{missing[missing > 0].to_string() if missing.sum() > 0 else 'No missing values found.'}")
    report.append(f"\n7. Duplicate records: {duplicates}")
    report.append(f"\n8. Unique values (sample):\n{uniques.head(10).to_string()}")
    report.append(f"\n9. Numerical columns ({len(num_cols)}):\n{', '.join(num_cols)}")
    report.append(f"\n10. Categorical columns ({len(cat_cols)}):\n{', '.join(cat_cols)}")
    report.append(f"\n11. Basic statistical summary:\n{stats_summary}")
    report.append(f"\n12. Target/attrition column: {target_col}")
    
    report.append(f"\n--- SPECIFIC VARIABLES CHECK ---")
    for k, v in found_vars.items():
        report.append(f"{k}: {v}")
        
    report_text = "\n".join(report)
    
    print(report_text)
    
    # Save report
    os.makedirs('outputs/eda', exist_ok=True)
    with open('outputs/eda/validation_report.txt', 'w') as f:
        f.write(report_text)
    print("\nReport saved to outputs/eda/validation_report.txt")

if __name__ == "__main__":
    validate_data('data/HR-Employee-Attrition.csv')
