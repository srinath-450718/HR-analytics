import pandas as pd
import numpy as np
import os

def run_eda(input_path, output_dir):
    print("Starting EDA...")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    
    report = []
    report.append("=== EXPLORATORY DATA ANALYSIS REPORT ===\n")
    
    # 1. Descriptive Statistics for numerical variables
    num_cols = df.select_dtypes(include=[np.number]).columns
    desc_stats = df[num_cols].describe().T[['mean', '50%', 'min', 'max', 'std', '25%', '75%']]
    desc_stats.columns = ['Mean', 'Median', 'Min', 'Max', 'Std Dev', 'Q1', 'Q3']
    
    report.append("1. Descriptive Statistics:")
    report.append(desc_stats.to_string())
    report.append("\n" + "-"*50 + "\n")
    
    # Transform Attrition to numeric for easy agg
    df['Attrition_Num'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    projects = 'NumCompaniesWorked'
    overtime = 'OverTime'
    
    # 2. Base Distributions
    report.append("2. Variable Analysis:")
    
    report.append(f"\nAttrition Rate: {(df['Attrition_Num'].mean() * 100):.2f}%")
    report.append(f"\nDepartment Breakdown:\n{df['Department'].value_counts(normalize=True).round(4) * 100}")
    report.append(f"\nOvertime Breakdown:\n{df['OverTime'].value_counts(normalize=True).round(4) * 100}")
    
    report.append("\n" + "-"*50 + "\n")
    report.append("3. Relationship Investigations:\n")
    
    # Projects and satisfaction
    report.append("A) Projects (NumCompaniesWorked) vs JobSatisfaction (Mean):")
    proj_sat = df.groupby(projects)['JobSatisfaction'].mean().round(2)
    report.append(proj_sat.to_string())
    
    # Overtime and satisfaction
    report.append("\nB) Overtime vs JobSatisfaction (Mean):")
    ot_sat = df.groupby(overtime)['JobSatisfaction'].mean().round(2)
    report.append(ot_sat.to_string())
    
    # Projects and attrition
    report.append("\nC) Projects (NumCompaniesWorked) vs Attrition Rate (%):")
    proj_att = (df.groupby(projects)['Attrition_Num'].mean() * 100).round(2)
    report.append(proj_att.to_string())
    
    # Overtime and attrition
    report.append("\nD) Overtime vs Attrition Rate (%):")
    ot_att = (df.groupby(overtime)['Attrition_Num'].mean() * 100).round(2)
    report.append(ot_att.to_string())
    
    # Satisfaction and attrition
    report.append("\nE) JobSatisfaction vs Attrition Rate (%):")
    sat_att = (df.groupby('JobSatisfaction')['Attrition_Num'].mean() * 100).round(2)
    report.append(sat_att.to_string())
    
    report.append("\nNote: 'Average monthly hours' was not present as a variable with variance in this dataset (StandardHours was constant at 80 and dropped). Therefore, its specific relationships were skipped. Salary is represented by MonthlyIncome in the descriptive stats.")
    
    report.append("\n" + "-"*50 + "\n")
    
    # 3. Correlation Matrix
    report.append("4. Correlation Matrix (Top 10 correlating pairs with Attrition_Num, excluding Attrition_Num itself):")
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    # Get highest correlations with Attrition
    attrition_corr = corr_matrix['Attrition_Num'].sort_values()
    report.append(attrition_corr.drop('Attrition_Num').to_string())
    
    # Save text report
    report_path = os.path.join(output_dir, 'eda_report.txt')
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
        
    print("\n".join(report))

    print(f"\nEDA completed. Text report saved to '{output_dir}'.")

if __name__ == "__main__":
    run_eda('outputs/cleaned_data/HR-Employee-Attrition_cleaned.csv', 'outputs/eda/')
