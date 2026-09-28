import pandas as pd
import numpy as np
import os

def run_workload_analysis(input_path, output_dir):
    print("Starting Workload Analysis...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return
        
    report = []
    report.append("=== WORKLOAD ANALYSIS REPORT ===\n")
    
    # 0. Temporary vs Chronic Overload Validation
    report.append("CRITICAL DATASET LIMITATION REGARDING CHRONIC OVERLOAD:")
    report.append("An inspection of the dataset confirms that there are NO valid time-series or date variables ")
    report.append("available to track employee data historically over time. Therefore, we CANNOT distinguish ")
    report.append("between temporary workload spikes and chronic overload. This dataset supports cross-sectional ")
    report.append("workload analysis only. Potential overload is identified below using defensible point-in-time ")
    report.append("indicators (e.g., Overtime and poor Work-Life Balance).\n")
    report.append("-" * 60 + "\n")
    
    # Mapping Attrition for aggregation if not already mapped
    if df['Attrition'].dtype == 'object':
        df['Attrition_Num'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    else:
        df['Attrition_Num'] = df['Attrition']
        
    # 1. Workload distribution
    report.append("1. Workload Distribution (Workload_Category):")
    wl_dist = df['Workload_Category'].value_counts(normalize=True).round(4) * 100
    report.append(f"Normal Workload (0): {wl_dist[0]:.2f}%")
    report.append(f"High Workload (1): {wl_dist[1]:.2f}%\n")
    
    # 2. Project workload
    report.append("2. Project Workload (using NumCompaniesWorked & High_Company_Moves_Indicator as proxy):")
    moves_dist = df['High_Company_Moves_Indicator'].value_counts(normalize=True).round(4) * 100
    report.append(f"Low/Normal Job Transitions (<5): {moves_dist[0]:.2f}%")
    report.append(f"High Job Transitions (>=5): {moves_dist[1]:.2f}%\n")
    
    # 3. Monthly working hours
    report.append("3. Monthly Working Hours:")
    report.append("As identified in Data Cleaning, the 'StandardHours' column possessed zero variance ")
    report.append("(all values = 80) and was dropped. True varying average monthly hours are NOT available.\n")
    
    # 4. Overtime
    report.append("4. Overtime Distribution:")
    ot_dist = df['Overtime_Indicator'].value_counts(normalize=True).round(4) * 100
    report.append(f"No Overtime (0): {ot_dist[0]:.2f}%")
    report.append(f"Yes Overtime (1): {ot_dist[1]:.2f}%\n")
    
    # 5. Workload versus satisfaction
    report.append("5. Workload Versus Job Satisfaction:")
    wl_sat = df.groupby('Workload_Category')['JobSatisfaction'].mean().round(2)
    report.append(f"Normal Workload Mean Satisfaction: {wl_sat[0]}")
    report.append(f"High Workload Mean Satisfaction: {wl_sat[1]}\n")
    
    # 6. Workload versus attrition
    report.append("6. Workload Versus Attrition Rate:")
    wl_att = df.groupby('Workload_Category')['Attrition_Num'].mean().round(4) * 100
    report.append(f"Normal Workload Attrition Rate: {wl_att[0]:.2f}%")
    report.append(f"High Workload Attrition Rate: {wl_att[1]:.2f}%\n")
    
    # 7. Department-level workload patterns
    report.append("7. Department-Level Workload Patterns (High Workload %):")
    dept_wl = (df.groupby('Department')['Workload_Category'].mean() * 100).round(2)
    for dept, pct in dept_wl.items():
        report.append(f"- {dept}: {pct}%")
    report.append("")
        
    # 8. High-workload groups (by JobRole)
    report.append("8. High-Workload Groups (by JobRole):")
    role_wl = (df.groupby('JobRole')['Workload_Category'].mean() * 100).sort_values(ascending=False).round(2)
    report.append("Top 5 Roles experiencing High Workload:")
    for role, pct in role_wl.head(5).items():
        report.append(f"- {role}: {pct}%")
    report.append("")
    
    # Burnout Risk Indicator
    burnout_pct = df['Workload_Risk_Indicator'].mean() * 100
    report.append("9. Extreme Burnout Risk Group:")
    report.append(f"{burnout_pct:.2f}% of the organization falls into the extreme Workload_Risk_Indicator ")
    report.append("category (High Workload + Poor Job Satisfaction).")

    report_text = "\n".join(report)
    
    # Save the report
    report_path = os.path.join(output_dir, 'workload_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
        
    print(report_text)
    print(f"\nWorkload analysis completed. Text report saved to: {report_path}")

if __name__ == "__main__":
    engineer_path = 'outputs/feature_engineered/HR-Employee-Attrition_features.csv'
    run_workload_analysis(engineer_path, 'outputs/workload_analysis/')
