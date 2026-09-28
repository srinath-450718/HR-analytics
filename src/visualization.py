import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_visualizations(input_path, output_dir):
    print("Starting Data Visualization...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return
        
    # Ensure Attrition is numeric for plotting proportions
    if 'Attrition' in df.columns and df['Attrition'].dtype == 'object':
        df['Attrition_Num'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    else:
        df['Attrition_Num'] = df['Attrition']
        
    sns.set_theme(style="whitegrid")
    
    # 1. Workload-related variable by department
    plt.figure(figsize=(10, 6))
    dept_wl = df.groupby('Department')['Workload_Category'].mean().reset_index()
    dept_wl['Workload_Category'] *= 100
    sns.barplot(data=dept_wl, x='Department', y='Workload_Category', palette='Blues_d')
    plt.title("High Workload Percentage by Department", fontsize=14, fontweight='bold')
    plt.xlabel("Department", fontsize=12)
    plt.ylabel("Percentage of Employees with High Workload (%)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_workload_by_department.png'))
    plt.close()
    
    # 2. Number of projects (NumCompaniesWorked) versus employee satisfaction
    plt.figure(figsize=(10, 6))
    proj_sat = df.groupby('NumCompaniesWorked')['JobSatisfaction'].mean().reset_index()
    sns.lineplot(data=proj_sat, x='NumCompaniesWorked', y='JobSatisfaction', marker='o', color='purple', linewidth=2)
    plt.title("Job Satisfaction by Number of Prior Companies (Projects Proxy)", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Companies Worked", fontsize=12)
    plt.ylabel("Average Job Satisfaction (1-4)", fontsize=12)
    plt.ylim(1, 4)
    plt.xticks(range(0, 10))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_projects_vs_satisfaction.png'))
    plt.close()
    
    # 3. Average monthly hours versus employee satisfaction
    # Since monthly hours is missing, we use Overtime as the closest workload time-metric
    plt.figure(figsize=(8, 6))
    ot_sat = df.groupby('OverTime')['JobSatisfaction'].mean().reset_index()
    sns.barplot(data=ot_sat, x='OverTime', y='JobSatisfaction', palette='Set2')
    plt.title("Job Satisfaction by Overtime Status\n(Proxy for Monthly Hours due to dataset limitations)", fontsize=14, fontweight='bold')
    plt.xlabel("Overtime Status", fontsize=12)
    plt.ylabel("Average Job Satisfaction (1-4)", fontsize=12)
    plt.ylim(1, 4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_overtime_vs_satisfaction.png'))
    plt.close()
    
    # 4. Overtime versus employee attrition
    plt.figure(figsize=(8, 6))
    ot_att = df.groupby('OverTime')['Attrition_Num'].mean().reset_index()
    ot_att['Attrition_Num'] *= 100
    sns.barplot(data=ot_att, x='OverTime', y='Attrition_Num', palette='Reds_d')
    plt.title("Employee Attrition Rate by Overtime Status", fontsize=14, fontweight='bold')
    plt.xlabel("Overtime Status", fontsize=12)
    plt.ylabel("Attrition Rate (%)", fontsize=12)
    for i, v in enumerate(ot_att['Attrition_Num']):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_overtime_vs_attrition.png'))
    plt.close()
    
    # 5. Satisfaction versus attrition
    plt.figure(figsize=(10, 6))
    sat_att = df.groupby('JobSatisfaction')['Attrition_Num'].mean().reset_index()
    sat_att['Attrition_Num'] *= 100
    sns.barplot(data=sat_att, x='JobSatisfaction', y='Attrition_Num', palette='viridis')
    plt.title("Employee Attrition Rate by Job Satisfaction Level", fontsize=14, fontweight='bold')
    plt.xlabel("Job Satisfaction (1=Low, 4=High)", fontsize=12)
    plt.ylabel("Attrition Rate (%)", fontsize=12)
    for i, v in enumerate(sat_att['Attrition_Num']):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_satisfaction_vs_attrition.png'))
    plt.close()

    print(f"5 visualizations successfully generated and saved to {output_dir}")

if __name__ == "__main__":
    engineer_path = 'outputs/feature_engineered/HR-Employee-Attrition_features.csv'
    generate_visualizations(engineer_path, 'outputs/visualizations/')
