import pandas as pd
import numpy as np
import os

def engineer_features(input_path, output_dir):
    print("Starting Feature Engineering...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return
        
    print(f"Original columns: {df.shape[1]}")
    
    # 1. High_Company_Moves_Indicator
    # Justification: EDA showed attrition spikes significantly at 5+ companies worked.
    df['High_Company_Moves_Indicator'] = (df['NumCompaniesWorked'] >= 5).astype(int)
    
    # 2. Overtime_Indicator
    # Justification: Simple binary encoding for ML algorithms, mapping 'Yes' to 1.
    if 'OverTime' in df.columns:
        df['Overtime_Indicator'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    
    # 3. Workload_Category
    # Justification: Combines Overtime and WorkLifeBalance to identify true heavy workload burden.
    # 1 (High Workload) if OverTime is Yes and WorkLifeBalance is poor/low (<= 2). 0 otherwise.
    df['Workload_Category'] = ((df['Overtime_Indicator'] == 1) & (df['WorkLifeBalance'] <= 2)).astype(int)
    
    # 4. Workload_Risk_Indicator
    # Justification: Identifies "burnout risk" by combining High Workload with Low JobSatisfaction (<=2).
    df['Workload_Risk_Indicator'] = ((df['Workload_Category'] == 1) & (df['JobSatisfaction'] <= 2)).astype(int)
    
    # 5. Tenure_Per_Company
    # Justification: Total Working Years divided by NumCompaniesWorked (min 1 to avoid division by zero).
    # Identifies job hopping speed/stability.
    df['Tenure_Per_Company'] = df['TotalWorkingYears'] / df['NumCompaniesWorked'].replace(0, 1)
    
    print(f"Engineered columns added: High_Company_Moves_Indicator, Overtime_Indicator, Workload_Category, Workload_Risk_Indicator, Tenure_Per_Company")
    print(f"Final columns: {df.shape[1]}")
    
    # Save the feature engineered dataset
    output_path = os.path.join(output_dir, 'HR-Employee-Attrition_features.csv')
    df.to_csv(output_path, index=False)
    print(f"\nFeature engineered dataset saved to: {output_path}")

if __name__ == "__main__":
    engineer_features('outputs/cleaned_data/HR-Employee-Attrition_cleaned.csv', 'outputs/feature_engineered/')
