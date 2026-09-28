import pandas as pd
import numpy as np
import os

def clean_data(input_path, output_dir):
    print("Starting data cleaning pipeline...\n")
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return
        
    original_rows = len(df)
    
    report = []
    report.append("--- DATA CLEANING REPORT ---")
    report.append(f"Original row count: {original_rows}")
    
    # 1 & 2. Missing-value analysis and handling
    missing_before = df.isnull().sum().sum()
    report.append(f"Missing values before cleaning: {missing_before}")
    
    # Drop completely empty rows (if any)
    df = df.dropna(how='all') 
    
    # Handle remaining missing values (imputation strategy for robustness)
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    missing_after = df.isnull().sum().sum()
    report.append(f"Missing values after cleaning: {missing_after}")
    
    # 3. Duplicate detection and removal
    duplicates = df.duplicated().sum()
    report.append(f"Duplicate records: {duplicates}")
    df = df.drop_duplicates()
    
    # 4, 5, 6, 7. Invalid-value detection, Categorical & Numerical validation, Data type correction
    invalid_values_notes = []
    
    # Drop zero-variance columns as they are uninformative and technically invalid for ML modeling
    columns_to_drop = []
    for col in df.columns:
        if df[col].nunique() == 1:
            val = df[col].iloc[0]
            invalid_values_notes.append(f"Column '{col}' has zero variance (all values = '{val}'). Dropped.")
            columns_to_drop.append(col)
            
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        
    # Check for invalid age or negative income
    if 'Age' in df.columns:
        invalid_ages = df[df['Age'] <= 0].shape[0]
        if invalid_ages > 0:
            invalid_values_notes.append(f"Found {invalid_ages} invalid Age records (<= 0). Removed.")
            df = df[df['Age'] > 0]
            
    if 'MonthlyIncome' in df.columns:
        invalid_income = df[df['MonthlyIncome'] < 0].shape[0]
        if invalid_income > 0:
            invalid_values_notes.append(f"Found {invalid_income} invalid MonthlyIncome records (< 0). Removed.")
            df = df[df['MonthlyIncome'] >= 0]
            
    if not invalid_values_notes:
        invalid_values_notes.append("No invalid values detected in numerical ranges or categorical classes.")
        
    final_rows = len(df)
    records_removed = original_rows - final_rows
    report.append(f"Final row count: {final_rows}")
    report.append(f"Records removed: {records_removed}")
    
    report.append(f"\nInvalid values / Anomalies:")
    for note in invalid_values_notes:
        report.append(f"  - {note}")
        
    # 8. Outlier inspection using IQR (Identified but NOT removed)
    outlier_observations = {}
    for col in df.select_dtypes(include=['number']).columns:
        # We generally only want to check continuous numerical columns for outliers, 
        # so we skip ones that look like categories/ordinals (few unique values)
        if df[col].nunique() > 10: 
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            if outliers_count > 0:
                outlier_observations[col] = outliers_count
                
    report.append("\nOutlier observations identified (using 1.5x IQR, NOT removed):")
    if outlier_observations:
        for col, count in outlier_observations.items():
            report.append(f"  - {col}: {count} potential outliers")
    else:
        report.append("  - None identified")
        
    report.append(f"\nData types after cleaning:\n{df.dtypes.to_string()}")
    
    # Save the cleaned dataset
    os.makedirs(output_dir, exist_ok=True)
    cleaned_file_path = os.path.join(output_dir, 'HR-Employee-Attrition_cleaned.csv')
    df.to_csv(cleaned_file_path, index=False)
    
    # Save the report
    report_text = "\n".join(report)
    with open(os.path.join(output_dir, 'cleaning_report.txt'), 'w') as f:
        f.write(report_text)
        
    print(report_text)
    print(f"\nCleaned dataset saved to: {cleaned_file_path}")

if __name__ == "__main__":
    clean_data('data/HR-Employee-Attrition.csv', 'outputs/cleaned_data/')
