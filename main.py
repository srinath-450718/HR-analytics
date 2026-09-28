import os
import sys
import traceback

# Ensure the src directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_validation import validate_data
from data_cleaning import clean_data
from eda import run_eda
from feature_engineering import engineer_features
from workload_analysis import run_workload_analysis
from visualization import generate_visualizations
from model import train_model
from evaluation import evaluate_model

def main():
    print("="*60)
    print("HR ANALYTICS — END-TO-END PIPELINE EXECUTION")
    print("="*60)
    
    # Define file paths
    raw_data_path = 'data/HR-Employee-Attrition.csv'
    cleaned_data_dir = 'outputs/cleaned_data/'
    cleaned_data_path = os.path.join(cleaned_data_dir, 'HR-Employee-Attrition_cleaned.csv')
    
    eda_dir = 'outputs/eda/'
    
    feature_engineered_dir = 'outputs/feature_engineered/'
    feature_engineered_path = os.path.join(feature_engineered_dir, 'HR-Employee-Attrition_features.csv')
    
    workload_analysis_dir = 'outputs/workload_analysis/'
    visualizations_dir = 'outputs/visualizations/'
    
    model_output_path = 'models/random_forest_model.pkl'
    model_results_dir = 'outputs/model_results/'
    
    try:
        # MODULE 1: Data Validation
        print("\n>>> Running Module 1: Data Validation...")
        validate_data(raw_data_path)
        
        # MODULE 2: Data Cleaning
        print("\n>>> Running Module 2: Data Cleaning...")
        clean_data(raw_data_path, cleaned_data_dir)
        
        # MODULE 3: EDA
        print("\n>>> Running Module 3: Exploratory Data Analysis...")
        run_eda(cleaned_data_path, eda_dir)
        
        # MODULE 4: Feature Engineering
        print("\n>>> Running Module 4: Feature Engineering...")
        engineer_features(cleaned_data_path, feature_engineered_dir)
        
        # MODULE 5: Workload Analysis
        print("\n>>> Running Module 5: Workload Analysis...")
        run_workload_analysis(feature_engineered_path, workload_analysis_dir)
        
        # MODULE 6: Visualization
        print("\n>>> Running Module 6: Data Visualization...")
        generate_visualizations(feature_engineered_path, visualizations_dir)
        
        # MODULE 7: Model Training
        print("\n>>> Running Module 7: Random Forest Training...")
        train_model(feature_engineered_path, model_output_path)
        
        # MODULE 8: Model Evaluation
        print("\n>>> Running Module 8: Random Forest Evaluation...")
        evaluate_model(feature_engineered_path, model_output_path, model_results_dir)
        
        # MODULES 9 & 10 are already saved as Markdown reports (key_insights.md and action_plan.md)
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("Modules 9 (Insights) and 10 (Action Plan) are preserved in outputs/insights/ and outputs/action_plan/")
        print("="*60)
        
    except Exception as e:
        print("\n[!] FATAL ERROR ENCOUNTERED IN PIPELINE EXECUTION:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
