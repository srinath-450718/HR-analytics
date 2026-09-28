import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def evaluate_model(data_path, model_path, output_dir):
    print("Starting Model Evaluation Pipeline...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset to re-create the exact test set
    df = pd.read_csv(data_path)
    
    target_col = 'Attrition'
    y = df[target_col].map({'Yes': 1, 'No': 0}) if df[target_col].dtype == 'object' else df[target_col]
    
    cols_to_drop = [target_col]
    if 'Attrition_Num' in df.columns: cols_to_drop.append('Attrition_Num')
    if 'EmployeeNumber' in df.columns: cols_to_drop.append('EmployeeNumber')
    if 'OverTime' in df.columns and 'Overtime_Indicator' in df.columns: cols_to_drop.append('OverTime')
        
    X = df.drop(columns=cols_to_drop)
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Needs to match random_state=42 and stratify=y from training
    _, X_test, _, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load model
    try:
        rf_model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Generate predictions
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    # Calculate feature importances
    feature_importances = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Build text report
    report = []
    report.append("=== RANDOM FOREST MODEL EVALUATION REPORT ===\n")
    
    report.append("1. Core Metrics:")
    report.append(f"- Accuracy:   {acc:.4f}")
    report.append(f"- Precision:  {prec:.4f}")
    report.append(f"- Recall:     {rec:.4f}")
    report.append(f"- F1-Score:   {f1:.4f}")
    report.append(f"- ROC-AUC:    {roc_auc:.4f}\n")
    
    report.append("2. The Problem with Accuracy on Imbalanced Datasets:")
    report.append("Employee attrition is an inherently imbalanced problem (most employees stay). ")
    report.append("If a dataset has an 84% retention rate, a 'dumb' model that blindly predicts ")
    report.append("'No' for everyone would achieve 84% Accuracy, completely failing to identify ")
    report.append("a single flight risk. Therefore, Accuracy alone is insufficient. We must look at:")
    report.append("- Precision: Out of all employees flagged as leaving, how many actually left?")
    report.append("- Recall: Out of all employees who ACTUALLY left, how many did we successfully flag?\n")
    
    report.append("3. Confusion Matrix:")
    report.append(f"True Negatives (Stayed, Predicted Stayed): {cm[0,0]}")
    report.append(f"False Positives (Stayed, Predicted Left):  {cm[0,1]}")
    report.append(f"False Negatives (Left, Predicted Stayed):  {cm[1,0]}")
    report.append(f"True Positives (Left, Predicted Left):     {cm[1,1]}\n")
    
    report.append("4. Classification Report:\n")
    report.append(cr)
    
    # Save the report
    with open(os.path.join(output_dir, 'comprehensive_evaluation.txt'), 'w') as f:
        f.write("\n".join(report))
        
    print("\n".join(report))
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 8))
    top_20_features = feature_importances.head(20)
    sns.barplot(x='Importance', y='Feature', data=top_20_features, palette='viridis')
    plt.title("Top 20 Random Forest Feature Importances", fontsize=14, fontweight='bold')
    plt.xlabel("Relative Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rf_feature_importance.png'))
    plt.close()
    
    print(f"\nEvaluation complete. Report and chart saved to: {output_dir}")

if __name__ == "__main__":
    engineer_path = 'outputs/feature_engineered/HR-Employee-Attrition_features.csv'
    model_path = 'models/random_forest_model.pkl'
    evaluate_model(engineer_path, model_path, 'outputs/model_results/')
