import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(input_path, model_output_path):
    print("Starting Model Training Pipeline...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return
        
    # 2. Select features and separate target
    # Target column is 'Attrition' (or Attrition_Num if it exists)
    target_col = 'Attrition'
    y = df[target_col].map({'Yes': 1, 'No': 0}) if df[target_col].dtype == 'object' else df[target_col]
    
    # Drop target and uninformative/leaky variables
    cols_to_drop = [target_col]
    if 'Attrition_Num' in df.columns:
        cols_to_drop.append('Attrition_Num')
    if 'EmployeeNumber' in df.columns:
        cols_to_drop.append('EmployeeNumber')
    
    # Drop original OverTime if Overtime_Indicator exists to prevent duplication
    if 'OverTime' in df.columns and 'Overtime_Indicator' in df.columns:
        cols_to_drop.append('OverTime')
        
    X = df.drop(columns=cols_to_drop)
    
    # 4 & 5. Encode categorical variables & handle numericals
    # Random Forest handles numerical variables well without scaling.
    # We will one-hot encode remaining categorical variables.
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # 6 & 7. Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # 8. Train RandomForestClassifier
    # We use a balanced class weight because attrition is imbalanced (~16% Yes)
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        max_depth=10
    )
    rf_model.fit(X_train, y_train)
    
    # 9. Generate predictions
    y_pred = rf_model.predict(X_test)
    
    # 10. Generate prediction probabilities
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 11. Calculate feature importance
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importances.head(10).to_string(index=False))
    
    # Save results to a report file
    os.makedirs('outputs/model_results', exist_ok=True)
    with open('outputs/model_results/rf_evaluation.txt', 'w') as f:
        f.write("=== RANDOM FOREST EVALUATION ===\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\nTop 10 Features:\n")
        f.write(feature_importances.head(10).to_string(index=False))
        
    # 12. Save the trained model using joblib
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(rf_model, model_output_path)
    
    # Save the feature names so we know the exact columns needed for future predictions
    joblib.dump(list(X_train.columns), model_output_path.replace('.pkl', '_features.pkl'))
    
    print(f"\nModel saved successfully to: {model_output_path}")

if __name__ == "__main__":
    engineer_path = 'outputs/feature_engineered/HR-Employee-Attrition_features.csv'
    train_model(engineer_path, 'models/random_forest_model.pkl')
