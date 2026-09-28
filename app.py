import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import io

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Helper Functions for Dynamic Column Detection ---
def detect_columns(df):
    cols = [c.lower() for c in df.columns]
    mapping = {
        'satisfaction': None,
        'projects': None,
        'working_hours': None,
        'overtime': None,
        'department': None,
        'attrition': None
    }
    
    for c_orig in df.columns:
        c = c_orig.lower()
        if 'satisfaction' in c:
            mapping['satisfaction'] = c_orig
        elif 'project' in c:
            mapping['projects'] = c_orig
        elif 'hour' in c or 'time' in c:
            if 'over' in c:
                mapping['overtime'] = c_orig
            elif 'monthly' in c or 'standard' in c or 'working' in c:
                mapping['working_hours'] = c_orig
        elif 'department' in c or 'sales' in c:
            mapping['department'] = c_orig
        elif 'attrition' in c or 'left' in c or 'turnover' in c:
            mapping['attrition'] = c_orig
            
    return mapping

# --- Application Layout ---
st.sidebar.title("HR Analytics Configuration")
uploaded_file = st.sidebar.file_uploader("Upload HR Dataset (CSV)", type="csv")
use_sample = st.sidebar.button("Use Sample Dataset (if available)")
run_analysis = st.sidebar.button("Run Analysis")

st.title("HR Analytics Dashboard")
st.subheader("Dynamic Employee Workload, Satisfaction & Attrition Analysis")

if not uploaded_file and not use_sample:
    st.info("Upload an HR CSV dataset to begin analysis.")
else:
    try:
        if uploaded_file:
            df_raw = pd.read_csv(uploaded_file)
            filename = uploaded_file.name
        else:
            df_raw = pd.read_csv('data/HR-Employee-Attrition.csv')
            filename = "HR-Employee-Attrition.csv (Sample)"
            
        st.write(f"**Loaded File:** {filename}")
        st.write(f"**Rows:** {df_raw.shape[0]} | **Columns:** {df_raw.shape[1]}")
        st.dataframe(df_raw)
        
        mapping = detect_columns(df_raw)
        st.sidebar.markdown("### Detected Variables")
        for k, v in mapping.items():
            st.sidebar.markdown(f"**{k.title()}**: `{v if v else 'Not available'}`")
            
        if run_analysis:
            with st.spinner("Running End-to-End Analysis..."):
                # Data Validation
                st.header("2. Data Validation")
                df_clean = df_raw.copy()
                st.write(f"Missing Values: {df_clean.isnull().sum().sum()}")
                st.write(f"Duplicates: {df_clean.duplicated().sum()}")
                df_clean.drop_duplicates(inplace=True)
                
                # Section 1 - Dataset Overview
                st.header("1. Dataset Overview")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                col1.metric("Total Employees", len(df_clean))
                if mapping['satisfaction']:
                    col2.metric("Avg Satisfaction", f"{df_clean[mapping['satisfaction']].mean():.2f}")
                else:
                    col2.metric("Avg Satisfaction", "N/A")
                    
                if mapping['working_hours']:
                    col3.metric("Avg Monthly Hours", f"{df_clean[mapping['working_hours']].mean():.1f}")
                else:
                    col3.metric("Avg Monthly Hours", "N/A")
                    
                if mapping['projects']:
                    col4.metric("Avg Projects", f"{df_clean[mapping['projects']].mean():.1f}")
                else:
                    col4.metric("Avg Projects", "N/A")
                    
                # Attrition
                if mapping['attrition']:
                    attr_col = mapping['attrition']
                    if df_clean[attr_col].dtype == 'object':
                        # assume Yes/No
                        att_rate = (df_clean[attr_col] == 'Yes').mean() * 100
                    else:
                        att_rate = df_clean[attr_col].mean() * 100
                    col6.metric("Attrition Rate", f"{att_rate:.1f}%")
                else:
                    col6.metric("Attrition Rate", "N/A")
                
                st.markdown("---")
                
                # Section 2 - Workload Analysis
                st.header("2. Workload Analysis")
                if mapping['department'] and mapping['overtime']:
                    fig_wl, ax_wl = plt.subplots(figsize=(10, 4))
                    sns.countplot(data=df_clean, x=mapping['department'], hue=mapping['overtime'], ax=ax_wl)
                    ax_wl.set_title("Overtime Distribution by Department")
                    plt.xticks(rotation=45)
                    st.pyplot(fig_wl)
                else:
                    st.info("Department or Overtime variable not detected. Cannot generate this chart.")
                st.markdown("---")
                
                # Section 3 - Satisfaction Analysis
                st.header("3. Satisfaction Analysis")
                colA1, colA2 = st.columns(2)
                with colA1:
                    if mapping['projects'] and mapping['satisfaction']:
                        fig_proj, ax_proj = plt.subplots(figsize=(6, 4))
                        sns.boxplot(data=df_clean, x=mapping['projects'], y=mapping['satisfaction'], ax=ax_proj)
                        ax_proj.set_title("Projects vs Satisfaction")
                        st.pyplot(fig_proj)
                    else:
                        st.info("Projects or Satisfaction variable missing.")
                with colA2:
                    if mapping['overtime'] and mapping['satisfaction']:
                        fig_ot, ax_ot = plt.subplots(figsize=(6, 4))
                        sns.barplot(data=df_clean, x=mapping['overtime'], y=mapping['satisfaction'], ax=ax_ot)
                        ax_ot.set_title("Overtime vs Satisfaction")
                        st.pyplot(fig_ot)
                    else:
                        st.info("Overtime or Satisfaction variable missing.")
                st.markdown("---")
                
                # Section 4 - Attrition Analysis
                st.header("4. Attrition Analysis")
                colB1, colB2 = st.columns(2)
                with colB1:
                    if mapping['overtime'] and mapping['attrition']:
                        fig_att_ot, ax_att_ot = plt.subplots(figsize=(6, 4))
                        sns.countplot(data=df_clean, x=mapping['overtime'], hue=mapping['attrition'], ax=ax_att_ot)
                        ax_att_ot.set_title("Overtime vs Attrition")
                        st.pyplot(fig_att_ot)
                    else:
                        st.info("Overtime or Attrition variable missing.")
                with colB2:
                    if mapping['satisfaction'] and mapping['attrition']:
                        fig_att_sat, ax_att_sat = plt.subplots(figsize=(6, 4))
                        sns.countplot(data=df_clean, x=mapping['satisfaction'], hue=mapping['attrition'], ax=ax_att_sat)
                        ax_att_sat.set_title("Satisfaction vs Attrition")
                        st.pyplot(fig_att_sat)
                    else:
                        st.info("Satisfaction or Attrition variable missing.")
                st.markdown("---")
                
                # Random Forest Training
                if mapping['attrition']:
                    st.header("5. Machine Learning (Random Forest)")
                    # Prepare Data
                    X = df_clean.drop(columns=[mapping['attrition']])
                    y = df_clean[mapping['attrition']]
                    
                    if y.dtype == 'object':
                        y = (y == 'Yes').astype(int)
                        
                    # basic encoding
                    X = pd.get_dummies(X, drop_first=True)
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)
                    
                    colA, colB, colC, colD = st.columns(4)
                    colA.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
                    colB.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
                    colC.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
                    colD.metric("F1-Score", f"{f1_score(y_test, y_pred):.3f}")
                    
                    st.subheader("Feature Importance")
                    importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': rf.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig, ax = plt.subplots(figsize=(10,5))
                    sns.barplot(data=importance, x='Importance', y='Feature', ax=ax)
                    st.pyplot(fig)
                    
                st.success("Analysis Complete!")
                
    except Exception as e:
        st.error(f"Error processing dataset: {e}")
