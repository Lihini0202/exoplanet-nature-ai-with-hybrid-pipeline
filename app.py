import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.utils import to_categorical
import xgboost as xgb

# Streamlit app setup
st.title("Kepler Exoplanet Analysis with Hybrid Pipeline")
st.write("Analyzing exoplanet data using ACO, CS, WOA, and CNN models.")

# Load and preprocess data with ACO feature selection (cached for efficiency)
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('exoplanets.csv')
    features = ['koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 
                'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
                'koi_steff', 'koi_slogg', 'koi_srad', 'ra', 'dec', 'koi_kepmag',
                'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
    error_features = ['koi_period_err1', 'koi_time0bk_err1', 'koi_impact_err1', 'koi_duration_err1',
                      'koi_depth_err1', 'koi_prad_err1', 'koi_teq_err1', 'koi_insol_err1']
    target = 'koi_disposition'
    
    df_clean = df[[target]].dropna().merge(df[features + error_features], left_index=True, right_index=True, how='left')
    df_clean[features + error_features] = df_clean[features + error_features].fillna(0)
    le = LabelEncoder()
    df_clean[target] = le.fit_transform(df_clean[target])
    
    # Adjust to match the 15 features the model was trained on (verify with Cell 4)
    aco_selected_features = ['koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_teq', 
                            'koi_insol', 'koi_model_snr', 'koi_steff', 'koi_fpflag_nt', 
                            'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_impact_err1', 
                            'koi_duration_err1', 'koi_depth_err1']  # Trimmed to 15
    X = df_clean[aco_selected_features].values
    y = df_clean[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, le.classes_, df_clean

X_train, X_test, y_train, y_test, classes, df_clean = load_and_preprocess_data()

# Load pre-trained CS model
@st.cache_resource
def load_cs_model():
    return joblib.load('cs_model.joblib')

clf_cs = load_cs_model()
y_pred_cs = clf_cs.predict(X_test)
cs_f1 = f1_score(y_test, y_pred_cs, average='weighted')

# Display model performance (pre-computed or calculated)
st.write("### Model Performance")
st.write(f"ACO F1-Score: 0.8677 (Pre-computed from Cell 3)")
st.write(f"CS F1-Score: {cs_f1:.4f} (Calculated from loaded model)")
st.write("WOA F1-Score: 0.8064 (Pre-computed from Cell 6)")
st.write("CNN F1-Score: 0.8815 (Pre-computed from Cell 7)")

# Robustness (pre-computed from Cell 8)
st.write("### Robustness Analysis")
st.write("Robust F1-Score (mean ± std):")
st.write("ACO: 0.7459 ± 0.0055")
st.write("CS: 0.6981 ± 0.0035")
st.write("WOA: 0.5828 ± 0.0099")
st.write("CNN: N/A (Not computed)")

# Habitability Insights (from Cell 9)
teq_idx, insol_idx = 4, 5  # Adjusted indices based on new feature order
habitability = np.where((X_test[:, teq_idx] < 0) & (np.abs(X_test[:, insol_idx] - 1) < 0.5), 1, 0)  # Scaled data
habitable_count = np.sum(habitability)
total_count = len(habitability)
st.write("### Habitability Insights")
st.write(f"Potentially Habitable Exoplanets: {habitable_count} out of {total_count} (~{habitable_count/total_count:.2%})")

# Visualization: Confusion Matrix for CS (from Cell 10)
st.write("### Confusion Matrix for CS")
cm = confusion_matrix(y_test, y_pred_cs)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
ax.set_title('Confusion Matrix for CS')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
st.pyplot(fig)

# Conclusion (from Cell 12)
st.write("### Conclusion and Future Work")
st.write("- CS achieved the highest F1-score (0.8994), while ACO showed the best robustness (0.7459 robust F1).")
st.write("- Only ~0.49% of exoplanets are potentially habitable based on teq and insol thresholds.")
st.write("- Future Work: Enhance WOA with noise-robust layers, integrate TESS data with FA clustering, and expand SHAP analysis.")
