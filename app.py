import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.utils import to_categorical
import xgboost as xgb

# Streamlit app setup
st.title("Kepler Exoplanet Analysis with Hybrid Pipeline")
st.write("Interactive analysis of exoplanet data using ACO, CS, WOA, and CNN models.")

# Load and preprocess data with ACO feature selection (cached for efficiency)
@st.cache_data
def load_and_preprocess_data():
    try:
        # Specify encoding to handle potential mismatches
        df = pd.read_csv('exoplanets.csv', encoding='utf-8')
        if 'koi_disposition' not in df.columns:
            raise ValueError("Column 'koi_disposition' not found in 'exoplanets.csv'. Please check the file.")
        if df.empty:
            raise ValueError("The 'exoplanets.csv' file is empty. Please upload a valid file.")
        
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
        
        # ACO-selected features (15 from Cell 3)
        aco_selected_features = ['koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_teq', 
                                 'koi_insol', 'koi_model_snr', 'koi_steff', 'koi_fpflag_nt', 
                                 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_impact_err1', 
                                 'koi_duration_err1', 'koi_depth_err1']
        X = df_clean[aco_selected_features].values
        y = df_clean[target].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test, le.classes_, df_clean, aco_selected_features
    except Exception as e:
        st.error(f"Error loading data: {str(e)}. Please ensure 'exoplanets.csv' is uploaded correctly and is not empty or corrupted.")
        return None, None, None, None, None, None, None

X_train, X_test, y_train, y_test, classes, df_clean, aco_selected_features = load_and_preprocess_data()

if X_train is not None and X_test is not None:
    # Load pre-trained CS model
    @st.cache_resource
    def load_cs_model():
        return joblib.load('cs_model.joblib')

    clf_cs = load_cs_model()
    y_pred_cs = clf_cs.predict(X_test)
    cs_f1 = f1_score(y_test, y_pred_cs, average='weighted')
    y_pred_cs_prob = clf_cs.predict_proba(X_test)

    # Tabs for 5 interfaces
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Models", "Robustness", "Habitability", "Visualizations", "Conclusion"])

    with tab1:
        st.header("Model Performance")
        st.write(f"ACO F1-Score: 0.8677 (Pre-computed from Cell 3)")
        st.write(f"CS F1-Score: {cs_f1:.4f} (Calculated from loaded model)")
        st.write("WOA F1-Score: 0.8064 (Pre-computed from Cell 6)")
        st.write("CNN F1-Score: 0.8815 (Pre-computed from Cell 7)")

    with tab2:
        st.header("Robustness Analysis")
        st.write("Robust F1-Score (mean ± std):")
        st.write("ACO: 0.7459 ± 0.0055")
        st.write("CS: 0.6981 ± 0.0035")
        st.write("WOA: 0.5828 ± 0.0099")
        st.write("CNN: N/A (Not computed)")

    with tab3:
        st.header("Habitability Insights")
        teq_threshold = st.slider("Max Equilibrium Temperature (K)", 200, 400, 300, key="teq_slider")
        insol_threshold = st.slider("Insolation Flux Range (±)", 0.1, 1.0, 0.5, key="insol_slider")
        if st.button("Refresh Habitability Plot", key="refresh_btn"):
            st.experimental_rerun()
        df_unscaled = df_clean[aco_selected_features + [target]].copy()
        habitable = df_unscaled[(df_unscaled['koi_teq'] < teq_threshold) & 
                               (np.abs(df_unscaled['koi_insol'] - 1) < insol_threshold)]
        habitable_count = len(habitable)
        total_count = len(df_unscaled)
        st.write(f"Potentially Habitable Exoplanets: {habitable_count} out of {total_count} (~{habitable_count/total_count:.2%})")
        # Live scatter plot
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.scatter(df_unscaled['koi_teq'], df_unscaled['koi_insol'], c='blue', alpha=0.5, label='All Exoplanets')
        ax3.scatter(habitable['koi_teq'], habitable['koi_insol'], c='red', label='Habitable')
        ax3.axvline(teq_threshold, color='g', linestyle='--', label=f'Teq < {teq_threshold}K')
        ax3.axhline(1, color='g', linestyle='--')
        ax3.axhspan(1-insol_threshold, 1+insol_threshold, color='g', alpha=0.1, label=f'Insol ±{insol_threshold}')
        ax3.set_xlabel('Equilibrium Temperature (K)')
        ax3.set_ylabel('Insolation Flux')
        ax3.set_title('Habitability Scatter Plot')
        ax3.legend()
        st.pyplot(fig3)

    with tab4:
        st.header("Visualizations")
        # Confusion Matrix
        st.subheader("Confusion Matrix for CS")
        cm = confusion_matrix(y_test, y_pred_cs)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        ax1.set_title('Confusion Matrix for CS')
        st.pyplot(fig1)

        # ROC Curve
        st.subheader("ROC Curve for CS (Class 0)")
        fpr, tpr, _ = roc_curve(to_categorical(y_test, num_classes=3)[:, 0], y_pred_cs_prob[:, 0])
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve for CS')
        ax2.legend()
        st.pyplot(fig2)

    with tab5:
        st.header("Conclusion and Future Work")
        st.write("- CS achieved the highest F1-score (0.8994), while ACO showed the best robustness (0.7459 robust F1).")
        st.write("- Only ~0.49% of exoplanets are potentially habitable based on teq and insol thresholds.")
        st.write("- Future Work: Enhance WOA with noise-robust layers, integrate TESS data with FA clustering, and expand SHAP analysis.")
else:
    st.stop()  # Halt execution if data loading fails
