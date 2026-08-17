import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Page Setup
st.set_page_config(
    page_title="Industrial Equipment Maintenance Profiler",
    page_icon="⚙️",
    layout="wide"
)

# Load Model Artifacts
@st.cache_resource
def load_artifacts():
    return joblib.load('model.pkl')

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
type_encoder = artifacts['type_encoder']

st.title("⚙️ Industrial Equipment Failure & Maintenance Profiler")
st.markdown("Real-time telemetry monitoring, risk prediction, and failure mode diagnosis using Machine Learning.")

# Sidebar Navigation
mode = st.sidebar.radio("Select Operating Mode", ["Single Equipment Telemetry", "Batch Dataset Prediction"])

if mode == "Single Equipment Telemetry":
    st.subheader("🔧 Sensor Inputs & Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prod_type = st.selectbox("Product Quality Type", options=['L', 'M', 'H'], index=0, help="L: Low, M: Medium, H: High")
        air_temp = st.number_input("Air Temperature [K]", min_value=290.0, max_value=310.0, value=300.0, step=0.1)
        proc_temp = st.number_input("Process Temperature [K]", min_value=300.0, max_value=325.0, value=310.0, step=0.1)
        
    with col2:
        rot_speed = st.number_input("Rotational Speed [rpm]", min_value=1000, max_value=3000, value=1500, step=10)
        torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0, step=0.5)
        tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=300, value=120, step=1)

    # Derived Feature Calculation
    temp_diff = proc_temp - air_temp
    power_w = torque * (rot_speed * (2 * np.pi / 60))
    overstrain = tool_wear * torque
    
    type_encoded = type_encoder.transform([prod_type])[0]
    
    input_data = pd.DataFrame([[
        type_encoded, air_temp, proc_temp, rot_speed, torque, tool_wear,
        temp_diff, power_w, overstrain
    ]], columns=artifacts['feature_cols'])
    
    input_scaled = scaler.transform(input_data)
    
    if st.button("Analyze Telemetry"):
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Failure Probability", f"{prob * 100:.2f}%")
            if pred == 1:
                st.error("🚨 ALERT: Machine Failure Likely Detected!")
            else:
                st.success("✅ Machine Operating Safely")
                
        with res_col2:
            st.markdown("#### Potential Failure Diagnostics")
            reasons = []
            if tool_wear >= 200:
                reasons.append("⚠️ **Tool Wear Failure (TWF)**: High cumulative wear time.")
            if temp_diff < 8.6 and rot_speed < 1380:
                reasons.append("⚠️ **Heat Dissipation Failure (HDF)**: Low temp differential and low speed.")
            if power_w < 3500 or power_w > 9000:
                reasons.append("⚠️ **Power Failure (PWF)**: Power output out of operational bounds.")
            if overstrain > 11000:
                reasons.append("⚠️ **Overstrain Failure (OSF)**: High product of tool wear and torque.")
                
            if reasons:
                for r in reasons:
                    st.write(r)
            else:
                st.write("No specific physical failure condition threshold exceeded.")

elif mode == "Batch Dataset Prediction":
    st.subheader("📁 Upload CSV for Batch Diagnostic Analytics")
    uploaded_file = st.file_uploader("Upload CSV matching AI4I dataset schema", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        batch_df.columns = batch_df.columns.str.strip().str.replace('[', '', regex=False).str.replace(']', '', regex=False)
        
        # Feature Engineering
        batch_df['Temp_Difference'] = batch_df['Process temperature K'] - batch_df['Air temperature K']
        batch_df['Power_W'] = batch_df['Torque Nm'] * (batch_df['Rotational speed rpm'] * (2 * np.pi / 60))
        batch_df['Overstrain_Index'] = batch_df['Tool wear min'] * batch_df['Torque Nm']
        
        X_batch = batch_df[artifacts['feature_cols']].copy()
        X_batch['Type'] = type_encoder.transform(X_batch['Type'])
        
        X_batch_scaled = scaler.transform(X_batch)
        
        batch_df['Failure_Prediction'] = model.predict(X_batch_scaled)
        batch_df['Failure_Probability'] = model.predict_proba(X_batch_scaled)[:, 1]
        
        st.write(f"Analyzed {len(batch_df)} rows.")
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Predicted Total Failures", int(batch_df['Failure_Prediction'].sum()))
        col_m2.metric("Failure Rate", f"{(batch_df['Failure_Prediction'].mean() * 100):.2f}%")
        
        fig = px.histogram(batch_df, x="Failure_Probability", nbins=30, title="Probability Distribution of Machine Failures")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(batch_df[['Product ID', 'Type', 'Rotational speed rpm', 'Torque Nm', 'Tool wear min', 'Failure_Prediction', 'Failure_Probability']])