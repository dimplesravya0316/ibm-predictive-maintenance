import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# --- PDF REPORT INTEGRATION ---
try:
    from pdf_report import generate_pdf_report
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IBM Predictive Maintenance Profiler",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN CLEAN CSS & ANIMATIONS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main .block-container {
        animation: fadeInUp 0.5s ease-out;
    }

    @keyframes alertPulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.4); }
        70% { box-shadow: 0 0 0 12px rgba(255, 75, 75, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
    }
    .alert-box-danger {
        background: rgba(255, 75, 75, 0.08);
        border: 1px solid #FF4B4B;
        border-radius: 10px;
        padding: 16px;
        animation: alertPulse 2s infinite;
        margin-bottom: 15px;
    }

    .alert-box-success {
        background: rgba(0, 200, 83, 0.08);
        border: 1px solid #00C853;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 15px;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.25s ease;
    }
    .stButton>button:hover {
        transform: scale(1.015);
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# --- ARTIFACT LOADING ---
@st.cache_resource
def load_model_artifacts():
    return joblib.load('model.pkl')

try:
    artifacts = load_model_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    type_encoder = artifacts['type_encoder']
    feature_cols = artifacts['feature_cols']
except Exception:
    st.error("⚠️️ Failed to load `model.pkl`. Ensure you have run `python train_model.py` successfully.")
    st.stop()

# --- HEADER ---
st.title("⚙️ Industrial Equipment Maintenance Profiler")
st.caption("AI-Powered Predictive Maintenance, Real-Time Sensor Diagnostics & Telemetry Analytics")

# --- SIDEBAR NAVIGATION & DEMO PRESETS ---
st.sidebar.markdown("## 🎛️ Control Panel")
mode = st.sidebar.radio("Operating Mode", ["Single Equipment Telemetry", "Batch Dataset Analytics"])

if mode == "Single Equipment Telemetry":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧪 Quick Presets for Demo")
    if st.sidebar.button("🟢 Scenario 1: Healthy Equipment"):
        st.session_state['type'] = 'L'
        st.session_state['air'] = 300.0
        st.session_state['proc'] = 310.0
        st.session_state['speed'] = 1500
        st.session_state['torque'] = 38.0
        st.session_state['wear'] = 45

    if st.sidebar.button("🚨 Scenario 2: Critical Tool Wear (TWF)"):
        st.session_state['type'] = 'L'
        st.session_state['air'] = 300.0
        st.session_state['proc'] = 311.0
        st.session_state['speed'] = 1450
        st.session_state['torque'] = 52.0
        st.session_state['wear'] = 225

    if st.sidebar.button("🔥 Scenario 3: Thermal Stress (HDF)"):
        st.session_state['type'] = 'M'
        st.session_state['air'] = 304.0
        st.session_state['proc'] = 308.5
        st.session_state['speed'] = 1250
        st.session_state['torque'] = 48.0
        st.session_state['wear'] = 110

    if st.sidebar.button("⚡ Scenario 4: Overstrain Failure (OSF)"):
        st.session_state['type'] = 'H'
        st.session_state['air'] = 299.0
        st.session_state['proc'] = 310.0
        st.session_state['speed'] = 1380
        st.session_state['torque'] = 68.0
        st.session_state['wear'] = 195

# ==========================================
# MODE 1: SINGLE EQUIPMENT TELEMETRY
# ==========================================
if mode == "Single Equipment Telemetry":
    st.markdown("### 🔧 Live Sensor Telemetry Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prod_type = st.selectbox(
            "Product Quality Type", 
            options=['L', 'M', 'H'], 
            index=['L', 'M', 'H'].index(st.session_state.get('type', 'L')), 
            help="L: Low (50%), M: Medium (30%), H: High (20%)"
        )
        air_temp = st.number_input("Air Temperature [K]", min_value=290.0, max_value=315.0, value=st.session_state.get('air', 300.0), step=0.1)
        proc_temp = st.number_input("Process Temperature [K]", min_value=300.0, max_value=325.0, value=st.session_state.get('proc', 310.0), step=0.1)
        
    with col2:
        rot_speed = st.number_input("Rotational Speed [rpm]", min_value=1000, max_value=3000, value=st.session_state.get('speed', 1500), step=10)
        torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=st.session_state.get('torque', 40.0), step=0.5)
        
    with col3:
        tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=300, value=st.session_state.get('wear', 120), step=1)
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("RUN DIAGNOSIS ⚡")

    # Feature Engineering
    temp_diff = proc_temp - air_temp
    power_w = torque * (rot_speed * (2 * np.pi / 60))
    overstrain = tool_wear * torque
        
    type_encoded = type_encoder.transform([prod_type])[0]
    
    input_df = pd.DataFrame([[
        type_encoded, air_temp, proc_temp, rot_speed, torque, tool_wear,
        temp_diff, power_w, overstrain
    ]], columns=feature_cols)
    
    input_scaled = scaler.transform(input_df)
    
    if analyze_btn or 'analyzed' in st.session_state:
        st.session_state['analyzed'] = True
        
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        
        # Derived Health Score & RUL Calculation
        health_score = max(0, int(100 - (prob * 100)))
        rul_hours = max(0, int((240 - tool_wear) * 0.8)) if pred == 0 else max(1, int(12 * (1 - prob)))

        reasons = []
        twf_risk = min(1.0, tool_wear / 200.0) * (100 if tool_wear >= 200 else 25)
        hdf_risk = 90.0 if (temp_diff < 8.6 and rot_speed < 1380) else 10.0
        pwf_risk = 85.0 if (power_w < 3500 or power_w > 9000) else 5.0
        osf_risk = min(1.0, overstrain / 11000.0) * (95 if overstrain > 11000 else 15)

        if tool_wear >= 200:
            reasons.append("⚠️ **Tool Wear Failure (TWF)**: Cumulative operating limit exceeded.")
        if temp_diff < 8.6 and rot_speed < 1380:
            reasons.append("⚠️ **Heat Dissipation Failure (HDF)**: Thermal gradient below dissipation capacity.")
        if power_w < 3500 or power_w > 9000:
            reasons.append("⚠️ **Power Failure (PWF)**: Wattage outside nominal operating range.")
        if overstrain > 11000:
            reasons.append("⚠️ **Overstrain Failure (OSF)**: Critical torque load on worn tool.")
            
        st.markdown("---")

        # --- EXECUTIVE KPI SUMMARY ROW ---
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        status_label = "CRITICAL" if pred == 1 else "HEALTHY"
        status_delta = "Action Required" if pred == 1 else "Nominal"
        
        kpi1.metric("Equipment Status", status_label, delta=status_delta, delta_color="inverse" if pred == 1 else "normal")
        kpi2.metric("Overall Health Score", f"{health_score}/100", delta=f"{'-' if prob>0.3 else '+'}{abs(health_score-85)}%")
        kpi3.metric("Est. Remaining Life (RUL)", f"{rul_hours} Hours")
        kpi4.metric("Generated Power Output", f"{power_w/1000:.2f} kW")
        kpi5.metric("Overstrain Index", f"{overstrain:.0f} Nm·min")

        st.markdown("<br>", unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown("#### Risk Assessment Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Failure Risk (%)", 'font': {'size': 18, 'color': "white"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#FF4B4B" if prob > 0.5 else "#00C6FF"},
                    'bgcolor': "#1E222D",
                    'borderwidth': 2,
                    'bordercolor': "#2E3440",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(0, 200, 83, 0.2)'},
                        {'range': [35, 65], 'color': 'rgba(255, 170, 0, 0.2)'},
                        {'range': [65, 100], 'color': 'rgba(255, 75, 75, 0.2)'}
                    ],
                }
            ))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with res_col2:
            st.markdown("#### Diagnostic Summary")
            
            if pred == 1 or prob > 0.5:
                st.markdown("""
                <div class="alert-box-danger">
                    <h3 style="color: #FF4B4B; margin:0;">🚨 WARNING: High Failure Risk</h3>
                    <p style="margin-top:8px; color: #E0E0E0;">Telemetry signals elevated risk of component degradation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-box-success">
                    <h3 style="color: #00C853; margin:0;">✅ STATUS: Operational Normal</h3>
                    <p style="margin-top:8px; color: #E0E0E0;">Equipment is running within healthy physical thresholds.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("**Identified Physical Threshold Exceedances:**")
            if reasons:
                for r in reasons:
                    st.write(r)
            else:
                st.write("• No critical physical thresholds exceeded.")

        st.markdown("---")
        
        # --- FEATURE IMPORTANCE & FAILURE BREAKDOWN ---
        fcol1, fcol2 = st.columns(2)
        
        with fcol1:
            st.markdown("#### 📊 Model Feature Contributions")
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
                feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=True)
                
                fig_imp = px.bar(
                    feat_imp_df, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h',
                    title="Global Random Forest Feature Weights",
                    color_discrete_sequence=['#00C6FF']
                )
                fig_imp.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_imp, use_container_width=True)

        with fcol2:
            st.markdown("#### 💡 AI Prescriptive Maintenance Recommendations")
            if pred == 1:
                if tool_wear >= 200:
                    st.error("🔧 **Tool Replacement:** Tool wear exceeds threshold. Schedule tool change immediately.")
                if temp_diff < 8.6:
                    st.warning("❄️ **Cooling Maintenance:** Inspect coolant fluid and thermal dissipation channels.")
                if power_w > 9000:
                    st.warning("⚡ **Load Throttling:** Reduce torque or speed to prevent electrical overload.")
                if overstrain > 11000:
                    st.error("🛑 **Load Reduction:** Critical stress on tool interface. Unload machine.")
            else:
                st.success("👍 Machine healthy. Perform normal operational checks according to schedule.")

            if PDF_SUPPORT:
                st.markdown("<br>", unsafe_allow_html=True)
                pdf_bytes = generate_pdf_report(
                    prod_type, air_temp, proc_temp, rot_speed, torque, tool_wear, prob, reasons
                )
                st.download_button(
                    label="📄 Download Official PDF Inspection Report",
                    data=pdf_bytes,
                    file_name="equipment_maintenance_report.pdf",
                    mime="application/pdf"
                )

# ==========================================
# MODE 2: BATCH DATASET ANALYTICS
# ==========================================
elif mode == "Batch Dataset Analytics":
    st.markdown("### 📁 Batch Fleet Telemetry Analysis")
    uploaded_file = st.file_uploader("Upload CSV matching AI4I 2020 schema", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        batch_df.columns = batch_df.columns.str.strip().str.replace('[', '', regex=False).str.replace(']', '', regex=False)
        
        batch_df['Temp_Difference'] = batch_df['Process temperature K'] - batch_df['Air temperature K']
        batch_df['Power_W'] = batch_df['Torque Nm'] * (batch_df['Rotational speed rpm'] * (2 * np.pi / 60))
        batch_df['Overstrain_Index'] = batch_df['Tool wear min'] * batch_df['Torque Nm']
        
        X_batch = batch_df[feature_cols].copy()
        X_batch['Type'] = type_encoder.transform(X_batch['Type'])
        X_batch_scaled = scaler.transform(X_batch)
        
        batch_df['Predicted Failure'] = model.predict(X_batch_scaled)
        batch_df['Failure Risk (%)'] = (model.predict_proba(X_batch_scaled)[:, 1] * 100).round(2)
        
        st.markdown("<br>", unsafe_allow_html=True)
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Total Equipment Logs", len(batch_df))
        mcol2.metric("Predicted Failures", int(batch_df['Predicted Failure'].sum()))
        mcol3.metric("Fleet Failure Rate", f"{(batch_df['Predicted Failure'].mean() * 100):.2f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        fig = px.histogram(
            batch_df, 
            x="Failure Risk (%)", 
            nbins=30, 
            title="Fleet Failure Risk Probability Distribution",
            color_discrete_sequence=['#00C6FF']
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Detailed Prediction Logs")
        st.dataframe(
            batch_df[['Product ID', 'Type', 'Rotational speed rpm', 'Torque Nm', 'Tool wear min', 'Predicted Failure', 'Failure Risk (%)']], 
            use_container_width=True
        )