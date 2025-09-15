import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json

# ----------------------------
# Load model + metadata
# ----------------------------
MODEL_PATH = "artifacts/xgb_model.joblib"    # renamed clean version
META_PATH = "artifacts/model_meta.json"

model = joblib.load(MODEL_PATH)

# Load metadata if available
try:
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    FEATURES = meta["features"]
    THRESHOLD = meta.get("threshold", 0.5)
except:
    FEATURES = [
        "age","gender","iop","cct","heart_rate","bp_sys",
        "screen_time_h","sleep_h","blink_per_min","pupil_mm",
        "iop_cct_ratio","screen_sleep_ratio","age_iop","bp_screen"
    ]
    THRESHOLD = 0.5

st.set_page_config(page_title="OcuSight: Glaucoma Risk", page_icon="🩺")
st.title("🩺 OcuSight — Glaucoma Risk Predictor")

st.markdown("""
This is a demo of **OcuSight**, an AI prototype that predicts glaucoma risk 
from clinical + wearable features.  
*(Hackathon dataset, not a real medical device)*
""")


# ----------------------------
# Input form
# ----------------------------
st.header("Patient Data Input")

inputs = {}
col1, col2 = st.columns(2)

with col1:
    inputs["age"] = st.number_input("Age", 18, 90, 50)
    inputs["gender"] = st.selectbox("Gender (0=Male, 1=Female)", [0,1], index=0)
    inputs["iop"] = st.number_input("Intraocular Pressure (IOP, mmHg)", 8.0, 40.0, 16.0)
    inputs["cct"] = st.number_input("Corneal Thickness (µm)", 420.0, 650.0, 540.0)
    inputs["heart_rate"] = st.number_input("Heart Rate (bpm)", 45, 120, 72)

with col2:
    inputs["bp_sys"] = st.number_input("Systolic BP (mmHg)", 90, 180, 120)
    inputs["screen_time_h"] = st.number_input("Daily Screen Time (hours)", 0.0, 14.0, 5.0)
    inputs["sleep_h"] = st.number_input("Sleep Hours", 3.0, 10.0, 7.0)
    inputs["blink_per_min"] = st.number_input("Blink Rate (per min)", 5, 40, 18)
    inputs["pupil_mm"] = st.number_input("Pupil Size (mm)", 2.0, 6.0, 3.2)


# Derived engineered features
inputs["iop_cct_ratio"] = round(inputs["iop"] / (inputs["cct"]+1e-3), 4)
inputs["screen_sleep_ratio"] = round((inputs["screen_time_h"]+1) / (inputs["sleep_h"]+1e-3), 4)
inputs["age_iop"] = round(np.sqrt(inputs["age"] * inputs["iop"]), 2)
inputs["bp_screen"] = round(np.log1p(inputs["bp_sys"] * inputs["screen_time_h"]), 2)

# Ensure all features are present in correct order
X_input = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=FEATURES)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Risk"):
    proba = model.predict_proba(X_input)[:,1][0]
    pred = 1 if proba >= THRESHOLD else 0

    st.subheader("Prediction Result")
    st.write(f"**Glaucoma Risk Probability:** {proba:.2f}")
    if pred == 1:
        st.error("⚠️ High Risk — patient may need clinical evaluation")
    else:
        st.success("✅ Low Risk")

    # ----------------------------
    # SHAP explanation
    # ----------------------------
    st.subheader("Feature Contributions (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)

    shap_df = pd.DataFrame({
        "Feature": X_input.columns,
        "Value": X_input.iloc[0].values,
        "SHAP Impact": shap_values[0]
    }).sort_values("SHAP Impact", key=abs, ascending=False)

    st.table(shap_df.head(6))
