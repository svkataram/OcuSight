OcuSight - Glaucoma Risk Prediction

OcuSight is a machine learning prototype that predicts glaucoma risk using clinical and wearable features.
The project demonstrates an end-to-end workflow with benchmarking of models, explainability using SHAP, and an interactive Streamlit application.

Dataset

Synthetic dataset of 25,000 patients (~32% glaucoma risk).

Features include clinical measures (IOP, corneal thickness, blood pressure, heart rate, pupil size) and lifestyle factors (screen time, sleep, blink rate), with additional engineered ratios.

Methods and Results

Models tested: Logistic Regression, Random Forest, SVM (RBF), MLP, and XGBoost.

XGBoost achieved the best performance: ~87% accuracy, 0.98 precision, 0.83 AUC.

SHAP confirmed clinically relevant drivers such as corneal thickness and intraocular pressure.

Streamlit Application

Run locally:

pip install -r requirements.txt
streamlit run app.py


Or try the live demo here:
👉 https://ocusight.streamlit.app/
