# **OcuSight — Glaucoma Risk Prediction**

OcuSight is a machine learning prototype that predicts glaucoma risk using clinical and wearable features.  
The project demonstrates an end-to-end workflow with model benchmarking, explainability using SHAP.  

---

## **Dataset**
- **25,000 synthetic patient records** (~32% glaucoma risk).  
- Features include:  
  - **Clinical**: intraocular pressure (IOP), corneal thickness (CCT), blood pressure, heart rate, pupil size  
  - **Lifestyle**: screen time, sleep hours, blink rate  
  - **Engineered**: IOP/CCT ratio, screen/sleep ratio, age × IOP, BP × screen  

---

## **Methods and Results**
- Models tested: Logistic Regression, Random Forest, SVM (RBF), MLP, **XGBoost**  
- **XGBoost achieved the best performance**:  
  - Accuracy: ~87%  
  - Precision: 0.98  
  - AUC: 0.83  
- SHAP confirmed key clinical drivers: **corneal thickness, intraocular pressure, blood pressure, blink rate**  

---

## **Streamlit Application**
Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
---
## **Or try the live demo here:**

```bash
https://ocusight.streamlit.app
