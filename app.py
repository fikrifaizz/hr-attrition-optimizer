import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import shap

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="HR Retention Optimizer",
    page_icon="🛡️",
    layout="wide"
)

# --- LOAD ARTIFACTS (CACHE AGAR CEPAT) ---
@st.cache_resource
def load_artifacts():
    # Path relatif terhadap app.py
    model_path = Path("data/models/xgboost_final.pkl")
    cols_path = Path("data/models/model_columns.pkl")
    
    if not model_path.exists():
        st.error("Model belum ditemukan. Jalankan src/modeling/train.py dulu!")
        return None, None
        
    model = joblib.load(model_path)
    model_cols = joblib.load(cols_path)
    return model, model_cols

model, model_features = load_artifacts()

# --- BUSINESS LOGIC: PRESCRIPTIVE ENGINE ---
# Mapping fitur ke solusi bisnis (Knowledge Base)
SOLUTIONS = {
    'overtime_yes': "🔴 **High Burnout Risk:** Lakukan audit beban kerja tim. Pertimbangkan hiring staff tambahan atau rotasi tugas.",
    'monthlyincome': "💰 **Compensation Gap:** Cek benchmark gaji pasar. Pertimbangkan penyesuaian gaji atau berikan Performance Bonus.",
    'stockoptionlevel': "📉 **Ownership Issue:** Karyawan merasa kurang memiliki perusahaan. Tawarkan opsi saham (ESOP) atau bonus jangka panjang.",
    'jobrole_sales representative': "🏃 **High Turnover Role:** Posisi Sales Rep sangat rentan. Perbaiki skema komisi dan insentif target.",
    'totalworkingyears': "👶 **Junior Retention:** Risiko tinggi di tahun-tahun awal. Perkuat program Mentoring & Onboarding.",
    'distancefromhome': "🚗 **Commute Stress:** Jarak rumah jauh. Tawarkan opsi **Work From Home (WFH)** atau tunjangan transportasi.",
    'yearsatcompany': "⏳ **Loyalty Fade:** Kejenuhan setelah durasi tertentu. Tawarkan rotasi divisi (Internal Mobility).",
    'jobinvolvement': "💤 **Disengagement:** Kurang terlibat. Manager perlu lakukan 'Stay Interview' untuk mencari aspirasi karir."
}

def get_recommendations(top_features):
    """Menerjemahkan fitur risiko menjadi kalimat solusi."""
    recs = []
    for feature in top_features:
        # Cek apakah fitur ada di dictionary solusi (bisa substring match)
        for key, sol in SOLUTIONS.items():
            if key in feature.lower():
                recs.append(sol)
                break
    return list(set(recs)) # Hapus duplikat

# --- UI: SIDEBAR INPUT ---
st.sidebar.header("📝 Input Employee Data")

def user_input_features():
    # Definisikan input form (Default value diambil dari median/mode dataset)
    # Kelompok 1: Demografi
    st.sidebar.subheader("Demographics")
    age = st.sidebar.slider("Age", 18, 60, 30)
    distance = st.sidebar.slider("Distance From Home (km)", 1, 30, 5)
    marital = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    
    # Kelompok 2: Pekerjaan
    st.sidebar.subheader("Job Details")
    dept = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    role = st.sidebar.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician", 
        "Manufacturing Director", "Healthcare Representative", "Manager", 
        "Sales Representative", "Research Director", "Human Resources"
    ])
    overtime = st.sidebar.selectbox("OverTime", ["No", "Yes"])
    
    # Kelompok 3: Kompensasi & Karir
    st.sidebar.subheader("Compensation & Tenure")
    income = st.sidebar.number_input("Monthly Income ($)", 1000, 20000, 5000)
    stock = st.sidebar.selectbox("Stock Option Level", [0, 1, 2, 3])
    job_level = st.sidebar.slider("Job Level", 1, 5, 2)
    total_years = st.sidebar.number_input("Total Working Years", 0, 40, 8)
    years_at_company = st.sidebar.number_input("Years at Company", 0, 40, 5)
    years_since_promotion = st.sidebar.number_input("Years Since Last Promotion", 0, 15, 1)
    
    # Kelompok 4: Kepuasan (1-4)
    st.sidebar.subheader("Satisfaction & Performance")
    env_sat = st.sidebar.slider("Environment Satisfaction", 1, 4, 3)
    job_sat = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
    wlb = st.sidebar.slider("Work Life Balance", 1, 4, 3)
    involvement = st.sidebar.slider("Job Involvement", 1, 4, 3)

    # Susun data menjadi DataFrame (sesuai format raw sebelum encoding)
    data = {
        'age': age,
        'distancefromhome': distance,
        'maritalstatus': marital,
        'department': dept,
        'jobrole': role,
        'overtime': overtime,
        'monthlyincome': income,
        'stockoptionlevel': stock,
        'joblevel': job_level,
        'totalworkingyears': total_years,
        'yearsatcompany': years_at_company,
        'yearssincelastpromotion': years_since_promotion,
        'environmentsatisfaction': env_sat,
        'jobsatisfaction': job_sat,
        'worklifebalance': wlb,
        'jobinvolvement': involvement,
        # Default values untuk kolom lain yang tidak diinput user (agar tidak error)
        'education': 3, 'numcompaniesworked': 2, 'percentsalaryhike': 15, 
        'performancerating': 3, 'relationshipsatisfaction': 3, 
        'trainingtimeslastyear': 3, 'yearsincurrentrole': 2, 'yearswithcurrmanager': 2,
        'dailyrate': 800, 'hourlyrate': 60, 'monthlyrate': 14000, 'gender': 'Male', 
        'educationfield': 'Life Sciences', 'businesstravel': 'Travel_Rarely'
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- MAIN PAGE ---
st.title("🛡️ Employee Retention Prescriptive System")
st.markdown("""
Sistem ini menggunakan **XGBoost Classifier** untuk memprediksi risiko *attrition* dan memberikan rekomendasi intervensi berbasis **SHAP Values**.
""")

# Tombol Prediksi
if st.button("🔍 Analyze Risk Profile"):
    
    # 1. Preprocessing Input (Sama persis dengan Training)
    # One-Hot Encoding
    input_encoded = pd.get_dummies(input_df)
    
    # Align Columns: Pastikan kolom input SAMA PERSIS dengan kolom model
    # (Mengisi 0 untuk kolom yang hilang, membuang kolom ekstra)
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)
    
    # 2. Prediksi
    prediction_prob = float(model.predict_proba(input_encoded)[0][1]) 
    prediction_class = int(prediction_prob > 0.5)
    
    # 3. Tampilan Hasil Utama
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Probability")
        if prediction_prob > 0.7:
            st.error(f"⚠️ **CRITICAL RISK: {prediction_prob:.1%}**")
        elif prediction_prob > 0.4:
            st.warning(f"⚠️ **MODERATE RISK: {prediction_prob:.1%}**")
        else:
            st.success(f"✅ **LOW RISK: {prediction_prob:.1%}**")
            
        st.progress(prediction_prob)
        st.caption("Probabilitas karyawan ini akan meninggalkan perusahaan.")

    with col2:
        st.subheader("Financial Impact Estimation")
        cost = 75000 # Asumsi biaya penggantian
        expected_loss = cost * prediction_prob
        st.metric(label="Expected Loss", value=f"${expected_loss:,.0f}", delta="-Risk Cost")
        st.markdown(f"*Based on avg replacement cost of ${cost:,}*")

    st.divider()

    # 4. SHAP Interpretation (The "WHY")
    st.subheader("📊 Why this prediction? (Top Risk Drivers)")
    
    # Hitung SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_encoded)
    
    # Buat DataFrame untuk plotting
    feature_importance = pd.DataFrame({
        'feature': model_features,
        'shap_value': shap_values[0]
    }).sort_values(by='shap_value', ascending=True) # Sort untuk bar chart
    
    # Ambil Top 5 Driver yang menaikkan risiko (SHAP positif)
    top_risk_drivers = feature_importance.tail(5)
    
    # Plotting Manual (Lebih stabil di Streamlit dibanding shap.force_plot)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['red' if x > 0 else 'green' for x in top_risk_drivers['shap_value']]
    ax.barh(top_risk_drivers['feature'], top_risk_drivers['shap_value'], color=colors)
    ax.set_xlabel("Impact on Risk (SHAP Value)")
    st.pyplot(fig)
    
    # 5. Prescriptive Actions (The "WHAT TO DO")
    st.subheader("💊 Recommended Interventions")
    
    # Ambil nama fitur yang nilai SHAP-nya positif (Mendorong risiko)
    risky_features = feature_importance[feature_importance['shap_value'] > 0]['feature'].tolist()
    # Ambil Top 3 paling berpengaruh
    top_3_risky = risky_features[-3:] 
    
    recommendations = get_recommendations(top_3_risky)
    
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("No critical risk factors detected. Keep maintaining good engagement!")

else:
    st.info("👈 Silakan atur profil karyawan di sidebar dan klik 'Analyze Risk Profile'.")