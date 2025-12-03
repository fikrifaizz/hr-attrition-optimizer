# Employee Retention Prescriptive System

> **Identifikasi karyawan berisiko resign lebih awal & rekomendasikan strategi retensi yang personal.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)

## Konteks Bisnis
Tingginya angka perputaran karyawan (*turnover*) merugikan perusahaan hingga **$75,000 per kejadian** (biaya rekrutmen & pelatihan ulang). Analitik HR tradisional hanya mampu memprediksi *siapa* yang mungkin keluar, namun gagal memberikan solusi preskriptif mengenai *apa yang harus dilakukan*.

Proyek ini menjawab tantangan tersebut dengan membangun **Sistem Analitik Preskriptif** yang mampu:
1.  Memprediksi risiko *attrition* dengan **Recall 81%** (Meminimalkan risiko karyawan keluar tanpa terdeteksi).
2.  Mengestimasi dampak finansial (*Financial Loss*) per karyawan.
3.  Merekomendasikan intervensi spesifik berdasarkan nilai **SHAP (Explainable AI)**.

## Fitur Utama
* **End-to-End ETL Pipeline:** Pembersihan & transformasi data otomatis dari sumber mentah hingga siap model.
* **Imbalanced Learning Strategy:** Mengoptimalkan XGBoost menggunakan parameter `scale_pos_weight` untuk menangani ketimpangan kelas data.
* **Interactive Dashboard:** Aplikasi Streamlit bagi Manajer HR untuk mensimulasikan skenario risiko dan melihat estimasi biaya.
* **Actionable Insights:** Menerjemahkan output model teknis menjadi tindakan bisnis nyata (Contoh: "Review Jam Lembur" vs "Sesuaikan Gaji").

## Struktur Proyek
```bash
hr-retention-system/
├── data/               # Data Mentah (Raw) & Hasil Proses (Processed)
├── notebooks/          # Arena Eksperimen (LogReg vs RF vs XGBoost)
├── src/                # Kode Produksi
│   ├── etl/            # Skrip Extract-Transform-Load
│   └── modeling/       # Skrip Training & Penyimpanan Model
├── app.py              # Antarmuka Dashboard Streamlit
└── requirements.txt    # Daftar Pustaka (Dependencies)
```

## Cara Menjalankan

1.  **Clone repositori ini**
    ```bash
    git clone https://github.com/fikrifaizz/hr-retention-system.git
    cd hr-retention-system
    ```

2.  **Install Dependensi**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan ETL Pipeline** (Opsional, data contoh sudah diproses)
    ```bash
    python -m src.run_pipeline
    ```

4.  **Latih Model (Retrain)**
    ```bash
    python -m src.modeling.train
    ```

5.  **Buka Dashboard**
    ```bash
    streamlit run app.py
    ```

## Performa Model
Setelah melalui tahap eksperimen membandingkan Logistic Regression, Random Forest, dan XGBoost, model **XGBoost** dipilih sebagai *Champion Model*.

| Metrik | Skor | Catatan |
| :--- | :--- | :--- |
| **Recall (Kelas Churn)** | **81%** | Prioritas utama untuk meminimalkan False Negatives (Kecolongan). |
| **AUC-ROC** | **0.77** | Kemampuan pemisahan kelas yang baik. |
| **Precision** | 26% | *Trade-off* yang diterima demi menangkap risiko maksimal (Biaya preventif < Biaya penggantian). |

---
*Dibuat oleh Fikri Faiz Zulfadhli - Data Science Enthusiast*