# ============================================================
# TEST MODEL MLP - PENGUJIAN DATA BARU (UTS KECERDASAN BUATAN)
# Klasifikasi Penyakit Jantung (Heart Disease UCI)
# ============================================================

import joblib
import numpy as np
import pandas as pd

# ============================================================
# 1. LOAD MODEL DAN SCALER
# ============================================================

model = joblib.load("model_mlp.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("feature_names.pkl")

print("============================================================")
print(" SISTEM PREDIKSI PENYAKIT JANTUNG (MLP CLASSIFIER)")
print(" REFERENSI JURNAL JAIC 2025")
print("============================================================\n")

# ============================================================
# 2. INPUT DATA PASIEN BARU
# ============================================================

print("Masukkan parameter klinis pasien:")
print("------------------------------------------------------------")

try:
    age = float(input("1.  Usia (tahun)                                      : "))
    sex = int(input("2.  Jenis Kelamin (1 = Laki-laki, 0 = Perempuan)       : "))
    cp = int(input("3.  Tipe Nyeri Dada (1: Typical, 2: Atypical, 3: Non-anginal, 4: Asymptomatic): "))
    trestbps = float(input("4.  Tekanan Darah Istirahat (mmHg)                    : "))
    chol = float(input("5.  Kadar Kolesterol Serum (mg/dL)                    : "))
    fbs = int(input("6.  Gula Darah Puasa > 120 mg/dL (1 = Ya, 0 = Tidak)  : "))
    restecg = int(input("7.  Hasil EKG Istirahat (0: Normal, 1: ST-T, 2: LVH)  : "))
    thalach = float(input("8.  Detak Jantung Maksimal Tercapai (bpm)             : "))
    exang = int(input("9.  Angina Induksi Olahraga (1 = Ya, 0 = Tidak)       : "))
    oldpeak = float(input("10. ST Depression Induksi Olahraga                    : "))
    slope = int(input("11. Kemiringan Segmen ST Puncak (1: Upsloping, 2: Flat, 3: Downsloping): "))
    ca = int(input("12. Jumlah Pembuluh Darah Utama Fluoroskopi (0 - 3)   : "))
    thal = int(input("13. Thalassemia (3: Normal, 6: Fixed Defect, 7: Reversible Defect): "))

    # ============================================================
    # 3. MEMBENTUK DATA INPUT
    # ============================================================

    data_baru = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])[features]

    # ============================================================
    # 4. STANDARDISASI
    # ============================================================

    data_scaled = scaler.transform(data_baru)

    # ============================================================
    # 5. PREDIKSI
    # ============================================================

    prediksi = model.predict(data_scaled)[0]
    probabilitas = model.predict_proba(data_scaled)[0]

    # ============================================================
    # 6. HASIL
    # ============================================================

    print("\n============================================================")
    print("HASIL PREDIKSI DIAGNOSIS KLINIS")
    print("============================================================")
    print(f"Probabilitas Pasien Sehat        : {probabilitas[0] * 100:.2f}%")
    print(f"Probabilitas Pasien Sakit Jantung: {probabilitas[1] * 100:.2f}%")
    print("------------------------------------------------------------")
    if prediksi == 1:
        print("HASIL AKHIR : POSITIF BERISIKO PENYAKIT JANTUNG (1)")
        print("Keterangan  : Pasien menunjukkan pola risiko tinggi kardiovaskular.")
    else:
        print("HASIL AKHIR : NEGATIF / JANTUNG SEHAT (0)")
        print("Keterangan  : Parameter fisiologis pasien dalam batas aman.")
    print("============================================================\n")

except Exception as e:
    print(f"\n[ERROR] Masukan tidak valid: {e}")
