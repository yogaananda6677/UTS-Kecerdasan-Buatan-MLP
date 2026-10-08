# ============================================================
# TRAIN MODEL MLP - PREDIKSI PENYAKIT JANTUNG (UTS KECERDASAN BUATAN)
# Referensi Jurnal:
# "Implementation of Deep Learning with Multilayer Perceptron (MLP) 
#  for Heart Disease Prediction Using the SMOTE-ENN Technique"
# Penulis: Erik Saputra & Erliyan Redy Susanto (JAIC Vol. 9 No. 3, 2025)
# Dataset: UCI Machine Learning Repository (heart.csv)
# ============================================================

import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTEENN
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. MEMBACA DATASET
# ============================================================

FILE_DATASET = "heart.csv"
data = pd.read_csv(FILE_DATASET)

print("=" * 65)
print("1. DATASET PENYAKIT JANTUNG (UCI REPOSITORY)")
print("=" * 65)
print(data.head())
print("\nJumlah Data Total :", len(data))
print("Jumlah Fitur       :", len(data.columns) - 1)
print("Daftar Kolom       :", data.columns.tolist())
print("\nDistribusi Target (0: Sehat, 1: Penyakit Jantung):")
print(data["target"].value_counts())

# ============================================================
# 2. MENENTUKAN FITUR DAN TARGET
# ============================================================

# 13 Fitur Prediktor klinis sesuai Tabel 1 pada Jurnal Referensi
features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

X = data[features]
y = data["target"]

# ============================================================
# 3. MEMBAGI DATA TRAINING DAN TESTING (80 : 20)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\n" + "=" * 65)
print("2. PEMBAGIAN DATASET (80 : 20)")
print("=" * 65)
print("Jumlah Data Training :", len(X_train))
print("Jumlah Data Testing  :", len(X_test))

# ============================================================
# 4. STANDARDISASI FITUR (STANDARD SCALER)
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 5. MEMBUAT MODEL MLP CLASSIFIER (SESUAI ARSITEKTUR JURNAL)
# ============================================================
# Arsitektur Jurnal:
# - Input Layer   : 13 Neuron
# - Hidden Layer 1: 64 Neuron (Aktivasi: ReLU)
# - Hidden Layer 2: 32 Neuron (Aktivasi: ReLU)
# - Output Layer  : 1 Neuron  (Aktivasi: Sigmoid / Logistic)
# - Optimizer     : Adam (Learning Rate = 0.001)

model_baseline = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
)

# ============================================================
# 6. TRAINING MODEL BASELINE
# ============================================================

print("\n" + "=" * 65)
print("3. TRAINING MODEL BASELINE MLP (TANPA SMOTE-ENN)")
print("=" * 65)
model_baseline.fit(X_train_scaled, y_train)
print("Training Baseline Selesai.")
print("Jumlah Iterasi Konvergensi:", model_baseline.n_iter_)

# ============================================================
# 7. EVALUASI MODEL BASELINE
# ============================================================

y_pred_base = model_baseline.predict(X_test_scaled)
y_prob_base = model_baseline.predict_proba(X_test_scaled)[:, 1]

acc_base = accuracy_score(y_test, y_pred_base)
prec_base = precision_score(y_test, y_pred_base)
rec_base = recall_score(y_test, y_pred_base)
f1_base = f1_score(y_test, y_pred_base)
auc_base = roc_auc_score(y_test, y_prob_base)
cm_base = confusion_matrix(y_test, y_pred_base)

print("\nHasil Evaluasi Data Testing (Baseline MLP):")
print(f" - Accuracy  : {acc_base:.4f} ({acc_base*100:.2f}%)")
print(f" - Precision : {prec_base:.4f} ({prec_base*100:.2f}%)")
print(f" - Recall    : {rec_base:.4f} ({rec_base*100:.2f}%)")
print(f" - F1-Score  : {f1_base:.4f} ({f1_base*100:.2f}%)")
print(f" - ROC-AUC   : {auc_base:.4f} ({auc_base*100:.2f}%)")
print("\nConfusion Matrix:")
print(cm_base)

# ============================================================
# 8. METODE PROPOSED JURNAL: MLP DENGAN SMOTE-ENN RESAMPLING
# ============================================================

print("\n" + "=" * 65)
print("4. PENERAPAN TEKNIK SMOTE-ENN (SESUAI PROPOSED METHOD JURNAL)")
print("=" * 65)

smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train_scaled, y_train)

print(f"Jumlah Data Latih Sebelum Resampling: {len(X_train_scaled)}")
print(f"Jumlah Data Latih Sesudah SMOTE-ENN : {len(X_train_res)}")

model_smote = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
)
model_smote.fit(X_train_res, y_train_res)

y_pred_smote = model_smote.predict(X_test_scaled)
y_prob_smote = model_smote.predict_proba(X_test_scaled)[:, 1]

acc_smote = accuracy_score(y_test, y_pred_smote)
prec_smote = precision_score(y_test, y_pred_smote)
rec_smote = recall_score(y_test, y_pred_smote)
f1_smote = f1_score(y_test, y_pred_smote)
auc_smote = roc_auc_score(y_test, y_prob_smote)
cm_smote = confusion_matrix(y_test, y_pred_smote)

print("\nHasil Evaluasi Data Testing (MLP + SMOTE-ENN):")
print(f" - Accuracy  : {acc_smote:.4f} ({acc_smote*100:.2f}%)")
print(f" - Precision : {prec_smote:.4f} ({prec_smote*100:.2f}%)")
print(f" - Recall    : {rec_smote:.4f} ({rec_smote*100:.2f}%)")
print(f" - F1-Score  : {f1_smote:.4f} ({f1_smote*100:.2f}%)")
print(f" - ROC-AUC   : {auc_smote:.4f} ({auc_smote*100:.2f}%)")
print("\nConfusion Matrix:")
print(cm_smote)

# ============================================================
# 9. TABEL KOMPARASI: JURNAL VS IMPLEMENTASI SENDIRI
# ============================================================

print("\n" + "=" * 65)
print("5. TABEL PERBANDINGAN: HASIL JURNAL VS HASIL IMPLEMENTASI SENDIRI")
print("=" * 65)

comparison_table = [
    {
        "Metode": "Baseline MLP (Tanpa SMOTE-ENN)",
        "Akurasi_Jurnal": "86.00%",
        "Akurasi_Proyek": f"{acc_base*100:.2f}%",
        "Presisi_Jurnal": "87.00%",
        "Presisi_Proyek": f"{prec_base*100:.2f}%",
        "Recall_Jurnal": "87.00%",
        "Recall_Proyek": f"{rec_base*100:.2f}%",
        "F1_Jurnal": "87.00%",
        "F1_Proyek": f"{f1_base*100:.2f}%",
        "AUC_Jurnal": "90.00%",
        "AUC_Proyek": f"{auc_base*100:.2f}%",
    },
    {
        "Metode": "Proposed MLP (Dengan SMOTE-ENN)",
        "Akurasi_Jurnal": "89.47%",
        "Akurasi_Proyek": f"{acc_smote*100:.2f}%",
        "Presisi_Jurnal": "77.78%",
        "Presisi_Proyek": f"{prec_smote*100:.2f}%",
        "Recall_Jurnal": "100.00%",
        "Recall_Proyek": f"{rec_smote*100:.2f}%",
        "F1_Jurnal": "87.50%",
        "F1_Proyek": f"{f1_smote*100:.2f}%",
        "AUC_Jurnal": "97.00%",
        "AUC_Proyek": f"{auc_smote*100:.2f}%",
    }
]

df_comp = pd.DataFrame(comparison_table)
print(df_comp.to_string(index=False))

# ============================================================
# 10. MENYIMPAN MODEL, SCALER, DAN DATA KOMPARASI
# ============================================================

joblib.dump(model_baseline, "model_mlp_baseline.pkl")
joblib.dump(model_smote, "model_mlp.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "feature_names.pkl")

# Simpan metadata perbandingan untuk digunakan oleh Web App
metrics_data = {
    "jurnal": {
        "judul": "Implementation of Deep Learning with Multilayer Perceptron (MLP) for Heart Disease Prediction Using the SMOTE-ENN Technique",
        "penulis": "Erik Saputra & Erliyan Redy Susanto",
        "institusi": "Universitas Teknokrat Indonesia",
        "jurnal_nama": "Journal of Applied Informatics and Computing (JAIC)",
        "edisi": "Vol. 9, No. 3, June 2025, pp. 1034-1041",
        "doi": "10.30871/jaic.v9i3.9337",
        "dataset_sumber": "UCI Machine Learning Repository (Heart Disease Cleveland)",
        "baseline": {
            "accuracy": 86.00,
            "precision": 87.00,
            "recall": 87.00,
            "f1_score": 87.00,
            "roc_auc": 90.00
        },
        "smote_enn": {
            "accuracy": 89.47,
            "precision": 77.78,
            "recall": 100.00,
            "f1_score": 87.50,
            "roc_auc": 97.00
        }
    },
    "proyek": {
        "dataset_total": len(data),
        "train_count": len(X_train),
        "test_count": len(X_test),
        "arsitektur": "MLP (Input: 13, Hidden 1: 64 [ReLU], Hidden 2: 32 [ReLU], Output: 1 [Sigmoid])",
        "baseline": {
            "accuracy": round(acc_base * 100, 2),
            "precision": round(prec_base * 100, 2),
            "recall": round(rec_base * 100, 2),
            "f1_score": round(f1_base * 100, 2),
            "roc_auc": round(auc_base * 100, 2),
            "cm": cm_base.tolist()
        },
        "smote_enn": {
            "accuracy": round(acc_smote * 100, 2),
            "precision": round(prec_smote * 100, 2),
            "recall": round(rec_smote * 100, 2),
            "f1_score": round(f1_smote * 100, 2),
            "roc_auc": round(auc_smote * 100, 2),
            "cm": cm_smote.tolist()
        }
    }
}

with open("metrics_comparison.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

print("\n" + "=" * 65)
print("Model dan data metrik berhasil disimpan ke disk.")
print("=" * 65)
