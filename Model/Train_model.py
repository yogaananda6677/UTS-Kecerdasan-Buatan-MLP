# ============================================================
# TRAIN MODEL MLP (MULTILAYER PERCEPTRON)
# Klasifikasi Penyakit Jantung (Heart Disease UCI Cleveland)
# Proyek Ujian Tengah Semester (UTS) - Kecerdasan Buatan
# D3 Manajemen Informatika - PSDKU Polinema Kediri
# ============================================================

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. MEMBACA DATASET
# ============================================================

FILE_DATASET = "heart.csv"
data = pd.read_csv(FILE_DATASET)

print("=" * 65)
print("1. DATASET PENYAKIT JANTUNG (UCI CLEVELAND)")
print("=" * 65)
print(data.head())
print("\nJumlah Data Total :", len(data))
print("Jumlah Fitur       :", len(data.columns) - 1)
print("Daftar Kolom       :", list(data.columns))
print("\nDistribusi Target (0: Sehat, 1: Penyakit Jantung):")
print(data["target"].value_counts())

# Penanganan Nilai Kosong (Imputasi jika ada)
data = data.ffill().bfill()

# ============================================================
# 2. MENENTUKAN FITUR DAN TARGET
# ============================================================

features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

X = data[features]
y = data["target"]

# ============================================================
# 3. MEMBAGI DATA TRAINING DAN TESTING (80% : 20%)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 65)
print("2. PEMBAGIAN DATASET (80% LATIH : 20% UJI)")
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
# 5. MEMBUAT MODEL MLP CLASSIFIER (SESUAI MODUL & JURNAL ACUAN)
# ============================================================
# Arsitektur Jaringan:
# - Input Layer   : 13 Neuron (Fitur Klinis)
# - Hidden Layer 1: 64 Neuron (Aktivasi: ReLU)
# - Hidden Layer 2: 32 Neuron (Aktivasi: ReLU)
# - Output Layer  : 1 Neuron  (Klasifikasi Biner)
# - Optimizer     : Adam (Learning Rate = 0.001)

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
)

# ============================================================
# 6. TRAINING MODEL MLP
# ============================================================

print("\n" + "=" * 65)
print("3. TRAINING MODEL MULTILAYER PERCEPTRON (MLP)")
print("=" * 65)
model.fit(X_train_scaled, y_train)
print("Training MLP Selesai.")
print("Jumlah Iterasi Konvergensi:", model.n_iter_)

# ============================================================
# 7. EVALUASI MODEL PADA DATA TESTING
# ============================================================

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\nHasil Evaluasi Data Testing:")
print(f" - Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
print(f" - Precision : {prec:.4f} ({prec*100:.2f}%)")
print(f" - Recall    : {rec:.4f} ({rec*100:.2f}%)")
print(f" - F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
print(f" - ROC-AUC   : {auc:.4f} ({auc*100:.2f}%)")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Sehat (0)", "Penyakit Jantung (1)"]))

# ============================================================
# 8. TABEL KOMPARASI: HASIL JURNAL VS HASIL PROYEK MANDIRI
# ============================================================

print("=" * 65)
print("4. TABEL KOMPARASI HASIL: JURNAL ACUAN VS PROYEK MANDIRI")
print("=" * 65)

# Nilai dari artikel jurnal acuan (JAIC 2025 - Model MLP Baseline)
jurnal_acc = 86.00
jurnal_prec = 87.00
jurnal_rec = 87.00
jurnal_f1 = 87.00
jurnal_auc = 90.00

comparison_rows = [
    {
        "Metrik Evaluasi": "Akurasi (Accuracy)",
        "Hasil Jurnal Acuan": f"{jurnal_acc:.2f}%",
        "Hasil Proyek Mandiri": f"{acc * 100:.2f}%",
        "Selisih": f"{(acc * 100 - jurnal_acc):+.2f}%"
    },
    {
        "Metrik Evaluasi": "Presisi (Precision)",
        "Hasil Jurnal Acuan": f"{jurnal_prec:.2f}%",
        "Hasil Proyek Mandiri": f"{prec * 100:.2f}%",
        "Selisih": f"{(prec * 100 - jurnal_prec):+.2f}%"
    },
    {
        "Metrik Evaluasi": "Recall (Sensitivitas)",
        "Hasil Jurnal Acuan": f"{jurnal_rec:.2f}%",
        "Hasil Proyek Mandiri": f"{rec * 100:.2f}%",
        "Selisih": f"{(rec * 100 - jurnal_rec):+.2f}%"
    },
    {
        "Metrik Evaluasi": "F1-Score",
        "Hasil Jurnal Acuan": f"{jurnal_f1:.2f}%",
        "Hasil Proyek Mandiri": f"{f1 * 100:.2f}%",
        "Selisih": f"{(f1 * 100 - jurnal_f1):+.2f}%"
    },
    {
        "Metrik Evaluasi": "ROC-AUC",
        "Hasil Jurnal Acuan": f"{jurnal_auc:.2f}%",
        "Hasil Proyek Mandiri": f"{auc * 100:.2f}%",
        "Selisih": f"{(auc * 100 - jurnal_auc):+.2f}%"
    }
]

df_comp = pd.DataFrame(comparison_rows)
print(df_comp.to_string(index=False))

# ============================================================
# 9. MENYIMPAN MODEL, SCALER, DAN DATA METRIK
# ============================================================

joblib.dump(model, "model_mlp.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "feature_names.pkl")

metrics_data = {
    "jurnal": {
        "judul": "Implementation of Deep Learning with Multilayer Perceptron (MLP) for Heart Disease Prediction Using the SMOTE-ENN Technique",
        "penulis": "Erik Saputra & Erliyan Redy Susanto",
        "institusi": "Universitas Teknokrat Indonesia",
        "jurnal_nama": "Journal of Applied Informatics and Computing (JAIC)",
        "edisi": "Vol. 9, No. 3, June 2025, pp. 1034-1041",
        "doi": "10.30871/jaic.v9i3.9337",
        "dataset_sumber": "UCI Machine Learning Repository (Cleveland Heart Disease)",
        "metrik": {
            "accuracy": jurnal_acc,
            "precision": jurnal_prec,
            "recall": jurnal_rec,
            "f1_score": jurnal_f1,
            "roc_auc": jurnal_auc
        }
    },
    "proyek": {
        "dataset_total": len(data),
        "train_count": len(X_train),
        "test_count": len(X_test),
        "arsitektur": "MLP (13 Input -> 64 Hidden 1 [ReLU] -> 32 Hidden 2 [ReLU] -> 1 Output)",
        "metrik": {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "roc_auc": round(auc * 100, 2),
            "cm": cm.tolist()
        }
    }
}

with open("metrics_comparison.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

print("\n" + "=" * 65)
print("Model 'model_mlp.pkl', scaler, dan 'metrics_comparison.json' tersimpan.")
print("=" * 65)
