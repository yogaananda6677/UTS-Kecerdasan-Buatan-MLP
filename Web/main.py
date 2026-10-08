import json
import os
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Inisialisasi Path Direktori
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
DATASET_PATH = BASE_DIR / "heart.csv"

# Inisialisasi FastAPI App
app = FastAPI(
    title="CardioMLP Analytics - Prediksi Penyakit Jantung (MLP)",
    description="Sistem Cerdas Klasifikasi Risiko Penyakit Jantung Berbasis Multilayer Perceptron & SMOTE-ENN (UTS Kecerdasan Buatan)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount Static Files & Templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Muat Model, Scaler, dan Metrik Komparasi
try:
    model_smote = joblib.load(MODELS_DIR / "model_mlp.pkl")
    model_baseline = joblib.load(MODELS_DIR / "model_mlp_baseline.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    with open(MODELS_DIR / "metrics_comparison.json", "r") as f:
        metrics_data = json.load(f)
except Exception as e:
    raise RuntimeError(f"Gagal memuat artefak model Machine Learning: {e}")

# Kamus Data Deskripsi Fitur Klinis (Sesuai UCI Cleveland)
FEATURE_INFO = {
    "age": {"label": "Usia", "unit": "Tahun", "desc": "Usia biologis pasien saat pemeriksaan klinis"},
    "sex": {"label": "Jenis Kelamin", "unit": "Biner", "desc": "1 = Laki-laki, 0 = Perempuan"},
    "cp": {"label": "Tipe Nyeri Dada", "unit": "Skala 1-4", "desc": "1: Typical Angina, 2: Atypical Angina, 3: Non-anginal, 4: Asymptomatic"},
    "trestbps": {"label": "Tekanan Darah Istirahat", "unit": "mmHg", "desc": "Tekanan darah sistolik saat istirahat di rumah sakit"},
    "chol": {"label": "Kolesterol Serum", "unit": "mg/dL", "desc": "Kadar total kolesterol darah (nilai normal < 200 mg/dL)"},
    "fbs": {"label": "Gula Darah Puasa > 120", "unit": "Biner", "desc": "1 = Ya (>120 mg/dL), 0 = Tidak (indikator skrining diabetes)"},
    "restecg": {"label": "Hasil EKG Istirahat", "unit": "Skala 0-2", "desc": "0: Normal, 1: Kelainan gelombang ST-T, 2: Hipertrofi ventrikel kiri"},
    "thalach": {"label": "Detak Jantung Maksimum", "unit": "bpm", "desc": "Detak jantung puncak tercapai selama uji treadmill"},
    "exang": {"label": "Angina Induksi Latihan", "unit": "Biner", "desc": "1 = Mengalami nyeri dada saat olahraga/latihan, 0 = Tidak"},
    "oldpeak": {"label": "Depresi ST Induksi Latihan", "unit": "Skala", "desc": "Tingkat depresi segmen ST relatif terhadap fase istirahat"},
    "slope": {"label": "Kemiringan Segmen ST", "unit": "Skala 1-3", "desc": "1: Menanjak (Upsloping), 2: Rata (Flat), 3: Menurun (Downsloping)"},
    "ca": {"label": "Jumlah Pembuluh Utama", "unit": "0-3", "desc": "Jumlah pembuluh koroner utama terdeteksi pewarnaan fluoroskopi"},
    "thal": {"label": "Thalassemia (Perfusi)", "unit": "Kategori", "desc": "3: Normal, 6: Cacat Menetap (Fixed Defect), 7: Cacat Reversibel"}
}

# Schema Request Pydantic
class HeartDiagnosisRequest(BaseModel):
    model_type: str = Field(default="smote", description="Pilihan model: 'smote' atau 'baseline'")
    age: float = Field(..., ge=1, le=120, description="Usia pasien")
    sex: float = Field(..., ge=0, le=1, description="Jenis kelamin: 1 (Laki-laki) atau 0 (Perempuan)")
    cp: float = Field(..., ge=1, le=4, description="Tipe nyeri dada: 1..4")
    trestbps: float = Field(..., ge=50, le=300, description="Tekanan darah istirahat (mmHg)")
    chol: float = Field(..., ge=50, le=700, description="Kolesterol serum (mg/dL)")
    fbs: float = Field(..., ge=0, le=1, description="Gula darah puasa: 1 (>120 mg/dL) atau 0")
    restecg: float = Field(..., ge=0, le=2, description="Hasil EKG: 0, 1, atau 2")
    thalach: float = Field(..., ge=50, le=250, description="Detak jantung maksimum (bpm)")
    exang: float = Field(..., ge=0, le=1, description="Angina latihan: 1 (Ya) atau 0 (Tidak)")
    oldpeak: float = Field(..., ge=0, le=10, description="Depresi ST")
    slope: float = Field(..., ge=1, le=3, description="Slope segmen ST: 1, 2, atau 3")
    ca: float = Field(..., ge=0, le=3, description="Jumlah pembuluh darah utama: 0..3")
    thal: float = Field(..., ge=1, le=7, description="Thalassemia: 3 (Normal), 6 (Fixed), 7 (Reversible)")

def evaluasi_rekomendasi_medis(prediksi: int, prob_sakit: float) -> Dict[str, Any]:
    if prediksi == 1:
        rekomendasi = [
            "Lakukan pemeriksaan kardiologi lanjutan (Echocardiography, Angiografi Koroner, atau CT Calcium Score).",
            "Konsultasikan hasil diagnosis ke dokter spesialis jantung dan pembuluh darah (Sp.JP).",
            "Terapkan diet rendah lemak jenuh, batasi natrium/garam, dan hentikan kebiasaan merokok seketika.",
            "Kelola faktor stres dan pantau tekanan darah serta profil lipid secara berkala setiap bulan."
        ]
        status_text = "Positif Berisiko Penyakit Jantung"
        status_badge = "badge-danger"
        kesimpulan = f"Model Artificial Neural Network (MLP) mendeteksi pola fisiologis yang mengindikasikan risiko tinggi penyakit jantung dengan probabilitas keyakinan {prob_sakit:.1f}%."
    else:
        rekomendasi = [
            "Pertahankan pola hidup sehat dengan aktivitas aerobik moderat minimal 150 menit per minggu.",
            "Lakukan pemeriksaan berkala kesehatan jantung (Medical Check-Up tahunan).",
            "Jaga indeks massa tubuh ideal dan pertahankan konsumsi serat sayuran serta asam lemak omega-3.",
            "Pantau tekanan darah dan kadar gula darah puasa secara rutin."
        ]
        status_text = "Negatif / Kardiovaskular Aman (Sehat)"
        status_badge = "badge-success"
        kesimpulan = f"Parameter fisiologis dan kardiovaskular pasien saat ini berada dalam rentang ambang aman dengan probabilitas sehat {(100.0 - prob_sakit):.1f}%."

    return {
        "status_text": status_text,
        "status_badge": status_badge,
        "kesimpulan": kesimpulan,
        "rekomendasi": rekomendasi
    }

# ============================================================
# ROUTING HALAMAN WEB (HTML RESPONSE)
# ============================================================

@app.get("/", response_class=HTMLResponse, name="index")
async def page_index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "feature_info": FEATURE_INFO,
            "metrics": metrics_data
        }
    )

@app.get("/komparasi", response_class=HTMLResponse, name="komparasi")
async def page_komparasi(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="komparasi.html",
        context={
            "metrics": metrics_data
        }
    )

@app.get("/arsitektur", response_class=HTMLResponse, name="arsitektur")
async def page_arsitektur(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="arsitektur.html",
        context={
            "feature_info": FEATURE_INFO,
            "metrics": metrics_data
        }
    )

# ============================================================
# API ENDPOINTS (JSON RESPONSE)
# ============================================================

@app.post("/api/predict")
async def api_predict(payload: HeartDiagnosisRequest):
    try:
        model_type = payload.model_type.lower()
        req_dict = payload.model_dump()

        input_values = []
        for feat in feature_names:
            if feat not in req_dict:
                raise HTTPException(status_code=400, detail=f"Fitur '{feat}' tidak ditemukan pada input")
            input_values.append(float(req_dict[feat]))

        df_input = pd.DataFrame([input_values], columns=feature_names)
        input_scaled = scaler.transform(df_input)

        active_model = model_smote if model_type == "smote" else model_baseline
        prediction = int(active_model.predict(input_scaled)[0])
        probabilities = active_model.predict_proba(input_scaled)[0]
        prob_sehat = round(float(probabilities[0]) * 100, 2)
        prob_sakit = round(float(probabilities[1]) * 100, 2)

        analisis = evaluasi_rekomendasi_medis(prediction, prob_sakit)

        return JSONResponse(content={
            "success": True,
            "prediksi": prediction,
            "label": "Penyakit Jantung" if prediction == 1 else "Sehat",
            "model_digunakan": "MLP + SMOTE-ENN (Proposed Jurnal)" if model_type == "smote" else "Baseline MLP",
            "probabilitas": {
                "sehat": prob_sehat,
                "sakit": prob_sakit
            },
            "analisis": analisis
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/api/metrics")
async def api_metrics():
    return JSONResponse(content=metrics_data)

@app.get("/health")
async def api_health():
    return {"status": "ok", "app": "CardioMLP Analytics", "framework": "FastAPI"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
