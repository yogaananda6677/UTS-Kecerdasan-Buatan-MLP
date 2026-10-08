# Proyek Ujian Tengah Semester (UTS) - Kecerdasan Buatan
## Klasifikasi Risiko Penyakit Jantung Menggunakan Multilayer Perceptron (MLP) Berbasis Jurnal Ilmiah Nasional

**Program Studi**: D3 Manajemen Informatika  
**Kampus**: PSDKU Polinema di Kota Kediri  
**Mata Kuliah**: Kecerdasan Buatan (Machine Learning / Deep Learning)  
**Metode**: Artificial Neural Network (ANN) - Multilayer Perceptron (MLP) dengan Aktivasi ReLU  
**Framework Web**: Python FastAPI + Uvicorn (Asynchronous High Performance)  
**Repositori GitHub**: [https://github.com/yogaananda6677/UTS-Kecerdasan-Buatan-MLP](https://github.com/yogaananda6677/UTS-Kecerdasan-Buatan-MLP)

---

## 1. Identitas Jurnal Acuan (Referensi Valid)

Penelitian ini mereplikasi dan mengkomparasikan hasil dari artikel jurnal ilmiah nasional terakreditasi:

- **Judul Artikel**: *Implementation of Deep Learning with Multilayer Perceptron (MLP) for Heart Disease Prediction Using the SMOTE-ENN Technique*
- **Penulis**: Erik Saputra, Erliyan Redy Susanto
- **Afiliasi**: Program Studi Informatika, Fakultas Teknik dan Ilmu Komputer, Universitas Teknokrat Indonesia
- **Publikasi**: *Journal of Applied Informatics and Computing (JAIC)*, Vol. 9, No. 3, June 2025, pp. 1034–1041
- **Identifikasi**: e-ISSN: 2548-6861 | Akreditasi SINTA | **DOI**: `10.30871/jaic.v9i3.9337`
- **Fokus Acuan Komparasi**: Mengacu pada pengujian model **Multilayer Perceptron (MLP)** pada dataset Cleveland Heart Disease.

---

## 2. Deskripsi Dataset

- **Sumber**: UCI Machine Learning Repository - *Cleveland Heart Disease Dataset*
- **Jumlah Data**: 303 baris data pasien
- **Jumlah Fitur**: 13 variabel fisiologis & klinis kardiologi:
  1. `age`: Usia pasien (tahun)
  2. `sex`: Jenis kelamin (1 = Laki-laki, 0 = Perempuan)
  3. `cp`: Tipe nyeri dada (1: Typical Angina, 2: Atypical Angina, 3: Non-anginal, 4: Asymptomatic)
  4. `trestbps`: Tekanan darah istirahat saat masuk rumah sakit (mmHg)
  5. `chol`: Kadar kolesterol serum (mg/dL)
  6. `fbs`: Gula darah puasa > 120 mg/dL (1 = Benar, 0 = Salah)
  7. `restecg`: Hasil elektrokardiogram istirahat (0: Normal, 1: ST-T wave abnorm, 2: Left ventricular hypertrophy)
  8. `thalach`: Detak jantung maksimum yang tercapai saat uji treadmill (bpm)
  9. `exang`: Angina yang diinduksi oleh latihan/olahraga (1 = Ya, 0 = Tidak)
  10. `oldpeak`: Depresi ST yang diinduksi oleh latihan relatif terhadap istirahat
  11. `slope`: Kemiringan segmen ST latihan puncak (1: Menanjak/Upsloping, 2: Datar/Flat, 3: Menurun/Downsloping)
  12. `ca`: Jumlah pembuluh darah utama (0-3) yang diwarnai dengan fluoroskopi
  13. `thal`: Status perfusi talasemia (3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect)
- **Target**: `target` (0 = Bebas Penyakit Jantung / Normal, 1 = Terindikasi Penyakit Jantung)
- **Distribusi Target**:
  - Kelas 0 (Sehat): 164 sampel (54.1%)
  - Kelas 1 (Penyakit Jantung): 139 sampel (45.9%)
- **Data Preprocessing**: Imputasi data hilang (`ca` & `thal`), Standard Scaling (`StandardScaler`), dan pembagian Stratified Train-Test Split (80% Latih : 20% Uji) sesuai modul kuliah.

---

## 3. Arsitektur & Hyperparameter Multilayer Perceptron (MLP)

Model dibangun murni menggunakan pustaka **Scikit-Learn `MLPClassifier`** sesuai alur modul praktikum Bab 6:

| Komponen Topologi | Konfigurasi Model | Keterangan Fungsional |
| :--- | :--- | :--- |
| **Input Layer** | 13 Neuron | Menerima 13 fitur klinis kardiologi terstandarisasi |
| **Hidden Layer 1** | 64 Neuron (ReLU) | Ekstraksi fitur non-linear tahap pertama |
| **Hidden Layer 2** | 32 Neuron (ReLU) | Representasi fitur laten risiko kardiovaskular |
| **Output Layer** | 1 Neuron (Sigmoid / Binary Log-Loss) | Estimasi probabilitas risiko penyakit jantung |
| **Optimizer** | Adam (`learning_rate=0.001`) | Adaptive Moment Estimation |
| **Loss Function** | Binary Cross-Entropy / Log-Loss | Pengukuran galat optimasi probabilitas biner |
| **Maks Iterasi** | 1000 Iterasi | Menjamin konvergensi penuh konveks |
| **Pembagian Data** | Train 80% (242 baris), Test 20% (61 baris) | Stratified train-test split (`random_state=42`) |

---

## 4. Komparasi Evaluasi: Jurnal Acuan vs Pengujian Mandiri (Soal UTS No. 8)

Tabel berikut menyajikan hasil komparasi kuantitatif antara model MLP pada jurnal acuan dan model MLP yang diimplementasikan mandiri pada proyek ini:

| Metrik Evaluasi | Hasil Jurnal Acuan (MLP) | Hasil Proyek Mandiri (MLP) | Selisih (Delta) | Keterangan / Analisis |
| :--- | :---: | :---: | :---: | :--- |
| **Akurasi (Accuracy)** | 86.00% | **85.25%** | **-0.75%** | Sangat mendekati dan konsisten dengan hasil jurnal acuan |
| **Presisi (Precision)** | 87.00% | **77.14%** | -9.86% | Menekan false negative pada kasus kardiovaskular |
| **Recall (Sensitivitas)** | 87.00% | **96.43%** | **+9.43%** | Model proyek unggul lebih tinggi dalam mendeteksi pasien berisiko |
| **F1-Score** | 87.00% | **85.71%** | -1.29% | Keseimbangan harmonik presisi dan recall tetap stabil |
| **ROC-AUC** | 90.00% | **93.40%** | **+3.40%** | Kemampuan diskriminasi kelas probabilitas lebih optimal |

### Analisis Kritis Perbedaan Hasil (Jawaban Soal No. 8):
1. **Persamaan**: Keduanya menggunakan dataset UCI Cleveland Heart Disease yang sama (303 sampel, 13 fitur), topologi tersembunyi yang seragam (64 dan 32 neuron ReLU), optimizer Adam dengan learning rate 0.001, serta pembagian data 80% latih dan 20% uji dengan `StandardScaler`.
2. **Perbedaan & Keunggulan**: Model mandiri kita mencatatkan nilai **Recall 96.43%** (hanya 1 false negative dari 28 pasien sakit), yang secara klinis jauh lebih aman untuk skrining jantung dibanding jurnal (87.00%).
3. **Penyebab Variasi Numerik**: Perbedaan framework (Jurnal menggunakan TensorFlow/Keras sedangkan proyek menggunakan Scikit-Learn MLPClassifier) serta ukuran data uji sebesar 61 baris di mana selisih 1 pasien benar menggeser akurasi sebesar ~1.64%.

---

## 5. Struktur Direktori Proyek

```text
UTS/
├── Model/                                  # Folder Pelatihan & Eksperimen Model
│   ├── Train_model.py                      # Script pelatihan model MLP (Standar Modul)
│   ├── Test_model.py                       # Script evaluasi CLI interaktif
│   ├── heart.csv                           # Dataset UCI Cleveland Heart Disease
│   ├── model_mlp.pkl                       # Model MLP tersimpan
│   ├── scaler.pkl                          # Objek StandardScaler terlatih
│   ├── feature_names.pkl                   # Metadata nama fitur
│   ├── metrics_comparison.json             # Hasil metrik evaluasi komparasi
│   └── jurnal/
│       ├── Implementation_MLP_Heart_Disease_JAIC.pdf  # PDF Jurnal Acuan Resmi (SINTA)
│       └── ringkasan_jurnal.md             # Catatan analisis metodologi jurnal
│
├── Web/                                    # Folder Aplikasi Web FastAPI (Deploy-Ready)
│   ├── main.py                             # Backend FastAPI server & API routing
│   ├── app.py                              # Entrypoint runner Uvicorn
│   ├── Procfile                            # Konfigurasi deployment hosting cloud
│   ├── requirements.txt                    # Dependensi pustaka Python
│   ├── run.sh                              # Script bash runner lokal
│   ├── heart.csv                           # Salinan dataset
│   ├── models/                             # Artefak model hasil pelatihan
│   │   ├── model_mlp.pkl
│   │   ├── scaler.pkl
│   │   ├── feature_names.pkl
│   │   └── metrics_comparison.json
│   ├── static/                             # Aset frontend (CSS & JS)
│   └── templates/                          # Template antarmuka Jinja2
│       ├── base.html                       # Layout dasar berstandar akademik Polinema
│       ├── index.html                      # Halaman diagnosis mandiri & shortcut preset
│       ├── komparasi.html                  # Halaman komparasi jurnal vs mandiri (No. 8)
│       └── arsitektur.html                 # Halaman detail topologi MLP & kamus data
└── README.md                               # Dokumentasi komprehensif proyek
```

---

## 6. Panduan Menjalankan Proyek Secara Lokal

### Prasyarat
- Python 3.10 atau versi lebih baru
- Virtual environment disarankan

### Langkah 1: Pelatihan Model (Folder `Model/`)
Buka terminal dan masuk ke folder `UTS/Model`:
```bash
cd "UTS/Model"
python3 Train_model.py
```
*Output akan menampilkan proses pelatihan, evaluasi metrik, dan menyimpan file `model_mlp.pkl`.*

Untuk menguji performa model via CLI:
```bash
python3 Test_model.py
```

### Langkah 2: Menjalankan Aplikasi Web FastAPI (Folder `Web/`)
Pindah ke folder `UTS/Web`:
```bash
cd "../Web"
pip install -r requirements.txt
./run.sh
# Atau langsung jalankan:
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```
Aplikasi akan aktif di:
- Antarmuka Web : **`http://localhost:5000`**
- Dokumentasi API Interaktif (Swagger UI) : **`http://localhost:5000/docs`**
- Dokumentasi API ReDoc : **`http://localhost:5000/redoc`**

---

## 7. Panduan Deployment Online (Sesuai Soal UTS No. 4)

Proyek ini telah dikonfigurasi siap pakai untuk hosting cloud gratis (seperti **Render.com** atau **Railway.app**):

### Deployment ke Render.com
1. Buka [dashboard.render.com](https://dashboard.render.com) dan klik **New Web Service**.
2. Hubungkan repository GitHub: `https://github.com/yogaananda6677/UTS-Kecerdasan-Buatan-MLP`.
3. Tentukan konfigurasi:
   - **Root Directory**: `Web`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Klik **Create Web Service**. URL publik web akan aktif dalam beberapa menit.
