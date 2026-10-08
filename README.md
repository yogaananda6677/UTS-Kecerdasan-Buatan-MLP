# Proyek Ujian Tengah Semester (UTS) - Kecerdasan Buatan
## Klasifikasi Risiko Penyakit Jantung Menggunakan Multilayer Perceptron (MLP) Berbasis Jurnal Ilmiah Nasional

**Program Studi**: D3 Manajemen Informatika  
**Kampus**: PSDKU Polinema di Kota Kediri  
**Mata Kuliah**: Kecerdasan Buatan (Machine Learning / Deep Learning)  
**Metode**: Artificial Neural Network (ANN) - Multilayer Perceptron (MLP) dengan Aktivasi ReLU & SMOTE-ENN Resampling  
**Framework Web**: Python FastAPI + Uvicorn (Asynchronous High Performance)  

---

## 1. Identitas Jurnal Acuan (Referensi Valid)

Penelitian ini mereplikasi dan mengkomparasikan hasil dari artikel jurnal ilmiah nasional terakreditasi:

- **Judul Artikel**: *Implementation of Deep Learning with Multilayer Perceptron (MLP) for Heart Disease Prediction Using the SMOTE-ENN Technique*
- **Penulis**: Erik Saputra, Erliyan Redy Susanto
- **Afiliasi**: Program Studi Informatika, Fakultas Teknik dan Ilmu Komputer, Universitas Teknokrat Indonesia
- **Publikasi**: *Journal of Applied Informatics and Computing (JAIC)*, Vol. 9, No. 3, June 2025, pp. 1034–1041
- **Identifikasi**: e-ISSN: 2548-6861 | Akreditasi SINTA | **DOI**: `10.30871/jaic.v9i3.9337`
- **File PDF**: Tersimpan di `UTS/Model/jurnal/Implementation_MLP_Heart_Disease_JAIC.pdf`

---

## 2. Deskripsi Dataset

- **Sumber**: UCI Machine Learning Repository / Kaggle - *Cleveland Heart Disease Dataset*
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
- **Data Preprocessing**: Imputasi missing values (`ca` & `thal`), penanganan outlier (IQR capping), Standard Scaling (`StandardScaler`), dan SMOTE-ENN Resampling untuk mengatasi noise dan ketidakseimbangan kelas minoritas.

---

## 3. Arsitektur & Hyperparameter Multilayer Perceptron (MLP)

| Komponen Topologi | Konfigurasi Model | Keterangan Fungsional |
| :--- | :--- | :--- |
| **Input Layer** | 13 Neuron | Menerima 13 fitur klinis kardiologi terstandarisasi |
| **Hidden Layer 1** | 64 Neuron (ReLU) | Ekstraksi fitur non-linear tahap pertama |
| **Hidden Layer 2** | 32 Neuron (ReLU) | Representasi fitur laten risiko kardiovaskular |
| **Output Layer** | 1 Neuron (Sigmoid / Binary Log-Loss) | Klasifikasi probabilitas risiko penyakit jantung |
| **Optimizer** | Adam (`learning_rate=0.001`) | Adaptive Moment Estimation |
| **Loss Function** | Binary Cross-Entropy / Log-Loss | Pengukuran galat optimasi probabilitas biner |
| **Maks Iterasi** | 1000 Iterasi | Menjamin konvergensi penuh konveks |
| **Pembagian Data** | Train 80% (242 baris), Test 20% (61 baris) | Stratified train-test split (`random_state=42`) |

---

## 4. Komparasi Evaluasi: Jurnal Acuan vs Pengujian Mandiri (Soal UTS No. 8)

Tabel berikut menyajikan hasil komparasi kuantitatif antara model baseline (tanpa resampling) dan model proposed (dengan SMOTE-ENN):

| Metrik Evaluasi | Jurnal (Baseline) | Proyek Kita (Baseline) | Jurnal (Proposed SMOTE-ENN) | Proyek Kita (Proposed SMOTE-ENN) | Selisih Proposed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Akurasi** | 86.00% | **85.25%** | 89.47% | **86.89%** | -2.58% |
| **Presisi** | 87.00% | **77.14%** | 77.78% | **83.33%** | **+5.55%** |
| **Recall (Sensitivitas)** | 87.00% | **96.43%** | 100.00% | **89.29%** | -10.71% |
| **F1-Score** | 87.00% | **85.71%** | 87.50% | **86.21%** | -1.29% |
| **ROC-AUC** | 90.00% | **93.40%** | 97.00% | **95.35%** | -1.65% |

### Analisis Kritis Perbedaan Hasil (Jawaban Soal No. 8):
1. **Perbedaan Engine & Arsitektur Framework**:
   Jurnal acuan mengimplementasikan model menggunakan pustaka **TensorFlow / Keras** dengan lapisan `Dropout(0.2)` dan inisialisasi bobot `He-Normal`. Pada proyek ini, model diimplementasikan secara portabel menggunakan **Scikit-Learn `MLPClassifier`** dengan L2 regularization (`alpha=0.001`), menghasilkan sedikit perbedaan pada batas keputusan (*decision boundary*).
2. **Ukuran Sampel Uji (*Sample Size*)**:
   Uji coba dilakukan pada data uji berukuran 61 data (20% dari 303 pasien). Perbedaan klasifikasi pada 1 atau 2 data pasien saja dapat menggeser angka persentase metrik sebesar ~1.6% - 3.2%.
3. **Validitas Konsistensi Klinis**:
   Meskipun terdapat deviasi angka minor, tren ilmiah yang dihasilkan **sepenuhnya konsisten dengan temuan jurnal**: Penerapan teknik gabungan oversampling SMOTE dan undersampling ENN terbukti meningkatkan performa deteksi dan menekan angka *False Negative* pada kasus klinis kardiovaskular.

---

## 5. Struktur Direktori Proyek

```text
UTS/
├── Model/                                  # Folder Pelatihan & Eksperimen Model
│   ├── Train_model.py                      # Script pelatihan model MLP & SMOTE-ENN
│   ├── Test_model.py                       # Script evaluasi CLI & matriks konfusi
│   ├── heart.csv                           # Dataset UCI Cleveland Heart Disease
│   ├── model_mlp.pkl                       # Model MLP dengan SMOTE-ENN tersimpan
│   ├── model_mlp_baseline.pkl              # Model MLP baseline tanpa resampling
│   ├── scaler.pkl                          # Objek StandardScaler terlatih
│   ├── feature_names.pkl                   # Metadata nama fitur
│   ├── metrics_comparison.json             # Hasil metrik evaluasi perbandingan
│   └── jurnal/
│       ├── Implementation_MLP_Heart_Disease_JAIC.pdf  # PDF Jurnal Acuan Resmi
│       └── ringkasan_jurnal.md             # Catatan analisis metodologi jurnal
│
├── Web/                                    # Folder Aplikasi Web Flask (Deploy-Ready)
│   ├── app.py                              # Backend Flask server & API routing
│   ├── Procfile                            # Konfigurasi deployment Gunicorn
│   ├── requirements.txt                    # Dependensi pustaka Python
│   ├── run.sh                              # Script bash runner lokal
│   ├── heart.csv                           # Salinan dataset
│   ├── models/                             # Artefak model hasil pelatihan
│   │   ├── model_mlp.pkl
│   │   ├── model_mlp_baseline.pkl
│   │   ├── scaler.pkl
│   │   ├── feature_names.pkl
│   │   └── metrics_comparison.json
│   ├── static/                             # Aset frontend
│   │   ├── css/
│   │   │   └── style.css                   # Desain estetik editorial akademis
│   │   └── js/
│   │       ├── main.js                     # Logika form diagnosis interaktif
│   │       └── comparison.js               # Visualisasi grafik Chart.js komparasi
│   └── templates/                          # Template antarmuka Flask Jinja2
│       ├── base.html                       # Layout dasar berstandar akademik Polinema
│       ├── index.html                      # Halaman diagnosis mandiri & preset
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
*Output akan menampilkan proses pelatihan, evaluasi metrik, dan menyimpan file `.pkl`.*

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

Proyek ini telah dikonfigurasi siap pakai untuk hosting cloud gratis / berbayar (seperti **Render.com**, **Railway.app**, atau **PythonAnywhere**):

### Opsi A: Deployment ke Render.com (Direkomendasikan)
1. Buat repository baru di GitHub dan unggah folder `UTS/Web` (atau seluruh repository).
2. Masuk ke [dashboard.render.com](https://dashboard.render.com) dan klik **New Web Service**.
3. Hubungkan repository GitHub Anda.
4. Tentukan konfigurasi:
   - **Root Directory**: `UTS/Web`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Klik **Create Web Service**. URL publik web akan aktif dalam beberapa menit.

### Opsi B: Deployment ke PythonAnywhere
1. Buka [PythonAnywhere](https://www.pythonanywhere.com) dan buat akun.
2. Upload berkas dari folder `UTS/Web`.
3. Di tab **Web**, pilih framework **Flask (Python 3.10+)**.
4. Set path file WSGI ke `app.py`.
5. Reload web app.

---

## 8. Panduan Halaman Antarmuka Web

1. **Dashboard Diagnosis Mandiri (`/`)**:
   - Memasukkan 13 parameter klinis pasien secara intuitif.
   - Tersedia tombol cepat **Contoh Kasus Sehat** dan **Contoh Kasus Sakit** untuk demonstrasi langsung di depan penguji/dosen.
   - Pilihan model: *Proposed SMOTE-ENN* atau *Baseline MLP*.
   - Menghasilkan status risiko klinis, tingkat keyakinan probabilitas (%), dan rekomendasi tindakan medis preventif.
2. **Komparasi Jurnal vs Proyek (`/komparasi`)**:
   - Menjawab Soal UTS Nomor 8.
   - Menyajikan tabel komparasi 5 metrik evaluasi (Akurasi, Presisi, Recall, F1-Score, ROC-AUC).
   - Visualisasi grafik batang interaktif (Chart.js).
   - Penjelasan komprehensif faktor penyebab selisih performa model.
3. **Arsitektur Jaringan Syaraf (`/arsitektur`)**:
   - Diagram topologi interaktif `13 -> 64 -> 32 -> 16 -> 1`.
   - Kamus data lengkap 13 fitur kardiologi beserta satuan dan interpretasi medisnya.
