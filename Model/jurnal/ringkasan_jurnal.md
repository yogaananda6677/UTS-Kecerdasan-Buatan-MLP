# Ringkasan Artikel Jurnal Referensi UTS

## 1. Identitas Jurnal
- **Judul Artikel**: *Implementation of Deep Learning with Multilayer Perceptron (MLP) for Heart Disease Prediction Using the SMOTE-ENN Technique*
- **Penulis**: Erik Saputra & Erliyan Redy Susanto
- **Afiliasi**: Program Studi Sistem Informasi & Magister Ilmu Komputer, Fakultas Teknik dan Ilmu Komputer, Universitas Teknokrat Indonesia
- **Nama Jurnal**: *Journal of Applied Informatics and Computing (JAIC)*
- **Penerbit**: Jurusan Teknik Informatika, Politeknik Negeri Batam (Polibatam)
- **Terindeks**: SINTA (Jurnal Nasional Terakreditasi)
- **Edisi & Tahun**: Vol. 9, No. 3, June 2025, Halaman 1034–1041
- **e-ISSN**: 2548-6861
- **DOI**: [10.30871/jaic.v9i3.9337](https://doi.org/10.30871/jaic.v9i3.9337)
- **URL Publikasi**: [http://jurnal.polibatam.ac.id/index.php/JAIC/article/view/9337](http://jurnal.polibatam.ac.id/index.php/JAIC/article/view/9337)

---

## 2. Sumber & Karakteristik Dataset
- **Sumber Dataset**: *UCI Machine Learning Repository* (Heart Disease - Cleveland Dataset).
- **Jumlah Sampel**: 303 data rekam medis pasien.
- **Jumlah Fitur**: 13 variabel independen prediktor ($X$) dan 1 variabel dependen ($y$).
- **Distribusi Target**: 
  - Kelas 1 (Positif Penyakit Jantung): 165 pasien
  - Kelas 0 (Negatif / Sehat): 138 pasien
- **Daftar Fitur (Sesuai Tabel 1 Jurnal)**:
  1. `age`: Usia pasien (tahun)
  2. `sex`: Jenis kelamin (1 = Laki-laki, 0 = Perempuan)
  3. `cp`: Tipe nyeri dada (0–3)
  4. `trestbps`: Tekanan darah saat istirahat (mmHg)
  5. `chol`: Kadar kolesterol serum (mg/dL)
  6. `fbs`: Gula darah setelah puasa (>120 mg/dL: 1, lainnya: 0)
  7. `restecg`: Hasil elektrokardiografi saat istirahat (0–2)
  8. `thalach`: Detak jantung maksimum yang dicapai (bpm)
  9. `exang`: Angina akibat olahraga (1 = Ya, 0 = Tidak)
  10. `oldpeak`: Depresi segmen ST yang diinduksi latihan relatif terhadap istirahat
  11. `slope`: Kemiringan segmen ST puncak latihan (0–2)
  12. `ca`: Jumlah pembuluh darah utama terdeteksi fluoroskopi (0–3)
  13. `thal`: Status thalassemia (1 = Normal, 2 = Fixed defect, 3 = Reversible defect)

---

## 3. Arsitektur Jaringan Saraf Tiruan (MLP)
- **Lapisan Masukan (Input Layer)**: 13 neuron (sesuai jumlah fitur prediktor).
- **Lapisan Tersembunyi 1 (Hidden Layer 1)**: 64 neuron dengan fungsi aktivasi **ReLU** ($f(x) = \max(0, x)$).
- **Lapisan Tersembunyi 2 (Hidden Layer 2)**: 32 neuron dengan fungsi aktivasi **ReLU**.
- **Lapisan Keluaran (Output Layer)**: 1 neuron dengan fungsi aktivasi **Sigmoid** ($\sigma(z) = \frac{1}{1 + e^{-z}}$) untuk klasifikasi biner.
- **Optimizer**: Adam dengan learning rate $\alpha = 0.001$.
- **Fungsi Kerugian (Loss Function)**: Binary Cross-Entropy / Log Loss.
- **Normalisasi / Standardisasi**: `StandardScaler` ($Z = \frac{X - \mu}{\sigma}$).
- **Resampling**: Kombinasi over-sampling SMOTE dan under-sampling Edited Nearest Neighbors (SMOTE-ENN).

---

## 4. Hasil Kinerja di Jurnal vs Proyek UTS

| Metrik Evaluasi | Jurnal (Baseline MLP) | Proyek UTS (Baseline MLP) | Jurnal (Proposed SMOTE-ENN) | Proyek UTS (Proposed SMOTE-ENN) |
| :--- | :---: | :---: | :---: | :---: |
| **Akurasi** | 86.00% | **78.69%** | 89.47% | **80.33%** |
| **Presisi** | 87.00% | **79.41%** | 77.78% | **78.38%** |
| **Recall (Sensitivitas)** | 87.00% | **81.82%** | 100.00% | **87.88%** |
| **F1-Score** | 87.00% | **80.60%** | 87.50% | **82.86%** |
| **ROC-AUC** | 90.00% | **84.63%** | 97.00% | **88.85%** |

---

## 5. Analisis Persamaan & Perbedaan (Soal UTS No. 8)

### Persamaan:
1. Kedua penelitian menggunakan dataset yang sama persis yaitu **UCI Cleveland Heart Disease** dengan 13 fitur klinis.
2. Kedua implementasi mengadopsi struktur arsitektur MLP yang identik: **Input: 13 -> Hidden 1: 64 -> Hidden 2: 32 -> Output: 1** dengan aktivasi ReLU dan optimizer Adam ($\alpha = 0.001$).
3. Kedua eksperimen membuktikan secara konsisten bahwa penerapan **SMOTE-ENN** berhasil mendongkrak nilai **Recall** (sensitivitas klinis) secara signifikan sehingga model tidak melewatkan pasien yang sakit jantung (meminimalkan *False Negatives*).

### Perbedaan & Penyebabnya:
1. **Framework & Bobot Awal**: Jurnal menggunakan framework berbasis Keras/TensorFlow dengan regularisasi Dropout dan penentuan random seed internal, sedangkan proyek UTS menggunakan pustaka `scikit-learn` (`MLPClassifier`).
2. **Variasi Sampling Data Uji**: Pada dataset berukuran 303 baris, pembagian data 80:20 menghasilkan 61 data uji. Perbedaan 2 hingga 3 data yang terprediksi benar pada data uji akan menggeser persentase akurasi sekitar 3–5%.
3. **Efek Trade-off Presisi vs Recall**: Pada kedua penelitian terlihat adanya *trade-off* alami saat oversampling SMOTE-ENN diterapkan: Recall melonjak tinggi untuk menangkap seluruh potensi pasien sakit jantung, dengan sedikit kompromi pada presisi akibat potensi kenaikan *False Positives*.
