# DESIGN SYSTEM: CARDIOPREDICT & ML ANALYTICS
Dial: ENERGY 1 / RHYTHM 2 / MOTION 1

## 1. Direction & Identity
- **Product**: Portal inferensi klinis kardiologi dan evaluasi model Multilayer Perceptron (MLP) berbasis dataset UCI Cleveland dan studi komparasi JAIC 2025.
- **Audience**: Penguji akademis, mahasiswa informatika medis, dan klinisi evaluasi.
- **Visual Stance**: Academic Journal meets Clinical Pathology Report. Tidak ada dekorasi kosong. Setiap elemen visual merepresentasikan data nyata atau kontrol fungsional.
- **Theme**: Light-first clinical surface. Tidak menggunakan dark mode artifisial tanpa konteks operasional ruang medis.

---

## 2. Token Palet & Kontras (WCAG AA Strict)

### Surfaces & Neutral Grid
- `--bg-page`: `#f8fafc` (Slate 50)
- `--bg-surface`: `#ffffff` (Pure White)
- `--bg-surface-subtle`: `#f1f5f9` (Slate 100)
- `--border-subtle`: `#e2e8f0` (Slate 200, garis pemisah 1px padat)
- `--border-focus`: `#0f172a` (Slate 900)

### Typographic Contrast
- `--text-heading`: `#0f172a` (Slate 900, rasio kontras 19.5:1 terhadap putih)
- `--text-body`: `#334155` (Slate 700, rasio kontras 9.6:1 terhadap putih)
- `--text-caption`: `#475569` (Slate 600, rasio kontras 5.9:1 terhadap putih)
- Batas bawah teks keterangan medis: `#64748b` (Slate 500, rasio 4.6:1, minimum absolut). Dilarang menggunakan abu-abu lebih terang dari ini untuk teks aktif.

### Diagnostic Accents
- `--accent-cardio`: `#9f1239` (Rose 800, indikator risiko terdeteksi, rasio 7.1:1)
- `--accent-cardio-bg`: `#fff1f2` (Rose 50)
- `--accent-cardio-border`: `#fecdd3` (Rose 200)
- `--status-normal`: `#166534` (Emerald 800, indikator kondisi sehat, rasio 7.3:1)
- `--status-normal-bg`: `#f0fdf4` (Emerald 50)
- `--status-normal-border`: `#bbf7d0` (Emerald 200)
- `--status-warn`: `#9a3412` (Amber 800, batas fisiologis kritis, rasio 5.5:1)
- Dilarang menambahkan warna ungu, cyan neon, gradasi pelangi, atau glow orbs.

---

## 3. Tipografi & Skala Teks

### Font Stack
- **Headings & Jurnal**: `'Lora', Georgia, serif` (Wibawa akademis publikasi ilmiah)
- **UI & Label Medis**: `'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif` (Keterbacaan rekam medis)
- **Data Metrik & Tensor**: `'JetBrains Mono', SFMono-Regular, Menlo, monospace` (Tabular numbers, tidak bergeser saat angka berubah)

### Skala Hierarki
| Peran | Font | Ukuran | Weight | Line Height | Tracking |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Page Title (H1)** | Lora | 28px - 32px | 700 | 1.25 | -0.02em |
| **Section Title (H2)** | Lora | 20px - 22px | 700 | 1.3 | -0.01em |
| **Card Header (H3)** | Lora | 16px - 18px | 600 | 1.35 | normal |
| **Form Section Tag** | Inter | 11px | 700 | 1.0 | +0.05em (UPPERCASE) |
| **Input Label** | Inter | 12px | 600 | 1.2 | normal |
| **Body / Description** | Inter | 13px - 14px | 400 | 1.5 | normal |
| **Clinical Metric Unit** | Inter | 10px - 11px | 500 | 1.2 | normal |
| **Metric Values** | Mono | 13px - 24px | 600 | 1.1 | normal |

---

## 4. Geometri & Spasial

### Radius Sudut
- Main Container / Card Besar: `16px` (`rounded-2xl`)
- Fieldset / Kartu Sub-bagian: `12px` (`rounded-xl`)
- Tombol, Input Form, Dropdown: `8px` (`rounded-lg`)
- Tag Status / Dot Indikator: `9999px` (`rounded-full`)
- Dilarang membuat tombol dan kartu berbentuk kapsul lonjong (pill button/card).

### Elevasi & Bayangan
- Permukaan datar bertingkat: Garis batas fisik (`1px solid #e2e8f0`) menggantikan bayangan tebal.
- Shadow hanya digunakan pada tingkat terendah: `box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.05)`.
- Dilarang menggunakan backdrop blur bertumpuk di lebih dari satu elemen utama (hanya diizinkan pada sticky header nav bar).

---

## 5. Pola Komponen Klinis

### 1. Form 13 Parameter Klinis
- Dikelompokkan ke dalam 4 fieldset logis rekam medis:
  1. Identitas & Tekanan Darah (`age`, `sex`, `trestbps`)
  2. Profil Lipid & Glikemik (`chol`, `fbs`, `restecg`)
  3. Kardiak & Elektrokardiogram Latihan (`thalach`, `exang`, `oldpeak`, `slope`)
  4. Patologi Koroner & Perfusi (`ca`, `thal`)
- Setiap input wajib mencantumkan satuan klinis (tahun, mmHg, mg/dL, bpm, skala ST).
- Focus state: border Slate 900 dengan ring tipis 1px (`ring-1 ring-slate-900`), tidak menggunakan ring biru neon.

### 2. Panel Output Inferensi (Focal Point)
- Menampilkan dua level informasi:
  1. **Keputusan Kategori**: Kotak penegasan status diagnosis ("Terindikasi Risiko Penyakit Jantung" vs "Kondisi Bebas Risiko").
  2. **Tingkat Keyakinan Model**: Persentase probabilitas sigmoid matematis dari output layer neural network.
- Rekomendasi klinis yang jelas, ringkas, dan dapat ditindaklanjuti.

### 3. Tabel Evaluasi Komparatif
- Format kolom tabel perbandingan:
  - Metrik (Accuracy, Precision, Recall, F1, ROC-AUC)
  - Baseline Jurnal (JAIC 2025)
  - Model Proyek Mandiri
  - Selisih Delta
- Sel yang memiliki keunggulan klinis signifikan (Recall +9.43%) diberi aksen hijau emerald subtle untuk menandai keberhasilan penekanan False Negative pada skrining jantung.

---

## 6. Standar Copywriting & Bahasa

- Menggunakan Bahasa Indonesia formal akademis dan istilah kardiologi baku.
- Dilarang menggunakan tanda baca em dash di seluruh antarmuka. Gunakan tanda koma, titik dua, atau kurung.
- Dilarang menggunakan frasa pemasaran generik AI ("revolusioner", "ajaib", "otomatis mulus", "kecerdasan tak tertandingi").
- Label tombol harus bersifat prediktif dan instruksional:
  - Benar: "Analisis Risiko Pasien", "Muat Sampel Kasus Sehat", "Lihat Naskah Jurnal"
  - Salah: "Mulai", "Coba Sekarang", "Jelajahi", "Submit"
