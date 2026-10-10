# 🎓 PANDUAN PENGUMPULAN TUGAS & DEPLOYMENT STREAMLIT

Dokumen ini menjelaskan tata cara pengumpulan tugas NLP, jaminan keamanan data rahasia (*privacy protection*), serta panduan deploy aplikasi ke Streamlit Cloud.

---

## 🔒 1. Penanganan Data Rahasia & Konfidensial (NIM & Nama)

Karena dataset asli `nilai_mhs_IP IPK.csv` berisi identitas pribadi mahasiswa (Nama Lengkap dan NIM) serta berukuran sangat besar (**127 MB**), langkah-langkah sanitasi data berikut telah diterapkan:

1. **Penghapusan Data Identitas Pribadi (PII)**:
   - Kolom `nama` telah **dihapus 100%**.
   - Kolom `nim` telah **disamarkan secara pseudonim** menjadi ID anonim berurutan (contoh: `MHS_000001`, `MHS_000002`, dst.) sehingga relasi data tetap valid secara metodologis tanpa mengungkap siapa mahasiswanya.
2. **Pencegahan Kebocoran ke GitHub & Publik**:
   - File `.gitignore` telah dikonfigurasi untuk **mengabaikan file `nilai_mhs_IP IPK.csv`**. File asli ini **TIDAK AKAN PERNAH** terunggah ke repositori Git/GitHub.
   - GitHub memiliki batasan ukuran file maksimal **100 MB**. Dengan mengabaikan file 127 MB ini, proses `git push` akan berjalan lancar tanpa error batas kuota.
3. **Dataset Bersih yang Dilampirkan**:
   - `data/academic_course_stats.csv`: Dataset agregasi statistik matkul (hanya memuat nama matkul, SKS, rata-rata IP, dan persentase nilai D & E). **Ukuran hanya ~600 KB, 100% aman dan publik**.
   - `data/nilai_mhs_anonim_sample.csv`: Sampel 10.000 baris data nilai yang telah disanitasi tanpa nama dan tanpa NIM asli.
   - `data/dataset_nlp_intent.csv`: Dataset teks 1.229 sampel pertanyaan intent mahasiswa untuk melatih model NLP.

---

## 📁 2. Struktur Folder Proyek yang Rapi

Seluruh berkas telah dipisahkan ke dalam folder khusus sesuai fungsinya:

```
NLP/
├── app.py                         # File Utama Web Streamlit (wajib di root untuk deploy)
├── requirements.txt               # Daftar pustaka untuk Streamlit Cloud
├── README.md                      # Dokumentasi ringkas proyek
├── .gitignore                     # Filter file rahasia & file besar 127MB
├── .env.example                   # Contoh konfigurasi OpenRouter API Key
│
├── 01_laporan_dan_tugas/          # 🎓 BERKAS UNTUK DIKUMPULKAN KE DOSEN
│   ├── Agentic_Campus_NLP.ipynb   # Jupyter Notebook lengkap (kode, output run, & visualisasi)
│   ├── LAPORAN_TUGAS_NLP.md       # Makalah / Laporan lengkap siap cetak/PDF
│   └── PANDUAN_PENGUMPULAN_DAN_DEPLOY.md # Petunjuk ini
│
├── 02_pipeline_nlp/               # ⚙️ MODUL PIPELINE NLP
│   ├── nlp_preprocessing.py       # Preprocessing Bahasa Indonesia (Sastrawi)
│   ├── create_nlp_dataset.py      # Generator dataset pertanyaan NLP
│   ├── train_nlp_models.py        # Pelatihan TF-IDF + ML & Embedding + BiLSTM
│   └── anonymize_dataset.py       # Script sanitasi NIM & Nama Mahasiswa
│
├── 03_agent_engine/               # 🤖 ENGINE AGENT & ANALYTICS
│   ├── agentic_campus_bot.py      # Otak agen cerdas & OpenRouter LLM
│   └── build_academic_stats.py    # Agregator nilai matkul dari CSV
│
├── data/                          # 📊 DATASET BERSIH & AMAN
│   ├── academic_course_stats.csv  # Data agregat nilai kampus (dipakai di Streamlit)
│   ├── dataset_nlp_intent.csv     # Dataset 1,229 pertanyaan NLP
│   ├── daftar_jurusan.json        # Daftar 62 jurusan unik
│   └── nilai_mhs_anonim_sample.csv # Sampel nilai bebas nama/NIM asli
│
├── models/                        # 🧠 MODEL TERLATIH (Bobot Keras & Scikit-Learn)
│   ├── dl_bilstm.keras            # Model Deep Learning BiLSTM
│   ├── dl_tokenizer.joblib        # Tokenizer Deep Learning
│   ├── ml_classifier.joblib       # Model Machine Learning (Linear SVM)
│   ├── tfidf_vectorizer.joblib    # Vektorisasi TF-IDF
│   ├── label_encoder.joblib       # Label encoder kelas intent
│   └── evaluation_results.json    # Ringkasan metrik akurasi & F1-score
│
└── reports/                       # 📈 GRAFIK EVALUASI MODEL
    ├── confusion_matrix_ml.png    # Matriks konfusi Machine Learning
    ├── confusion_matrix_dl.png    # Matriks konfusi Deep Learning
    └── training_history_dl.png    # Kurva akurasi & loss pelatihan BiLSTM
```

---

## 📦 3. Berkas yang Dikumpulkan ke Dosen / E-Learning

Jika dosen meminta pengumpulan tugas dalam bentuk **ZIP** atau file mandiri:

1. **Opsi A: Mengumpulkan Folder Tugas Lengkap (.ZIP)**
   - Buat file `.zip` dari folder `NLP/`, **KECUALI** file mentah `nilai_mhs_IP IPK.csv` (jangan disertakan karena berisi data pribadi dan ukurannya 127 MB).
   - Ukuran ZIP akhir hanya sekitar **15 - 20 MB** (sangat ringan dan mudah diunggah ke Google Classroom / LMS Kampus).

2. **Opsi B: Mengumpulkan Berkas Utama Saja**
   - File Notebook: `01_laporan_dan_tugas/Agentic_Campus_NLP.ipynb`
   - File Laporan: `01_laporan_dan_tugas/LAPORAN_TUGAS_NLP.md` (dapat di-convert ke PDF menggunakan VS Code atau browser).
   - Link Deploy Streamlit (jika sudah dideploy online).

---

## 🚀 4. Panduan Deploy ke Streamlit Community Cloud (Gratis)

Streamlit Community Cloud memungkinkan dosen dan teman kampus membuka aplikasi web agen ini langsung dari browser tanpa perlu menginstall Python di laptop mereka.

### Langkah 1: Buat Repositori GitHub
1. Buat repository baru di [GitHub](https://github.com/new), beri nama misalnya: `agentic-campus-nlp`.
2. Buka terminal di folder `d:\tempat nugas\NLP`, lalu jalankan:
   ```bash
   git init
   git add .
   git commit -m "feat: Agentic Campus NLP Course Project"
   git branch -M main
   git remote add origin https://github.com/USERNAME-KAMU/agentic-campus-nlp.git
   git push -u origin main
   ```
   *(Catatan: Berkat `.gitignore`, file 127 MB tidak akan ikut ter-push, sehingga proses push selesai dalam hitungan detik).*

### Langkah 2: Deploy di Streamlit Cloud
1. Kunjungi [share.streamlit.io](https://share.streamlit.io/) dan login menggunakan akun GitHub kamu.
2. Klik tombol **"New app"**.
3. Isi kolom pengaturan:
   - **Repository**: `USERNAME-KAMU/agentic-campus-nlp`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. *(Opsional)* Jika ingin OpenRouter langsung aktif untuk semua pengunjung tanpa perlu mengetik API key:
   - Klik **"Advanced settings..."**
   - Pada kolom **Secrets**, masukkan:
     ```toml
     OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxx"
     ```
   - Klik **Save**.
5. Klik **"Deploy!"**.
6. Tunggu 1–2 menit hingga proses build selesai. Aplikasi kamu akan langsung online dengan URL publik (contoh: `https://agentic-campus-nlp.streamlit.app`).
