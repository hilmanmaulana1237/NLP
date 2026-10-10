# 🎓 Agentic Campus: Smart Academic Course Advisor & Study Coach Agent

Proyek Tugas Mata Kuliah Natural Language Processing (NLP) untuk studi kasus **"Agentic Campus"** berbasis dataset akademik universitas (`nilai_mhs_IP IPK.csv`) dengan agen chatbot cerdas terintegrasi OpenRouter LLM.

---

## 🌟 Fitur Utama
1. **Model NLP Intent Recognition**:
   - **TF-IDF + Linear SVM (Machine Learning)**: Akurasi **100%** | F1-Score: **1.0000**.
   - **Word Embedding + Bidirectional LSTM (Deep Learning)**: Akurasi **99.59%** | F1-Score: **0.9959**.
2. **Preprocessing Bahasa Indonesia Lengkap (PySastrawi)**:
   - Case Folding, Cleansing, Tokenizing, Stopword Removal, dan Stemming Nazief & Adriani.
3. **Analisis Riil 1 Juta Data Nilai Mahasiswa**:
   - Agregasi 5,659 mata kuliah di 62 program studi.
   - Analisis rasio nilai D & E (tingkat tidak lulus), nilai rawan (C, D, E), dan rata-rata IP matkul.
4. **Perlindungan Data Pribadi (100% Anonymized)**:
   - Kolom nama dihapus total, NIM disamarkan menjadi kode pseudonim `MHS_xxxxxx`.
   - File mentah 127 MB diabaikan oleh `.gitignore` sehingga tidak akan bocor ke GitHub dan tidak melanggar limit 100MB GitHub.
5. **AI Agent Bot dengan OpenRouter LLM & Local Fallback**:
   - Menjawab pertanyaan mata kuliah tersulit, memberikan rekomendasi belajar solutif, serta menyajikan visualisasi interaktif.
6. **Deployment Ready untuk Streamlit Cloud**:
   - Dilengkapi `requirements.txt` dan struktur siap deploy 1-klik.

---

## 📁 Struktur Folder Proyek

```
NLP/
├── app.py                         # File Utama Web Streamlit (wajib di root untuk deploy)
├── requirements.txt               # Dependencies untuk Streamlit Cloud
├── README.md                      # Dokumentasi ini
├── .gitignore                     # Mencegah upload data rahasia & file 127MB
├── .env.example                   # Contoh konfigurasi OpenRouter API Key
│
├── 01_laporan_dan_tugas/          # 🎓 KHUSUS PENGUMPULAN TUGAS KE DOSEN
│   ├── Agentic_Campus_NLP.ipynb   # Jupyter Notebook lengkap siap dikumpulkan
│   ├── LAPORAN_TUGAS_NLP.md       # Laporan/Makalah resmi tugas NLP
│   └── PANDUAN_PENGUMPULAN_DAN_DEPLOY.md # Panduan pengumpulan & cara deploy online
│
├── 02_pipeline_nlp/               # ⚙️ MODUL PIPELINE NLP
│   ├── nlp_preprocessing.py       # Preprocessing Bahasa Indonesia (Sastrawi)
│   ├── create_nlp_dataset.py      # Generator dataset pertanyaan NLP
│   ├── train_nlp_models.py        # Pelatihan TF-IDF + ML & Embedding + BiLSTM
│   └── anonymize_dataset.py       # Script pembersih NIM & Nama Mahasiswa
│
├── 03_agent_engine/               # 🤖 ENGINE AGENT & ANALYTICS
│   ├── agentic_campus_bot.py      # Otak agen cerdas & OpenRouter LLM
│   └── build_academic_stats.py    # Agregator nilai matkul dari CSV
│
├── data/                          # 📊 DATASET BERSIH & AMAN
│   ├── academic_course_stats.csv  # Data agregat nilai kampus (dipakai di Streamlit)
│   ├── dataset_nlp_intent.csv     # Dataset 1,229 pertanyaan NLP
│   ├── daftar_jurusan.json        # 62 jurusan unik
│   └── nilai_mhs_anonim_sample.csv # Sampel data anonim bebas nama/NIM asli
│
├── models/                        # 🧠 MODEL TERLATIH
│   ├── dl_bilstm.keras
│   ├── dl_tokenizer.joblib
│   ├── ml_classifier.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── label_encoder.joblib
│   └── evaluation_results.json
│
└── reports/                       # 📈 GRAFIK EVALUASI MODEL
    ├── confusion_matrix_ml.png
    ├── confusion_matrix_dl.png
    └── training_history_dl.png
```

---

## 🚀 Cara Menjalankan

### 1. Menjalankan Web App Streamlit (Lokal)
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di browser Anda di `http://localhost:8501`.

### 2. Membuka Jupyter Notebook
Buka file `01_laporan_dan_tugas/Agentic_Campus_NLP.ipynb` di VS Code atau Jupyter Lab.

### 3. Deploy ke Streamlit Community Cloud (Online & Gratis)
1. Push repo ini ke GitHub kamu (`git init && git add . && git commit -m "init" && git push`).
2. Kunjungi [share.streamlit.io](https://share.streamlit.io/) dan pilih repositori kamu.
3. Set **Main file path**: `app.py`.
4. Masukkan Secret `OPENROUTER_API_KEY` di Advanced Settings (opsional).
5. Klik **Deploy**!
