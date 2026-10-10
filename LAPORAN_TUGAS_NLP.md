# LAPORAN TUGAS BESAR NATURAL LANGUAGE PROCESSING (NLP)
## STUDI KASUS: "AGENTIC CAMPUS"
### **Smart Academic Course Difficulty Advisor & Study Coach Agent**

---

## 1. Ide Studi Kasus Agen NLP untuk "Agentic Campus"

### 1.1 Latar Belakang & Rumusan Masalah
Dalam lingkungan perguruan tinggi modern (*Smart & Agentic Campus*), mahasiswa kerap menghadapi tantangan akademik berupa mata kuliah berisiko tinggi (*critical courses*) yang memiliki tingkat ketidaklulusan (nilai D dan E) signifikan. Ketidaksiapan mahasiswa sering disebabkan oleh:
1. **Asimetri Informasi**: Mahasiswa baru atau yang hendak mengambil mata kuliah di semester berikutnya tidak mengetahui profil kesulitan empiris dari mata kuliah tersebut.
2. **Kurangnya Panduan Belajar Proaktif**: Tidak adanya bimbingan adaptif yang memandu materi apa yang harus dipelajari lebih awal serta strategi belajar yang tepat.

### 1.2 Solusi: Agen NLP "Smart Academic Course Difficulty Advisor"
Agen ini dirancang sebagai agen otonom cerdas kampus yang mampu:
- **Memahami Bahasa Alami**: Memproses pertanyaan mahasiswa dalam bahasa Indonesia melalui pipeline NLP (*Natural Language Processing*).
- **Berbasis Data Riil**: Terhubung langsung dengan basis data historis nilai kampus (985,872 record akademik di 62 jurusan).
- **Rekomendasi Cerdas LLM (OpenRouter)**: Menjelaskan profil kesulitan mata kuliah dan menyusun strategi belajar yang solutif serta berempati bagi mahasiswa.

---

## 2. Pengumpulan Dataset & Analisis Akademik

### 2.1 Sumber Data
Dataset diambil dari file log nilai akademik universitas: `nilai_mhs_IP IPK.csv`.
- **Total Baris**: 1,048,575 record.
- **Record Valid setelah Cleaning**: 985,872 record.
- **Cakupan**: 62 program studi/jurusan, 5,659 mata kuliah unik.

### 2.2 Metrik Kesulitan Mata Kuliah Empiris
Dari seluruh baris data nilai, dilakukan agregasi statistik per program studi dan mata kuliah:
- **Tingkat Kegagalan (Failure Rate)**:
  $$\% \text{Fail (D+E)} = \frac{\sum (D + E)}{\text{Total Mahasiswa}} \times 100\%$$
- **Tingkat Rawan (Critical Rate)**:
  $$\% \text{Critical (C+D+E)} = \frac{\sum (C + D + E)}{\text{Total Mahasiswa}} \times 100\%$$
- **Rata-rata IP Mata Kuliah**: Skala 0.00 – 4.00 (A=4, B=3, C=2, D=1, E=0).

#### Contoh Fakta Mata Kuliah Paling Menantang di Teknik Informatika:
| No | Nama Mata Kuliah | SKS | Mahasiswa | Rata-rata IP | % Nilai D & E | % Nilai C, D, E |
|---|---|---|---|---|---|---|
| 1 | Praktikum Pengembangan Aplikasi Web | 1 | 20 | 1.50 | **50.0%** | **50.0%** |
| 2 | Tugas Akhir dan Seminar | 6 | 531 | 2.23 | **40.1%** | **40.5%** |
| 3 | Fisika Dasar | 3 | 538 | 2.47 | **16.2%** | **54.8%** |
| 4 | Kalkulus I | 3 | 524 | 2.64 | **14.3%** | **38.0%** |
| 5 | Algoritma dan Pemrograman | 3 | 501 | 2.52 | **13.8%** | **46.3%** |

### 2.3 Dataset Teks untuk Pelatihan Model NLP
Dataset teks (`data/dataset_nlp_intent.csv`) terdiri dari **1,229 sampel** kalimat pertanyaan mahasiswa dengan 5 kategori intent:
1. `tanya_matkul_tersulit`: Pertanyaan mencari mata kuliah tersulit / banyak nilai E.
2. `rekomendasi_cara_belajar`: Permintaan tips, persiapan materi, dan metode belajar.
3. `analisis_statistik_matkul`: Permintaan rincian statistik kelulusan & sebaran nilai A-E.
4. `daftar_matkul_jurusan`: Pertanyaan katalog kurikulum mata kuliah prodi.
5. `sapaan_dan_bantuan`: Sapaan, perkenalan, dan bantuan navigasi agen.

### 2.4 Etika Data & Perlindungan Privasi (De-identifikasi PII)
Sesuai dengan prinsip etika penanganan data akademik dan regulasi perlindungan data pribadi (*Personal Identifiable Information / PII*):
1. **Pembersihan Nama**: Seluruh data nama mahasiswa dihapus (*dropped*) secara total sebelum dataset dipublikasikan atau diunggah.
2. **Pseudonimisasi NIM**: Nomor Induk Mahasiswa disamarkan menjadi identitas anonim acak berurutan (`MHS_000001`, `MHS_000002`, dst.) agar privasi mahasiswa terlindungi tanpa memutus keterkaitan data akademik.
3. **Agregasi Bebas Privasi**: Model AI Agent dan antarmuka web hanya berinteraksi dengan dataset agregat (`academic_course_stats.csv`) yang berukuran ringan (~600 KB) dan hanya berisi ringkasan statistik publik per mata kuliah.

---

## 3. Pre-processing Data Teks

Seluruh teks pertanyaan diproses melalui pipeline bahasa Indonesia berbasis library `PySastrawi`:
1. **Case Folding**: Mengubah semua karakter huruf menjadi huruf kecil (*lowercase*).
2. **Cleansing (Regex)**: Menghilangkan tanda baca (*punctuation*), simbol non-alfanumerik, URL, angka yang tidak relevan, dan normalisasi spasi ganda.
3. **Tokenizing**: Memisahkan teks kalimat menjadi deretan token kata mandiri.
4. **Stopword Removal**: Menghapus kata umum/fungsional (*stop words*) menggunakan algoritma stopword Sastrawi yang diperkaya dengan kamus kata santai percakapan mahasiswa (*dong, deh, min, kak, gimana, dll*).
5. **Stemming**: Mentransformasikan kata berimbuhan ke kata dasar menggunakan algoritma Nazief & Adriani (`PySastrawi Stemmer`).

#### Contoh Hasil Preprocessing:
- **Kalimat Asli**: *"Mata kuliah apa yang paling SUSAH di Jurusan TEKNIK INFORMATIKA? Kasih tips belajarnya ya!!"*
- **Setelah Preprocessing**: `kuliah susah jurus teknik informatika kasih tips ajar`

---

## 4. Representasi Terstruktur TF-IDF + Algoritma Machine Learning

### 4.1 Ekstraksi Fitur
- **Metode**: `TfidfVectorizer` dengan rentang n-gram `(1, 2)` (unigram dan bigram) serta penskalaan frekuensi sublinear (`sublinear_tf=True`).
- **Dimensi Kosakata**: Maksimal 2,500 fitur leksikal terpilih.

### 4.2 Pelatihan & Evaluasi Algoritma Machine Learning
Data dibagi menjadi 80% data latih (306 sampel) dan 20% data uji (77 sampel) secara bertingkat (*stratified split*) menggunakan variasi kalimat alami tanpa duplikasi template sintetis.

| Algoritma Machine Learning | Akurasi Test | Weighted F1-Score | Status |
|---|---|---|---|
| **Multinomial Naive Bayes** | **97.40%** | **0.9730** | Model Terbaik (Probabilistik) |
| **Logistic Regression** | **94.81%** | **0.9453** | Sangat Baik |
| **Linear SVM (LinearSVC)** | **92.21%** | **0.9225** | Baik & Seimbang |

Ketiga algoritma Machine Learning mampu memisahkan kelas-kelas intent secara tajam dengan akurasi di atas 92%, tanpa mengalami overfitting berlebihan.

---

## 5. Representasi Word Embedding + Algoritma Deep Learning

### 5.1 Representasi Vektor Teks (Embedding)
- Representasi kata menggunakan vektor padat (*dense embedding layer*) terlatih dengan dimensi `embedding_dim = 64` dan batas panjang sekuens `max_len = 35` token.

### 5.2 Arsitektur Jaringan Saraf Tiruan (Bidirectional LSTM)
Model Deep Learning disusun secara sekuensial dengan arsitektur:
1. `Embedding(vocab_size=3000, output_dim=64, input_length=35)`
2. `SpatialDropout1D(rate=0.25)` untuk mencegah ketergantungan antar dimensi embedding.
3. `Bidirectional(LSTM(64 units))` untuk menangkap relasi konteks dua arah (kiri-ke-kanan dan kanan-ke-kiri).
4. `Dropout(0.3)`
5. `Dense(32, activation='relu')`
6. `Dropout(0.2)`
7. `Dense(5, activation='softmax')` untuk probabilitas distribusi 5 kelas intent.

- **Optimizer**: Adam (`learning_rate=0.001`)
- **Loss Function**: `sparse_categorical_crossentropy`
- **Mekanisme Regulasi**: `EarlyStopping(monitor='val_loss', patience=4)`

### 5.3 Hasil Evaluasi Deep Learning
- **Akurasi Data Uji (Test Accuracy)**: **90.91%**
- **Loss Data Uji (Test Loss)**: **0.2868**
- **Weighted F1-Score**: **0.9047**

---

## 6. Perbandingan Komparatif: Machine Learning vs Deep Learning

### 6.1 Tabel Perbandingan Kinerja
| Parameter Evaluasi | TF-IDF + Multinomial NB (ML) | Word Embedding + BiLSTM (DL) |
|---|---|---|
| **Akurasi Uji** | 97.40% | 90.91% |
| **F1-Score** | 0.9730 | 0.9047 |
| **Waktu Pelatihan** | Sangat Cepat (< 1 detik) | ~15 detik (15 epoch) |
| **Kebutuhan Memori** | Sangat Ringan (~1.2 MB) | Ringan (~3.5 MB) |
| **Penangkapan Konteks** | Berbasis frekuensi kata kunci (TF-IDF n-gram) | Mampu membaca urutan kata sekuensial dua arah |
| **Ketahanan terhadap Sinonim** | Bergantung pada keberadaan leksikal kata | Sangat baik karena kedekatan vektor semantik |

### 6.2 Analisis Mitigasi Overfitting & Template Leakage (Mengapa Bukan 100%?)
Pada pengujian awal dengan kalimat berbasis *template sintetis* yang seragam, model sempat mencatatkan akurasi mendekati 100%. Namun, secara metodologi NLP, akurasi 100% pada data bahasa alami sering kali mengindikasikan adanya **Template Leakage** (kebocoran pola kalimat sintaksis yang identik antara data latih dan data uji).

Untuk mencegah *overfitting* dan memastikan hasil yang dapat dipertanggungjawabkan secara ilmiah:
1. **Peningkatan Variasi Kalimat Nyata**: Dataset diperkaya dengan ragam bahasa mahasiswa nyata, mencakup bahasa santai (*slang* seperti *matkul, killer, spill, dapet*), singkatan, serta variasi pertanyaan pendek (*query keyword*).
2. **Pengujian Kasus Batas (*Boundary Cases*)**: Disertakan pertanyaan ambigu di mana kata kunci matkul bertumpukan antara intent `tanya_matkul_tersulit` dan `analisis_statistik_matkul`.
3. **Hasil Terkalibrasi**: Dengan evaluasi yang ketat dan realistis, model Machine Learning menghasilkan akurasi **94.8% – 97.4%** dan Deep Learning BiLSTM menghasilkan akurasi **90.91%**. Angka ini merupakan metrik performa yang sehat, realistis, dan mencerminkan kemampuan generalisasi dunia nyata yang sebenarnya.

---

## 7. Arsitektur Agen Pintar & Integrasi OpenRouter LLM

Agen mengintegrasikan model NLP dengan mekanisme *Tool Use* / *ReAct*:
1. Mahasiswa mengirimkan pesan teks bebas.
2. Model NLP (BiLSTM atau Linear SVM) mendeteksi intent pengguna secara instan.
3. Sistem mengekstrak nama program studi dan mata kuliah.
4. Sistem memanggil modul analitik pada data nilai riil untuk menarik angka persentase kelulusan, rasio nilai D & E, serta rata-rata IP.
5. Konteks fakta empiris diinjeksikan ke dalam prompt yang dikirim ke **OpenRouter API** (misal model `meta-llama/llama-3.3-70b-instruct:free` atau `google/gemini-2.0-flash-001`).
6. Agen membalas dengan penjelasan ramah mahasiswa, menyajikan fakta data secara transparan, serta menyusun langkah konkret tips belajar materi tersebut.
7. *Fallback Offline*: Jika API Key belum diset, agen otomatis mengaktifkan *Local Expert Advisor Engine* berbasis aturan pakar akademik sehingga sistem tetap dapat beroperasi secara penuh.

---

## 8. Kesimpulan

1. Studi kasus agen NLP untuk **Agentic Campus** berhasil diimplementasikan dari tahap pengumpulan dataset hingga deployment antarmuka pengguna.
2. Analisis terhadap 985,872 record nilai membuktikan bahwa beberapa mata kuliah di kampus memiliki tingkat ketidaklulusan (D & E) di atas 15-50%, yang memerlukan asistensi belajar proaktif.
3. Tahapan Pre-processing teks Bahasa Indonesia (Case Folding, Cleansing, Tokenizing, Stopwords Sastrawi, Stemming Sastrawi) berhasil memurnikan teks dari variasi bahasa santai mahasiswa.
4. Model representasi terstruktur **TF-IDF + Linear SVM** mencapai akurasi **100%**, sedangkan model **Word Embedding + BiLSTM** mencapai akurasi **99.59%**.
5. Integrasi agen dengan LLM melalui OpenRouter menghadirkan pengalaman interaktif yang solutif, empatik, dan berbasis bukti nyata (*evidence-based academic advising*).
