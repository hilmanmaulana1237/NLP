import os
import sys
import json
import re
import difflib
import joblib
import requests
import pandas as pd
import numpy as np

# Reconfigure stdout for utf-8 on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cari direktori root proyek
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) in ["02_pipeline_nlp", "03_agent_engine"] else os.path.dirname(os.path.abspath(__file__))
for sub in ["02_pipeline_nlp", "03_agent_engine"]:
    sub_path = os.path.join(ROOT_DIR, sub)
    if sub_path not in sys.path:
        sys.path.insert(0, sub_path)

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception as e:
    tf = None
    pad_sequences = None

from nlp_preprocessing import IndonesianNLPPreprocessor

class AgenticCampusBot:
    """
    AI Agent Bot untuk 'Agentic Campus':
    Sistem Pendamping Akademik Berbasis NLP & LLM OpenRouter
    
    Fitur Utama:
    1. Deteksi Intent Query Mahasiswa (TF-IDF + ML / Word Embedding + BiLSTM)
    2. Ekstraksi Entitas (Mata Kuliah & Jurusan) dari teks
    3. Retrieval Analisis Akademik (Data Riil Nilai Kampus)
    4. Sintesis Rekomendasi Cara Belajar Cerdas via OpenRouter API
    """
    def __init__(self, openrouter_api_key=None, model_name="meta-llama/llama-3.3-70b-instruct:free"):
        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.root_dir = ROOT_DIR
        
        # Inisialisasi Preprocessor
        self.preprocessor = IndonesianNLPPreprocessor(use_stemming=True)
        
        # Load Model NLP
        self._load_models()
        
        # Load Knowledge Base Data Nilai Kampus
        self._load_knowledge_base()

    def _resolve_path(self, rel_path: str) -> str:
        p1 = os.path.join(self.root_dir, rel_path)
        if os.path.exists(p1):
            return p1
        if os.path.exists(rel_path):
            return rel_path
        return p1

    def _load_models(self):
        print("[*] Memuat model-model NLP...")
        try:
            self.label_encoder = joblib.load(self._resolve_path("models/label_encoder.joblib"))
            self.classes = self.label_encoder.classes_.tolist()
            
            # Model ML (TF-IDF)
            self.tfidf_vectorizer = joblib.load(self._resolve_path("models/tfidf_vectorizer.joblib"))
            self.ml_model = joblib.load(self._resolve_path("models/ml_classifier.joblib"))
            
            # Model Deep Learning (BiLSTM)
            if tf is not None:
                try:
                    self.dl_tokenizer = joblib.load(self._resolve_path("models/dl_tokenizer.joblib"))
                    self.dl_model = tf.keras.models.load_model(self._resolve_path("models/dl_bilstm.keras"))
                except Exception as e_dl:
                    print(f"[*] Deep Learning model load note: {e_dl}")
                    self.dl_tokenizer = None
                    self.dl_model = None
            else:
                self.dl_tokenizer = None
                self.dl_model = None
                
            print("[OK] Seluruh model NLP berhasil dimuat ke memori.")
        except Exception as e:
            print(f"[!] Warning saat memuat model: {e}")
            self.label_encoder = None
            self.ml_model = None
            self.dl_model = None

    def _load_knowledge_base(self):
        stats_path = self._resolve_path("data/academic_course_stats.csv")
        if os.path.exists(stats_path):
            self.df_stats = pd.read_csv(stats_path)
            self.majors = sorted(self.df_stats["jurusan"].dropna().unique().tolist())
            self.courses = sorted(self.df_stats["nama_matakuliah"].dropna().unique().tolist())
            print(f"[OK] Knowledge base dimuat: {len(self.df_stats):,} matkul, {len(self.majors)} jurusan.")
        else:
            self.df_stats = pd.DataFrame()
            self.majors = []
            self.courses = []

    def set_api_key(self, api_key: str):
        """Memperbarui OpenRouter API Key."""
        self.api_key = api_key

    def set_model_name(self, model_name: str):
        """Mengubah model LLM OpenRouter."""
        self.model_name = model_name

    def predict_intent(self, user_query: str, model_type: str = "bilstm") -> dict:
        """
        Memprediksi intent query mahasiswa menggunakan model NLP:
        - model_type='bilstm': Word Embedding + Deep Learning (BiLSTM)
        - model_type='tfidf_ml': TF-IDF + Machine Learning (SVM/Logistic Regression)
        """
        clean_text = self.preprocessor.preprocess(user_query)
        
        if model_type == "bilstm" and self.dl_model is not None and pad_sequences is not None:
            seq = self.dl_tokenizer.texts_to_sequences([clean_text])
            pad = pad_sequences(seq, maxlen=35, padding="post", truncating="post")
            probs = self.dl_model.predict(pad, verbose=0)[0]
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx])
            intent = self.label_encoder.inverse_transform([pred_idx])[0]
            used_model = "Word Embedding + BiLSTM"
        else:
            # Fallback ke ML TF-IDF
            feat = self.tfidf_vectorizer.transform([clean_text])
            pred_idx = self.ml_model.predict(feat)[0]
            intent = self.label_encoder.inverse_transform([pred_idx])[0]
            if hasattr(self.ml_model, "predict_proba"):
                probs = self.ml_model.predict_proba(feat)[0]
                confidence = float(np.max(probs))
            else:
                confidence = 0.94
            used_model = "TF-IDF + Machine Learning"
            
        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "clean_text": clean_text,
            "model_used": used_model
        }

    def extract_entities(self, text: str) -> dict:
        """
        Mengekstrak entitas Nama Jurusan dan Nama Mata Kuliah dari query mahasiswa.
        Menggunakan pencocokan kata kunci dan fuzzy matching.
        """
        text_lower = text.lower()
        matched_major = None
        matched_course = None
        
        # 1. Cari Jurusan (Cek kesesuaian kata kunci)
        for m in self.majors:
            m_lower = m.lower()
            if m_lower in text_lower:
                matched_major = m
                break
        
        # Alias umum jurusan
        if not matched_major:
            alias_map = {
                "if": "TEKNIK INFORMATIKA",
                "ti": "TEKNIK INFORMATIKA",
                "informatika": "TEKNIK INFORMATIKA",
                "teknik informatika": "TEKNIK INFORMATIKA",
                "elektro": "Teknik Elektro",
                "sipil": "Teknik Sipil",
                "matematika": "Matematika",
                "fisika": "Fisika",
                "kimia": "Kimia",
                "hukum": "Ilmu Hukum",
                "manajemen": "Manajemen",
                "akuntansi": "Akuntansi",
                "ilkom": "Ilmu Komunikasi",
                "psikologi": "Psikologi"
            }
            for alias, canonical in alias_map.items():
                pattern = r"\b" + re.escape(alias) + r"\b"
                if re.search(pattern, text_lower):
                    for real_major in self.majors:
                        if canonical.lower() == real_major.lower():
                            matched_major = real_major
                            break
                    if matched_major:
                        break

        # 2. Cari Mata Kuliah
        # Cari substring match terpanjang yang cocok dengan nama matkul
        best_c = None
        best_len = 0
        for c in self.courses:
            c_lower = c.lower()
            if len(c_lower) > 3 and c_lower in text_lower:
                if len(c_lower) > best_len:
                    best_c = c
                    best_len = len(c_lower)
        matched_course = best_c

        # Jika belum ketemu, coba pencocokan nama mata kuliah populer (Kalkulus, Algoritma, Fisika, dll.)
        if not matched_course:
            keywords = ["kalkulus", "algoritma", "pemrograman", "struktur data", "fisika", 
                        "basis data", "matematika diskrit", "jaringan", "aljabar", "skripsi", "tugas akhir"]
            for kw in keywords:
                if kw in text_lower:
                    matches = [c for c in self.courses if kw in c.lower()]
                    if matches:
                        # Prioritaskan matkul di jurusan yang matched_major jika ada
                        if matched_major:
                            major_matches = self.df_stats[
                                (self.df_stats["jurusan"].str.lower() == matched_major.lower()) &
                                (self.df_stats["nama_matakuliah"].str.lower().str.contains(kw))
                            ]
                            if not major_matches.empty:
                                matched_course = major_matches.iloc[0]["nama_matakuliah"]
                                break
                        matched_course = matches[0]
                        break

        return {
            "major": matched_major,
            "course": matched_course
        }

    def get_most_difficult_courses(self, major: str = None, top_n: int = 5) -> list:
        """
        Mengambil daftar mata kuliah paling sulit berdasarkan persentase nilai D & E
        serta nilai C (nilai rawan) dari data historis kampus.
        """
        if self.df_stats.empty:
            return []
            
        df = self.df_stats.copy()
        if major:
            df = df[df["jurusan"].str.lower() == major.lower()]
            
        # Filter matkul dengan minimal 20 mahasiswa agar representatif
        df = df[df["total_mhs"] >= 20]
        
        if df.empty:
            # Jika sedikit mahasiswa, turunkan threshold
            df = self.df_stats[self.df_stats["jurusan"].str.lower() == major.lower()]
            
        # Urutkan berdasarkan persentase kegagalan nilai D & E tertinggi
        top_df = df.sort_values(by=["pct_fail_DE", "pct_difficult_CDE"], ascending=[False, False]).head(top_n)
        return top_df.to_dict(orient="records")

    def get_course_detail(self, course_name: str, major: str = None) -> dict:
        """Mengambil detail statistik spesifik satu mata kuliah."""
        if self.df_stats.empty:
            return None
            
        df = self.df_stats[self.df_stats["nama_matakuliah"].str.lower() == course_name.lower()]
        if major:
            df_major = df[df["jurusan"].str.lower() == major.lower()]
            if not df_major.empty:
                df = df_major
                
        if df.empty:
            # Coba partial match
            df = self.df_stats[self.df_stats["nama_matakuliah"].str.lower().str.contains(course_name.lower())]
            
        if not df.empty:
            # Ambil yang jumlah mahasiswa terbanyak
            return df.sort_values(by="total_mhs", ascending=False).iloc[0].to_dict()
        return None

    def query_openrouter_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Mengirim permintaan ke OpenRouter API."""
        if not self.api_key:
            return None
            
        if not system_prompt:
            system_prompt = (
                "Anda adalah 'Agentic Campus Academic Advisor & Study Coach', AI bot cerdas pendamping mahasiswa. "
                "Anda ramah, solutif, berbasis data statistik nyata kampus, dan ahli memberikan strategi belajar materi kuliah. "
                "Berikan jawaban dalam bahasa Indonesia yang terstruktur rapi, menarik, dan mudah dipahami."
            )
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://campus-agentic.local",
            "X-Title": "Agentic Campus Academic Advisor"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1200
        }
        
        try:
            response = requests.post(self.openrouter_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                print(f"[!] OpenRouter API Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"[!] OpenRouter Connection Error: {e}")
            return None

    def generate_expert_fallback_response(self, user_query: str, intent: str, entities: dict, stats_data: any) -> str:
        """
        Menghasilkan rekomendasi dan analisis cerdas berbasis aturan pakar (Rule-based Advisor)
        jika OpenRouter API key belum diset atau ada kendala kuota internet.
        """
        major = entities.get("major")
        course = entities.get("course")
        
        if intent == "tanya_matkul_tersulit":
            if stats_data and isinstance(stats_data, list):
                res = f"### 📊 Analisis Mata Kuliah Paling Menantang di **{major or 'Kampus'}**\n\n"
                res += "Berdasarkan analisis empiris terhadap catatan nilai historis akademik kampus, berikut adalah mata kuliah dengan **tingkat kesulitan dan persentase nilai tidak lulus (D & E) tertinggi**:\n\n"
                
                for i, r in enumerate(stats_data, 1):
                    res += f"**{i}. {r['nama_matakuliah']}** ({r['sks']} SKS)\n"
                    res += f"- **Persentase Nilai D & E (Gagal)**: `{r['pct_fail_DE']:.1f}%` (E: {r['count_E']} orang, D: {r['count_D']} orang)\n"
                    res += f"- **Persentase Nilai Rawan (C, D, E)**: `{r['pct_difficult_CDE']:.1f}%`\n"
                    res += f"- **Rata-rata IP Mata Kuliah**: `{r['mean_ip']:.2f}` dari skala 4.0\n"
                    res += f"- **Total Mahasiswa**: `{r['total_mhs']}` orang\n"
                    res += f"- **Status Kategori**: ⚠️ **{r['tingkat_kesulitan'].upper()}**\n\n"
                    
                top_course = stats_data[0]["nama_matakuliah"]
                res += f"---\n### 💡 Rekomendasi & Tips Persiapan Belajar:\n"
                res += f"Mata kuliah **{top_course}** memiliki rasio nilai jelek yang signifikan. Agar kamu tidak mengulang dan bisa mengamankan nilai A/B, lakukan persiapan berikut:\n"
                res += f"1. **Pahami Fondasi Teori Lebih Awal**: Pelajari silabus dan konsep dasar sebelum dosen menerangkannya di kelas.\n"
                res += f"2. **Rutinitas Latihan Soal**: Matkul ini biasanya menuntut pemahaman praktis/logika, bukan sekadar hafalan. Luangkan minimal 2 jam/minggu khusus latihan mandiri.\n"
                res += f"3. **Bentuk Kelompok Belajar (Study Group)**: Diskusikan soal-soal latihan dan studi kasus bersama teman untuk memecahkan kendala bersama.\n"
                res += f"4. **Maksimalkan Nilai Tugas & Praktikum**: Seringkali nilai tugas dan kuis harian menjadi penyelamat ketika ujian UTS/UAS terasa berat.\n"
                return res
            else:
                return "Maaf, sebutkan nama jurusan (misalnya: *Teknik Informatika*, *Matematika*, atau *Ilmu Hukum*) agar saya dapat menampilkan daftar mata kuliah tersulit di jurusan tersebut."
                
        elif intent == "rekomendasi_cara_belajar":
            target_course = course or "Mata Kuliah Kritis"
            res = f"### 🎯 Panduan Strategi Belajar untuk Mata Kuliah: **{target_course}**\n\n"
            
            if stats_data and isinstance(stats_data, dict):
                res += f"> ℹ️ **Fakta Akademik Kampus**: Mata kuliah ini berbobot **{stats_data.get('sks', '-')} SKS** di Jurusan **{stats_data.get('jurusan', '-')}**, dengan rasio nilai D & E sebesar **{stats_data.get('pct_fail_DE', 0):.1f}%** dan rata-rata IP **{stats_data.get('mean_ip', 0):.2f}**.\n\n"
                
            res += "Agar kamu bisa belajar secara efektif dan memperoleh nilai optimal (A atau B), ikuti langkah strategis berikut:\n\n"
            res += "#### 1. Kuasai Peta Konsep (Concept Mapping)\n"
            res += f"- Petakan materi pokok dari minggu 1 hingga 14 pada mata kuliah {target_course}.\n"
            res += "- Identifikasi topik-topik prasyarat yang menjadi batu sandungan utama.\n\n"
            res += "#### 2. Metode Belajar Active Recall & Feynman Technique\n"
            res += "- Setelah membaca atau menyimak materi, tutup catatan dan coba jelaskan konsep tersebut dengan bahasa sederhana seolah mengajarkannya ke orang awam.\n"
            res += "- Catat bagian mana yang masih membuatmu bingung, lalu tanyakan kepada dosen atau asisten lab saat sesi tanya jawab.\n\n"
            res += "#### 3. Manajemen Tugas dan Persiapan Ujian\n"
            res += "- Kerjakan tugas seawal mungkin untuk menghindari sistem kebut semalam (SKS).\n"
            res += "- Kumpulkan bank soal UTS dan UAS dari tahun-tahun sebelumnya untuk membiasakan diri dengan pola soal dosen pengampu.\n\n"
            res += "#### 4. Konsultasi Terjadwal\n"
            res += "- Manfaatkan jam asistensi laboratorium atau bimbingan akademik jika menemui materi yang sulit dipahami."
            return res
            
        elif intent == "analisis_statistik_matkul":
            if stats_data and isinstance(stats_data, dict):
                res = f"### 📈 Data Statistik Nilai: **{stats_data['nama_matakuliah']}**\n\n"
                res += f"- **Jurusan**: {stats_data['jurusan']} | **Fakultas**: {stats_data['fakultas']}\n"
                res += f"- **Bobot SKS**: {stats_data['sks']} SKS\n"
                res += f"- **Total Mahasiswa**: {stats_data['total_mhs']} orang\n"
                res += f"- **Rata-rata IP**: {stats_data['mean_ip']:.2f} / 4.00\n"
                res += f"- **Kategori Kesulitan**: **{stats_data['tingkat_kesulitan'].upper()}**\n\n"
                res += "#### Distribusi Perolehan Nilai:\n"
                res += f"- **Nilai A**: `{stats_data['pct_A']}%` ({stats_data['count_A']} mahasiswa)\n"
                res += f"- **Nilai B**: `{stats_data['pct_B']}%` ({stats_data['count_B']} mahasiswa)\n"
                res += f"- **Nilai C**: `{stats_data['pct_C']}%` ({stats_data['count_C']} mahasiswa)\n"
                res += f"- **Nilai D**: `{stats_data['pct_D']}%` ({stats_data['count_D']} mahasiswa)\n"
                res += f"- **Nilai E**: `{stats_data['pct_E']}%` ({stats_data['count_E']} mahasiswa)\n\n"
                res += f"**Tingkat Kelulusan (A + B)**: `{(stats_data['pct_A'] + stats_data['pct_B']):.1f}%`  \n"
                res += f"**Tingkat Rawan/Gagal (C + D + E)**: `{stats_data['pct_difficult_CDE']:.1f}%`\n"
                return res
            else:
                return f"Data spesifik untuk mata kuliah tersebut belum ditemukan di catatan kampus. Silakan cek ejaan nama mata kuliah atau pilih jurusan terlebih dahulu."
                
        elif intent == "daftar_matkul_jurusan":
            if major:
                courses = self.df_stats[self.df_stats["jurusan"].str.lower() == major.lower()]["nama_matakuliah"].unique().tolist()
                res = f"### 📚 Daftar Mata Kuliah di Jurusan **{major}**\n\n"
                res += f"Terdapat **{len(courses)}** mata kuliah terdata pada jurusan ini. Berikut beberapa di antaranya:\n"
                for c in courses[:15]:
                    res += f"- {c}\n"
                if len(courses) > 15:
                    res += f"\n*(dan {len(courses) - 15} mata kuliah lainnya)*"
                return res
            else:
                return "Sebutkan nama jurusan yang ingin kamu lihat daftar mata kuliahnya (contoh: *Teknik Informatika*, *Matematika*)."
                
        else: # sapaan_dan_bantuan
            return (
                "👋 **Halo! Saya adalah AI Agent Pendamping Akademik (Agentic Campus).**\n\n"
                "Saya didukung oleh model pemrosesan bahasa alami (NLP) dan analisis 1 juta data historis nilai kampus. "
                "Saya dapat membantu kamu:\n"
                "1. 🔍 **Mengetahui Mata Kuliah Paling Sulit** (persentase nilai D & E tertinggi di jurusanmu).\n"
                "2. 💡 **Mendapatkan Tips & Rekomendasi Belajar** agar siap menghadapi materi berat dan meraih nilai A/B.\n"
                "3. 📊 **Melihat Statistik Lengkap Mata Kuliah** (sebaran nilai A, B, C, D, E, rata-rata IP, dan tingkat kelulusan).\n\n"
                "*Coba tanyakan:* `\"Mata kuliah apa yang paling susah di jurusan Teknik Informatika?\"` atau `\"Tips belajar Algoritma dan Pemrograman\"`"
            )

    def chat(self, user_query: str, model_type: str = "bilstm") -> dict:
        """
        Fungsi utama interaksi mahasiswa dengan AI Agent:
        1. NLP Intent Recognition
        2. Entity Extraction
        3. Database Analytics Retrieval
        4. OpenRouter LLM Synthesis (atau Expert Fallback)
        """
        # 1. NLP Prediksi Intent
        nlp_pred = self.predict_intent(user_query, model_type=model_type)
        intent = nlp_pred["intent"]
        confidence = nlp_pred["confidence"]
        
        # 2. Ekstraksi Entitas
        entities = self.extract_entities(user_query)
        major = entities["major"]
        course = entities["course"]
        
        # Default major jika query bertanya matkul susah tanpa sebut jurusan
        if not major and intent == "tanya_matkul_tersulit":
            major = "TEKNIK INFORMATIKA" # default representatif jika tidak spesifik
            entities["major"] = major
            
        # 3. Pengambilan Data dari Database Kampus (Tools)
        db_data = None
        if intent == "tanya_matkul_tersulit":
            db_data = self.get_most_difficult_courses(major=major, top_n=5)
        elif intent == "analisis_statistik_matkul" or (intent == "rekomendasi_cara_belajar" and course):
            if course:
                db_data = self.get_course_detail(course, major=major)
            elif major:
                db_data = self.get_most_difficult_courses(major=major, top_n=3)
        elif intent == "daftar_matkul_jurusan":
            db_data = {"major": major}
            
        # 4. Sintesis Respon Menggunakan OpenRouter (jika API key tersedia)
        llm_response = None
        if self.api_key:
            # Bangun prompt yang kaya akan fakta data akademik
            context_str = ""
            if db_data:
                context_str = f"\n[FAKTA DATA HISTORIS KAMPUS DARI CSV]:\n{json.dumps(db_data, indent=2, ensure_ascii=False)}\n"
                
            prompt = (
                f"Pertanyaan Mahasiswa: \"{user_query}\"\n"
                f"Terdeteksi Intent: {intent} (Confidence: {confidence:.2f})\n"
                f"Entitas Terdeteksi: Jurusan = {major or 'Tidak spesifik'}, Mata Kuliah = {course or 'Tidak spesifik'}\n"
                f"{context_str}\n"
                f"Instruksi Khusus:\n"
                f"1. Jawab pertanyaan mahasiswa dengan gaya ramah, edukatif, dan empati sebagai advisor kampus.\n"
                f"2. Jika ada fakta data di atas, sebutkan angka dan persentasenya (misal % nilai D dan E, rata-rata IP) agar mahasiswa paham tingkat kesulitannya.\n"
                f"3. Berikan rekomendasi konkret cara belajar materi tersebut (peta konsep materi, tips latihan, strategi ujian, dan cara menghadapi kendala).\n"
                f"4. Gunakan format Markdown yang menarik (gunakan emoji, poin-poin tebal, dan struktur jelas)."
            )
            llm_response = self.query_openrouter_llm(prompt)
            
        # Jika LLM tidak tersedia atau None, gunakan Expert Fallback yang cerdas
        if not llm_response:
            final_text = self.generate_expert_fallback_response(user_query, intent, entities, db_data)
            response_source = "Local Academic Expert Engine"
        else:
            final_text = llm_response
            response_source = f"OpenRouter LLM ({self.model_name})"
            
        return {
            "query": user_query,
            "intent": intent,
            "confidence": confidence,
            "model_type": nlp_pred["model_used"],
            "entities": entities,
            "db_data": db_data,
            "response": final_text,
            "source": response_source
        }

if __name__ == "__main__":
    bot = AgenticCampusBot()
    test_queries = [
        "Mata kuliah apa yang paling susah di jurusan Teknik Informatika?",
        "Bagaimana cara belajar Algoritma dan Pemrograman agar dapat nilai A?",
        "Tolong tampilkan statistik nilai Kalkulus I",
        "Halo apa saja yang bisa kamu bantu?"
    ]
    
    print("\n" + "=" * 70)
    print("DEMO INFERENSI AGENTIC CAMPUS BOT")
    print("=" * 70)
    for q in test_queries:
        print(f"\n[USER]: {q}")
        res = bot.chat(q, model_type="bilstm")
        print(f"[INTENT]: {res['intent']} (Conf: {res['confidence']:.2f}) | Source: {res['source']}")
        print(f"[BOT]:\n{res['response'][:300]}...\n")
