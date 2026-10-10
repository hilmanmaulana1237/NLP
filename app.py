import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys

# Tambahkan direktori modular ke sys.path agar impor selalu berhasil lokal & Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for subfolder in ["02_pipeline_nlp", "03_agent_engine"]:
    sub_path = os.path.join(BASE_DIR, subfolder)
    if sub_path not in sys.path:
        sys.path.insert(0, sub_path)

# Reconfigure stdout for utf-8
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agentic_campus_bot import AgenticCampusBot
from nlp_preprocessing import IndonesianNLPPreprocessor

# Page Configuration
st.set_page_config(
    page_title="Agentic Campus - Smart Academic Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    
    .main-header h1 {
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.3rem;
        color: #ffffff;
    }
    
    .main-header p {
        font-size: 1.05rem;
        opacity: 0.95;
        margin: 0;
    }
    
    .badge-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-right: 6px;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(4px);
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .chat-bubble-user {
        background: #f1f5f9;
        border-radius: 16px 16px 2px 16px;
        padding: 12px 18px;
        margin-bottom: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .chat-bubble-bot {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 4px 16px 16px 16px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Cache Bot Initialization
@st.cache_resource
def get_bot(api_key=None, model="meta-llama/llama-3.3-70b-instruct:free"):
    return AgenticCampusBot(openrouter_api_key=api_key, model_name=model)

# Cache Academic Dataset
@st.cache_data
def get_academic_stats():
    stats_path = "data/academic_course_stats.csv"
    if os.path.exists(stats_path):
        return pd.read_csv(stats_path)
    return pd.DataFrame()

# Cache Evaluation Metrics
@st.cache_data
def get_eval_results():
    path = "models/evaluation_results.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

df_stats = get_academic_stats()
eval_data = get_eval_results()

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/artificial-intelligence.png", width=64)
    st.title("Pengaturan Agen")
    st.caption("Agentic Campus • NLP & LLM Assistant")
    
    st.markdown("---")
    st.subheader("🔑 Konfigurasi OpenRouter")
    api_key_input = st.text_input(
        "OpenRouter API Key:",
        type="password",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        help="Masukkan API key OpenRouter untuk menghubungkan LLM. Jika kosong, sistem otomatis memakai Mesin Pakar Lokal (Offline Mode)."
    )
    
    model_choices = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-001",
        "deepseek/deepseek-chat",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-haiku",
        "qwen/qwen-2.5-72b-instruct"
    ]
    selected_model = st.selectbox("Pilih Model LLM:", model_choices, index=0)
    
    st.markdown("---")
    st.subheader("🧠 Engine NLP Intent")
    nlp_engine = st.radio(
        "Model Klasifikasi Teks:",
        ["Word Embedding + BiLSTM (Deep Learning)", "TF-IDF + Linear SVM (Machine Learning)"],
        index=0
    )
    model_type_key = "bilstm" if "BiLSTM" in nlp_engine else "tfidf_ml"
    
    st.markdown("---")
    if api_key_input:
        st.success("🟢 OpenRouter LLM Aktif")
    else:
        st.info("🟡 Mode Pakar Akademik Lokal (Tanpa API Key)")
        
    st.caption("Dikembangkan untuk Tugas Praktikum NLP: Agentic Campus.")

# Inisialisasi Bot dengan Config Terkini
bot = get_bot(api_key=api_key_input, model=selected_model)
bot.set_api_key(api_key_input)
bot.set_model_name(selected_model)

# Header Banner
st.markdown("""
<div class="main-header">
    <span class="badge-chip">AGENTIC CAMPUS</span>
    <span class="badge-chip">NLP & LLM ADVISOR</span>
    <span class="badge-chip">1 JUTA DATA MAHASISWA</span>
    <h1>🎓 Smart Academic Course Advisor</h1>
    <p>Agen Cerdas Pendamping Akademik: Menganalisis Mata Kuliah Paling Sulit & Memberikan Rekomendasi Belajar Berbasis NLP dan LLM OpenRouter.</p>
</div>
""", unsafe_allow_html=True)

# Tabs Navigation
tab_chat, tab_dashboard, tab_nlp = st.tabs([
    "💬 Tanya AI Agent (Chatbot)",
    "📊 Analisis Data Kampus (Dashboard)",
    "🧠 Laporan & Evaluasi Model NLP (Tugas Kuliah)"
])

# =============================================================================
# TAB 1: CHATBOT AI AGENT
# =============================================================================
with tab_chat:
    st.subheader("Konsultasi Mata Kuliah & Strategi Belajar")
    st.caption("Tanyakan mata kuliah apa yang paling sulit, alasan banyaknya nilai jelek, atau minta panduan belajar materi kuliah.")
    
    # Preset Prompt Chips
    st.markdown("**💡 Contoh Pertanyaan Populer:**")
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_q = None
    with col_p1:
        if st.button("📌 Matkul tersulit di Teknik Informatika?", use_container_width=True):
            preset_q = "Mata kuliah apa yang paling susah di jurusan Teknik Informatika?"
    with col_p2:
        if st.button("📌 Tips belajar Algoritma & Pemrograman?", use_container_width=True):
            preset_q = "Bagaimana cara belajar Algoritma dan Pemrograman agar dapat nilai A?"
    with col_p3:
        if st.button("📌 Statistik kelulusan Kalkulus I?", use_container_width=True):
            preset_q = "Tolong tampilkan statistik sebaran nilai untuk mata kuliah Kalkulus I"

    # Chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Halo! Saya adalah AI Agent Pendamping Akademik Kampus.**\n\n"
                    "Saya dapat membantu kamu menganalisis mata kuliah mana yang memiliki **tingkat nilai tidak lulus (D & E) tertinggi**, "
                    "serta memberikan **rekomendasi dan strategi belajar materi** agar kamu siap dan bisa mengamankan nilai A atau B!\n\n"
                    "Silakan tanyakan mata kuliah atau jurusan yang ingin kamu konsultasikan."
                ),
                "intent_info": None
            }
        ]

    # Display existing chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("intent_info"):
                info = msg["intent_info"]
                st.caption(f"🎯 **Intent Terdeteksi**: `{info['intent']}` (Confidence: `{info['confidence'] * 100:.1f}%`) | **Engine**: {info['model_type']} | **Sumber**: {info['source']}")
            st.markdown(msg["content"])

    # Chat Input
    user_input = st.chat_input("Ketik pertanyaanmu seputar mata kuliah, kesulitan nilai, atau tips belajar...")
    active_prompt = preset_q or user_input

    if active_prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("AI Agent sedang memproses query NLP & menganalisis data nilai kampus..."):
                result = bot.chat(active_prompt, model_type=model_type_key)
                
                # Intent Badge
                st.caption(f"🎯 **Intent Terdeteksi**: `{result['intent']}` (Confidence: `{result['confidence'] * 100:.1f}%`) | **Engine**: {result['model_type']} | **Sumber**: {result['source']}")
                
                # Render Bot Response
                st.markdown(result["response"])
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "intent_info": {
                        "intent": result["intent"],
                        "confidence": result["confidence"],
                        "model_type": result["model_type"],
                        "source": result["source"]
                    }
                })

# =============================================================================
# TAB 2: DASHBOARD ANALISIS AKADEMIK (1 JUTA DATA)
# =============================================================================
with tab_dashboard:
    st.subheader("Eksplorasi Data Nilai Akademik Kampus")
    st.caption("Data agregasi empiris dari 985,872 catatan nilai mahasiswa di 62 jurusan dan 5,659 mata kuliah.")
    
    if not df_stats.empty:
        # Filter Jurusan
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            all_majors = ["-- Semua Jurusan --"] + sorted(df_stats["jurusan"].unique().tolist())
            selected_jur = st.selectbox("Pilih Jurusan / Program Studi:", all_majors, index=all_majors.index("TEKNIK INFORMATIKA") if "TEKNIK INFORMATIKA" in all_majors else 0)
        with col_f2:
            min_students = st.slider("Filter Minimal Mahasiswa Pengambil Matkul:", min_value=10, max_value=200, value=30, step=10)
            
        filtered_df = df_stats.copy()
        if selected_jur != "-- Semua Jurusan --":
            filtered_df = filtered_df[filtered_df["jurusan"] == selected_jur]
        filtered_df = filtered_df[filtered_df["total_mhs"] >= min_students]
        
        # Metric KPI
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Mata Kuliah Terdata", f"{len(filtered_df):,}")
        with col_m2:
            avg_ip = filtered_df["mean_ip"].mean() if not filtered_df.empty else 0
            st.metric("Rata-rata IP Mata Kuliah", f"{avg_ip:.2f} / 4.00")
        with col_m3:
            avg_fail = filtered_df["pct_fail_DE"].mean() if not filtered_df.empty else 0
            st.metric("Rata-rata % Nilai D & E", f"{avg_fail:.1f}%")
        with col_m4:
            count_hard = (filtered_df["tingkat_kesulitan"] == "Sulit").sum() if not filtered_df.empty else 0
            st.metric("Mata Kuliah Kategori Sulit", f"{count_hard}")
            
        st.markdown("---")
        
        # Visualisasi Top Mata Kuliah Paling Sulit (Tingkat D & E Tertinggi)
        st.subheader(f"🏆 Top 10 Mata Kuliah Paling Sulit (Persentase Nilai D & E Tertinggi)")
        top_hard = filtered_df.sort_values(by="pct_fail_DE", ascending=False).head(10)
        
        if not top_hard.empty:
            fig_bar = px.bar(
                top_hard,
                x="pct_fail_DE",
                y="nama_matakuliah",
                orientation="h",
                color="pct_fail_DE",
                color_continuous_scale="Reds",
                labels={"pct_fail_DE": "% Mahasiswa Dapat Nilai D & E", "nama_matakuliah": "Mata Kuliah"},
                title=f"Mata Kuliah dengan Persentase Nilai Tidak Lulus Tertinggi ({selected_jur})",
                text="pct_fail_DE"
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Tabel Rinci
            st.markdown("**📋 Tabel Rincian Statistik Mata Kuliah:**")
            display_cols = ["nama_matakuliah", "sks", "total_mhs", "mean_ip", "pct_A", "pct_B", "pct_C", "pct_D", "pct_E", "pct_fail_DE", "tingkat_kesulitan"]
            display_df = top_hard[display_cols].rename(columns={
                "nama_matakuliah": "Nama Mata Kuliah",
                "sks": "SKS",
                "total_mhs": "Total Mhs",
                "mean_ip": "Rata-rata IP",
                "pct_fail_DE": "% Gagal (D+E)",
                "tingkat_kesulitan": "Status"
            })
            st.dataframe(display_df, use_container_width=True)
            
        st.markdown("---")
        # Detail Distribusi Nilai Satu Mata Kuliah (Pie Chart)
        st.subheader("🔍 Cek Detail Sebaran Nilai Suatu Mata Kuliah")
        course_list = sorted(filtered_df["nama_matakuliah"].unique().tolist())
        if course_list:
            selected_course = st.selectbox("Pilih Mata Kuliah untuk Lihat Detail:", course_list)
            course_row = filtered_df[filtered_df["nama_matakuliah"] == selected_course].iloc[0]
            
            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.markdown(f"### **{course_row['nama_matakuliah']}**")
                st.markdown(f"- **Jurusan**: {course_row['jurusan']} | **SKS**: {course_row['sks']}")
                st.markdown(f"- **Total Mahasiswa**: {course_row['total_mhs']} orang")
                st.markdown(f"- **Rata-rata IP**: `{course_row['mean_ip']:.2f}` dari 4.00")
                st.markdown(f"- **Tingkat Kesulitan**: `⚠️ {course_row['tingkat_kesulitan'].upper()}`")
                st.markdown(f"- **Persentase Nilai D & E**: `{course_row['pct_fail_DE']:.1f}%`")
                st.markdown(f"- **Persentase Nilai Rawan (C, D, E)**: `{course_row['pct_difficult_CDE']:.1f}%`")
                
            with c_right:
                grade_labels = ["Nilai A", "Nilai B", "Nilai C", "Nilai D", "Nilai E"]
                grade_values = [course_row["count_A"], course_row["count_B"], course_row["count_C"], course_row["count_D"], course_row["count_E"]]
                colors = ["#10b981", "#3b82f6", "#f59e0b", "#f97316", "#ef4444"]
                
                fig_pie = go.Figure(data=[go.Pie(labels=grade_labels, values=grade_values, hole=0.4, marker=dict(colors=colors))])
                fig_pie.update_layout(title=f"Distribusi Nilai Mahasiswa ({course_row['nama_matakuliah']})", height=320)
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("Data agregasi akademik belum ditemukan. Silakan jalankan `python build_academic_stats.py`.")

# =============================================================================
# TAB 3: LAPORAN & EVALUASI MODEL NLP (TUGAS KULIAH)
# =============================================================================
with tab_nlp:
    st.subheader("Dokumentasi Praktikum & Evaluasi Model NLP")
    st.caption("Implementasi lengkap 4 komponen tugas kuliah sesuai panduan.")
    
    # Accordion 1: Ide Studi Kasus
    with st.expander("📌 1. Ide Studi Kasus Agen NLP untuk 'Agentic Campus'", expanded=True):
        st.markdown("""
        - **Nama Agen**: **Agentic Campus: Smart Academic Course Difficulty Advisor & Study Coach Agent**
        - **Latar Belakang**: 
          Di perguruan tinggi, mahasiswa sering kali tidak mengetahui mata kuliah mana yang memiliki tingkat kegagalan tinggi (tingkat nilai D dan E) 
          sebelum mengambil mata kuliah tersebut, sehingga kurang mempersiapkan diri. Agen ini bertindak sebagai asisten pintar kampus yang mengolah 
          pertanyaan bahasa alami mahasiswa, mengekstraksi informasi mata kuliah, mengaitkannya dengan data nilai akademik riil (1 juta record), 
          serta memberikan rekomendasi cara belajar yang actionable dari model LLM (OpenRouter).
        - **Pipeline NLP Agent**:
          1. **Input Teks**: Query pertanyaan mahasiswa (misal: *"Matkul apa yang paling susah di jurusan Teknik Informatika?"*).
          2. **Preprocessing**: Case Folding, Cleansing, Tokenizing, Stopword Removal (Sastrawi), dan Stemming (Sastrawi).
          3. **Klasifikasi Teks**: Model NLP mengklasifikasikan intent pertanyaan (5 kelas: `tanya_matkul_tersulit`, `rekomendasi_cara_belajar`, `analisis_statistik_matkul`, `daftar_matkul_jurusan`, `sapaan_dan_bantuan`).
          4. **Retrieval Tool**: Mengambil fakta empiris nilai historis (persentase D & E, IP rata-rata, distribusi grade) dari database kampus.
          5. **LLM Synthesis**: Mengirim prompt terstruktur ke model OpenRouter untuk menghasilkan penjelasan ramah dan panduan belajar materi.
        """)

    # Accordion 2: Pre-processing Data
    with st.expander("⚙️ 2. Pre-processing Data Teks (Sastrawi)", expanded=True):
        st.markdown("""
        Tahapan Pre-processing Data Teks Bahasa Indonesia yang diimplementasikan:
        1. **Case Folding**: Mengubah semua huruf teks menjadi huruf kecil (*lowercase*).
        2. **Cleansing**: Menghapus URL, mention, tanda baca, simbol non-alfanumerik, dan normalisasi spasi.
        3. **Tokenization**: Memecah kalimat menjadi deretan token kata tunggal.
        4. **Stopword Removal**: Menghilangkan kata umum/kata sambung yang tidak memiliki bobot semantik (*PySastrawi Stopwords* + *custom stopwords*).
        5. **Stemming**: Mengembalikan setiap kata berimbuhan ke bentuk dasarnya (*root word*) menggunakan algoritma Nazief & Adriani dari *PySastrawi Stemmer*.
        """)
        
        st.markdown("**Contoh Uji Coba Interaktif Preprocessing:**")
        sample_input = st.text_input("Uji Teks Preprocessing:", value="Mata kuliah apa yang paling SUSAH di Jurusan TEKNIK INFORMATIKA? Kasih tips belajarnya ya!!")
        if sample_input:
            p = IndonesianNLPPreprocessor(use_stemming=True)
            steps = p.get_preprocessing_steps(sample_input)
            
            c1, c2 = st.columns(2)
            with c1:
                st.code(f"Original        : {steps['original']}\nCase Folding    : {steps['case_folding']}\nCleansing       : {steps['cleansing']}\nTokenizing      : {steps['tokenizing']}", language="text")
            with c2:
                st.code(f"Stopword Removed: {steps['stopword_removed']}\nStemming        : {steps['stemming']}\nFinal Result    : {steps['final_text']}", language="text")

    # Accordion 3: Evaluasi TF-IDF + Machine Learning
    with st.expander("📊 3. Representasi Terstruktur TF-IDF + Algoritma Machine Learning", expanded=True):
        st.markdown("""
        - **Ekstraksi Fitur**: `TfidfVectorizer` dengan kombinasi n-gram `(1, 2)` (unigram & bigram), `sublinear_tf=True`, dan batas maksimal 2,500 fitur.
        - **Algoritma Machine Learning**: Diuji 3 algoritma populer:
          1. **Linear SVM (LinearSVC)**
          2. **Logistic Regression**
          3. **Multinomial Naive Bayes**
        """)
        
        if eval_data and "all_ml_algorithms" in eval_data:
            ml_comp = []
            for name, m in eval_data["all_ml_algorithms"].items():
                ml_comp.append({"Algoritma": name, "Akurasi": f"{m['accuracy'] * 100:.2f}%", "Weighted F1-Score": f"{m['f1_score']:.4f}"})
            st.table(pd.DataFrame(ml_comp))
            
        c_ml1, c_ml2 = st.columns([1, 1])
        with c_ml1:
            if os.path.exists("reports/confusion_matrix_ml.png"):
                st.image("reports/confusion_matrix_ml.png", caption="Confusion Matrix Model Machine Learning (Linear SVM)")
        with c_ml2:
            st.markdown("""
            **Analisis Hasil Machine Learning:**
            - Representasi TF-IDF mampu memisahkan kelas-kelas intent secara tajam karena kata kunci pembeda (seperti *susah, nilai, tips, belajar, daftar, statistik*) memiliki bobot TF-IDF tinggi.
            - Model Linear SVM dan Logistic Regression mencapai akurasi hingga **100%** pada test set karena ruang vektor teks intent sangat terpisah secara linear.
            """)

    # Accordion 4: Evaluasi Word Embedding + Deep Learning
    with st.expander("🧠 4. Representasi Word Embedding + Algoritma Deep Learning (BiLSTM)", expanded=True):
        st.markdown("""
        - **Representasi Teks**: Tokenizer Keras + Sequence Padding (`maxlen=35`).
        - **Arsitektur Deep Learning**:
          1. **Embedding Layer**: `Embedding(vocab_size=3000, output_dim=64, input_length=35)`
          2. **Spatial Dropout**: `SpatialDropout1D(rate=0.25)`
          3. **Bidirectional LSTM**: `Bidirectional(LSTM(64 units))`
          4. **Dropout & Dense Layer**: `Dropout(0.3) -> Dense(32, ReLU) -> Dropout(0.2)`
          5. **Output Layer**: `Dense(5 classes, activation='softmax')`
          6. **Optimizer**: Adam (`learning_rate=0.001`), **Loss**: `sparse_categorical_crossentropy`
        """)
        
        c_dl1, c_dl2 = st.columns([1, 1])
        with c_dl1:
            if os.path.exists("reports/confusion_matrix_dl.png"):
                st.image("reports/confusion_matrix_dl.png", caption="Confusion Matrix Model Deep Learning (Word Embedding + BiLSTM)")
        with c_dl2:
            if os.path.exists("reports/training_history_dl.png"):
                st.image("reports/training_history_dl.png", caption="Grafik Kurva Loss & Akurasi Training BiLSTM")
                
        if eval_data and "models_evaluated" in eval_data:
            dl_metrics = eval_data["models_evaluated"].get("Word Embedding + BiLSTM", {})
            st.markdown(f"""
            **Hasil Evaluasi Deep Learning pada Test Set:**
            - **Akurasi Test**: `{dl_metrics.get('accuracy', 0) * 100:.2f}%`
            - **F1-Score**: `{dl_metrics.get('f1_score', 0):.4f}`
            - **Loss Test**: `{dl_metrics.get('loss', 0):.4f}`
            """)

    # Accordion 5: Tabel Perbandingan ML vs DL
    with st.expander("⚖️ 5. Perbandingan Komparatif: Machine Learning vs Deep Learning", expanded=True):
        if eval_data and "models_evaluated" in eval_data:
            comp_rows = []
            for m_name, m_info in eval_data["models_evaluated"].items():
                comp_rows.append({
                    "Model": m_name,
                    "Akurasi": f"{m_info.get('accuracy', 0) * 100:.2f}%",
                    "F1-Score": f"{m_info.get('f1_score', 0):.4f}",
                    "Karakteristik & Keunggulan": (
                        "Sangat cepat dalam training dan inferensi, interpretasi bobot TF-IDF transparan, efisien untuk dataset berbasis kata kunci." 
                        if "TF-IDF" in m_name else 
                        "Mampu menangkap urutan konteks kata (bidirectional context), generalisasi semantik lebih fleksibel terhadap sinonim kata."
                    )
                })
            st.table(pd.DataFrame(comp_rows))

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>© 2026 Agentic Campus Academic Advisor • NLP Course Project • Dikembangkan dengan Streamlit & OpenRouter</p>", unsafe_allow_html=True)
