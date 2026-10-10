import pandas as pd
import numpy as np
import os
import random

def build_balanced_realistic_nlp_dataset(output_path="data/dataset_nlp_intent.csv"):
    """
    Dataset NLP Intent dengan variasi bahasa alami mahasiswa Indonesia
    (Formal, Santai, Gaul, Singkatan) yang seimbang dan bebas dari template leakage.
    """
    print("[*] Membangun dataset NLP seimbang & realistis...")
    
    majors = ["Teknik Informatika", "Teknik Elektro", "Matematika", "Fisika", "Kimia", "Manajemen", "Ilmu Hukum", "Sistem Informasi", "Psikologi", "Akuntansi"]
    courses = ["Kalkulus I", "Kalkulus II", "Algoritma dan Pemrograman", "Struktur Data", "Fisika Dasar", "Dasar Pemrograman", "Logika Informatika", "Basis Data", "Jaringan Komputer", "Pengantar Analisis Riil"]
    
    # Kumpulan pola kalimat realistis
    intent_samples = {
        "tanya_matkul_tersulit": [
            "Mata kuliah apa yang paling susah di jurusan {m}?",
            "Matkul paling killer di jurusan {m} apaan ya min?",
            "Spill mata kuliah tersulit di {m} dong kak",
            "Di jurusan {m}, matkul apa yang banyak dapet nilai E?",
            "Matkul apa aja yang sering bikin mahasiswa {m} ngulang semester?",
            "Kenapa banyak mahasiswa {m} dapet nilai jelek di semester awal?",
            "Mata kuliah dengan persentase kegagalan tertinggi di {m} apa ya?",
            "Kak mau tanya matkul tersulit di program studi {m} apa saja?",
            "Ada matkul horor ga di jurusan {m} yang tingkat kelulusannya kecil?",
            "Mata kuliah apa yang paling bikin IPK anjlok di jurusan {m}?",
            "Banyak yang ga lulus di matkul apa sih anak {m}?",
            "MK apa yang paling rawan dapet nilai D dan E di prodi {m}?",
            "Tolong list 5 mata kuliah tersulit di {m} berdasarkan nilai mahasiswa",
            "Mata kuliah manakah di prodi {m} yang mencatatkan tingkat kelulusan paling minim?",
            "Matkul paling ditakuti sama mahasiswa baru di {m} apa aja?",
            "matkul paling susah {m}",
            "matkul killer {m}",
            "matkul nilai E terbanyak di {m}",
            "daftar matkul tersulit {m}",
            "matkul persentase D dan E tertinggi {m}"
        ],
        "rekomendasi_cara_belajar": [
            "Bagaimana cara belajar mata kuliah {c} agar dapat nilai A?",
            "Spill tips dapet nilai bagus di matkul {c} dong kak",
            "Minta saran dong cara survive dan lulus di matkul {c}",
            "Ada tips belajar efektif buat mata kuliah {c} ga ya?",
            "Saya takut dapet nilai jelek di {c}, ada saran persiapan belajarnya?",
            "Gimana cara memahami konsep materi {c} dari dasar min?",
            "Beri saya panduan belajar mandiri untuk menghadapi ujian {c}",
            "Materi apa yang paling penting dipelajari terlebih dahulu di {c}?",
            "Bisa kasih tips bagaimana cara lulus mata kuliah {c} tanpa mengulang?",
            "Bagaimanakah strategi pembelajaran yang efektif untuk menguasai materi {c}?",
            "Trik belajar dan latihan soal buat matkul {c} gimana ya?",
            "Ada rekomendasi buku atau video tutorial buat belajar {c}?",
            "Gimana cara dapet nilai maksimal pas praktikum dan ujian {c}?",
            "Langkah persiapan apa saja sebelum mengambil mata kuliah {c}?",
            "Tips membagi waktu antara teori dan tugas besar pada matkul {c}",
            "tips belajar {c}",
            "cara lulus {c}",
            "strategi belajar {c}",
            "panduan belajar materi {c}",
            "cara cepat paham konsep {c}"
        ],
        "analisis_statistik_matkul": [
            "Berapa persentase kelulusan mata kuliah {c} di jurusan {m}?",
            "Tolong tampilkan statistik sebaran nilai untuk mata kuliah {c}",
            "Berapa rata-rata nilai IP mahasiswa pada matkul {c}?",
            "Berapa banyak mahasiswa yang dapat nilai E dan D di {c}?",
            "Apakah mata kuliah {c} di {m} termasuk kategori sulit?",
            "Tampilkan distribusi nilai A, B, C, D, dan E untuk {c}",
            "Berapa persen sih yang dapet A di mata kuliah {c}?",
            "Cek dong berapa mahasiswa yang ngulang di matkul {c}",
            "Rata-rata IP mahasiswa di matkul {c} jurusan {m} berapa ya?",
            "Tingkat kelulusan di matkul {c} beneran rendah ya min?",
            "Bisa lihat rekapitulasi data nilai mata kuliah {c} ga kak?",
            "Persentase nilai A B C D E di matkul {c} prodi {m} berapa?",
            "Berapa total mahasiswa yang pernah menempuh mata kuliah {c}?",
            "Tampilkan grafik atau sebaran perolehan nilai mata kuliah {c}",
            "Cek status tingkat kesulitan mata kuliah {c} berdasarkan data",
            "statistik nilai {c}",
            "distribusi nilai {c}",
            "persentase kelulusan {c}",
            "rata-rata IP {c}",
            "rekapitulasi nilai {c} {m}"
        ],
        "daftar_matkul_jurusan": [
            "Mata kuliah apa saja yang ada di jurusan {m}?",
            "Tolong tampilkan daftar seluruh mata kuliah di prodi {m}",
            "Apa saja kurikulum dan mata kuliah wajib di jurusan {m}?",
            "Bisa lihat daftar nama matkul yang tersedia untuk prodi {m}?",
            "Ada matkul apa aja di semester awal jurusan {m}?",
            "Mau liat daftar katalog mata kuliah di prodi {m} dong",
            "Struktur kurikulum dan mata kuliah peminatan di {m} apa aja?",
            "Matkul wajib di program studi {m} apa saja ya kak?",
            "Spill katalog mata kuliah jurusan {m} semester 1 sampai 8",
            "Info daftar mata kuliah yang ditawarkan pada jurusan {m}",
            "daftar matkul {m}",
            "kurikulum prodi {m}",
            "list matkul jurusan {m}",
            "katalog matkul {m}",
            "mata kuliah yang ada di {m}"
        ],
        "sapaan_dan_bantuan": [
            "Halo apa kabar agen cerdas kampus?",
            "Hai selamat pagi bot asisten akademik",
            "Selamat siang kak mau tanya-tanya seputar perkuliahan",
            "Halo bot, kamu bisa membantu saya apa saja di kampus ini?",
            "Apa fungsi dan kemampuan dari AI Agentic Campus ini?",
            "Hai bot cerdas perkenalkan dirimu dong",
            "Halo asisten akademik saya butuh bantuan untuk konsultasi nilai",
            "Makasih banyak ya infonya sangat membantu",
            "Terima kasih atas saran dan tips belajarnya ya bot",
            "Permisi mau konsultasi mengenai persiapan mata kuliah",
            "Halo apakah kamu bisa menganalisis nilai mahasiswa di kampus?",
            "Bantuan apa saja yang bisa kamu sediakan untuk mahasiswa?",
            "Selamat malam asisten akademik kampus",
            "Tes bot apakah sistem aktif dan online?",
            "Halo salam kenal ya asisten kampus"
        ]
    }
    
    rows = []
    
    for intent, templates in intent_samples.items():
        for tpl in templates:
            if "{m}" in tpl and "{c}" in tpl:
                for _ in range(6):
                    rows.append({"text": tpl.format(m=random.choice(majors), c=random.choice(courses)), "intent": intent})
            elif "{m}" in tpl:
                for m in majors:
                    rows.append({"text": tpl.format(m=m), "intent": intent})
            elif "{c}" in tpl:
                for c in courses:
                    rows.append({"text": tpl.format(c=c), "intent": intent})
            else:
                rows.append({"text": tpl, "intent": intent})
                
    df = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Dataset tersimpan di {output_path} ({len(df)} sampel).")
    print("     Distribusi per kelas intent:")
    print(df["intent"].value_counts().to_string())
    return df

if __name__ == "__main__":
    build_balanced_realistic_nlp_dataset()
