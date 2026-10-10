import pandas as pd
import numpy as np
import os
import json

def process_academic_data(csv_path="nilai_mhs_IP IPK.csv", output_dir="data"):
    print(f"[*] Membaca data akademik dari {csv_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Baca kolom relevan
    cols = ["fakultas", "jurusan", "kode_matakuliah", "nama_matakuliah", "sks", "nilai_huruf"]
    df = pd.read_csv(csv_path, sep=";", usecols=cols)
    
    # Cleaning data dasar
    df = df.dropna(subset=["nama_matakuliah", "nilai_huruf", "jurusan"])
    df["nama_matakuliah"] = df["nama_matakuliah"].astype(str).str.strip()
    df["jurusan"] = df["jurusan"].astype(str).str.strip()
    df["fakultas"] = df["fakultas"].astype(str).str.strip()
    df["nilai_huruf"] = df["nilai_huruf"].astype(str).str.strip().str.upper()
    
    # Pemetaan bobot nilai standar 4.0
    grade_map = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "E": 0.0}
    df = df[df["nilai_huruf"].isin(grade_map.keys())]
    df["bobot"] = df["nilai_huruf"].map(grade_map)
    
    print(f"[*] Total record nilai valid: {len(df):,}")
    
    # Agregasi per Jurusan dan Mata Kuliah
    agg = df.groupby(["fakultas", "jurusan", "nama_matakuliah"]).agg(
        total_mhs=("bobot", "count"),
        sks=("sks", "first"),
        mean_ip=("bobot", "mean"),
        count_A=("nilai_huruf", lambda x: (x == "A").sum()),
        count_B=("nilai_huruf", lambda x: (x == "B").sum()),
        count_C=("nilai_huruf", lambda x: (x == "C").sum()),
        count_D=("nilai_huruf", lambda x: (x == "D").sum()),
        count_E=("nilai_huruf", lambda x: (x == "E").sum())
    ).reset_index()
    
    # Hitung persentase
    agg["pct_A"] = (agg["count_A"] / agg["total_mhs"] * 100).round(2)
    agg["pct_B"] = (agg["count_B"] / agg["total_mhs"] * 100).round(2)
    agg["pct_C"] = (agg["count_C"] / agg["total_mhs"] * 100).round(2)
    agg["pct_D"] = (agg["count_D"] / agg["total_mhs"] * 100).round(2)
    agg["pct_E"] = (agg["count_E"] / agg["total_mhs"] * 100).round(2)
    
    # Tingkat kegagalan (Fail Rate: D & E)
    agg["pct_fail_DE"] = ((agg["count_D"] + agg["count_E"]) / agg["total_mhs"] * 100).round(2)
    # Tingkat nilai bermasalah / rawan (Critical Rate: C, D, & E)
    agg["pct_difficult_CDE"] = ((agg["count_C"] + agg["count_D"] + agg["count_E"]) / agg["total_mhs"] * 100).round(2)
    agg["mean_ip"] = agg["mean_ip"].round(2)
    
    # Label Tingkat Kesulitan Akademik Objektif (Ground Truth)
    # Kriteria:
    # - "Sulit"  : % D+E >= 15% ATAU % C+D+E >= 40% ATAU Rata-rata IP < 2.5 (untuk matkul dengan N >= 15)
    # - "Sedang" : % D+E antara 5% - 15% ATAU % C+D+E antara 20% - 40%
    # - "Mudah"  : % D+E < 5% DAN % C+D+E < 20% DAN Rata-rata IP >= 3.2
    def assign_difficulty(row):
        if row["total_mhs"] < 10:
            return "Sedang" # data sampel sedikit
        if row["pct_fail_DE"] >= 15.0 or row["pct_difficult_CDE"] >= 40.0 or row["mean_ip"] < 2.5:
            return "Sulit"
        elif row["pct_fail_DE"] >= 5.0 or row["pct_difficult_CDE"] >= 20.0 or row["mean_ip"] < 3.2:
            return "Sedang"
        else:
            return "Mudah"
            
    agg["tingkat_kesulitan"] = agg.apply(assign_difficulty, axis=1)
    
    # Simpan hasil agregasi
    agg_path = os.path.join(output_dir, "academic_course_stats.csv")
    agg.to_csv(agg_path, index=False)
    print(f"[OK] Berhasil menyimpan statistik matkul ke: {agg_path}")
    print(f"    Total mata kuliah terdata: {len(agg):,}")
    print("    Distribusi Tingkat Kesulitan:")
    print(agg["tingkat_kesulitan"].value_counts().to_string())
    
    # Simpan daftar jurusan unik untuk dropdown / filter
    majors = sorted(agg["jurusan"].unique().tolist())
    with open(os.path.join(output_dir, "daftar_jurusan.json"), "w", encoding="utf-8") as f:
        json.dump(majors, f, ensure_ascii=False, indent=2)
    print(f"[OK] Berhasil menyimpan {len(majors)} jurusan unik ke: data/daftar_jurusan.json")
    
    return agg

if __name__ == "__main__":
    process_academic_data()
