import pandas as pd
import hashlib
import os

def anonymize_dataset(input_csv="nilai_mhs_IP IPK.csv", output_dir="data", sample_size=10000):
    """
    Menghapus dan menganonimkan data rahasia (PII: NIM dan Nama Mahasiswa):
    1. Kolom 'nama' dihapus sepenuhnya.
    2. Kolom 'nim' dimasking / disamarkan menjadi ID anonim (contoh: MHS_000001)
       secara konsisten sehingga relasi data tetap valid tanpa membocorkan identitas.
    3. Menghasilkan sampel bersih dan aman untuk pengumpulan tugas dan GitHub.
    """
    print(f"[*] Membaca data untuk proses anonimisasi dari {input_csv}...")
    if not os.path.exists(input_csv):
        print(f"[!] File {input_csv} tidak ditemukan.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Baca kolom-kolom yang ada
    df = pd.read_csv(input_csv, sep=";", dtype=str)
    print(f"[*] Total baris data asli: {len(df):,}")
    print(f"[*] Kolom asli: {df.columns.tolist()}")
    
    # 1. Hapus kolom identitas pribadi (Nama)
    if "nama" in df.columns:
        df = df.drop(columns=["nama"])
        print("[OK] Kolom 'nama' berhasil dihapus (100% anonymized).")
        
    # 2. Samarkan NIM menjadi ID Anonim (MHS_000001, dst)
    if "nim" in df.columns:
        unique_nims = df["nim"].dropna().unique()
        nim_to_anon = {nim: f"MHS_{i+1:06d}" for i, nim in enumerate(unique_nims)}
        df["nim_anonim"] = df["nim"].map(nim_to_anon).fillna("MHS_UNKNOWN")
        df = df.drop(columns=["nim"])
        
        # Pindahkan nim_anonim ke kolom pertama
        cols = ["nim_anonim"] + [c for c in df.columns if c != "nim_anonim"]
        df = df[cols]
        print(f"[OK] {len(unique_nims):,} NIM unik berhasil disamarkan menjadi kode anonim (MHS_xxxxxx).")
        
    # Bersihkan whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
        
    # Simpan sampel bersih dan aman (10.000 baris) untuk pengumpulan tugas / GitHub
    sample_path = os.path.join(output_dir, "nilai_mhs_anonim_sample.csv")
    df_sample = df.head(sample_size)
    df_sample.to_csv(sample_path, sep=";", index=False)
    print(f"[OK] Sampel bersih tersimpan di: {sample_path} ({len(df_sample):,} baris, bebas data rahasia).")
    
    # Simpan informasi metadata kolom
    print(f"[*] Kolom akhir setelah anonimisasi: {df.columns.tolist()}")
    return df

if __name__ == "__main__":
    anonymize_dataset()
