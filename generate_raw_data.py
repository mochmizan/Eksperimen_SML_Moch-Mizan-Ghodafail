"""
Jalankan script ini SEKALI untuk membuat wine_raw.csv di root repo.
  python generate_raw_data.py
"""
import pandas as pd
import urllib.request
import os

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
OUTPUT_FILE = "wine_raw.csv"

def generate_data():
    print(f"Mengunduh dataset dari {URL}...")
    try:
        # Baca langsung dari URL (delimiter di dataset wine UCI adalah semicolon ';')
        df = pd.read_csv(URL, sep=';')
        
        # Buat binary target 'quality_label'
        # Asumsi: >= 6 is 'good', < 6 is 'bad'
        df['quality_label'] = df['quality'].apply(lambda x: 'good' if x >= 6 else 'bad')
        
        # Hapus kolom quality asli agar sesuai dengan klasifikasi biner
        df.drop(columns=['quality'], inplace=True)
        
        # Standarisasi nama kolom agar tidak ada spasi (ganti dengan underscore)
        df.columns = [c.replace(' ', '_') for c in df.columns]
        
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"{OUTPUT_FILE} berhasil dibuat. Shape: {df.shape}")
        print("Sampel data:")
        print(df.head())
        
    except Exception as e:
        print(f"Gagal mengunduh dataset: {e}")

if __name__ == "__main__":
    generate_data()
