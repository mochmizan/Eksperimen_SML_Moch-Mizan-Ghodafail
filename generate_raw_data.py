"""
Jalankan script ini SEKALI untuk membuat breast_cancer_raw.csv di root repo.
  python generate_raw_data.py
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

OUTPUT_FILE = "breast_cancer_raw.csv"

def generate_data():
    print("Memuat Breast Cancer Wisconsin dataset dari scikit-learn...")
    try:
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        
        # Buat target string agar lebih informatif (malignant/benign)
        # di sklearn, 0 = malignant, 1 = benign
        df['target'] = pd.Series(data.target).map({0: 'malignant', 1: 'benign'})
        
        # Standarisasi nama kolom agar tidak ada spasi (ganti dengan underscore)
        df.columns = [c.replace(' ', '_') for c in df.columns]
        
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"{OUTPUT_FILE} berhasil dibuat. Shape: {df.shape}")
        print("Sampel data:")
        print(df.head())
        
    except Exception as e:
        print(f"Gagal memuat dataset: {e}")

if __name__ == "__main__":
    generate_data()
