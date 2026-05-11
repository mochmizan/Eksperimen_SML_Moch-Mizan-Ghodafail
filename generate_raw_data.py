"""
Jalankan script ini SEKALI untuk membuat iris_raw.csv di root repo.
  python generate_raw_data.py
"""
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[t] for t in iris.target]
df.to_csv('iris_raw.csv', index=False)
print(f"iris_raw.csv berhasil dibuat. Shape: {df.shape}")
print(df.head())
