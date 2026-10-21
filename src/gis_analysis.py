import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


data = {
    'IPK': [3.8, 2.9, 3.5, 2.2, 3.9, 3.1, 2.7, 3.6, 3.4, 2.5],
    'Jam_Belajar': [25, 10, 20, 5, 30, 15, 8, 22, 18, 12],
    'Keaktifan_Organisasi': [8, 4, 7, 2, 9, 6, 3, 8, 7, 4],
    'Layak_Beasiswa': [1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df[['IPK', 'Jam_Belajar', 'Keaktifan_Organisasi']]
y = df['Layak_Beasiswa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

print("=== HASIL MODEL MACHINE LEARNING (STUDI KASUS 4) ===")
print(f"Akurasi Model pada Data Uji: {akurasi * 100:.2f}%\n")

mahasiswa_baru = pd.DataFrame({
    'IPK': [3.7],
    'Jam_Belajar': [24],
    'Keaktifan_Organisasi': [8]
})

prediksi_baru = model.predict(mahasiswa_baru)
status = "LAYAK Menerima Beasiswa" if prediksi_baru[0] == 1 else "TIDAK LAYAK Menerima Beasiswa"
print(f"Prediksi untuk Mahasiswa Baru (IPK 3.7, Belajar 24 jam, Organisasi 8):")
print(f"Status: {status}")