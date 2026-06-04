import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_california_housing(file_id='13CH1nErq2Grw691XgqPLidTwwqLgOZtV'):
    """
    Fungsi otomatis untuk memuat, membersihkan, dan mentransformasi 
    dataset California Housing Prices agar siap dilatih oleh model Machine Learning.
    
    Parameter:
    - file_id (str): ID file Google Drive tempat dataset berada.
    
    Return:
    - X_train, X_test, y_train, y_test: Dataset yang sudah dipisah dan diskalakan.
    """
    
    # ==========================================
    # Fase 0: Memuat Data
    # ==========================================
    url = f'https://drive.google.com/uc?id={file_id}'
    print("Mengunduh dataset dari Google Drive...")
    df = pd.read_csv(url)
    
    # ==========================================
    # Tahap 1: Menghapus Data Duplikat
    # ==========================================
    df = df.drop_duplicates()
    
    # ==========================================
    # Tahap 2: Deteksi dan Penanganan Outlier
    # ==========================================
    # Menghapus harga rumah yang disensor pada batas atas ($500,001)
    df = df[df['median_house_value'] < 500000]

    # Memfilter outlier pada 'median_income' menggunakan metode IQR
    Q1 = df['median_income'].quantile(0.25)
    Q3 = df['median_income'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df['median_income'] >= lower_bound) & (df['median_income'] <= upper_bound)]
    
    # ==========================================
    # Tahap 3: Menangani Data Kosong (Missing Values)
    # ==========================================
    # Mengisi nilai NaN di 'total_bedrooms' dengan nilai median
    median_bedrooms = df['total_bedrooms'].median()
    df['total_bedrooms'] = df['total_bedrooms'].fillna(median_bedrooms)

    # ==========================================
    # Tahap 4: Encoding Data Kategorikal
    # ==========================================
    # Mengubah teks 'ocean_proximity' menjadi variabel biner numerik
    df = pd.get_dummies(df, columns=['ocean_proximity'], dtype=int)

    # ==========================================
    # Persiapan Pemisahan Data
    # ==========================================
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    # Pemisahan data latih (80%) dan data uji (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==========================================
    # Tahap 5: Normalisasi / Standarisasi Fitur
    # ==========================================
    # Memilih fitur numerik asli (tidak termasuk hasil encoding)
    num_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                'total_bedrooms', 'population', 'households', 'median_income']

    scaler = StandardScaler()

    # Fit & transform pada data training, transform saja pada data testing
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    print("Preprocessing selesai! Data siap untuk dilatih.")
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, train_path='train_processed.csv', test_path='test_processed.csv'):
    """
    Fungsi opsional untuk menyimpan data yang sudah diproses ke dalam format CSV.
    """
    train_processed = X_train.copy()
    train_processed['median_house_value'] = y_train
    
    test_processed = X_test.copy()
    test_processed['median_house_value'] = y_test
    
    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)
    print(f"Data berhasil disimpan di {train_path} dan {test_path}")

# ==========================================
# Blok Eksekusi Utama (Bisa digunakan untuk testing skrip)
# ==========================================
if __name__ == "__main__":
    # 1. Jalankan fungsi preprocessing
    X_train, X_test, y_train, y_test = preprocess_california_housing()
    
    # 2. Periksa dimensi hasil kembalian data
    print(f"Dimensi X_train: {X_train.shape}")
    print(f"Dimensi X_test: {X_test.shape}")
    
    # 3. (Opsional) Simpan ke file lokal
    save_processed_data(X_train, X_test, y_train, y_test)