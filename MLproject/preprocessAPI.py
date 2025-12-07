"""
Script untuk preprocessing data mentah sebelum dikirim ke API
Fungsi helper untuk mengolah data input user menjadi format yang sesuai dengan model
"""

import pandas as pd
import numpy as np
import requests
import json
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


def data_preprocessing(data):
    """
    Melakukan preprocessing pada data mentah sebelum prediksi
    CATATAN: Untuk production, seharusnya load model PCA/Scaler yang sudah di-fit saat training
    Karena tidak ada file tersimpan, kita gunakan transformasi sederhana
    
    Args:
        data (Pandas DataFrame): DataFrame dengan data mentah dari user
        
    Returns:
        Pandas DataFrame: Data yang sudah diproses dan siap untuk prediksi
    """
    
    print("\n[PREPROCESSING] Memulai preprocessing data...")
    print("  ⚠️  PERINGATAN: Menggunakan transformasi simplified (tanpa PCA fitted)")
    print("     Untuk hasil optimal, load PCA/Scaler yang sudah di-fit saat training")
    
    # 1. Label Encoding untuk fitur kategorikal
    print("  1. Label Encoding untuk fitur kategorikal...")
    categorical_features = ['Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
    
    # Mapping manual berdasarkan training data
    credit_mix_mapping = {'Good': 1, 'Standard': 2, 'Bad': 3}
    payment_min_mapping = {'No': 0, 'Yes': 1, 'NM': 2}
    payment_behaviour_mapping = {
        'Low_spent_Small_value_payments': 0,
        'High_spent_Large_value_payments': 1,
        'High_spent_Medium_value_payments': 2,
        'Low_spent_Medium_value_payments': 3,
        'Low_spent_Large_value_payments': 4,
        '!@9#%8': 5  # Unknown/invalid
    }
    
    if 'Credit_Mix' in data.columns:
        data['Credit_Mix'] = data['Credit_Mix'].map(credit_mix_mapping).fillna(1)
    if 'Payment_of_Min_Amount' in data.columns:
        data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].map(payment_min_mapping).fillna(0)
    if 'Payment_Behaviour' in data.columns:
        data['Payment_Behaviour'] = data['Payment_Behaviour'].map(payment_behaviour_mapping).fillna(0)
    
    # 2. Normalisasi Age (estimasi range: 18-80)
    print("  2. Normalisasi Age (0-1)...")
    if 'Age' in data.columns:
        # Gunakan range fixed untuk konsistensi
        age_min, age_max = 18, 80
        data['Age'] = (data['Age'] - age_min) / (age_max - age_min)
        data['Age'] = data['Age'].clip(0, 1)  # Pastikan dalam range [0, 1]
    
    # 3. Standardisasi dan dimensi reduction untuk grup fitur 1
    print("  3. Transformasi grup fitur 1 (simplified)...")
    pca_features_1 = [
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
        'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment'
    ]
    
    if all(col in data.columns for col in pca_features_1):
        # Standardisasi sederhana (Z-score dengan mean/std estimasi)
        # Nilai-nilai ini harus diganti dengan mean/std dari training data yang sebenarnya
        means_1 = [4, 5, 8, 5, 15, 12]  # Estimasi mean
        stds_1 = [2, 2, 4, 3, 10, 8]    # Estimasi std
        
        for i, col in enumerate(pca_features_1):
            data[col] = (data[col] - means_1[i]) / stds_1[i]
        
        # Simulasi PCA dengan kombinasi linear sederhana
        data['pc1_1'] = -0.4 * data['Num_Bank_Accounts'] + 0.2 * data['Interest_Rate']
        data['pc1_2'] = 0.3 * data['Num_Credit_Card'] - 0.1 * data['Num_of_Loan']
        data['pc1_3'] = 0.1 * data['Delay_from_due_date'] + 0.05 * data['Num_of_Delayed_Payment']
        data['pc1_4'] = -0.2 * data['Interest_Rate'] + 0.1 * data['Num_of_Loan']
        data['pc1_5'] = 0.1 * data['Num_Bank_Accounts'] + 0.05 * data['Num_Credit_Card']
        
        # Drop kolom asli
        data = data.drop(columns=pca_features_1)
    
    # 4. Standardisasi dan dimensi reduction untuk grup fitur 2
    print("  4. Transformasi grup fitur 2 (simplified)...")
    pca_features_2 = [
        'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt',
        'Monthly_Inhand_Salary', 'Monthly_Balance', 'Amount_invested_monthly',
        'Total_EMI_per_month'
    ]
    
    if all(col in data.columns for col in pca_features_2):
        # Standardisasi sederhana
        means_2 = [20, 6, 2000, 3000, 300, 300, 100]  # Estimasi mean
        stds_2 = [15, 4, 1500, 2000, 400, 250, 80]    # Estimasi std
        
        for i, col in enumerate(pca_features_2):
            data[col] = (data[col] - means_2[i]) / stds_2[i]
        
        # Simulasi PCA dengan kombinasi linear sederhana
        data['pc2_1'] = 0.3 * data['Outstanding_Debt'] + 0.2 * data['Monthly_Balance']
        data['pc2_2'] = -0.1 * data['Changed_Credit_Limit'] + 0.05 * data['Total_EMI_per_month']
        
        # Drop kolom asli
        data = data.drop(columns=pca_features_2)
    
    # 5. Drop Credit_History_Age (jika ada)
    if 'Credit_History_Age' in data.columns:
        data = data.drop(columns=['Credit_History_Age'])
    
    # 6. Reorder kolom sesuai dengan urutan yang diharapkan model
    print("  5. Menyusun ulang kolom sesuai signature model...")
    expected_columns = [
        "Age", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour",
        "pc1_1", "pc1_2", "pc1_3", "pc1_4", "pc1_5",
        "pc2_1", "pc2_2"
    ]
    
    # Pastikan semua kolom ada
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0  # Isi dengan 0 jika kolom tidak ada
    
    # Pilih hanya kolom yang dibutuhkan dengan urutan yang benar
    data = data[expected_columns]
    
    print("✓ Preprocessing selesai!")
    print(f"  Shape data hasil: {data.shape}")
    print(f"  Kolom: {list(data.columns)}")
    
    return data


def prediction(data):
    """
    Melakukan prediksi menggunakan API endpoint
    
    Args:
        data (str): Data dalam format JSON string
        
    Returns:
        array: Hasil prediksi (Good, Standard, atau Poor)
    """
    
    print("\n[PREDIKSI] Mengirim request ke API...")
    
    # URL endpoint dari model yang sedang di-serve
    url = "http://127.0.0.1:5004/invocations"
    headers = {"Content-Type": "application/json"}
    
    try:
        # Kirim POST request
        response = requests.post(url, data=data, headers=headers)
        
        if response.status_code == 200:
            print("✓ Request berhasil!")
            
            # Parse response
            response_json = response.json()
            
            # Ambil predictions
            if isinstance(response_json, dict) and "predictions" in response_json:
                predictions = response_json["predictions"]
            else:
                predictions = response_json
            
            print(f"  Raw predictions: {predictions}")
            
            # Decode hasil prediksi
            # 0 = Good, 1 = Poor, 2 = Standard (sesuai dengan LabelEncoder)
            label_mapping = {0: "Good", 1: "Poor", 2: "Standard"}
            
            # Konversi angka ke label
            final_result = [label_mapping.get(pred, f"Unknown({pred})") for pred in predictions]
            
            print("✓ Prediksi selesai!")
            
            return final_result
            
        else:
            print(f"✗ Request gagal dengan status code: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("✗ Gagal terhubung ke server!")
        print("  Pastikan model server berjalan di http://127.0.0.1:5002")
        print("\n  Jalankan command ini di terminal terpisah:")
        print('  mlflow models serve -m "models:/credit-scoring/1" --port 5002 --no-conda')
        return None
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None


def prepare_payload(data_df):
    """
    Mengubah DataFrame menjadi JSON payload untuk API
    
    Args:
        data_df (Pandas DataFrame): DataFrame yang sudah diproses
        
    Returns:
        str: JSON string dalam format dataframe_split
    """
    
    print("\n[PAYLOAD] Membuat JSON payload...")
    
    # Konversi DataFrame ke format JSON yang diinginkan
    json_output = {
        "dataframe_split": {
            "columns": data_df.columns.tolist(),
            "data": data_df.values.tolist()
        }
    }
    
    # Konversi ke JSON string
    data_json = json.dumps(json_output)
    
    print("✓ Payload berhasil dibuat!")
    print(f"  Jumlah sample: {len(data_df)}")
    print(f"  Jumlah fitur: {len(data_df.columns)}")
    
    return data_json


# Fungsi utama untuk end-to-end inference
def inference_pipeline(raw_data, columns):
    """
    Pipeline lengkap dari data mentah hingga prediksi
    
    Args:
        raw_data (list): Data mentah dari user dalam bentuk list
        columns (list): Nama-nama kolom yang sesuai dengan raw_data
        
    Returns:
        array: Hasil prediksi
    """
    
    print("="*60)
    print("INFERENCE PIPELINE - CREDIT SCORING")
    print("="*60)
    
    # 1. Konversi data ke DataFrame
    print("\n[STEP 1] Konversi data ke DataFrame...")
    df = pd.DataFrame([raw_data], columns=columns)
    print("✓ Data berhasil dikonversi")
    print(f"  Shape: {df.shape}")
    
    # 2. Preprocessing
    print("\n[STEP 2] Preprocessing data...")
    processed_data = data_preprocessing(data=df)
    
    # 3. Prepare payload
    print("\n[STEP 3] Prepare JSON payload...")
    payload = prepare_payload(processed_data)
    
    # 4. Prediksi
    print("\n[STEP 4] Melakukan prediksi...")
    result = prediction(payload)
    
    print("\n" + "="*60)
    print("HASIL AKHIR")
    print("="*60)
    
    if result:
        for i, pred in enumerate(result):
            print(f"  Sample {i+1}: {pred}")
    else:
        print("  Prediksi gagal!")
    
    print("="*60)
    
    return result


# Contoh penggunaan
if __name__ == "__main__":
    
    # Definisi kolom (urutan harus sesuai dengan data)
    columns = [
        "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", 
        "Age", "Num_Bank_Accounts", "Num_Credit_Card",
        "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", 
        "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Outstanding_Debt", "Monthly_Inhand_Salary", 
        "Monthly_Balance", "Amount_invested_monthly", 
        "Total_EMI_per_month", "Credit_History_Age"
    ]
    
    # Contoh data mentah dari user
    data = [
        "Good",           # Credit_Mix
        "No",             # Payment_of_Min_Amount
        "Low_spent_Small_value_payments",  # Payment_Behaviour
        23,               # Age
        3,                # Num_Bank_Accounts
        4,                # Num_Credit_Card
        3,                # Interest_Rate
        4,                # Num_of_Loan
        3,                # Delay_from_due_date
        7,                # Num_of_Delayed_Payment
        11.27,            # Changed_Credit_Limit
        5,                # Num_Credit_Inquiries
        809.98,           # Outstanding_Debt
        1824.80,          # Monthly_Inhand_Salary
        186.26,           # Monthly_Balance
        236.64,           # Amount_invested_monthly
        49.50,            # Total_EMI_per_month
        216               # Credit_History_Age
    ]
    
    # Jalankan inference pipeline
    result = inference_pipeline(data, columns)
