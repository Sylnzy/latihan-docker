"""
Script untuk melakukan validasi dan prediksi model menggunakan run_id
Latihan: Prediksi Melalui run_id
"""

import mlflow
import pandas as pd
import json
from mlflow.models import validate_serving_input

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# TODO: Ganti dengan run_id dari model yang sudah dilatih
# Untuk mendapatkan run_id:
# 1. Buka MLflow UI di http://127.0.0.1:5000
# 2. Pilih experiment "Latihan Credit Scoring"
# 3. Klik pada run yang diinginkan
# 4. Copy run_id dari halaman detail
run_id = "313ddd611aed4ad3b4aa533582fd65f9"  # Ganti dengan run_id Anda

# Membuat URI model
model_uri = f'runs:/{run_id}/model'

print("="*50)
print("LATIHAN PREDIKSI MENGGUNAKAN RUN_ID")
print("="*50)

# ============================================
# OPSI 1: Menggunakan Data Manual (Tanpa File)
# ============================================
print("\n[OPSI 1] Membuat data input manual...")

# Format data input manual
data = {
    "columns": [
        "Age",
        "Credit_Mix",
        "Payment_of_Min_Amount",
        "Payment_Behaviour",
        "pc1_1",
        "pc1_2",
        "pc1_3",
        "pc1_4",
        "pc1_5",
        "pc2_1",
        "pc2_2"
    ],
    "data": [
        [
            0.7142857142857142,
            1,
            1,
            3,
            -0.4381534490735855,
            0.1711382783346808,
            0.0773630019922211,
            -0.0401910461904993,
            0.049590092121234,
            -0.1448249280763024,
            -0.0606673847105827
        ],
        [
            0.4523809523809524,
            2,
            2,
            5,
            0.4778277065736656,
            -0.1050643177401745,
            -0.185971337955209,
            -0.3789896489656689,
            0.1718128833126148,
            -0.2417658875286998,
            0.0066502389514661
        ],
        [
            0.4999999999999999,
            3,
            1,
            1,
            -0.2172441359177029,
            0.0068230171993729,
            0.0319554863404481,
            -0.0402970285419156,
            0.0861914654821804,
            0.779882880094656,
            0.1309092693530689
        ],
        [
            0.4523809523809524,
            1,
            1,
            6,
            -0.6893954147621222,
            0.1842207645520866,
            0.1887210184373613,
            -0.1923419820570049,
            0.0561499495340574,
            0.5840348959263311,
            0.0673148184570155
        ],
        [
            0.8333333333333333,
            3,
            0,
            5,
            -0.287145204288708,
            -0.2462885871184559,
            0.1247759677401836,
            -0.0544947505767573,
            0.0941141473487672,
            0.0526512725593595,
            -0.1827263807556981
        ]
    ]
}

# Mengubah format data menjadi format yang diterima oleh model
# Model menggunakan signature "dataframe_split" bukan "inputs"
input_data = {"dataframe_split": data}

print("✓ Data input siap")
print(f"  Jumlah kolom: {len(data['columns'])}")
print(f"  Jumlah sample: {len(data['data'])}")

# ============================================
# VALIDASI MODEL
# ============================================
print("\n[VALIDASI] Memvalidasi input dengan model...")
try:
    validation_result = validate_serving_input(model_uri, input_data)
    print("✓ Validasi berhasil! Model siap menerima input.")
    if validation_result:
        print(f"  Hasil validasi: {validation_result}")
except Exception as e:
    print(f"✗ Validasi gagal: {str(e)}")
    print("\nPastikan:")
    print("  1. MLflow tracking server berjalan di http://127.0.0.1:5000")
    print("  2. run_id sudah benar")
    print("  3. Format data sesuai dengan signature model")
    exit(1)

# ============================================
# PREDIKSI
# ============================================
print("\n[PREDIKSI] Memuat model dan melakukan prediksi...")
try:
    # Memuat model dari MLflow
    model = mlflow.pyfunc.load_model(model_uri)
    print("✓ Model berhasil dimuat")
    
    # Mengubah data menjadi DataFrame
    df = pd.DataFrame(
        input_data["dataframe_split"]["data"], 
        columns=input_data["dataframe_split"]["columns"]
    )
    
    print(f"✓ Data dikonversi ke DataFrame")
    print(f"  Shape: {df.shape}")
    
    # Melakukan prediksi
    predictions = model.predict(df)
    
    print("\n" + "="*50)
    print("HASIL PREDIKSI")
    print("="*50)
    print(predictions)
    
    # Menampilkan prediksi per sample
    print("\nDetail prediksi per sample:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i+1}: {pred}")
    
    print("\n✓ Prediksi selesai!")
    
except Exception as e:
    print(f"✗ Prediksi gagal: {str(e)}")
    exit(1)

# ============================================
# OPSI 2: Menggunakan Artefak dari MLflow
# ============================================
print("\n" + "="*50)
print("[OPSI 2] Prediksi menggunakan artefak MLflow")
print("="*50)

try:
    # Mencoba memuat serving_input_example dari artefak
    artifact_uri = f'runs:/{run_id}/model/serving_input_example.json'
    print(f"\nMencoba memuat artefak: {artifact_uri}")
    
    # Download artifact
    local_path = mlflow.artifacts.download_artifacts(artifact_uri)
    
    with open(local_path, 'r') as f:
        artifact_data = json.load(f)
    
    print("✓ Artefak berhasil dimuat")
    
    # Validasi dengan artefak
    validation_result = validate_serving_input(model_uri, artifact_data)
    print("✓ Validasi dengan artefak berhasil")
    
    # Prediksi dengan artefak
    if "dataframe_split" in artifact_data:
        df_artifact = pd.DataFrame(
            artifact_data["dataframe_split"]["data"],
            columns=artifact_data["dataframe_split"]["columns"]
        )
        predictions_artifact = model.predict(df_artifact)
        
        print("\nHasil prediksi dari artefak:")
        print(predictions_artifact)
    
except Exception as e:
    print(f"ℹ Artefak tidak tersedia atau error: {str(e)}")
    print("  (Ini normal jika model tidak menyimpan serving_input_example)")

print("\n" + "="*50)
print("LATIHAN SELESAI")
print("="*50)
print("\nKesimpulan:")
print("✓ Validasi input sebelum prediksi penting untuk memastikan kualitas")
print("✓ Prediksi dengan run_id mudah untuk testing & development")
print("✓ Untuk production, gunakan MLflow Model Serving atau API")
