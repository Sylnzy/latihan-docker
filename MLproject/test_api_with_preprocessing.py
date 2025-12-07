"""
Testing API dengan Data Mentah + Preprocessing
Script untuk menguji endpoint API model dengan data mentah yang belum diproses
"""

import pandas as pd
from preprocessAPI import inference_pipeline

print("="*70)
print("TESTING API DENGAN DATA MENTAH")
print("Credit Scoring Model - MLflow REST API")
print("="*70)

# Definisi kolom input (data mentah dari user)
columns = [
    "Credit_Mix", 
    "Payment_of_Min_Amount", 
    "Payment_Behaviour", 
    "Age", 
    "Num_Bank_Accounts", 
    "Num_Credit_Card",
    "Interest_Rate", 
    "Num_of_Loan", 
    "Delay_from_due_date", 
    "Num_of_Delayed_Payment", 
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries", 
    "Outstanding_Debt", 
    "Monthly_Inhand_Salary", 
    "Monthly_Balance", 
    "Amount_invested_monthly", 
    "Total_EMI_per_month", 
    "Credit_History_Age"
]

print("\n📋 INFORMASI INPUT DATA")
print("-"*70)
print(f"Jumlah fitur: {len(columns)}")
print(f"Fitur kategorkal: Credit_Mix, Payment_of_Min_Amount, Payment_Behaviour")
print(f"Fitur numerik: {len(columns) - 3}")

# ============================================
# TEST CASE 1: Data dengan Credit Mix "Good"
# ============================================
print("\n\n" + "="*70)
print("TEST CASE 1: Customer dengan Credit Mix 'Good'")
print("="*70)

data_1 = [
    "Good",                                # Credit_Mix
    "No",                                  # Payment_of_Min_Amount
    "Low_spent_Small_value_payments",      # Payment_Behaviour
    23,                                    # Age
    3,                                     # Num_Bank_Accounts
    4,                                     # Num_Credit_Card
    3,                                     # Interest_Rate
    4,                                     # Num_of_Loan
    3,                                     # Delay_from_due_date
    7,                                     # Num_of_Delayed_Payment
    11.27,                                 # Changed_Credit_Limit
    5,                                     # Num_Credit_Inquiries
    809.98,                                # Outstanding_Debt
    1824.80,                               # Monthly_Inhand_Salary
    186.26,                                # Monthly_Balance
    236.64,                                # Amount_invested_monthly
    49.50,                                 # Total_EMI_per_month
    216                                    # Credit_History_Age
]

print("\n📊 Data Input:")
print("-"*70)
for col, val in zip(columns, data_1):
    print(f"  {col:30s}: {val}")

# Jalankan prediksi
result_1 = inference_pipeline(data_1, columns)

# ============================================
# TEST CASE 2: Data dengan Credit Mix "Standard"
# ============================================
print("\n\n" + "="*70)
print("TEST CASE 2: Customer dengan Credit Mix 'Standard'")
print("="*70)

data_2 = [
    "Standard",                            # Credit_Mix
    "Yes",                                 # Payment_of_Min_Amount
    "High_spent_Large_value_payments",     # Payment_Behaviour
    35,                                    # Age
    5,                                     # Num_Bank_Accounts
    6,                                     # Num_Credit_Card
    8,                                     # Interest_Rate
    6,                                     # Num_of_Loan
    10,                                    # Delay_from_due_date
    15,                                    # Num_of_Delayed_Payment
    25.50,                                 # Changed_Credit_Limit
    8,                                     # Num_Credit_Inquiries
    1500.00,                               # Outstanding_Debt
    3500.00,                               # Monthly_Inhand_Salary
    500.00,                                # Monthly_Balance
    400.00,                                # Amount_invested_monthly
    150.00,                                # Total_EMI_per_month
    300                                    # Credit_History_Age
]

print("\n📊 Data Input:")
print("-"*70)
for col, val in zip(columns, data_2):
    print(f"  {col:30s}: {val}")

# Jalankan prediksi
result_2 = inference_pipeline(data_2, columns)

# ============================================
# TEST CASE 3: Data dengan Credit Mix "Poor"
# ============================================
print("\n\n" + "="*70)
print("TEST CASE 3: Customer dengan Credit Mix 'Poor'")
print("="*70)

data_3 = [
    "Bad",                                 # Credit_Mix
    "NM",                                  # Payment_of_Min_Amount (Not Mentioned)
    "!@9#%8",                              # Payment_Behaviour (unusual)
    45,                                    # Age
    2,                                     # Num_Bank_Accounts
    2,                                     # Num_Credit_Card
    15,                                    # Interest_Rate (high)
    8,                                     # Num_of_Loan (many loans)
    25,                                    # Delay_from_due_date (very late)
    30,                                    # Num_of_Delayed_Payment (many delays)
    5.00,                                  # Changed_Credit_Limit (low)
    15,                                    # Num_Credit_Inquiries (high)
    5000.00,                               # Outstanding_Debt (high)
    2000.00,                               # Monthly_Inhand_Salary
    50.00,                                 # Monthly_Balance (low)
    50.00,                                 # Amount_invested_monthly (low)
    300.00,                                # Total_EMI_per_month (high)
    120                                    # Credit_History_Age
]

print("\n📊 Data Input:")
print("-"*70)
for col, val in zip(columns, data_3):
    print(f"  {col:30s}: {val}")

# Jalankan prediksi
result_3 = inference_pipeline(data_3, columns)

# ============================================
# RINGKASAN HASIL
# ============================================
print("\n\n" + "="*70)
print("📊 RINGKASAN HASIL TESTING")
print("="*70)

results = [
    ("Test Case 1 (Good Credit Mix)", result_1),
    ("Test Case 2 (Standard Credit Mix)", result_2),
    ("Test Case 3 (Poor Credit Mix)", result_3)
]

for test_name, result in results:
    if result:
        print(f"\n{test_name}:")
        print(f"  Prediksi: {result[0]}")
    else:
        print(f"\n{test_name}:")
        print(f"  Prediksi: GAGAL")

print("\n" + "="*70)
print("✓ Testing selesai!")
print("="*70)

print("\n💡 CATATAN:")
print("-"*70)
print("• Model menerima data MENTAH (belum diproses)")
print("• Preprocessing dilakukan otomatis oleh preprocessAPI.py")
print("• Data ditransformasi dengan:")
print("  - Label Encoding untuk fitur kategorikal")
print("  - Normalisasi Age (0-1)")
print("  - PCA untuk reduksi dimensi fitur numerik")
print("• Hasil prediksi: Good, Standard, atau Poor")
print("="*70)
