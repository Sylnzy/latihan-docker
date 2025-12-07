# 🚀 Prediksi API dengan Preprocessing Data Mentah

## 📖 Penjelasan Konsep

Pada latihan sebelumnya, kita menggunakan data yang **sudah diproses** (hasil PCA, normalisasi, dll). Namun di dunia nyata, **user tidak tahu** tentang preprocessing yang kita lakukan!

User hanya tahu data asli seperti:
- Age: 23
- Credit_Mix: "Good"
- Num_Bank_Accounts: 3
- dll.

**Masalah**: Model kita butuh data dalam format:
- Age: 0.714 (dinormalisasi)
- Credit_Mix: 1 (di-encode)
- pc1_1, pc1_2, ... (hasil PCA)

**Solusi**: Kita buat fungsi preprocessing yang mengubah data mentah → data siap prediksi!

---

## 📁 File yang Dibuat

### 1. `preprocessAPI.py`
**Fungsi utama:**
- `data_preprocessing(data)` - Melakukan preprocessing data mentah
- `prepare_payload(data_df)` - Mengubah DataFrame → JSON payload
- `prediction(data)` - Mengirim request ke API dan decode hasilnya
- `inference_pipeline(raw_data, columns)` - **Pipeline lengkap** dari data mentah → hasil prediksi

**Preprocessing yang dilakukan:**
1. **Label Encoding** untuk fitur kategorikal (Credit_Mix, Payment_of_Min_Amount, Payment_Behaviour)
2. **Normalisasi Age** (0-1)
3. **PCA Grup 1** (6 fitur → 5 komponen): Num_Bank_Accounts, Num_Credit_Card, Interest_Rate, dll.
4. **PCA Grup 2** (7 fitur → 2 komponen): Changed_Credit_Limit, Outstanding_Debt, Monthly_Balance, dll.
5. **Drop** Credit_History_Age
6. **Reorder** kolom sesuai signature model

### 2. `test_api_with_preprocessing.py`
Script testing dengan **3 test cases**:
- Test Case 1: Customer dengan Credit Mix "Good"
- Test Case 2: Customer dengan Credit Mix "Standard"
- Test Case 3: Customer dengan Credit Mix "Poor"

---

## 🎯 Cara Menggunakan

### Langkah 1: Pastikan MLflow Server Berjalan

**Terminal 1** - MLflow Tracking Server:
```bash
mlflow server --host 127.0.0.1 --port 5000
```

### Langkah 2: Serve Model sebagai API

**Terminal 2** - Model Serving:
```bash
cd "c:\Users\mhasa\Kuliah\dicoding\belajar\Membangun sistem machine learning\MLFlow\latihan\Latihan-CI\MLproject"
mlflow models serve -m "models:/credit-scoring/1" --port 5002 --no-conda
```

Tunggu sampai muncul:
```
[INFO] Listening at: http://127.0.0.1:5002
```

### Langkah 3: Jalankan Testing

**Terminal 3** - Testing:
```bash
cd "c:\Users\mhasa\Kuliah\dicoding\belajar\Membangun sistem machine learning\MLFlow\latihan\Latihan-CI\MLproject"
python test_api_with_preprocessing.py
```

---

## 📝 Contoh Penggunaan Manual

Jika Anda ingin membuat prediksi sendiri:

```python
from preprocessAPI import inference_pipeline

# Definisi kolom
columns = [
    "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", 
    "Age", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", 
    "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Outstanding_Debt", "Monthly_Inhand_Salary", 
    "Monthly_Balance", "Amount_invested_monthly", 
    "Total_EMI_per_month", "Credit_History_Age"
]

# Data mentah dari user (18 fitur)
data = [
    "Good",      # Credit_Mix
    "No",        # Payment_of_Min_Amount
    "Low_spent_Small_value_payments",  # Payment_Behaviour
    25,          # Age
    3,           # Num_Bank_Accounts
    4,           # Num_Credit_Card
    3,           # Interest_Rate
    4,           # Num_of_Loan
    3,           # Delay_from_due_date
    7,           # Num_of_Delayed_Payment
    11.27,       # Changed_Credit_Limit
    5,           # Num_Credit_Inquiries
    809.98,      # Outstanding_Debt
    1824.80,     # Monthly_Inhand_Salary
    186.26,      # Monthly_Balance
    236.64,      # Amount_invested_monthly
    49.50,       # Total_EMI_per_month
    216          # Credit_History_Age
]

# Jalankan prediksi (otomatis preprocessing + API call)
result = inference_pipeline(data, columns)
print(f"Hasil prediksi: {result[0]}")  # Output: "Good", "Standard", atau "Poor"
```

---

## 🔄 Alur Kerja (Workflow)

```
┌─────────────────────────────────────────────────────────────┐
│  USER INPUT (Data Mentah - 18 Fitur)                        │
│  Credit_Mix: "Good", Age: 23, Num_Bank_Accounts: 3, ...     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Konversi ke DataFrame                              │
│  pd.DataFrame([data], columns=columns)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Preprocessing (data_preprocessing)                 │
│  • Label Encoding (Credit_Mix, Payment_of_Min_Amount, ...)  │
│  • Normalisasi Age (0-1)                                    │
│  • PCA Grup 1: 6 fitur → 5 komponen (pc1_1 ... pc1_5)      │
│  • PCA Grup 2: 7 fitur → 2 komponen (pc2_1, pc2_2)         │
│  • Drop Credit_History_Age                                  │
│  Hasil: 11 fitur yang siap untuk model                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Prepare JSON Payload (prepare_payload)             │
│  Format: {"dataframe_split": {"columns": [...], ...}}       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Kirim ke API (prediction)                          │
│  POST http://127.0.0.1:5002/invocations                     │
│  Response: {"predictions": [0]} atau [1] atau [2]           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Decode Hasil                                       │
│  0 → "Good"                                                 │
│  1 → "Poor"                                                 │
│  2 → "Standard"                                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: Hasil Prediksi                                     │
│  "Good" / "Standard" / "Poor"                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 Poin Penting yang Dipelajari

### 1. **Konsistensi Preprocessing**
Preprocessing di **production** harus **sama persis** dengan **training**!

### 2. **User Experience**
User tidak perlu tahu tentang:
- PCA
- Label Encoding
- Normalisasi

User hanya input data mentah → sistem otomatis proses!

### 3. **API Integration**
- Model di-serve sebagai REST API (MLflow Models Serve)
- Aplikasi lain bisa akses via HTTP POST
- Format standar: JSON dengan `dataframe_split`

### 4. **Error Handling**
- Cek koneksi ke server
- Validasi response status code
- Decode hasil dengan benar

---

## 🔍 Troubleshooting

### Error: "Connection refused"
**Penyebab**: Model server belum berjalan

**Solusi**:
```bash
mlflow models serve -m "models:/credit-scoring/1" --port 5002 --no-conda
```

### Error: "Model not found"
**Penyebab**: Model belum didaftarkan di registry

**Solusi**:
1. Buka MLflow UI: http://127.0.0.1:5000
2. Pilih run yang diinginkan
3. Klik "Register Model"
4. Beri nama: `credit-scoring`

### Error: "Shape mismatch"
**Penyebab**: Preprocessing tidak konsisten

**Solusi**:
- Pastikan urutan kolom benar
- Cek apakah semua fitur ada
- Verifikasi hasil PCA (5+2 komponen)

---

## 📊 Struktur Data

### Input (Data Mentah - 18 Fitur):
```
Credit_Mix, Payment_of_Min_Amount, Payment_Behaviour,
Age, Num_Bank_Accounts, Num_Credit_Card, Interest_Rate,
Num_of_Loan, Delay_from_due_date, Num_of_Delayed_Payment,
Changed_Credit_Limit, Num_Credit_Inquiries, Outstanding_Debt,
Monthly_Inhand_Salary, Monthly_Balance, Amount_invested_monthly,
Total_EMI_per_month, Credit_History_Age
```

### Setelah Preprocessing (11 Fitur):
```
Age (normalized), Credit_Mix (encoded), 
Payment_of_Min_Amount (encoded), Payment_Behaviour (encoded),
pc1_1, pc1_2, pc1_3, pc1_4, pc1_5,  # PCA grup 1
pc2_1, pc2_2  # PCA grup 2
```

### Output:
```
"Good" / "Standard" / "Poor"
```

---

## 🎯 Next Steps

1. ✅ Testing dengan 3 test cases
2. ✅ Verifikasi preprocessing konsisten
3. ✅ Test API endpoint
4. 🔜 Integrasi dengan aplikasi web (FastAPI/Flask)
5. 🔜 Deployment ke production

---

## 📚 Resources

- MLflow Documentation: https://mlflow.org/docs/latest/
- Scikit-learn PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- Python Requests: https://docs.python-requests.org/

---

**Selamat! Anda sudah menguasai cara mengintegrasikan preprocessing dengan API endpoint!** 🎉
