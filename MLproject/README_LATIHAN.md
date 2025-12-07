# Panduan Latihan: Prediksi Melalui run_id

## Penjelasan Singkat

Latihan ini mengajarkan cara melakukan **validasi** dan **prediksi** model machine learning yang sudah tersimpan di MLflow menggunakan **run_id**. Ini adalah cara tercepat untuk testing model sebelum deployment production.

---

## Langkah-Langkah Pengerjaan

### 1. Persiapan - Jalankan MLflow Tracking Server

Pertama, pastikan MLflow tracking server berjalan:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Biarkan terminal ini tetap berjalan. Buka terminal baru untuk langkah selanjutnya.

---

### 2. Latih Model (Jika Belum Ada)

Jika belum punya model di MLflow, latih dulu dengan file `modelling.py`:

```bash
cd MLproject
python modelling.py
```

Setelah selesai, buka browser ke `http://127.0.0.1:5000` untuk melihat MLflow UI.

---

### 3. Dapatkan run_id

1. Buka MLflow UI di browser: `http://127.0.0.1:5000`
2. Klik experiment **"Latihan Credit Scoring"**
3. Klik salah satu run yang sudah berhasil
4. Copy **run_id** yang ada di halaman detail (contoh: `a1b2c3d4e5f6g7h8`)

---

### 4. Edit File Prediksi

Buka file `predict_with_runid.py` dan ganti baris ini:

```python
run_id = "YOUR_RUN_ID_HERE"  # Ganti dengan run_id Anda
```

Menjadi (contoh):

```python
run_id = "a1b2c3d4e5f6g7h8"  # run_id yang sudah di-copy
```

---

### 5. Jalankan Prediksi

```bash
python predict_with_runid.py
```

---

## Output yang Diharapkan

Jika berhasil, Anda akan melihat:

```
==================================================
LATIHAN PREDIKSI MENGGUNAKAN RUN_ID
==================================================

[OPSI 1] Membuat data input manual...
✓ Data input siap
  Jumlah kolom: 11
  Jumlah sample: 5

[VALIDASI] Memvalidasi input dengan model...
✓ Validasi berhasil! Model siap menerima input.

[PREDIKSI] Memuat model dan melakukan prediksi...
✓ Model berhasil dimuat
✓ Data dikonversi ke DataFrame
  Shape: (5, 11)

==================================================
HASIL PREDIKSI
==================================================
[0 1 2 0 1]  # Contoh hasil

Detail prediksi per sample:
  Sample 1: 0
  Sample 2: 1
  Sample 3: 2
  Sample 4: 0
  Sample 5: 1

✓ Prediksi selesai!
```

---

## Penjelasan Kode

### A. Validasi Model

```python
validate_serving_input(model_uri, input_data)
```

**Fungsi**: Memastikan format input sesuai dengan yang diharapkan model
- Mengecek struktur data
- Mengecek tipe data
- Mencegah error saat prediksi

### B. Load Model

```python
model = mlflow.pyfunc.load_model(model_uri)
```

**Fungsi**: Memuat model dari MLflow menggunakan run_id

### C. Prediksi

```python
predictions = model.predict(df)
```

**Fungsi**: Melakukan prediksi dengan data yang sudah divalidasi

---

## Struktur Data

Model ini menggunakan signature **"dataframe_split"**, bukan "inputs":

```json
{
  "dataframe_split": {
    "columns": ["Age", "Credit_Mix", ...],
    "data": [[0.714, 1, 1, 3, ...], ...]
  }
}
```

File `inputdata.json` sudah berisi sample data dengan format yang benar.

---

## Troubleshooting

### Error: "Connection refused"
- Pastikan MLflow server berjalan di port 5000
- Cek dengan: `netstat -ano | findstr :5000`

### Error: "Run not found"
- run_id salah atau tidak ada
- Cek lagi di MLflow UI

### Error: "Signature mismatch"
- Format data tidak sesuai
- Pastikan menggunakan "dataframe_split" bukan "inputs"

---

## Kesimpulan

✅ **Kelebihan prediksi dengan run_id:**
- Cepat untuk testing & development
- Mudah diimplementasikan
- Tidak perlu setup infrastruktur

❌ **Keterbatasan:**
- Tidak scalable untuk banyak request
- Harus load model setiap kali prediksi
- Sulit diintegrasikan dengan aplikasi lain

➡️ **Untuk Production**: Gunakan MLflow Model Serving atau build API dengan FastAPI

---

## File-File Penting

- `predict_with_runid.py` - Script utama untuk validasi & prediksi
- `inputdata.json` - Sample data untuk testing
- `modelling.py` - Script untuk melatih model
- `README_LATIHAN.md` - Panduan ini

---

## Challenge

Coba implementasikan latihan yang sama dengan framework **TensorFlow** dan bandingkan experience-nya!

Selamat berlatih! 🚀
