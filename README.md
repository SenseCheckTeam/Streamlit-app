# 🩺 Diagnosa Awal Berdasarkan Gejala (Streamlit App)

Aplikasi ini merupakan sistem diagnosa awal berbasis **model XLM-Roberta** yang memprediksi kemungkinan penyakit berdasarkan kategori pancaindra dan deskripsi gejala dari pengguna. Aplikasi ini dibangun menggunakan **Streamlit** dan model **Transformers dari Hugging Face**.

## 🚀 Fitur Utama

- Input kategori pancaindra: `mata`, `telinga`, `hidung`, `kulit`, `lidah`.
- Input deskripsi gejala bebas.
- Model klasifikasi teks menggunakan **XLM-Roberta**.
- Validasi diagnosis sesuai kategori.
- Penanganan jika tingkat kepercayaan diagnosis rendah.
- Rekomendasi penanganan pertama untuk diagnosis yang valid.

---

## 🧠 Teknologi yang Digunakan

- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [Transformers (Hugging Face)](https://huggingface.co/transformers/)
- [scikit-learn (joblib)](https://scikit-learn.org/stable/modules/model_persistence.html#persistence-example-with-joblib)

---

## 📁 Struktur File

```
.
├── app.py                  # Aplikasi Streamlit utama
├── model/                  # Folder berisi model XLM-Roberta fine-tuned
├── label_encoder.pkl       # Encoder label penyakit
├── label_map.json          # Mapping label ke diagnosis dan saran
├── requirements.txt        # Daftar dependensi Python
└── README.md               # Dokumentasi proyek
```

---

## 🏃 Cara Menjalankan Aplikasi

### 1. Clone repositori dan masuk ke direktori

```bash
git clone https://github.com/username/diagnosis-streamlit-app.git
cd diagnosis-streamlit-app
```

### 2. Install dependensi

Disarankan menggunakan virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Jalankan aplikasi Streamlit

```bash
streamlit run app.py
```

---

## 📝 Format `label_map.json`

```json
{
  "0": {
    "diagnosis": "Katarak",
    "saran": "Segera konsultasikan ke dokter mata untuk pemeriksaan lanjutan."
  },
  ...
}
```

---

## ⚠️ Catatan

- Diagnosis yang ditampilkan **bukan pengganti diagnosis medis profesional**.
- Model hanya bekerja optimal pada gejala yang sesuai dengan kategori pancaindra yang dipilih.
- Jika kepercayaan model di bawah 50%, maka dianggap tidak dapat memberikan diagnosis yang valid.

---

## 👨‍⚕️ Contoh Penggunaan

1. Pilih kategori: `mata`
2. Masukkan gejala: `Pandangan kabur dan silau di siang hari`
3. Klik **"Prediksi Diagnosis"**
4. Hasil diagnosis dan saran penanganan akan ditampilkan.

---

## 📌 Lisensi

Proyek ini bebas digunakan untuk keperluan edukasi dan pengembangan non-komersial.
