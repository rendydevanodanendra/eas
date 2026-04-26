# Automated Essay Scoring (AES) menggunakan IndoBERT

Proyek ini mengimplementasikan model Transformer **IndoBERT** untuk melakukan penilaian esai otomatis berdasarkan *Semantic Textual Similarity*. Model ini dilatih untuk memprediksi skor esai dengan membandingkan jawaban mahasiswa terhadap kunci jawaban dosen.

## 🚀 Fitur Utama
- **Fine-tuning IndoBERT**: Menggunakan model `indobert-base-p1` untuk tugas regresi.
- **Normalisasi Skor**: Mendukung penilaian dengan skala 0-5.
- **Aplikasi Interaktif**: Interface menggunakan Streamlit untuk mencoba penilaian secara real-time.
- **Evaluasi Performa**: Menggunakan metrik MAE dan RMSE (Hasil pelatihan menunjukkan MAE sekitar 0.05 - 0.06 pada skala 0-1).

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy)
- **Hugging Face Transformers** (PyTorch)
- **Scikit-learn** (Evaluasi & Splitting)
- **Streamlit** (Web Dashboard)

## 📊 Hasil Evaluasi
Berdasarkan hasil pengujian pada data testing, model mencapai:
- **MAE**: 0.0597 (Skala 0-1) atau sekitar **0.29** pada skala 0-5.
- **RMSE**: 0.0891 (Skala 0-1).

## 💻 Cara Menjalankan
1. Clone repositori:
   ```bash
   git clone [https://github.com/username/automated-essay-scoring-indobert.git](https://github.com/username/automated-essay-scoring-indobert.git)