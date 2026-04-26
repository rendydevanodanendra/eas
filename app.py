import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load Model
@st.cache_resource
def load_model():
    # Pastikan modelnya membaca folder hasil training kamu tadi
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = AutoModelForSequenceClassification.from_pretrained("./model_indobert_aes_terbaik")
    return tokenizer, model

tokenizer, model = load_model()

# Desain UI Web
st.set_page_config(page_title="AES IndoBERT", page_icon="🤖")
st.title("🤖 Aplikasi Penilai Esai Otomatis")
st.write("Dibuat dengan model IndoBERT hasil Fine-Tuning")

kunci = st.text_area("Masukkan Kunci Jawaban Dosen:")
jawaban = st.text_area("Masukkan Jawaban Mahasiswa:")

if st.button("Nilai Esai Ini", type="primary"):
    if kunci and jawaban:
        with st.spinner('AI sedang mikir...'):
            # Proses tokenisasi teks
            inputs = tokenizer(jawaban.lower(), kunci.lower(), return_tensors="pt", truncation=True, max_length=512)
            
            # Prediksi menggunakan model
            with torch.no_grad():
                outputs = model(**inputs)
                prediksi_skor = outputs.logits.item() * 5.0 # Skala 0 - 5
            
            st.success(f"### 🎉 Skor Prediksi AI: {prediksi_skor:.2f} / 5.00")
    else:
        st.warning("Eits, isi dulu kedua kolom teks di atas!")
