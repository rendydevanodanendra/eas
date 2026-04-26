import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. SETUP MODEL & DATABASE (EXCEL)
# ==========================================
@st.cache_resource
def load_system():
    # 1A. Load Model IndoBERT
    with st.spinner("Memuat Otak AI dari Hugging Face..."):
        tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
        # GANTI DENGAN USERNAME KAMU:
        model = AutoModelForSequenceClassification.from_pretrained("rendydevanodanendra/indobert-aes")
    
    # 1B. Load Database Kunci Jawaban (Dari Excel aslimu)
    # Pastikan file excel ini di-upload ke GitHub dan path-nya benar
    file_path = 'dataset question & answer skripsi - Raw (2).xlsx' 
    try:
        df_kunci = pd.read_excel(file_path, sheet_name='Kunci Jawaban')
        df_kunci = df_kunci.rename(columns={'jurusan': 'Jurusan', 'Mata Pelajaran': 'Mapel'})
        # Kita hanya butuh list kode soal untuk dipilih oleh user
        list_soal = df_kunci['Kode'].unique().tolist()
    except Exception as e:
        st.error(f"Gagal memuat database soal: {e}")
        df_kunci, list_soal = None, []
        
    return tokenizer, model, df_kunci, list_soal

tokenizer, model, df_kunci, list_soal = load_system()

# ==========================================
# 2. DESAIN ANTARMUKA APLIKASI
# ==========================================
st.set_page_config(page_title="Sistem AES Terintegrasi", layout="centered")

st.title("🎓 Sistem Penilaian Esai Otomatis (AES)")
st.markdown("---")

if df_kunci is not None:
    # Bagian Kiri: Siswa Memilih Soal
    st.subheader("Data Ujian")
    
    # User memilih kode soal (Sistem otomatis mencari kunci jawabannya)
    kode_terpilih = st.selectbox("Pilih Kode Soal yang Dikerjakan:", list_soal)
    
    # Cari baris data di Excel yang sesuai dengan kode soal terpilih
    data_soal = df_kunci[df_kunci['Kode'] == kode_terpilih].iloc[0]
    
    st.info(f"**Mata Pelajaran:** {data_soal['Mapel']} | **Kelas:** {data_soal['Kelas']}")
    st.write(f"**Pertanyaan:** {data_soal['Pertanyaan']}")
    
    # Kunci Jawaban disembunyikan (Sistem yang tahu)
    kunci_jawaban_sistem = str(data_soal['Jawaban_kunci']).lower()
    
    st.markdown("---")
    
    # Bagian Kanan: Input Jawaban Siswa
    st.subheader("Lembar Jawaban")
    jawaban_siswa = st.text_area("Ketik/Paste jawaban mahasiswa di sini:", height=150)
    
    if st.button("Nilai Secara Otomatis", type="primary", use_container_width=True):
        if jawaban_siswa.strip() == "":
            st.warning("Jawaban masih kosong!")
        else:
            with st.spinner('AI sedang membandingkan makna dengan kunci jawaban sistem...'):
                # 3. PROSES PENILAIAN AI
                inputs = tokenizer(
                    jawaban_siswa.lower(), 
                    kunci_jawaban_sistem, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Kembalikan ke skala 0-5
                    prediksi_skor = outputs.logits.item() * 5.0 
                    
                    # Cegah nilai minus atau lebih dari 5
                    prediksi_skor = max(0.0, min(5.0, prediksi_skor))
                
                # Tampilkan Hasil
                st.success("### ✅ Penilaian Selesai")
                col1, col2 = st.columns(2)
                col1.metric("Skor Prediksi AI", f"{prediksi_skor:.2f} / 5.00")
                
                # Opsional: Buka contekan kunci jawaban untuk dosen (Toggle)
                with st.expander("Lihat Referensi Kunci Jawaban (Mode Dosen)"):
                    st.write(data_soal['Jawaban_kunci'])
else:
    st.error("Sistem sedang perbaikan, database kunci jawaban tidak ditemukan.")
