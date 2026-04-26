import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ========================================================
# SKRIP PELATIHAN (FINE-TUNING) INDOBERT AES
# ========================================================

# 1. LOAD DATASET
# (Pastikan file excel ada di dalam folder 'data' jika di GitHub)
file_path = './data/dataset_ujian.xlsx' # Sesuaikan nama file ini nanti
df_kunci = pd.read_excel(file_path, sheet_name='Kunci Jawaban')
df_kunci = df_kunci.rename(columns={'jurusan': 'Jurusan', 'Mata Pelajaran': 'Mapel'})

nama_sheet_jawaban = ['DPK (TKJ)', 'MPP (RPL)', 'MPP (PPL) 2', 'MPP (TKJ-Telkom)']
list_df_jawaban = [pd.read_excel(file_path, sheet_name=s) for s in nama_sheet_jawaban]
df_jawaban_all = pd.concat(list_df_jawaban, ignore_index=True)

df_final = pd.merge(df_jawaban_all, df_kunci, on=['Kelas', 'Jurusan', 'Mapel', 'Kode'], how='left', suffixes=('_siswa', '_kunci'))
df_final = df_final.dropna(subset=['Jawaban_kunci', 'Nilai'])

# 2. PREPROCESSING & NORMALISASI
df_final['label_skor'] = df_final['Nilai'].astype(float) / 5.0
df_final['jawaban_bersih'] = df_final['Jawaban_siswa'].astype(str).str.lower()
df_final['kunci_bersih'] = df_final['Jawaban_kunci'].astype(str).str.lower()

# 3. SPLIT DATA
train_data, temp_data = train_test_split(df_final, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_labels = train_data['label_skor'].tolist()
val_labels = val_data['label_skor'].tolist()
test_labels = test_data['label_skor'].tolist()

# 4. TOKENISASI
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
def tokenize_pairs(dataframe):
    return tokenizer(
        dataframe['jawaban_bersih'].tolist(), dataframe['kunci_bersih'].tolist(),
        padding=True, truncation=True, max_length=512, return_tensors="pt"
    )

train_encodings = tokenize_pairs(train_data)
val_encodings = tokenize_pairs(val_data)
test_encodings = tokenize_pairs(test_data)

# 5. PYTORCH DATASET
class EssayDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = EssayDataset(train_encodings, train_labels)
val_dataset = EssayDataset(val_encodings, val_labels)
test_dataset = EssayDataset(test_encodings, test_labels)

# 6. INISIALISASI MODEL
model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=1)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    return {
        "MAE": mean_absolute_error(labels, predictions),
        "RMSE": np.sqrt(mean_squared_error(labels, predictions))
    }

# 7. TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="RMSE",
    greater_is_better=False
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=train_dataset, eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 8. JALANKAN TRAINING & SIMPAN MODEL
if __name__ == "__main__":
    print("Memulai proses fine-tuning...")
    trainer.train()
    print("\nEvaluasi Model pada Data Test:")
    print(trainer.evaluate(test_dataset))
    trainer.save_model("./model_indobert_aes_terbaik")
    print("Model berhasil disimpan di ./model_indobert_aes_terbaik")
