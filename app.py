import os
import re
import string
import gdown
import streamlit as st
import torch
import pandas as pd
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

# ===================== MODEL URL =====================
ASPEK_FOLDER = "petugas_model"
ASPEK_MODEL_FILES = {
    "config.json": "https://drive.google.com/uc?id=1-gaeEZS51znvhsvSdnez9XB-xKvR2Dih",
    "best_model.pt": "https://drive.google.com/uc?id=1USuLqLopkGwJY6EtBTARdor6qvHcowW_",
    "vocab.txt": "https://drive.google.com/uc?id=1Ur8adye08EcCoQ74YMIgFCPJenkdtqhA",
    "special_tokens_map.json": "https://drive.google.com/uc?id=1-lurkvcFx02DmjMqzIc9Z4LGRGRQ9AuS",
    "tokenizer_config.json": "https://drive.google.com/uc?id=1-tHk4S9UMk3xdosTpkgeCkxsP3JWB2xJ"
}

SENTIMEN_FOLDER = "sentimen_petugas_model"
SENTIMEN_MODEL_FILES = {
    "config.json": "https://drive.google.com/uc?id=1SQTrxi-SPteLDUsaSuyVJP6vz7B_HR-g",
    "best_model.pt": "https://drive.google.com/uc?id=1FyKVBFkNn5lGKyQ4nAI9hkE89swtoGp3",
    "vocab.txt": "https://drive.google.com/uc?id=1QZM0JNNP7M3MxdHnuWHoSKA6opJHXoqi",
    "special_tokens_map.json": "https://drive.google.com/uc?id=1ZYUsqSqWpR8NEtQ0MEX3hs_Le0UFd1HK",
    "tokenizer_config.json": "https://drive.google.com/uc?id=1-tHk4S9UMk3xdosTpkgeCkxsP3JWB2xJ"
}

KAMUS_CSV_URL = "https://drive.google.com/uc?id=1fGWZu5qVYJa-pv078spaLE4urs5zDDPV"
KAMUS_PATH = "kamus.csv"

# ===================== FUNGSI BANTUAN =====================
def download_model(model_folder, model_files):
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    for filename, url in model_files.items():
        path = os.path.join(model_folder, "pytorch_model.bin" if filename == "best_model.pt" else filename)
        if not os.path.exists(path):
            with st.spinner(f"Mengunduh {filename}..."):
                gdown.download(url, path, quiet=False)

def download_kamus():
    if not os.path.exists(KAMUS_PATH):
        with st.spinner("Mengunduh kamus slang..."):
            gdown.download(KAMUS_CSV_URL, KAMUS_PATH, quiet=False)

@st.cache_resource(show_spinner=True)
def load_tokenizer(folder):
    return BertTokenizer.from_pretrained(folder)

@st.cache_resource(show_spinner=True)
def load_model(folder):
    config = BertConfig.from_pretrained(folder)
    model = BertForSequenceClassification.from_pretrained(folder, config=config)
    model.eval()
    return model

@st.cache_resource
def load_kamus():
    df = pd.read_csv(KAMUS_PATH)
    return dict(zip(df['slang'], df['formal']))

def preprocess(text, kamus_slang):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([kamus_slang.get(word, word) for word in text.split()])
    return text.strip()

def predict_aspek_sentimen(text, kamus_slang, tokenizer_aspek, model_aspek, tokenizer_sentimen, model_sentimen):
    cleaned = preprocess(text, kamus_slang)

    inputs_aspek = tokenizer_aspek(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        output_aspek = model_aspek(**inputs_aspek)
    pred_aspek = torch.argmax(output_aspek.logits, dim=1).item()

    if pred_aspek == 1:
        inputs_sentimen = tokenizer_sentimen(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            output_sentimen = model_sentimen(**inputs_sentimen)
        pred_sentimen = torch.argmax(output_sentimen.logits, dim=1).item()
        sentimen = {2: "Positif", 0: "Negatif", 1: "Netral"}.get(pred_sentimen, "Tidak Diketahui")
        return "Petugas", sentimen
    else:
        return "Bukan Petugas", "-"

# ===================== APLIKASI STREAMLIT =====================
def main():
    st.set_page_config(page_title="Prediksi Aspek dan Sentimen", layout="wide")
    st.title("🕋 Prediksi Aspek & Sentimen - Layanan Petugas Haji")

    # Unduh model & kamus
    download_model(ASPEK_FOLDER, ASPEK_MODEL_FILES)
    download_model(SENTIMEN_FOLDER, SENTIMEN_MODEL_FILES)
    download_kamus()

    # Load model
    kamus_slang = load_kamus()
    tokenizer_aspek = load_tokenizer(ASPEK_FOLDER)
    model_aspek = load_model(ASPEK_FOLDER)
    tokenizer_sentimen = load_tokenizer(SENTIMEN_FOLDER)
    model_sentimen = load_model(SENTIMEN_FOLDER)

    # Upload file CSV
    st.subheader("📤 Upload File CSV (dengan kolom `text`)")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("⚠️ CSV harus memiliki kolom bernama 'text'")
            return

        hasil = []

        with st.spinner("🔍 Sedang memproses prediksi, mohon tunggu..."):
            for _, row in df.iterrows():
                aspek, sentimen = predict_aspek_sentimen(
                    row['text'], kamus_slang,
                    tokenizer_aspek, model_aspek,
                    tokenizer_sentimen, model_sentimen
                )
                hasil.append({"text": row['text'], "aspek": aspek, "sentimen": sentimen})

        df_hasil = pd.DataFrame(hasil)
        st.success("✅ Prediksi selesai!")
        st.dataframe(df_hasil)

        csv_hasil = df_hasil.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Hasil CSV",
            data=csv_hasil,
            file_name="hasil_prediksi_aspek_sentimen.csv",
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
