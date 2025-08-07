from wordcloud import WordCloud
import matplotlib.pyplot as plt

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

        # ===================== STATISTIK =====================
        st.subheader("📊 Statistik Hasil Prediksi")

        col1, col2 = st.columns(2)

        # Pie chart aspek
        with col1:
            st.markdown("### 🔍 Distribusi Aspek")
            aspek_counts = df_hasil['aspek'].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(aspek_counts, labels=aspek_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)

        # Pie chart sentimen (khusus Petugas)
        with col2:
            st.markdown("### 😊 Distribusi Sentimen (Petugas Saja)")
            petugas_df = df_hasil[df_hasil['aspek'] == "Petugas"]
            sentimen_counts = petugas_df['sentimen'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(sentimen_counts, labels=sentimen_counts.index, autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2)

        # ===================== WORDCLOUD =====================
        st.subheader("☁️ Wordcloud untuk Tweet Aspek Petugas")

        text_petugas = ' '.join(petugas_df['text'].astype(str).tolist())
        cleaned_text = preprocess(text_petugas, kamus_slang)

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.imshow(wordcloud, interpolation='bilinear')
        ax3.axis('off')
        st.pyplot(fig3)
