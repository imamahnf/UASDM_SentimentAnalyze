import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load pickle files (TF-IDF vectorizer dan model SVM)
with open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Hardcoded model accuracy (ganti dengan nilai akurasi aktual Anda)
model_accuracy = 0.85  # Contoh: 85% akurasi dari model pelatihan

# Load stopwords dan stemmer
stopword_factory = StopWordRemoverFactory()
stopwords = set(stopword_factory.get_stop_words())
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Function for preprocessing text
def preprocess_text(text):
    # Hapus mention, hashtag, URL, dan emoji
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Hapus mention
    text = re.sub(r'#\w+', '', text)  # Hapus hashtag
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Hapus URL
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "\U00002B50-\U00002B55"
        "\U000024C2-\U0001F251"
        "]", flags=re.UNICODE
    )
    text = emoji_pattern.sub(' ', text)  # Ganti emoji dengan spasi
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Hapus karakter non-alfabet
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi ganda
    
    # Case folding
    text = text.lower()
    
    # Hapus stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords])
    
    # Stemming
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    
    return text

# Streamlit UI
st.title("Aplikasi Analisis Sentimen")
st.sidebar.title("Informasi Model")
st.sidebar.write(f"Akurasi Model: {model_accuracy * 100:.2f}%")

st.write("Masukkan teks atau unggah file CSV untuk analisis sentimen (positif atau negatif).")

# Tab navigation: Single Text Input or File Upload
option = st.radio("Pilih mode input:", ("Teks Tunggal", "Unggah File CSV"))

# Case 1: Single Text Input
if option == "Teks Tunggal":
    user_input = st.text_area("Masukkan Teks", placeholder="Ketik teks di sini...")

    if st.button("Analisis Sentimen"):
        if user_input:
            # Preprocess text
            preprocessed_text = preprocess_text(user_input)
            
            # Extract features using TF-IDF
            features = tfidf_vectorizer.transform([preprocessed_text])
            
            # Predict sentiment and confidence
            prediction = svm_model.predict(features)[0]
            confidence_scores = svm_model.decision_function(features)
            confidence = np.max(confidence_scores)  # Ambil confidence tertinggi
            
            # Display result
            if prediction == 'positive':
                st.success(f"Hasil: Sentimen Positif 😊 (Confidence: {confidence:.2f})")
            elif prediction == 'negative':
                st.error(f"Hasil: Sentimen Negatif 😠 (Confidence: {confidence:.2f})")
        else:
            st.warning("Mohon masukkan teks untuk dianalisis!")

# Case 2: File Upload
elif option == "Unggah File CSV":
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Periksa ukuran file (limit 50 MB)
            if uploaded_file.size > 50 * 1024 * 1024:
                st.error("File terlalu besar! Harap unggah file dengan ukuran di bawah 50 MB.")
            else:
                # Load file CSV
                df = pd.read_csv(uploaded_file)
                st.write("Data yang diunggah:")
                st.dataframe(df.head())

                # Validasi kolom
                text_column = st.selectbox("Pilih kolom teks:", df.columns)
                if st.button("Analisis Sentimen CSV"):
                    # Preprocess all texts
                    df['preprocessed_text'] = df[text_column].astype(str).apply(preprocess_text)
                    
                    # Extract features using TF-IDF
                    features = tfidf_vectorizer.transform(df['preprocessed_text'])
                    
                    # Predict sentiments
                    df['sentiment'] = svm_model.predict(features)
                    
                    # Tambahkan kolom akurasi
                    df['model_accuracy'] = model_accuracy

                    # Display results
                    st.success("Analisis selesai! Berikut hasilnya:")
                    st.dataframe(df[[text_column, 'sentiment', 'model_accuracy']])
                    
                    # Visualize results with a pie chart
                    sentiment_counts = df['sentiment'].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')  # Equal aspect ratio ensures pie chart is circular.
                    st.pyplot(fig)
                    
                    # Download results as CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Unduh Hasil sebagai CSV",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
