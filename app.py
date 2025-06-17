# -*- coding: utf-8 -*-
"""
Weather Classification Streamlit App
Aplikasi untuk klasifikasi cuaca menggunakan deep learning
"""

import streamlit as st
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pickle
from pathlib import Path

# Konfigurasi halaman
st.set_page_config(
    page_title="Weather Classification App",
    page_icon="🌤️",
    layout="wide"
)

def default_theme():
    default_css = """
    <style>
        @keyframes gradient_animation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .stApp {
            background: linear-gradient(-45deg, #F8FAFC, #D9EAFD, #BCCCDC, #9AA6B2);
            background-size: 400% 400%;
            animation: gradient_animation 20s ease infinite;
        }

        [data-testid="stSidebar"]{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        [data-testid="stAppViewContainer"] > .main {
            background-color: rgba(255, 255, 255, 0.92);
            border-radius: 15px;
        }
        
        div[data-testid="stHeading"] h1,
        div[data-testid="stHeading"] h2,
        div[data-testid="stHeading"] h3,
        div[data-testid="stMarkdownContainer"] h3,
        div[data-testid="stMarkdownContainer"] h4,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li{
            color: #1a1a1a !important; 
        }

        button[data-testid="stBaseButton-primary"] {
            background-color: rgb(204, 219, 250, 0.80) !important
            color: white !important;
            border: none;
            border-radius: 10px;
        }

        button[data-testid="stBaseButton-primary"]:hover {
            background-color: rgb(169, 197, 255, 0.80)
        }

        [data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
            color: #1f2937;
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #111827;
        }

        div[data-baseweb="select"] > div {
            background-color: rgb(165, 192, 221, 0.3);
            border: 2px solid rgb(165, 192, 221, 0.1);
            border-radius: 8px;
            color: black;
            backdrop-filter: blur(50px);
            -webkit-backdrop-filter: blur(10px); 
            box-shadow: 0 4px 10px rgba(0,0,0,0.1); 
        }

        div[data-baseweb="progress-bar"] > div > div{
            background-color: #e0e0e0; /* Warna trek menjadi abu-abu muda */
            border-radius: 10px;
        }

        div[data-baseweb="progress-bar"] > div > div > div {
            background-color: #6ed965; /* Warna isian menjadi merah */
        }
        
        .stDataFrame div[data-testid="stHorizontalBlock"] {
            background: rgba(255, 255, 255, 0.5);
            border-radius: 10px;
            padding: 10px;
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }

        .stDataFrame td {
            font-size: 14px;
            border: 1px solid #ccc;
        }
    </style>
    """
    st.markdown(default_css, unsafe_allow_html=True)

default_theme()

def get_page_background_style(colors):
    """Membuat CSS untuk latar belakang gradien animasi dari 3 warna."""
    if len(colors) < 3:
        colors = ["#6dd5ed", "#2193b0", "#B0BEC5"]  # Fallback colors

    css = f"""
    <style>
    @keyframes gradientBackground {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    html, body, .stApp {{
        height: 100%;
        background: linear-gradient(-45deg, {colors[0]}, {colors[1]}, {colors[2]});
        background-size: 400% 400%;
        animation: gradientBackground 20s ease infinite;
    }}

    [data-testid="stAppViewContainer"] > .main {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
    }}
    </style>
    """

st.markdown(get_page_background_style(["#00c6ff", "#0072ff", "#00c9ff"]), unsafe_allow_html=True)

# Judul aplikasi
st.title("🌤️ Weather Classification App")
st.markdown("Aplikasi untuk mengklasifikasikan kondisi cuaca dari gambar menggunakan Deep Learning")

# Sidebar untuk navigasi
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Pilih Mode:", 
                                ["Prediksi Cuaca", "Training Model", "Tentang App"])

# Parameter global
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

@st.cache_resource
def load_trained_model():
    """Memuat model yang sudah dilatih"""
    repo_top = Path(__file__).resolve().parent
    try:
        model_path = repo_top / 'Model' / 'weather_classification_model.h5'
        class_names_path = repo_top / 'Model' / 'class_names.pkl'
        st.write(model_path)
        print("🔍 Mencoba memuat model dari:", model_path)
        print("📁 Ada file model?", model_path.exists())
        model = load_model(model_path)

        if class_names_path.exists():
            with open(class_names_path, 'rb') as f:
                class_names = pickle.load(f)
        else:
            class_names = ['cloudy', 'rain', 'shine', 'sunrise']

        return model, class_names
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        return None, None

def save_class_names(class_names):
    """Simpan class names ke file"""
    with open("./Model/class_names.pkl", 'wb') as f:
        pickle.dump(class_names, f)

def create_model(num_classes):
    """Membuat model MobileNetV2 untuk klasifikasi cuaca"""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Membekukan lapisan base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Menambahkan lapisan kustom
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy',  # Ubah ke sparse_categorical_crossentropy
                  metrics=['accuracy'])
    
    return model

def predict_weather_from_image(model, img, class_names):
    """Prediksi cuaca dari gambar"""
    # Resize dan preprocess gambar
    img_resized = img.resize(IMG_SIZE)
    
    # Konversi PIL ke numpy array
    img_array = np.array(img_resized)
    
    # Pastikan gambar dalam format RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Gambar sudah RGB
        pass
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # Gambar RGBA, konversi ke RGB
        img_array = img_array[:,:,:3]
    elif len(img_array.shape) == 2:
        # Gambar grayscale, konversi ke RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Normalisasi dan tambah batch dimension
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediksi
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(prediction) * 100
    
    return predicted_class_name, confidence, prediction

# Mode Prediksi Cuaca
if app_mode == "Prediksi Cuaca":
    st.header("🔮 Prediksi Kondisi Cuaca")
    
    # Cek apakah model sudah ada
    model, class_names = load_trained_model()
    
    if model is None:
        st.error("Model belum tersedia! Silakan latih model terlebih dahulu di tab 'Training Model'.")
        st.info("Atau pastikan file 'weather_classification_model.h5' tersedia di direktori yang sama.")
    else:
        st.success("Model berhasil dimuat!")
        st.info(f"Kelas yang tersedia: {', '.join(class_names)}")
        
        # Input gambar
        uploaded_file = st.file_uploader(
            "Upload gambar cuaca:", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload gambar untuk diprediksi kondisi cuacanya"
        )
        
        # Opsi untuk menggunakan kamera
        
        
        # Proses gambar yang diupload
        if uploaded_file is not None:
            # Pilih sumber gambar
            image_source = uploaded_file
            
            try:
                # Buka dan tampilkan gambar
                img = Image.open(image_source)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Gambar Input")
                    st.image(img, caption="Gambar yang diupload", use_container_width=True)
                
                with col2:
                    st.subheader("Hasil Prediksi")
                    
                    if st.button("Prediksi Cuaca", type="primary"):
                        with st.spinner("Sedang memproses..."):
                            predicted_class, confidence, all_predictions = predict_weather_from_image(
                                model, img, class_names
                            )
                            
                            st.success(f"**Prediksi: {predicted_class.upper()}**")
                            st.info(f"Tingkat Keyakinan: {confidence:.2f}%")
                            
                            # Tampilkan probabilitas semua kelas
                            st.subheader("Probabilitas Semua Kelas:")
                            for i, class_name in enumerate(class_names):
                                prob = all_predictions[0][i] * 100
                                st.write(f"• {class_name}: {prob:.2f}%")
                                st.progress(prob / 100)
                            
            except Exception as e:
                st.error(f"Error memproses gambar: {str(e)}")

# Mode Training Model
elif app_mode == "Training Model":
    st.header("🏋️ Training Model")
    st.warning("⚠️ Training model memerlukan dataset dan bisa memakan waktu lama!")
    
    # Input untuk URL dataset
    st.markdown("Link Dataset : https://www.kaggle.com/datasets/jehanbhathena/weather-dataset")
    
    epochs = st.slider("Jumlah Epochs:", min_value=1, max_value=50, value=5)
    
    if st.button("Mulai Training", type = "primary"):
        try:
            with st.spinner("Mengunduh dataset..."):
                # Download dataset
                # od.download(dataset_url)
                st.success("Dataset berhasil diunduh!")
            
            data_dir = './weather/dataset'
            
            if os.path.exists(data_dir):
                with st.spinner("Memproses data..."):
                    # Data augmentation - menggunakan tf.keras.utils.image_dataset_from_directory
                    train_ds = tf.keras.utils.image_dataset_from_directory(
                        data_dir,
                        validation_split=0.2,
                        subset="training",
                        seed=123,
                        image_size=IMG_SIZE,
                        batch_size=BATCH_SIZE
                    )
                    
                    val_ds = tf.keras.utils.image_dataset_from_directory(
                        data_dir,
                        validation_split=0.2,
                        subset="validation",
                        seed=123,
                        image_size=IMG_SIZE,
                        batch_size=BATCH_SIZE
                    )
                    
                    # Mendapatkan nama kelas
                    class_names = train_ds.class_names
                    num_classes = len(class_names)
                    
                    # Normalisasi data
                    normalization_layer = tf.keras.layers.Rescaling(1./255)
                    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
                    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
                    
                    # Optimasi performa
                    AUTOTUNE = tf.data.AUTOTUNE
                    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
                    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
                    
                    st.info(f"Kelas yang ditemukan: {class_names}")
                    st.info(f"Jumlah kelas: {num_classes}")
                    
                    # Tampilkan beberapa contoh gambar
                    st.subheader("Contoh Gambar dari Dataset:")
                    sample_batch = next(iter(train_ds))
                    sample_images, sample_labels = sample_batch
                    
                    cols = st.columns(4)
                    for i in range(min(4, len(sample_images))):
                        with cols[i]:
                            # Konversi tensor ke numpy dan denormalisasi
                            img = sample_images[i].numpy()
                            label_idx = int(sample_labels[i].numpy())
                            st.image(img, caption=f"{class_names[label_idx]}", use_container_width=True)
                
                with st.spinner("Membuat dan melatih model..."):
                    # Buat model
                    model = create_model(num_classes)
                    
                    # Progress bar untuk training
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Training
                    history = model.fit(
                        train_ds,
                        epochs=epochs,
                        validation_data=val_ds,
                        verbose=0
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Training selesai!")
                
                # Simpan model dan class names
                model.save('./Model/weather_classification_model.h5')
                save_class_names(class_names)
                st.success("Model berhasil disimpan sebagai 'weather_classification_model.h5'!")
                st.success("Class names berhasil disimpan sebagai 'class_names.pkl'!")
                
                # Tampilkan hasil training
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots()
                    ax.plot(history.history['accuracy'], label='Training Accuracy')
                    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
                    ax.set_title('Model Accuracy')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.legend()
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots()
                    ax.plot(history.history['loss'], label='Training Loss')
                    ax.plot(history.history['val_loss'], label='Validation Loss')
                    ax.set_title('Model Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    st.pyplot(fig)
                    
            else:
                st.error("Direktori dataset tidak ditemukan!")
                
        except Exception as e:
            st.error(f"Error saat training: {str(e)}")

# Mode Tentang App
else:
    st.header("ℹ️ Tentang Aplikasi")
    
    st.markdown("""
    ### Weather Classification App
    
    Aplikasi ini menggunakan **Deep Learning** dengan arsitektur **MobileNetV2** untuk mengklasifikasikan kondisi cuaca dari gambar.
    
    #### Fitur Utama:
    - 🔮 **Prediksi Cuaca**: Upload gambar atau ambil foto untuk prediksi
    - 🏋️ **Training Model**: Latih model dengan dataset custom
    - 📊 **Visualisasi**: Melihat hasil training dan probabilitas prediksi
    
    #### Teknologi yang Digunakan:
    - **TensorFlow/Keras**: Framework deep learning
    - **MobileNetV2**: Model pre-trained untuk transfer learning
    - **Streamlit**: Framework untuk web app
    - **PIL/OpenCV**: Pemrosesan gambar
    
    #### Cara Penggunaan:
    1. **Training**: Gunakan tab "Training Model" untuk melatih model dengan dataset
    2. **Prediksi**: Gunakan tab "Prediksi Cuaca" untuk mengklasifikasi gambar
    3. Upload gambar atau gunakan kamera untuk input
    
    #### Dataset:
    Aplikasi ini menggunakan dataset cuaca dari Kaggle yang berisi gambar dengan label:
    - ☁️ Cloudy (Berawan)
    - 🌧️ Rain (Hujan)  
    - ☀️ Shine (Cerah)
    - 🌅 Sunrise (Matahari Terbit)
    
    ---
    
    **Dibuat dengan ❤️ menggunakan Streamlit dan TensorFlow**
    """)
    
    # Informasi sistem
    st.subheader("Informasi Sistem")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"TensorFlow Version: {tf.__version__}")
        st.info(f"Python Version: {st.__version__}")
    
    with col2:
        if os.path.exists('weather_classification_model.h5'):
            st.success("Model tersedia ✅")
        else:
            st.warning("Model belum dilatih ⚠️")

# Footer
st.markdown("---")
st.markdown("**Note**: Pastikan Anda memiliki koneksi internet untuk mengunduh dataset dan model pre-trained.")
