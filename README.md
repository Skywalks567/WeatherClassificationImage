# WeatherClassificationImage
# Berikut adalah projek untuk penyelesaian UAS AI Semester 4 Tahun 2025
## Model AI yang digunakan
1. MobileNetV2
2. Lapisan Kustom (Dense Layers)
3. Adam Optimizer

## Problem yang diselesaikan ( Bisa apa Modelnya )
1. MobileNetV2
    MobileNetV2 merupakan Convolutional Neural Network(CNN) yang dikembangkan oleh google. Ini adalah model yang sudah dilatih sebelumnya pada dataset raksasa bernama ImageNet. 
    MobileNetV2 pada website aplikasi klasifikasi cuaca berdasarkan foto digunakan untuk: 
    - sebagai ekstraktor fitur karena model ini sudah sangat ahli dalam mengenali pola-pola visual umum.
    - Transfer learning: Mengunci pengetahuan tentang pengenalan pola dasar sehingga semua lapisan dari MobileNetV2 selain lapisan paling atas dan membekukannya.
    - Efisien: arsitektur ini dirancang agar ringan dan cepat sehingga cocok untuk aplikasi yang perlu berjalan di perangkat dengan sumber daya terbatas, termasuk aplikasi web.

2. Lapisan Kustom (Custom Layers)
    Lapisan kustom ini bukan merupakan jenis model yang terpisah, melainkan komponen yang dibangun di atas MobileNetV2 untuk membentuk satu model utuh yang baru.
    Lapisan Kustom pada website aplikasi klasifikasi cuaca berdasarkan foto yang digunakan: 
    - *GlobalAveragePooling2D()*: lapisan ini mengambil output dari MobileNetV2 dan meratakannya menjadi sebuah vektor tunggal yang  lebih sederhana. Ini merupakan cara modern dan efisien untuk mengurangi jumlah parameter dan mencegah overfitting.
    - *Dense(1024, activation='relu')*: lapisan _fully-connected standar_. Tujuannya untuk mempelajari kombinasi kompleks  dari fitur-fitur yang diberkan oleh lapisan sebelumnya.
    - *Dense(num_classes, activation='softmax')*: lapisan output dan paling krusial untuk klasifikasi 

3. Optimizer: Adam
    Adam (Adaptive Moment Estimation) merupakan algoritma optimasi yang digunakan selama proses training. Ini juga bukan bagian dari model, tetapi metode untuk melatih model.
    Optimizer: Adam pada website aplikasi klasifikasi cuaca berdasarkan foto yang digunakan untuk:
    - Mengatur proses belajar: saat training model akan menebak, lalu membandingkan tebakannya dengan jawaban yang benar, dan menghitung "kesalahan" (loss). Tugas Adam adalah memutuskan seberapa besar bobot di dalam lapisan dense kita harus diubah untuk mengurangi kesalahan tersebut diiterasi berikutnya.
    - Efisien dan Cepat: Adam dianggap sebagai salah satu optimizer paling efektif dan efisien untuk sebagian besar masalah Deep Learning, karena ia dapat menyesuaikan learning rate secara adaptif 


## Dataset nya apa (boleh disertakan linknya)
- "Link Dataset : https://www.kaggle.com/datasets/jehanbhathena/weather-dataset"