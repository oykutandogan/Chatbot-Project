# 📚 Üniversite Kütüphane Chatbotu

Bu proje, üniversite öğrencilerinin kütüphane hizmetleri hakkında hızlı ve doğru bilgi alabilmesini sağlamak amacıyla geliştirilmiş bir doğal dil işleme (NLP) tabanlı **chatbot sistemidir**.

Proje, iki farklı dil modeli (LLM) kullanılarak geliştirilmiştir:  
- **HuggingFace TF-IDF + BERT tabanlı sınıflandırıcı**
- **Falcon (FLAN-T5) tabanlı jeneratif model**

## 🚀 Özellikler

- ✅ Selamlama, vedalaşma, reddetme gibi temel etkileşimler
- 📘 Kütüphane kitapları, erişim, cezalar gibi bilgi sunumu
- 🧠 TF-IDF destekli niyet sınıflandırması
- 💬 LLM tabanlı doğal dil yanıt üretimi
- 📊 Model performans karşılaştırması (Precision, Recall, F1 Score)

---

## 📁 Proje Yapısı
├── app/

│ └── streamlit_app.py # Uygulamanın arayüzü

├── data/

│ └── chatbot_dataset_library.csv # Intent ve örnek veri seti

├── demo/

│ └── flan-t5_chat.png

│ └── huggingface_chat.png

│ └── interface.png

├── evaluation/

│ └── evaluation.py # Modellerin karşılaştırılması

├── models/

│ ├── __init__.py

│ ├── huggingface_model.py # TF-IDF + BERT tabanlı model

│ └── falcon_model.py # FLAN-T5 jeneratif model

├── requirements.py

└── README.md


## 🧠 Kullanılan Modeller

### 1. HuggingFaceModel (TF-IDF + BERT)
- Türkçe destekli `dbmdz/bert-base-turkish-cased` kullanılmıştır.
- TF-IDF ile intent belirlenir, ardından yanıt jenerasyonu yapılır.
- Belirli yanıt şablonları veya örnek benzerliğiyle çalışır.

### 2. FalconModel (FLAN-T5)
- Google tarafından sunulan `google/flan-t5-base` modeli kullanılmıştır.
- Sorulara açık uçlu ve doğal cevaplar üretir.
- Intent sınıflandırması yapmaz.


## 📊 Model Performans Karşılaştırması

### Değerlendirme Metrikleri:
- Precision, Recall, F1 Score
- `sklearn.metrics` kütüphanesi kullanıldı.

📝 Detaylı değerlendirme için bkz: `evaluation/huggingface_classification_report.csv`

## 🔁 Chatbot Akışı

![Chatbot Akışı](chatbot_akis_diyagrami.png)

1. Başlangıç
2. Selamlama kontrolü
3. Temel soru mu?
4. Evet → Temel soruya cevap  
   Hayır → Konuya özgü senaryo cevabı
5. Vedalaşma
