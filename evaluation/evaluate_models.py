import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models.huggingface_model import HuggingFaceModel
from models.falcon_model import FalconModel
import string
import numpy as np


# Veriyi yükle
df = pd.read_csv("data/chatbot_dataset_library.csv")

# Veri setini train/test olarak ayır
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# TF-IDF bazlı HuggingFaceModel
hf_model = HuggingFaceModel()

# TF-IDF vektörizeri tekrar eğit (evaluation amaçlı)
intent_examples = {}
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words=['english', 'turkish'])
for intent in train_df['Intent'].unique():
    examples = train_df[train_df['Intent'] == intent]['Example'].tolist()
    intent_examples[intent] = examples

all_examples = [ex for examples in intent_examples.values() for ex in examples]
vectorizer.fit(all_examples)
intent_vectors = {intent: vectorizer.transform(examples) for intent, examples in intent_examples.items()}

def normalize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def predict_intent_tfidf(text):
    norm_text = normalize(text)
    vec = vectorizer.transform([norm_text])
    best_score = 0
    best_intent = None
    for intent, vectors in intent_vectors.items():
        sim = cosine_similarity(vec, vectors).flatten()
        max_sim = np.max(sim)
        if max_sim > best_score:
            best_score = max_sim
            best_intent = intent
    return best_intent if best_score > 0.3 else "unknown"

true_labels = []
pred_hf = []
pred_falcon = []
pred_tfidf = []

falcon = FalconModel()

for _, row in test_df.iterrows():
    ex = row["Example"]
    label = row["Intent"]

    _, hf_intent = hf_model.find_best_match(ex)
    hf_intent = hf_intent if hf_intent else "unknown"

    tfidf_intent = predict_intent_tfidf(ex)

    # Falcon modeli için gerçek intent sınıflandırması yapılamaz, sadece çıktı üretilir
    # Bu nedenle burada sadece örnek yanıt bastırıyoruz
    print("\nUser:", ex)
    print("Falcon Response:", falcon.get_response(ex))

    true_labels.append(label)
    pred_hf.append(hf_intent)
    pred_tfidf.append(tfidf_intent)


from sklearn.metrics import classification_report

report_hf = classification_report(true_labels, pred_hf, output_dict=True)
report_tfidf = classification_report(true_labels, pred_tfidf, output_dict=True)

pd.DataFrame(report_hf).transpose().to_csv("huggingface_classification_report.csv")
pd.DataFrame(report_tfidf).transpose().to_csv("tfidf_classification_report.csv")


print("\n=== HuggingFace TF-IDF Model ===")
print(classification_report(true_labels, pred_hf))

print("\n=== TF-IDF Doğrudan Model ===")
print(classification_report(true_labels, pred_tfidf))