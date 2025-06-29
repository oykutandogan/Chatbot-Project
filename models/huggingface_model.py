import os
import pandas as pd
import string
import numpy as np
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class HuggingFaceModel:
    def __init__(self):
        try:
            load_dotenv()
            print("Model başlatılıyor...")

            # Veri kümesi yükleniyor
            print("Dataset yükleniyor...")
            self.dataset = pd.read_csv("data/chatbot_dataset_library.csv")

            # Intent örnekleri ve yanıtlarını ayır
            self.intent_examples = {}
            self.intent_responses = {}
            self.vectorizer = TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 2),
                stop_words=['english', 'turkish'],
                min_df=1,
                max_df=0.95
            )

            all_examples = []
            for intent in self.dataset['Intent'].unique():
                examples = self.dataset[self.dataset['Intent'] == intent]['Example'].fillna('').tolist()
                responses = self.dataset[self.dataset['Intent'] == intent]['Response'].fillna('').tolist()
                self.intent_examples[intent] = examples
                self.intent_responses[intent] = responses
                all_examples.extend(examples)

            # Vektörizer eğitimi
            print("Vektörizer eğitiliyor...")
            self.vectorizer.fit(all_examples)

            self.intent_vectors = {
                intent: self.vectorizer.transform(examples)
                for intent, examples in self.intent_examples.items()
            }

            # Selamlama ve veda
            self.greeting_words = {'merhaba', 'selam', 'günaydın', 'iyi günler', 'iyi akşamlar'}
            self.goodbye_words = {'görüşürüz', 'hoşça kal', 'güle güle', 'iyi günler', 'kendine iyi bak'}

            print("Dataset ve vektörler başarıyla yüklendi!")

            # Model yükleniyor
            print("Model yükleniyor...")
            self.model_name = "dbmdz/bert-base-turkish-cased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=150,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.5,
                do_sample=True,
                num_return_sequences=1
            )

            print("Model başarıyla yüklendi!")

        except Exception as e:
            print(f"Model başlatma hatası: {str(e)}")
            self.model = None
            self.tokenizer = None
            self.generator = None
            raise

    def normalize(self, text):
        if not isinstance(text, str):
            return ""
        return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

    def find_best_match(self, query):
        try:
            norm_query = self.normalize(query)
            query_vector = self.vectorizer.transform([norm_query])
            best_intent = None
            best_score = 0
            best_example = None

            for intent, vectors in self.intent_vectors.items():
                similarities = cosine_similarity(query_vector, vectors).flatten()
                max_sim = np.max(similarities)
                if max_sim > best_score:
                    best_score = max_sim
                    best_intent = intent
                    best_example = self.intent_examples[intent][np.argmax(similarities)]

            return (best_example, best_intent) if best_score > 0.3 else (None, None)
        except Exception as e:
            print(f"Eşleştirme hatası: {str(e)}")
            return None, None

    def generate_reply(self, user_input, matched_example, intent):
        try:
            if intent in self.intent_responses and self.intent_responses[intent]:
                return self.intent_responses[intent][0]

            prompt = f"""Soru: {user_input}
Bağlam: Kütüphane bilgi sistemi chatbot'u
Amaç: {intent}
Benzer Örnek: {matched_example}
Yanıt: Kütüphane bilgi sistemi chatbot'u olarak aşağıdaki bilgileri içeren net ve açıklayıcı bir yanıt ver:
1. Soruya doğrudan cevap
2. Gerekirse ek açıklamalar
3. Yardımcı kaynak önerisi
Yanıt:"""

            generated_text = self.generator(
                prompt,
                max_length=len(prompt.split()) + 100,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )[0]['generated_text']

            response = generated_text.split("Yanıt:")[-1].strip()
            response = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', response)

            # Aynı cümleleri temizle
            sentences = response.split('.')
            unique_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in unique_sentences:
                    unique_sentences.append(sentence)

            return '. '.join(unique_sentences)
        except Exception as e:
            print(f"Yanıt üretme hatası: {str(e)}")
            return "Üzgünüm, yanıt üretirken bir hata oluştu. Lütfen tekrar deneyin."

    def get_response(self, user_input):
        try:
            if self.model is None or self.tokenizer is None or self.generator is None:
                return "Model yüklenemedi. Lütfen sistem yöneticinize başvurun."

            norm_input = self.normalize(user_input)
            if any(greeting in norm_input for greeting in self.greeting_words):
                return "Merhaba! Size nasıl yardımcı olabilirim?"

            if any(goodbye in norm_input for goodbye in self.goodbye_words):
                return "Görüşmek üzere! Başka sorularınız olursa yine beklerim."

            best_match, intent = self.find_best_match(user_input)
            if best_match:
                return self.generate_reply(user_input, best_match, intent)
            else:
                return "Üzgünüm, bu konuda bilgim yok. Lütfen sorunuzu farklı bir şekilde ifade edin."

        except Exception as e:
            print(f"Yanıt üretme hatası: {str(e)}")
            return "Üzgünüm, sistemsel bir hata oluştu. Lütfen tekrar deneyin."