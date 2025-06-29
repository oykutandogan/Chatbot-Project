import os
import sys
import time
from datetime import datetime

# Ana dizini Python yoluna ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from dotenv import load_dotenv
import streamlit as st
from models.huggingface_model import HuggingFaceModel

# .env dosyasını yükle
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# API anahtarı kontrolü
if not os.getenv("HUGGINGFACE_API_KEY"):
    st.error("⚠️ Hugging Face API anahtarı (HUGGINGFACE_API_KEY) bulunamadı. Lütfen .env dosyasını kontrol edin.")
    st.stop()

# Sayfa ayarları
st.set_page_config(
    page_title="Kütüphane Bilgi Chatbotu",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar HTML ve Ayarlar
with st.sidebar:
    st.image("data/marmara_logo.png", use_container_width=True)
    st.markdown("""
    <div class="sidebar-section">
        <h4>📚 Hoş Geldiniz!</h4>
        <p>Bu chatbot, üniversite kütüphanesi ile ilgili tüm sorularınızı yanıtlamak için burada. 
                Aşağıdaki konularda bilgi alabilirsiniz:
                - Kitap arama ve erişim
                - E-kitap ve veritabanları
                - Kütüphane çalışma saatleri
                - Üyelik ve ödünç alma işlemleri
                - Gecikme cezaları
                - Akademik kaynaklara erişim
                - Kütüphane etkinlikleri ve duyurular
                - Rezerve oda ve çalışma alanı kullanımı</p>
    </div>

    <div class="sidebar-section">
        <h4>📢 Duyuru</h4>
        <p>Final haftasında kütüphane 23:00’a kadar açık!</p>
    </div>

    <div class="sidebar-section">
        <h4>🧾 İpucu</h4>
        <p>"Kitap ödünç alma süresi nedir?" gibi doğal sorular sorabilirsiniz.</p>
    </div>
    """, unsafe_allow_html=True)

    st.title("⚙️ Ayarlar")

    # Sohbet temizleme
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.rerun()

    # Sohbet istatistikleri
    st.markdown("### İstatistikler")
    if "messages" in st.session_state:
        st.markdown(f"Toplam mesaj: {len(st.session_state.messages)}")
        if st.session_state.messages:
            last_message_time = st.session_state.messages[-1].get("timestamp", "")
            if last_message_time:
                st.markdown(f"Son mesaj: {last_message_time}")

model_option = st.sidebar.selectbox("Kullanılacak Model", ["HuggingFace TF-IDF", "Falcon LLM"])

# Model başlatıcı
@st.cache_resource

def load_model():
    try:
        if model_option == "Falcon LLM":
            from models.falcon_model import FalconModel
            return FalconModel()
        else:
            from models.huggingface_model import HuggingFaceModel
            return HuggingFaceModel()
    except Exception as e:
        st.error(f"⚠️ Model başlatma hatası: {str(e)}")
        st.stop()

# Model yükleniyor
with st.spinner("Model yükleniyor..."):
    model = load_model()

# Başlık ve açıklama
st.title("Kütüphane Bilgi Chatbotu")
st.markdown("""
Bu chatbot, üniversite kütüphanesi ile ilgili sorularınızı yanıtlamak için tasarlanmıştır.""")

# Sohbet geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# Sohbet geçmişini göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("id") not in st.session_state.feedback:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍", key=f"like_{message['id']}"):
                    st.session_state.feedback[message["id"]] = "positive"
                    st.success("Geri bildiriminiz için teşekkürler!")
            with col2:
                if st.button("👎", key=f"dislike_{message['id']}"):
                    st.session_state.feedback[message['id']] = "negative"
                    st.error("Geri bildiriminiz için teşekkürler!")

# Yeni mesaj girişi
if prompt := st.chat_input("Mesajınızı yazın...", key="chat_input_unique"):
    message_id = str(int(time.time()))
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "id": message_id,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Yanıt üretimi
    with st.chat_message("assistant"):
        with st.spinner("Yanıt hazırlanıyor..."):
            try:
                response = model.get_response(prompt)
                st.markdown(response)

                assistant_message_id = str(int(time.time()))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "id": assistant_message_id,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            except Exception as e:
                error_message = f"⚠️ Bir hata oluştu: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "id": assistant_message_id,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })

# Özel CSS
st.markdown("""
<style>
    .sidebar-logo {
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .sidebar-logo img {
        width: 120px;
        border-radius: 12px;
        box-shadow: 0 0 8px rgba(255,255,255,0.2);
    }

    .sidebar-section {
        margin: 1rem 0;
        padding: 1rem;
        background-color: #2e5475;
        border-radius: 10px;
    }

    .sidebar-section h4 {
        color: #ffffff;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }

    .sidebar-section p {
        font-size: 0.9rem;
        color: #d6e6f2;
    }

    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .stChatMessage.user {
        background-color: #d6ecfa;
        color: #1c3b57;
        align-self: flex-end;
    }

    .stChatMessage.assistant {
        background-color: #ffffff;
        border-left: 5px solid #1c3b57;
    }

    .stButton>button {
        background-color: #1c3b57;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }

    .stButton>button:hover {
        background-color: #0e253b;
    }
</style>
""", unsafe_allow_html=True) 