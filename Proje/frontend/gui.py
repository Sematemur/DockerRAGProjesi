import streamlit as st
import requests

import os

API_URL_ASK = os.getenv("BACKEND_URL", "http://backend:8000") + "/ask"
API_URL_UPLOAD = os.getenv("BACKEND_URL", "http://backend:8000") + "/uploadfile/"


st.set_page_config(page_title="PDF Asistanı", layout="wide")
st.title("📄🧠 PDF Soru-Cevap Asistanı")

# PDF Yükleme
st.subheader("📤 PDF Yükle")
uploaded_file = st.file_uploader("PDF dosyanızı seçin", type=["pdf"])

if uploaded_file is not None:
    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
    response = requests.post(API_URL_UPLOAD, files=files)

    if response.status_code == 200:
        st.success(f"✅ PDF yüklendi: {uploaded_file.name}")
    else:
        st.error("❌ PDF yüklenemedi!")

# Chat geçmişi saklama
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("---")
st.subheader("💬 Soru Sor")

user_input = st.text_input("Bir soru yazın:")

if st.button("Sor"):
    if user_input:
        payload = {
            "input": user_input,
            "chat_history": [
                {"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
                for m in st.session_state.chat_history
            ]
        }

        response = requests.post(API_URL_ASK, json=payload)

        if response.status_code == 200:
            answer = response.json()["answer"]
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            st.error("❌ API'den yanıt alınamadı!")

# Geçmiş gösterimi
st.markdown("---")
st.subheader("🗂️ Sohbet Geçmişi")

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**👤 Kullanıcı:** {msg['content']}")
    else:
        st.markdown(f"**🤖 Asistan:** {msg['content']}")
