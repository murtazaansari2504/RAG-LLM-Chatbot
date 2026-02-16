import os
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

import pytesseract
pytesseract.pytesseract.tesseract_cmd = \
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"


import pytesseract
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from pypdf import PdfReader

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="RAG LLM Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG LLM Chatbot")
st.caption("Supports Text PDFs + Scanned PDFs (OCR) + Images")

# ======================================================
# TEXT PDF READER
# ======================================================
def extract_text_from_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text.strip()

# ======================================================
# OCR HANDLER
# ======================================================
def extract_text_from_ocr(uploaded_file):
    text = ""
    if uploaded_file.type == "application/pdf":
        images = convert_from_bytes(uploaded_file.read())
        for img in images:
            text += pytesseract.image_to_string(img)
    else:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
    return text.strip()

# ======================================================
# AUTO TEXT EXTRACTOR
# ======================================================
def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf_text(uploaded_file)
        if len(text) < 50:  # text kam mila → scanned PDF
            uploaded_file.seek(0)
            text = extract_text_from_ocr(uploaded_file)
    else:
        text = extract_text_from_ocr(uploaded_file)
    return text

# ======================================================
# LOAD MODELS
# ======================================================
@st.cache_resource
def load_models():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.llm = Ollama(
        model="deepseek-r1:1.5b",
        request_timeout=300
    )

load_models()

# ======================================================
# FILE UPLOAD
# ======================================================
uploaded_file = st.file_uploader(
    "Upload PDF or Image",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    with st.spinner("Reading document & building index..."):
        text = extract_text(uploaded_file)

        if not text:
            st.error("❌ Unable to extract text from this file.")
            st.stop()

        document = Document(text=text)
        index = VectorStoreIndex.from_documents([document])
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact"
        )

    st.success("✅ Document ready for chat!")

    # ======================================================
    # SESSION STATE
    # ======================================================
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask anything about the document."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ======================================================
    # CHAT
    # ======================================================
    if user_input := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_engine.query(user_input)
                answer = response.response
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("📄 Upload a PDF or image to start chatting.")
