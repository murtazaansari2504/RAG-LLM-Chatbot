# 🤖 RAG LLM Chatbot

This project is a **document-centric Retrieval-Augmented Generation (RAG) chatbot** built with **Streamlit**, capable of handling **text PDFs, scanned PDFs, and images**. It combines **OCR, semantic embeddings, and LLM-based responses** to create an interactive chat experience.

## Features

1. **Text and Scanned PDF Support**
   - Extracts text directly from text-based PDFs.
   - Uses **Tesseract OCR** for scanned PDFs or images.

2. **Automatic Text Extraction**
   - Automatically detects if a PDF is scanned and applies OCR if necessary.

3. **RAG (Retrieval-Augmented Generation)**
   - Converts document text into **vector embeddings** using `sentence-transformers/all-MiniLM-L6-v2`.
   - Uses an **Ollama LLM (`deepseek-r1:1.5b`)** to generate answers based on relevant document chunks.
   - Enables semantic search within the uploaded document.

4. **Interactive Chat Interface**
   - Streamlit chat UI for asking questions about uploaded documents.
   - Maintains session-based conversation history during the session.

5. **Compact and Efficient**
   - Lightweight solution without a database or user login for personal/single-user usage.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/murtazaansari2504/RAG-LLM-Chatbot.git
cd RAG-LLM-Chatbot
