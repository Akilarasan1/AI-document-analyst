import streamlit as st
import tempfile
from app import ask_question
from ocr_loader import load_image_document
from agent import ingest_documents

st.title("AI Document Analyst")

# Upload section
uploaded_file = st.file_uploader(
    "Upload a document",type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.success("File uploaded")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # OCR extraction
    docs = load_image_document(file_path)
    # print(docs)

    ingest_documents(docs)

    st.success("Document processed and stored")

st.divider()

question = st.text_input("Ask a question about the document")

if st.button("Ask"):

    answer = ask_question(question)

    st.write(answer)