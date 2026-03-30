import streamlit as st
import tempfile
from app import ask_question
from ocr_loader import load_image_document
from agent import ingest_documents
import threading

st.title("AI Document Analyst")

if "processed" not in st.session_state:
    st.session_state.processed = False

if "last_file" not in st.session_state:
    st.session_state.last_file = None


uploaded_file = st.file_uploader("Upload a document", type=["png", "jpg", "jpeg"])

if uploaded_file and uploaded_file.name != st.session_state.get("last_file"):

    # 🚨 Reset state
    st.session_state.pop("vectordb", None)
    st.session_state.processed = False
    st.session_state.last_file = uploaded_file.name

    st.success("File uploaded")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    docs = load_image_document(file_path)
    st.session_state["docs"] = docs

    st.success("Document loaded successfully (ready for questions)")

st.divider()

if "docs" in st.session_state:
    with st.form("question_form"):
        question = st.text_input("Ask a question about the document")
        submitted = st.form_submit_button("Ask")

        if submitted:
            docs = st.session_state.get("docs")
            vectordb = st.session_state.get("vectordb")

            if vectordb:
                answer = ask_question(question)  # use DB
            else:
                answer = ask_question(question, docs=docs)  # fallback
            st.write(answer)
else:
    st.info("📄 Please upload a document to start asking questions.")


docs = st.session_state.get("docs", None)
if docs and "vectordb" not in st.session_state and submitted:
    with st.spinner("Optimizing document for faster search..."):
        vectordb = ingest_documents(docs, reset_db=True)
        st.session_state["vectordb"] = vectordb
        st.session_state.processed = True

# "posthog>=2.4.0,<6.0.0"
