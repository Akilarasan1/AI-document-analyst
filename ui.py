import streamlit as st
import tempfile
from app import ask_question
from ocr_loader import load_image_document
from agent import ingest_documents

st.title("AI Document Analyst")

if "processed" not in st.session_state:
    st.session_state.processed = False

if "last_file" not in st.session_state:
    st.session_state.last_file = None


uploaded_file = st.file_uploader("Upload a document", type=["png", "jpg", "jpeg"])

if uploaded_file and uploaded_file.name != st.session_state.last_file:
    
    #  Step 1: Reset everything BEFORE doing anything
    st.session_state.pop("vectordb", None)
    st.session_state.processed = False
    st.session_state.last_file = uploaded_file.name

    st.success("File uploaded")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    docs = load_image_document(file_path)

    #  Step 2: Fresh ingestion
    vectordb = ingest_documents(docs, reset_db=True)

    #  Step 3: Store clean state
    st.session_state["vectordb"] = vectordb
    st.session_state.processed = True

    st.success("Document processed and stored")

st.divider()

with st.form("question_form"):
    question = st.text_input("Ask a question about the document")
    submitted = st.form_submit_button("Ask")
    if submitted:
        answer = ask_question(question)
        st.write(answer)

