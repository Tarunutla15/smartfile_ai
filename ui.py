# ui.py
import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000/v1"

st.set_page_config(page_title="SmartFile AI", layout="wide")
st.title("📄 SmartFile AI - Chat with Your PDFs")

# --- PDF Upload Section ---
st.subheader("Upload your PDF(s)")
uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Upload PDFs"):
    files = [("files", (file.name, file.getvalue(), "application/pdf")) for file in uploaded_files]
    with st.spinner("Uploading and processing PDFs..."):
        res = requests.post(f"{API_BASE_URL}/upload-pdf/", files=files)
    if res.status_code == 200:
        st.success(res.json()["message"])
    else:
        st.error(f"Upload failed: {res.text}")

# --- Chat Section ---
st.subheader("Chat with your PDFs")
query = st.text_input("Enter your question")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            res = requests.post(f"{API_BASE_URL}/chat/", json={"query": query})
        if res.status_code == 200:
            st.markdown(f"**Answer:** {res.json()['response']}")
        else:
            st.error(f"Error: {res.text}")
