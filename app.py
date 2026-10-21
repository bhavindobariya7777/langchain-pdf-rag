"""Run with: streamlit run app.py."""

import logging

import streamlit as st

logger = logging.getLogger(__name__)

from pdf_rag.config import Settings
from pdf_rag.service import RagService

st.set_page_config(page_title="PDF Knowledge Chat", page_icon="📚")
st.title("📚 PDF Knowledge Chat")
st.caption("LangChain · OpenAI · local ChromaDB")
st.info(
    "Place PDFs in documents/, then index them. Keep the daily worker running for automatic scans."
)

try:
    settings = Settings()
    service = RagService(settings)
except Exception:
    logger.exception("Initialization failed")
    st.error("Could not initialize. Check .env, your OpenAI API key, and the setup instructions.")
    st.stop()

with st.sidebar:
    st.header("Document index")
    st.write(f"Folder: {settings.documents_dir}")
    st.caption(f"Daily scan: {settings.scan_hour:02}:{settings.scan_minute:02} {settings.timezone}")
    if st.button("Scan PDFs now"):
        try:
            with st.spinner("Indexing PDFs…"):
                st.json(service.sync())
        except Exception:
            logger.exception("Request failed")
            st.error("Indexing failed. Check worker logs and API configuration.")

if "messages" not in st.session_state:
    st.session_state.messages = []
for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

if question := st.chat_input("Ask a question about your PDFs"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents…"):
                result = service.ask(question)
            st.markdown(result["answer"])
            for source in result["sources"]:
                with st.expander(f"[{source['id']}] {source['file']} — page {source['page']}"):
                    st.text(source["excerpt"])
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        except Exception:
            logger.exception("Request failed")
            st.error("Request failed. Check your API key, billing, connection, and terminal logs.")
