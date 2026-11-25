from pathlib import Path
import os
import re

import chromadb
import streamlit as st
from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError

from rag import answer, index, TOP_K, MAX_DISTANCE

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="uDream.AI Product Assistant", page_icon="💺")
st.title("uDream.AI Product Assistant")
st.write("Find answers about the OSIM OS-8258 chair, its controls, and the Well-Being app.")
st.caption("Independent class project · Answers use the supplied OSIM uDream.AI operation manual.")

with st.sidebar:
    st.header("Knowledge base")
    for path in sorted((ROOT / "documents").glob("*.pdf")):
        title = path.stem.split("_", 1)[-1].replace("_", " ")
        st.text(title)
    if st.button("Clear conversation"):
        st.session_state.messages = []
    with st.expander("Retrieval settings"):
        st.caption(f"Top {TOP_K} · Maximum cosine distance {MAX_DISTANCE:.2f} (lower is closer)")

def original_pdf_page(metadata):
    """Resolve a manual citation to a zero-based page in its bundled PDF."""
    source = metadata.get("source", "")
    if Path(source).name != source or Path(source).suffix.lower() != ".pdf":
        return None
    citation = re.fullmatch(r"Manual PDF page (\d+)", metadata.get("section", ""))
    if not citation:
        return None
    page_range = re.fullmatch(r"(\d+)-(\d+)_(.+)", Path(source).stem)
    first_page = int(page_range[1]) if page_range else 1
    page_index = int(citation[1]) - first_page
    if page_index < 0:
        return None
    return ROOT / "documents" / source, page_index


def show(message, message_index):
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message.get("sources"):
            with st.expander("View sources"):
                for source_index, row in enumerate(message["sources"]):
                    meta = row["metadata"]
                    st.write(f'[{row["number"]}] {meta["title"]} · {meta["section"]}')
                    st.caption(f'{row["id"]} · Cosine distance {row["distance"]:.3f}')
                    if meta.get("url"):
                        st.link_button("Open original document", meta["url"])
                    st.text(row["text"])
                    pdf_page = original_pdf_page(meta)
                    if pdf_page and st.checkbox(
                        "View original page", key=f"pdf-page-{message_index}-{source_index}"
                    ):
                        import pdfplumber

                        path, page_index = pdf_page
                        try:
                            with pdfplumber.open(path) as pdf:
                                page = pdf.pages[page_index].to_image(resolution=144)
                                st.image(page.original, caption=meta["section"])
                        except (OSError, IndexError, ValueError, pdfplumber.utils.exceptions.PdfminerException):
                            st.warning("The original PDF page could not be displayed.")

st.session_state.setdefault("messages", [])
if not st.session_state.messages:
    st.info("Try asking about the chair controls, app setup, or cleaning.")
for message_index, message in enumerate(st.session_state.messages):
    show(message, message_index)

try:
    key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
except (FileNotFoundError, KeyError):
    key = os.getenv("OPENAI_API_KEY")
if not key:
    st.warning("Add OPENAI_API_KEY to Streamlit secrets to start chatting. See the README for setup.")

@st.cache_resource
def get_database():
    return chromadb.PersistentClient(path=str(ROOT / "rag_chroma_db"))

question = st.chat_input("Ask about the OSIM uDream.AI", disabled=not key, max_chars=1000)
if question and question.strip():
    message = {"role": "user", "text": question.strip()}
    st.session_state.messages.append(message)
    show(message, len(st.session_state.messages) - 1)
    try:
        with st.spinner("Finding an answer in the product documents…"):
            client = OpenAI(api_key=key, timeout=30)
            collection, warnings = index(client, get_database(), ROOT / "documents")
            for warning in warnings:
                st.warning(warning)
            previous = next(
                (m["text"] for m in reversed(st.session_state.messages[:-1]) if m["role"] == "user"), ""
            )
            response, sources, timings = answer(client, collection, message["text"], previous)
            with st.sidebar.expander("Latest response timings"):
                st.json(timings)
    except AuthenticationError:
        response, sources = "OpenAI rejected the API key. Check your Streamlit secrets.", []
    except (APIConnectionError, RateLimitError):
        response, sources = "OpenAI is unavailable or the usage limit was reached. Please try again later.", []
    except ValueError as error:
        response, sources = str(error), []
    except Exception:
        response, sources = "The document search could not finish. Check the OpenAI connection and the local index, then retry.", []
    message = {"role": "assistant", "text": response, "sources": sources}
    st.session_state.messages.append(message)
    show(message, len(st.session_state.messages) - 1)
