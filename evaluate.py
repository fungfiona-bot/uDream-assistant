import argparse
import json
import os
import tomllib
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from rag import FALLBACK, MAX_DISTANCE, answer, index, retrieve

ROOT = Path(__file__).resolve().parent

CASES = [
    ("Direct", "Where does the manual explain app download?", "App Download"),
    ("Direct", "What parts of the chair can I adjust?", "Names of Parts & Functions or Preparation before use"),
    ("Direct", "What does the manual say about preparing the chair for use?", "Preparation before use"),
    ("Paraphrase", "How do I look after the chair and keep it clean?", "Product Care"),
    ("Paraphrase", "Where should I look if the chair stops working?", "Troubleshooting"),
    ("Misspelled", "How do I downlaod the app?", "App Download"),
    ("Multiple sections", "Where can I learn to use the chair controls and download the app?", "Names of Parts & Functions and App Download"),
    ("Multiple sections", "What should I know about setting up the chair and looking after it?", "Preparation before use and Product Care"),
    ("Irrelevant", "What is the capital of Iceland?", "Fallback"),
    ("Irrelevant", "How do I make sourdough bread?", "Fallback"),
    ("Vague", "Can it do that?", "Fallback without prior context"),
]

def offline_check(root):
    from rag import clean_text, overlapping_chunks, records
    documents = sorted((root / "documents").glob("*.pdf"))
    assert len(documents) == 5
    for document in documents:
        sections = list(records(document))
        start, end = map(int, document.stem.split("_", 1)[0].split("-"))
        assert {section for _, _, section, _ in sections} == {
            f"Manual PDF page {page}" for page in range(start, end + 1)
        }
        assert sections[0][2] == f"Manual PDF page {start}"
        assert sections[-1][2] == f"Manual PDF page {end}"
        assert any(body for _, _, _, body in sections)
    text = "A massage program uses the chair controls. " * 100
    segments = overlapping_chunks(clean_text(text))
    assert len(segments) > 1 and all(0 < len(segment) <= 850 for segment in segments)
    assert clean_text("Page 23\nExam-\nple   text") == "Example text"
    print(f"Offline checks passed for {len(documents)} documents, cleaning, and chunking.")

def evaluation_key(root):
    secrets = root / ".streamlit" / "secrets.toml"
    key = os.getenv("OPENAI_API_KEY")
    if not key and secrets.exists():
        key = tomllib.loads(secrets.read_text(encoding="utf-8")).get("OPENAI_API_KEY")
    return key


def streamlit_main():
    st.set_page_config(page_title="uDream Retrieval Evaluation", layout="wide")
    st.title("uDream Retrieval Evaluation")
    labels = [f"{category} | {question}" for category, question, _ in CASES]
    selection = st.selectbox("Test question", ["Custom question", *labels], index=1)
    if selection == "Custom question":
        question = st.text_input("Question", max_chars=1000).strip()
        expected = ""
    else:
        _, question, expected = CASES[labels.index(selection)]
    if expected:
        st.caption(f"Expected: {expected}")

    with st.sidebar:
        top_ks = sorted(st.multiselect("Top-K values", range(1, 11), default=[3, 5, 7]))
        threshold = st.slider("Maximum cosine distance", 0.0, 2.0, MAX_DISTANCE, 0.01)
        generate = st.checkbox("Generate an answer for each Top-K", value=False)

    config = (question, tuple(top_ks), threshold, generate)
    if st.button("Run comparison"):
        st.session_state.pop("evaluation_result", None)
        if not question or not top_ks:
            st.warning("Enter a question and select at least one Top-K value.")
            return
        key = evaluation_key(ROOT)
        if not key:
            st.warning("Configure OPENAI_API_KEY before running evaluation.")
            return

        import chromadb
        from openai import OpenAI

        try:
            with st.spinner("Searching the manual…"):
                client = OpenAI(api_key=key, timeout=30)
                database = chromadb.PersistentClient(path=str(ROOT / "rag_chroma_db"))
                collection, warnings = index(client, database, ROOT / "documents")
                rows, timings = retrieve(client, collection, question, top_k=max(top_ks))
                answers = {
                    k: answer(client, collection, question, top_k=k, threshold=threshold)
                    for k in top_ks
                } if generate else {}
            st.session_state.evaluation_result = (config, rows, timings, warnings, answers)
        except Exception as error:
            st.error(f"Evaluation could not finish ({type(error).__name__}). Check your API key, connection, and local index.")
            return

    result = st.session_state.get("evaluation_result")
    if not result or result[0] != config:
        return

    _, rows, timings, warnings, answers = result
    for warning in warnings:
        st.warning(warning)
    st.caption("Each Top-K comparison uses the first K results from one search at the largest selected K. Lower cosine distance indicates a closer match.")
    comparisons = []
    for k in top_ks:
        candidates = rows[:k]
        passing = [row for row in candidates if row["distance"] <= threshold]
        comparisons.append({
            "Top-K": k, "Retrieved": len(candidates), "Passing cutoff": len(passing),
            "Passing ranks": ", ".join(str(row["number"]) for row in passing) or "None",
        })
    st.dataframe(comparisons, hide_index=True)

    st.subheader("Retrieved passages")
    for row in rows:
        meta = row["metadata"]
        status = "Pass" if row["distance"] <= threshold else "Filtered"
        with st.expander(f'Rank {row["number"]} · Distance {row["distance"]:.4f} · {status} · {meta["title"]}'):
            st.caption(f'{meta["section"]} · {row["id"]}')
            st.text(row["text"])

    if answers:
        st.subheader("Generated answers")
        st.caption("Each answer uses its own retrieval at the selected Top-K.")
        for k, tab in zip(top_ks, st.tabs([f"Top-K {k}" for k in top_ks])):
            response, citations, _ = answers[k]
            with tab:
                st.markdown(response)
                for row in citations:
                    meta = row["metadata"]
                    st.caption(f'[{row["number"]}] {meta["title"]} · {meta["section"]}')
    with st.expander("Retrieval timings"):
        st.json(timings)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    root = Path(__file__).resolve().parent
    if parser.parse_args().offline:
        offline_check(root)
        return
    import chromadb
    from openai import OpenAI

    key = evaluation_key(root)
    if not key:
        raise SystemExit("Configure OPENAI_API_KEY before running live evaluation.")
    client = OpenAI(api_key=key, timeout=30)
    collection, warnings = index(client, chromadb.PersistentClient(path=str(root / "rag_chroma_db")), root / "documents")
    if warnings:
        print("\n".join(warnings))
    for category, question, expected in CASES:
        rows, _ = retrieve(client, collection, question, top_k=7)
        comparisons = {
            f"k={k}, distance<={limit}": [row["id"] for row in rows[:k] if row["distance"] <= limit]
            for k in (3, 5, 7) for limit in (0.55, 0.65, 0.75)
        }
        response, citations, timings = answer(client, collection, question)
        print(json.dumps(dict(category=category, question=question, expected=expected,
                              comparisons=comparisons, answer=response, citations=citations,
                              rejected=response == FALLBACK, timings=timings,
                              review={"retrieval": "pending", "accuracy": "pending",
                                      "source": "pending", "rejection": "pending"}),
                         ensure_ascii=False, indent=2))

if __name__ == "__main__":
    if get_script_run_ctx(suppress_warning=True):
        streamlit_main()
    else:
        main()
