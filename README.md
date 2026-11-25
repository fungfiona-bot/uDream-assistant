# uDream.AI Product Assistant - Nov 2026

**Project Overview:** This is the chatbot as OSIM uDream.AI massage chair (OS-8258) product assistant for the AI Builders Bootcamp final project. The chatbot answers questions about setup, controls, app pairing, care, and troubleshooting using the supplied operation manual.

**Problem it solved:** The operation manual contains the answers, but finding them can require searching several sections and reading tables or diagrams. The chatbot brings relevant passages into one answer and links each citation to its source excerpt and full PDF page

## RAG pipeline

The chatbot uses retrieval-augmented generation (RAG). It prepares the manual for semantic search by converting short passages into vectors and saving them in Chroma. When a user asks a question, the app converts the question into a vector using the same embedding model and looks up the closest passages. This helps it match questions phrased differently from the manual.  It filters the retrieved passages by cosine distance, then sends the question and the remaining evidence to the language model. The model uses those passages to generate a cited answer. If no passages pass the filter, the app returns a fallback message.

## Chatbot architecture

**Document → Load → Clean → Chunk → Embed → Store → Query → Retrieve → Filter → Generate → Cite**

| Stage | What happens in uDream |
|---|---|
| **Document** | Five PDFs in `documents/` cover parts and controls, preparation, app download, care, and troubleshooting. |
| **Load** | `rag.py` reads the PDFs page by page and records the original manual page number. It uses `pdfplumber` to extract the Bluetooth LED and troubleshooting tables as paired rows. |
| **Clean** | The app removes formatting noise and repairs some broken line endings. Searchable labels such as `[+] button` and `[▲] button` come directly from the updated PDF. |
| **Chunk** | Ordinary text is split into overlapping passages within each page. The packaged defaults are **850 characters** with **120 characters of overlap**. Each extracted table row stays together as one chunk. |
| **Embed** | OpenAI’s `text-embedding-3-small` converts each chunk into a numeric vector for semantic search. |
| **Store** | Chroma stores the vectors with their chunk text, IDs, and source metadata. The app can reuse the index when the indexed content and embedding model match. |
| **Query** | The user enters a question in Streamlit. For a recognized follow-up, the app can include the previous question in the search query. It embeds the query using the same model used for the chunks. |
| **Retrieve** | Chroma returns up to `TOP_K` of the closest chunks, ranked by cosine distance. The packaged default is **5**. Lower distances indicate closer matches. |
| **Filter** | The app keeps chunks at or below `MAX_DISTANCE`, which defaults to **0.75**. If no chunks pass, it returns the fallback message. |
| **Generate** | `rag.py` sends the question and numbered evidence to `gpt-5-mini` through `client.responses.create()`. The prompt asks the model to answer from the evidence, cite it, and end after answering. |
| **Cite** | The app checks that citation numbers refer to chunks supplied to the model. Users can read the excerpts in **View sources** and select **View original page** to see the full PDF page, including diagrams and tables. |

**Document through Store** prepares the searchable knowledge base. **Query through Cite** runs for each question. The separate `export_processed.py` command writes the cleaned text before chunking to `processed/` for inspection.

## Features Highlights
1.  Added row-aware PDF table extraction
2.  Added the prompt rule that ends replies after the answer
3.  Added citation-to-page mapping and an inline View original page control
4.  Added pdfplumbe for table extraction and PDF page previews.
5.  Added an offline export command and generated text files for all five manual PDFs.
6.  Added retrieval evaluator (similar to app.py in Lesson 17)


## Run locally

In the PyCharm terminal, open the folder containing `app.py` and install the dependencies.

```powershell
pip install openai chromadb numpy pandas matplotlib pypdf pdfplumber
```

For first-time setup, enter `OPENAI_API_KEY` in `.streamlit/secrets.toml`.   Keep that file out of version control. The app also accepts the key as an environment variable.

```powershell
streamlit run app.py
```

Ask a question such as **What does a fast blinking blue Bluetooth light mean?** Under the answer, open **View sources** for the cited excerpts. Select **View original page** beneath a PDF citation to display the full page, including diagrams and tables.

## Manual sources

| File | Original PDF viewer pages |
| --- | --- |
| `07-09_Names_of_Parts_and_Functions.pdf` | 7–9 |
| `10-16_Preparation_before_use.pdf` | 10–16 |
| `17-18_App_Download.pdf` | 17–18 |
| `19-19_Product_care_and_storage.pdf` | 19 |
| `20-22_Troubleshooting.pdf` | 20–22 |

Citations use the original PDF viewer numbering. For example, **Manual PDF page 8** opens page 2 of the bundled parts PDF.

## How the chatbot works

1. **Extract and clean.** `rag.py` loads PDF pages or TXT sections and cleans the extracted text. On manual page 8, the standard PDF extractor reads the control instructions directly, including searchable `[+] button`, `[-] button`, `[▲] button`, and `[▼] button` labels.
2. **Preserve table rows.** `bluetooth_led_rows()` uses `pdfplumber` for the four Bluetooth LED rows on page 8. The loader removes the flattened version of that table from the page text so the index uses the correctly paired states and meanings. `troubleshooting_records()` extracts the 14 troubleshooting rows on pages 20–22 and keeps each problem with its causes and actions.
3. **Chunk and index.** Ordinary text uses windows of up to **500 characters** with **75 characters of overlap** in the packaged defaults. Each extracted table row is kept together. The app embeds the chunks with `text-embedding-3-small` and stores them with page metadata in persistent Chroma. A content fingerprint lets later questions reuse the index when the indexed text and embedding model match.
4. **Retrieve and answer.** The app searches up to **5** chunks and keeps results within a cosine distance of **0.75**. The cosine distance in this chatbot is tailored to 0 to 2 scale, with lower number meaning closer in similarity (i.e. 0 = same direction, 2 = opposite direction).  It sends the question and retrieved evidence to `gpt-5-mini` using `client.responses.create()`. The model is instructed to cite the supplied evidence, use a fallback when evidence is insufficient, and **end after answering without follow-up offers**.
5. **Show sources.** The app checks that the reply's citation IDs refer to supplied chunks. The source viewer displays the excerpts and **renders the cited PDF page** on request.

## Inspect the processed text

Run this from the project folder to regenerate the readable text exports after changing a source document or the extraction logic.

```powershell
python export_processed.py
```

The exports in `processed/` show the cleaned text **before chunking**, with manual page headings and the paired table rows. The exporter shares the chatbot's extraction code and runs without an API key. The chatbot builds its search index from `documents/`. Refreshing the text exports is a separate command.


## Evaluation and deployment

Run `streamlit run evaluate.py` to open the retrieval evaluator (similar to app.py in Lesson 17). Select a sample or custom question, choose Top-K values from 1 to 10, set the maximum cosine distance, and click **Run comparison**. The results show how many passages each Top-K includes, which pass the cutoff, and the text and distance for each ranked passage. Enable **Generate an answer for each Top-K** to compare model replies and citations. This uses additional OpenAI calls. The original command-line report remains available with `python evaluate.py`, and `python evaluate.py --offline` checks document extraction without an API key. Check that the retrieved excerpts support the answers, that an out-of-scope question receives the fallback, and that PDF citations open the expected pages. 

