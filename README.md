# 📜 Latin Intertextuality Inspector

Interactive Streamlit app for quickly inspecting potential intertextual relationships between pairs of Latin text snippets. For every (original, paraphrased) pair it:

- Computes sentence embedding cosine similarity (SentenceTransformer).
- Computes a classification probability for intertextuality (sequence pair classifier).
- (If a background corpus is provided) Ranks the paraphrased text relative to all corpus segments when querying with the original text (lower rank = more similar compared to corpus items).
- Displays per‑pair progress bars and lets you download an annotated CSV.

The app loads Hugging Face models (defaults pre‑filled) and caches them between runs.

## ✨ Features
- Upload sentence pair CSV (required) and optional corpus CSV.
- Automatic model loading & GPU use if available.
- Per‑pair cosine similarity & probability scores.
- Dynamic rank-in-corpus (temporary, non-persisted insertion of the paraphrase) if a corpus is supplied.
- Download enriched results as CSV.

## 📁 Input CSV Formats

### 1. Sentence Pairs (required)
Required columns:
- `original`
- `paraphrased`

Optional columns (displayed if present):
- `case_id`
- `operation_description`

Example:
```csv
case_id,original,paraphrased,operation_description
1,arma virumque cano,arma virique cano,minor lexical variant
2,gallia est omnis divisa,gallia omnis in partes divisa,classical paraphrase
```

### 2. Corpus (optional)
Required columns:
- `id`
- `text`

Example:
```csv
id,text
VergAen-1,Arma virumque cano Troiae qui primus ab oris
CaesBG-1,Gallia est omnis divisa in partes tres
```
The corpus is embedded once; each pair’s paraphrased text is compared transiently (not added permanently).

## 🔧 Installation
```bash
pip install -r requirements.txt
```

Recommended Python version: 3.11.4 (other 3.11 patch versions usually work, but 3.11.4 is the tested baseline).

If you have a GPU & compatible PyTorch, install an appropriate torch build beforehand (optional but faster).

## 🚀 Run
```bash
streamlit run streamlit_app.py
```
Open the local URL Streamlit prints (usually http://localhost:8501).

## 🧪 Usage Flow
1. (Optional) Adjust model names in the sidebar fields (defaults use hosted Hugging Face models).
2. Upload the sentence pairs CSV (processing button enables once loaded).
3. (Optional) Upload a corpus CSV to enable rank computation.
4. Click “Process” to compute embeddings, probabilities, and (if corpus) ranks.
5. Scroll results; download the enriched CSV via the “Download Scored CSV” button.
