from __future__ import annotations  # keep list[str] type hints on Py ≤3.9
import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from corpus import CorpusWrapper


# ─────────── MODEL CONSTANTS ───────────
EMBED_MODEL_NAME = "julian-schelb/SPhilBerta-latin-intertextuality"  # "bowphs/PhilBerta"
# "julian-schelb/xlm-roberta-base-latin-intertextuality"
CLF_MODEL_NAME = "julian-schelb/PhilBerta-latin-intertextuality"
POS_CLASS_IDX = 1  # positive ("intertextual") is *first* label

# ─────────── CACHE LOADER ───────────


@st.cache_resource(show_spinner="🔄  Loading HF models …")
def load_models():
    """Load SentenceTransformer & classifier on one device (CPU in Streamlit Cloud, GPU if available)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    tokenizer = AutoTokenizer.from_pretrained(CLF_MODEL_NAME)
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        CLF_MODEL_NAME).to(device)
    clf_model.eval()
    return embedder, tokenizer, clf_model, device

# ─────────── HELPERS ───────────


def calc_cosine_similarity(embedder, original: str, paraphrase: str) -> float:
    """
    Compute cosine similarity between the embeddings of two input texts.
    """
    # Encode each input as a 1D vector
    original_vec = embedder.encode(
        original, convert_to_numpy=True).reshape(1, -1)
    paraphrase_vec = embedder.encode(
        paraphrase, convert_to_numpy=True).reshape(1, -1)

    # Compute cosine similarity between the two vectors
    similarity = cosine_similarity(original_vec, paraphrase_vec)
    return float(similarity[0, 0])


def calc_probability(tokenizer, model, original: str, paraphrase: str, device: str) -> float:
    """
    Compute P_positive for a single original/paraphrase pair.
    """
    inputs = tokenizer(
        original,
        paraphrase,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().cpu()

    return float(probs[POS_CLASS_IDX])

# ─────────── UI CONFIG ───────────
st.set_page_config(page_title="Inspector", layout="wide")
# st.title("📜 Intertextuality Quick-Check")

# ─────────── DATA LOADING ───────────

st.subheader("Model & Data Configuration")

df: Optional[pd.DataFrame] = None

if os.path.exists("test_cases.csv"):
    try:
        df = pd.read_csv("test_cases.csv")
    except Exception as e:
        pass

clf_model_name = st.text_input(
    "Name of the Classification Model:", CLF_MODEL_NAME
)
embed_model_name = st.text_input(
    "Name of the Sentence Transformer Model:", EMBED_MODEL_NAME
)

uploaded = st.file_uploader("File with Sentence Pairs:", type="csv")
if uploaded is not None:
    df = pd.read_csv(uploaded)
    with st.expander("Preview Sentence Pairs (first 5 rows)"):
        st.dataframe(df.head())

# Optional secondary corpus file (id,text)
corpus_df: Optional[pd.DataFrame] = None
corpus_file = st.file_uploader(
    "Optional Corpus File (two columns: id,text):", type="csv", key="corpus_upload"
)
if corpus_file is not None:
    try:
        corpus_df = pd.read_csv(corpus_file)
        required_cols = {"id", "text"}
        missing_corpus = required_cols - set(corpus_df.columns)
        if missing_corpus:
            st.error(
                f"Corpus CSV missing required column(s): {', '.join(missing_corpus)}"
            )
            corpus_df = None
        else:
            with st.expander("Preview Corpus (first 5 rows)"):
                st.dataframe(corpus_df.head())
    except Exception as e:
        st.error(f"Failed to read corpus file: {e}")

if df is None or df.empty:
    st.info("Upload a CSV or place **test_cases.csv** next to the script.")
    st.stop()

missing = {"original", "paraphrased"} - set(df.columns)
if missing:
    st.error("CSV missing required column(s): " + ", ".join(missing))
    st.stop()

# Enable the Process button only when both model names are specified and a CSV file is uploaded.
can_process = bool(
    clf_model_name and embed_model_name and uploaded is not None)
if not st.button("Process", disabled=not can_process):
    st.stop()

if can_process:
    # ─────────── MODEL INFERENCE ───────────
    embedder, tokenizer, clf_model, device = load_models()

    # Encode corpus if file uploaded
    corpus_wrapper = None
    if 'corpus_wrapper' in st.session_state:
        prev = st.session_state['corpus_wrapper']
        if corpus_df is not None and len(prev) != len(corpus_df):
            st.session_state.pop('corpus_wrapper')

    if corpus_df is not None and 'corpus_wrapper' not in st.session_state:
        with st.spinner('🧱 Indexing corpus (embedding segments) …'):
            try:
                corpus_wrapper = CorpusWrapper(corpus_df, embedder)
                st.session_state['corpus_wrapper'] = corpus_wrapper
                st.success(f"Corpus indexed: {len(corpus_wrapper)} segments.")
            except Exception as e:
                st.error(f"Failed to index corpus: {e}")
    elif 'corpus_wrapper' in st.session_state:
        corpus_wrapper = st.session_state['corpus_wrapper']
         

    # Calculate similarity scores
    with st.spinner("🔎 Scoring pairs …"):

        cosine_list = []
        prob_list = []
        rank_list = []
        total_list = []
        use_corpus_ranking = corpus_wrapper is not None and len(corpus_wrapper) > 0
        if use_corpus_ranking:
            corpus_embs = corpus_wrapper.embeddings  # (N, D)
            corpus_norm = corpus_wrapper.normalize
        else:
            corpus_embs = None
            corpus_norm = True
        for orig, para in zip(df["original"], df["paraphrased"]):
            # compute cosine similarity for a single pair
            sim = calc_cosine_similarity(embedder, orig, para)
            # compute probability for a single pair
            prob = calc_probability(tokenizer, clf_model, orig, para, device)
            cosine_list.append(round(sim, 3))
            prob_list.append(round(prob, 3))

            # Rank computation: treat paraphrase as candidate in temporary extended corpus for this query
            if use_corpus_ranking:
                rank, total, _score = corpus_wrapper.calc_rank(orig, para)
                rank_list.append(rank)
                total_list.append(total)
            else:
                rank_list.append(None)
                total_list.append(None)

        df["cosine_similarity"] = cosine_list
        df["P_positive"] = prob_list
        if use_corpus_ranking:
            df["paraphrase_rank"] = rank_list
            df["rank_total_docs"] = total_list

    # ─────────── LAYOUT ───────────
    st.subheader("Results")
    # st.markdown("Number of sentence pairs: {}".format(len(df)))

    for idx, row in df.iterrows():
        with st.container(border=True, key=f"row_{idx}"):
            # Divide the layout into two columns: left for text details, right for metrics.
            left_col, right_col = st.columns([2, 1])

            with left_col:
                st.markdown(
                    f"**Case ID:** {row.get('case_id', 'N/A')}  \n"
                    f"**Original Text:** {row['original']}  \n"
                    f"**Paraphrased Text:** {row['paraphrased']}  \n"
                    f"**Comment:** {row['operation_description']}"
                )

            with right_col:
                # Display cosine similarity
                sim_value = row["cosine_similarity"]
                sim_pct = max(-1, min(1, sim_value))  # Clamp the value
                st.progress(int(sim_pct * 100),
                            text=f"**Cosine Similarity:** `{sim_value:.3f}`")

                # Actual rank progress bar
                if corpus_wrapper is not None and len(corpus_wrapper) > 0 and 'paraphrase_rank' in df.columns:
                    rank_value = row.get('paraphrase_rank')
                    total_docs = row.get('rank_total_docs')
                    if rank_value is not None and total_docs:
                        if total_docs > 1:
                            rank_pct = int((1 - (rank_value - 1) / (total_docs - 1)) * 100)
                        else:
                            rank_pct = 100
                        st.progress(
                            rank_pct,
                            text=f"**Rank in Corpus Search:** `{rank_value}/{total_docs}`"
                        )
                    else:
                        st.progress(0, text="**Rank in Corpus Search:** `N/A`")
                else:
                    st.progress(0, text="**Rank in Corpus Search:** `N/A` (no corpus)")
   

                # Display probability
                prob_value = row["P_positive"]
                # Clamp the value to [0, 1]
                prob_pct = max(0, min(1, prob_value))
                st.progress(int(prob_pct * 100),
                            text=f"**Probability:** `{prob_value:.3f}`")


    # ─────────── DOWNLOAD BUTTON ───────────
    st.download_button(
        "💾 Download Scored CSV",
        data=df.to_csv(index=False).encode(),
        file_name="results_with_scores.csv",
        mime="text/csv",
    )
