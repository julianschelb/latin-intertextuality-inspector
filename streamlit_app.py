from __future__ import annotations  # keep list[str] type hints on Py ≤3.9
import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics.pairwise import cosine_similarity


# ─────────── MODEL CONSTANTS ───────────
EMBED_MODEL_NAME = "julian-schelb/SPhilBerta-latin-intertextuality"  # "bowphs/PhilBerta"
# "julian-schelb/xlm-roberta-base-latin-intertextuality"
CLF_MODEL_NAME = "julian-schelb/PhilBerta-latin-intertextuality"
POS_CLASS_IDX = 1  # positive ("intertextual") is *first* label
COLOR_MAP = LinearSegmentedColormap.from_list(
    "light_blues", ["#ffffff", "#2676b8"])
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


# =================== ATTENTION WIEGHTS ===================


# def get_avg_attention_per_token(
#     tokenizer, model, para: str, orig: str, device: str,
#     max_tokens: int = 512, filter_special_tokens: bool = True
# ) -> tuple[list, list]:
#     """Return two lists of (token, average_attention_received) pairs for para and orig separately."""
#     enc = tokenizer(para.strip(), orig.strip(), return_tensors="pt",
#                     truncation=True, max_length=max_tokens).to(device)

#     with torch.no_grad():
#         attn = model(**enc, output_attentions=True).attentions[-4]
#         attn = attn[0].mean(dim=0).cpu().numpy()  # (seq_len, seq_len)

#     input_ids = enc["input_ids"][0][:max_tokens]
#     tokens = tokenizer.convert_ids_to_tokens(input_ids)
#     attn = attn[:len(tokens), :len(tokens)]

#     # Compute attention received
#     avg_received = attn.mean(axis=0)

#     # Find separator token index to split para and orig
#     sep_id = 2  # tokenizer.sep_token_id
#     print(input_ids)
#     sep_indices = (input_ids == sep_id).nonzero(as_tuple=True)[0].tolist()

#     if len(sep_indices) < 1:
#         raise ValueError(
#             "Could not find separator token to split para and orig.")

#     # para ends at first [SEP], orig starts after
#     split_index = sep_indices[0] + 1

#     # Split tokens, attention scores, and ids
#     para_parts = list(
#         zip(tokens[:split_index], avg_received[:split_index], input_ids[:split_index]))
#     orig_parts = list(
#         zip(tokens[split_index:], avg_received[split_index:], input_ids[split_index:]))

#     if filter_special_tokens:
#         special_ids = tokenizer.all_special_ids
#         para_parts = [(tok, score) for tok, score,
#                       tok_id in para_parts if tok_id.item() not in special_ids]
#         orig_parts = [(tok, score) for tok, score,
#                       tok_id in orig_parts if tok_id.item() not in special_ids]

#     return para_parts, orig_parts


# def attention_tokens_to_html(token_attention: list, cmap: str = "Blues") -> str:
#     """Render tokens as colored boxes with pill-style visual grouping for subwords."""
#     import matplotlib.cm as cm
#     import matplotlib.colors as mcolors

#     tokens, scores = zip(*token_attention)

#     # Normalize attention scores
#     norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
#     colormap = COLOR_MAP  # cm.get_cmap(cmap)

#     html = ""
#     for i, (token, score) in enumerate(token_attention):
#         rgba = colormap(norm(score))
#         hex_color = mcolors.to_hex(rgba)
#         clean_token = token.replace("Ġ", "").replace("▁", "").replace("##", "")

#         is_start = (
#             i == 0
#             or token.startswith("Ġ")
#             or token.startswith("▁")
#             or token.startswith("<")
#             or token.startswith("[")
#         )
#         is_end = (
#             i == len(tokens) - 1
#             or tokens[i + 1].startswith("Ġ")
#             or tokens[i + 1].startswith("▁")
#             or tokens[i + 1].startswith("<")
#             or tokens[i + 1].startswith("[")
#         )

#         # Border radius logic
#         if is_start and is_end:
#             border_radius = "6px"
#         elif is_start:
#             border_radius = "6px 0 0 6px"
#         elif is_end:
#             border_radius = "0 6px 6px 0"
#         else:
#             border_radius = "0"

#         # Padding logic
#         if is_start or is_end:
#             padding = "2px 6px"
#         else:
#             padding = "2px 4px"

#         # Add space between word groups
#         if is_start and i != 0:
#             html += " "

#         html += f'<span style="background-color:{hex_color}; padding:{padding}; margin:1px 0px; border-radius:{border_radius}; display:inline-block;">{clean_token}</span>'

#     return html


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

    with st.spinner("🔎 Scoring pairs …"):

        cosine_list = []
        prob_list = []
        for orig, para in zip(df["original"], df["paraphrased"]):
            # compute cosine similarity for a single pair
            sim = calc_cosine_similarity(embedder, orig, para)
            # compute probability for a single pair
            prob = calc_probability(tokenizer, clf_model, orig, para, device)
            cosine_list.append(round(sim, 3))
            prob_list.append(round(prob, 3))

        df["cosine_similarity"] = cosine_list
        df["P_positive"] = prob_list

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
                # st.markdown(
                #     f"**Cosine Similarity:** `{sim_value:.3f}`  \n"
                # )

                # Display probability
                prob_value = row["P_positive"]
                # Clamp the value to [0, 1]
                prob_pct = max(0, min(1, prob_value))
                st.progress(int(prob_pct * 100),
                            text=f"**Probability:** `{prob_value:.3f}`")
                # st.markdown(
                #     f"**Probability of Intertextuality:** `{prob_value:.3f}`  \n"
                # )

            # # Add a popover button for Attention Weights using an expander.
            # with st.expander("Attention Weights"):
            #     with st.spinner("Computing attention …"):
            #         st.markdown("**Attention Weights:**")
            #         weights_para, weight_orig = get_avg_attention_per_token(
            #             tokenizer, clf_model, row["paraphrased"], row["original"], device)

            #         st.markdown(weights_para)
            #         st.markdown(weight_orig)

            #         # Display attention weights for original texts
            #         html = attention_tokens_to_html(weight_orig)
            #         st.markdown(html, unsafe_allow_html=True)

            #         # Display attention weights for paraphrased texts
            #         html = attention_tokens_to_html(weights_para)
            #         st.markdown(html, unsafe_allow_html=True)

    # ─────────── DOWNLOAD BUTTON ───────────
    st.download_button(
        "💾 Download Scored CSV",
        data=df.to_csv(index=False).encode(),
        file_name="results_with_scores.csv",
        mime="text/csv",
    )
