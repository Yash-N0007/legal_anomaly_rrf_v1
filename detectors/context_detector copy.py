# detectors/context_detector.py
from sentence_transformers import SentenceTransformer, util
import numpy as np
from utils.pdf_tools import extract_sentences, split_long_sentences, normalize_scores, export_results, highlight_pdf

MODEL = "intfloat/e5-base-v2"
model = SentenceTransformer(MODEL)

def run_context(pdf_path):
    sents = extract_sentences(pdf_path)
    sents = split_long_sentences(sents)

    embs = model.encode(sents, normalize_embeddings=True)
    sim_scores = []
    for i, e in enumerate(embs):
        left = util.cos_sim(e, embs[i - 1])[0][0] if i > 0 else 1
        right = util.cos_sim(e, embs[i + 1])[0][0] if i < len(embs) - 1 else 1
        sim_scores.append(1 - (left + right) / 2)
    zs = normalize_scores(sim_scores)
    export_results(sents, zs, "results/context_scores.csv")
    highlight_pdf(pdf_path, sents, zs, "results/context_highlighted.pdf")
    return sents, zs
