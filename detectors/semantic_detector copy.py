# detectors/semantic_detector.py
from sentence_transformers import SentenceTransformer
import numpy as np
from utils.pdf_tools import extract_sentences,split_long_sentences, normalize_scores, export_results, highlight_pdf

MODEL = "nlpaueb/legal-bert-base-uncased"
model = SentenceTransformer(MODEL)

def run_semantic(pdf_path):
    sents = extract_sentences(pdf_path)
    sents = split_long_sentences(sents)

    embs = model.encode(sents, normalize_embeddings=True)
    centroid = np.mean(embs, axis=0)
    dist = np.linalg.norm(embs - centroid, axis=1)
    zs = normalize_scores(dist)
    export_results(sents, zs, "results/semantic_scores.csv")
    highlight_pdf(pdf_path, sents, zs, "results/semantic_highlighted.pdf")
    return sents, zs
