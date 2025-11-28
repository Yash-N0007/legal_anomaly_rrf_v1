# app.py
import gradio as gr
import numpy as np
import pandas as pd
from detectors.ppl_detector import run_ppl
from detectors.semantic_detector import run_semantic
from detectors.context_detector import run_context
from detectors.fusion import run_fusion

def analyze_pdf(pdf_file):
    pdf_path = pdf_file.name
    print(f"[Processing] {pdf_path}")

    # Run detectors sequentially
    sents_ppl, ppl_scores = run_ppl(pdf_path)
    sents_sem, sem_scores = run_semantic(pdf_path)
    sents_ctx, ctx_scores = run_context(pdf_path)

    # Sanity check for alignment
    if not (len(sents_ppl) == len(sents_sem) == len(sents_ctx)):
        min_len = min(len(sents_ppl), len(sents_sem), len(sents_ctx))
        sents = sents_ppl[:min_len]
        ppl_scores, sem_scores, ctx_scores = (
            np.array(ppl_scores[:min_len]),
            np.array(sem_scores[:min_len]),
            np.array(ctx_scores[:min_len]),
        )
    else:
        sents = sents_ppl

    # Fuse scores
    fused = run_fusion(pdf_path, sents, ppl_scores, sem_scores, ctx_scores)

    # Save combined CSV for reference
    df = pd.DataFrame({
        "Sentence": sents,
        "PPL_Score": ppl_scores,
        "Semantic_Score": sem_scores,
        "Context_Score": ctx_scores,
        "RRF_Fused": fused,
    })
    df.to_csv("results/final_rrf_scores.csv", index=False)
    print("[Saved] results/final_rrf_scores.csv")

    return "results/rrf_highlighted.pdf", "results/final_rrf_scores.csv"

# ---------------- GRADIO INTERFACE ---------------- #
demo = gr.Interface(
    fn=analyze_pdf,
    inputs=gr.File(label="Upload Legal PDF"),
    outputs=[
        gr.File(label="Highlighted PDF (RRF)"),
        gr.File(label="All Scores CSV")
    ],
    title="Clause-Level Legal Anomaly Detector (Reciprocal Rank Fusion)",
    description="Uploads a legal document and highlights anomalous clauses using Pseudo-Perplexity, Semantic, and Contextual scores combined via RRF."
)

if __name__ == "__main__":
    demo.launch()
