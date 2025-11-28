# detectors/fusion.py
import numpy as np
import pandas as pd
from utils.pdf_tools import highlight_pdf, export_results

def reciprocal_rank_fusion(score_arrays, k=60):
    """score_arrays: list of np.arrays (normalized per detector)."""
    n = len(score_arrays[0])
    rrf = np.zeros(n)
    for arr in score_arrays:
        ranks = np.argsort(np.argsort(arr))
        for i, rank in enumerate(ranks):
            rrf[rank] += 1 / (k + i + 1)
    return rrf

def run_fusion(pdf_path, sents, ppl, sem, ctx):
    fused = reciprocal_rank_fusion([ppl, sem, ctx])
    export_results(sents, fused, "results/rrf_scores.csv")
    highlight_pdf(pdf_path, sents, fused, "results/rrf_highlighted.pdf")
    return fused
