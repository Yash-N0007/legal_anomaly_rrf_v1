import fitz  # PyMuPDF
import pandas as pd
from nltk.tokenize import sent_tokenize
import os

# ========== TEXT EXTRACTION ==========

def extract_sentences(pdf_path):
    """Extract text from all pages of a PDF and split into sentences."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + " "
    sentences = sent_tokenize(text)
    return sentences


# ========== LONG SENTENCE HANDLER ==========

def split_long_sentences(sentences, max_chars=1000):
    """
    Split sentences longer than max_chars into smaller chunks
    so they never exceed 512 BERT tokens.
    """
    new_sentences = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if len(s) > max_chars:
            # Split by punctuation and bullet markers
            parts = s.replace("•", ". ").replace(";", ". ").split(". ")
            buffer = ""
            for p in parts:
                if len(buffer) + len(p) < max_chars:
                    buffer += p + ". "
                else:
                    new_sentences.append(buffer.strip())
                    buffer = p + ". "
            if buffer:
                new_sentences.append(buffer.strip())
        else:
            new_sentences.append(s)
    return new_sentences


# ========== NORMALIZATION ==========

def normalize_scores(scores):
    """Normalize numeric scores to [0, 1]."""
    s_min, s_max = min(scores), max(scores)
    if s_max == s_min:
        return [0.5 for _ in scores]
    return [(s - s_min) / (s_max - s_min) for s in scores]


# ========== PDF HIGHLIGHTER ==========

import numpy as np

def highlight_pdf(pdf_path, sentences, scores, output_path):
    doc = fitz.open(pdf_path)

    # compute dynamic percentile cutoffs
    low_th = np.percentile(scores, 60)
    mid_th = np.percentile(scores, 80)

    for page in doc:
        for sent, score in zip(sentences, scores):
            areas = page.search_for(sent)
            for rect in areas:
                if score <= low_th:       # bottom 1/3 → green
                    color = (0, 1, 0)
                elif score <= mid_th:     # middle 1/3 → yellow
                    color = (1, 1, 0)
                else:                     # top 1/3 → orange/red
                    color = (1, 0.65, 0)
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=color)
                annot.update()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    doc.close()
    print(f"[Saved] {output_path}")



# ========== CSV EXPORTER ==========

def export_results(sentences, scores, csv_path):
    """Save sentences and their scores to a CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame({"Sentence": sentences, "Score": scores})
    df.to_csv(csv_path, index=False)
    print(f"[Exported] {csv_path}")
