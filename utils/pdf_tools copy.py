# utils/pdf_tools.py
import fitz, os, pandas as pd, numpy as np
from nltk.tokenize import sent_tokenize

def extract_sentences(pdf_path):
    """Extract sentences from all PDF pages."""
    doc = fitz.open(pdf_path)
    text = " ".join(page.get_text("text") for page in doc)
    return [s for s in sent_tokenize(text) if len(s.split()) > 3]

def normalize_scores(scores):
    """Return z-score normalized values."""
    arr = np.array(scores)
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

def export_results(sentences, scores, out_csv):
    """Save sentence-score pairs."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame({"Sentence": sentences, "Score": scores}).to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

def highlight_pdf(pdf_path, sentences, scores, out_pdf):
    """Highlight sentences by score intensity."""
    doc = fitz.open(pdf_path)
    p75, p90 = np.percentile(scores, [75, 90])
    for page in doc:
        for s, sc in zip(sentences, scores):
            for rect in page.search_for(s):
                if sc < p75:
                    color = (0, 1, 0)
                elif sc < p90:
                    color = (1, 1, 0)
                else:
                    color = (1, 0.5, 0)
                a = page.add_highlight_annot(rect)
                a.set_colors(stroke=color)
                a.update()
    doc.save(out_pdf)
    print(f"[Highlighted PDF] {out_pdf}")

#New
def split_long_sentences(sentences, max_chars=1000):
    """Split sentences longer than max_chars into smaller chunks."""
    new_sentences = []
    for s in sentences:
        if len(s) > max_chars:
            # Break large sentences on punctuation or bullet separators
            parts = s.replace("•", ". ").split(". ")
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
