import torch, math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils.pdf_tools import extract_sentences, split_long_sentences, normalize_scores, export_results, highlight_pdf
import math as m

MODEL = "law-ai/InLegalBERT"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForMaskedLM.from_pretrained(MODEL).to(device).eval()

def pseudo_perplexity(sentence):
    if not sentence.strip():
        return float("nan")

    ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    n = ids.size(1) - 2
    if n <= 0:
        return float("nan")

    loss = 0
    with torch.no_grad():
        for i in range(1, n + 1):
            masked = ids.clone()
            masked[0, i] = tokenizer.mask_token_id
            logits = model(masked).logits
            logp = torch.nn.functional.log_softmax(logits[0, i], dim=-1)
            loss += -logp[ids[0, i]].item()

    return m.exp(loss / n)

def run_ppl(pdf_path):
    sents = extract_sentences(pdf_path)
    sents = split_long_sentences(sents)

    scores = [pseudo_perplexity(s) for s in tqdm(sents, desc="Computing PPL")]
    sents, scores = zip(*[(s, p) for s, p in zip(sents, scores) if not m.isnan(p)])

    zs = normalize_scores(scores)
    export_results(sents, zs, "results/ppl_scores.csv")
    highlight_pdf(pdf_path, sents, zs, "results/ppl_highlighted.pdf")
    return sents, zs
