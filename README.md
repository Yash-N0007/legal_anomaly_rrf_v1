# 🧠 Legal Document Anomaly Detection using Reciprocal Rank Fusion (RRF)

This repository implements a clause-level anomaly detection system for legal documents using three independent detectors based on transformer models.   
The system highlights unusual or risky sentences in PDF legal documents using statistical, semantic, and contextual deviation signals, and merges them using **Reciprocal Rank Fusion (RRF)** for a final anomaly score.

---

## 📌 Features

- Extract sentences from PDF legal documents  
- Handles long sentences with chunking  
- Three independent anomaly detectors:
  - **Pseudo-Perplexity (PPL)** – InLegalBERT  
  - **Semantic Deviation** – LegalBERT  
  - **Contextual Deviation** – E5-base-v2  
- Z-score or min–max normalization  
- Final ranking with **RRF**  
- Exports:
  - CSV score sheets  
  - Highlighted PDF (Green / Yellow / Orange)

---

## 🏗️ System Architecture

[ PDF Document ]
        ↓
 Text Extraction
        ↓
┌─────────────── Pipeline ───────────────┐
│                                         │
│  • PPL Detector (InLegalBERT)           │
│  • Semantic Detector (LegalBERT)        │
│  • Context Detector (E5-base-v2)        │
│                                         │
└───────────────────┬─────────────────────┘
                    ↓
      Reciprocal Rank Fusion (RRF)
                    ↓
      Highlighted PDF + Score CSVs


## 🧪 Detectors

### **1. Pseudo-Perplexity Detector (InLegalBERT)**  
Computes token-level pseudo-perplexity using masked LM scoring.  
Higher PPL = sentence is linguistically unusual.

### **2. Semantic Deviation Detector (LegalBERT)**  
Computes distance from the embedding centroid of the document.  
Higher distance = semantically out-of-distribution.

### **3. Contextual Deviation Detector (E5-base-v2)**  
Computes similarity with left and right neighboring sentences.  
Lower similarity = context break / misplaced clause.

---

## 🔀 Reciprocal Rank Fusion (RRF)

The system uses RRF to merge rankings from all detectors which improves robustness and reduces dependence on any single model.

---

## 🎨 Output Highlighting

The final PDF highlights clauses based on anomaly severity:

| Risk Level | Color |
|-----------|--------|
| Low       | 🟢 Green |
| Medium    | 🟡 Yellow |
| High      | 🟠 Orange |

Color thresholds are based on percentiles of the final RRF scores.
