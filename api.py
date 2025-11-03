# api.py (BiasBuster 2.2 draft backend)

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification

# ---------- setup nltk once ----------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ✅ new: NLTK 3.9+ also needs 'punkt_tab'
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ---------- load models ----------
SENTIMENT_MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
sent_tokenizer_hf = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
sent_model_hf = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
sentiment_pipe = pipeline(
    "text-classification",
    model=sent_model_hf,
    tokenizer=sent_tokenizer_hf,
    top_k=None,
    truncation=True,
)

# bias model (onnx)
BIAS_MODEL_NAME = "protectai/distilroberta-bias-onnx"
bias_tokenizer = AutoTokenizer.from_pretrained(BIAS_MODEL_NAME)
bias_model = ORTModelForSequenceClassification.from_pretrained(BIAS_MODEL_NAME)

LABELS = ["BIASED", "NEUTRAL"]  # expected output order from this model


class AnalyseRequest(BaseModel):
    text: str


class SentenceReport(BaseModel):
    sentence: str
    tone_label: str          # "positive"/"neutral"/"negative"
    tone_score: float        # -1..1
    tone_confidence: float   # 0..1
    bias_class: str          # "low"/"medium"/"high"
    bias_raw: float          # 0..1 biased probability (after calibration)
    fairness_label: str      # "balanced"/"slightly skewed"/"unbalanced"
    note: str                # short human-facing explanation
    toneColor: str           # frontend color hint
    biasColor: str
    fairColor: str


class SummaryBlock(BaseModel):
    overall_tone: str          # "Positive", "Neutral", "Negative"
    avg_polarity: float
    avg_bias_level: str        # "Low", "Medium", "High"
    bias_rate: float           # % of sentences flagged medium/high
    fairness_overall: str      # "Balanced", "Slightly Skewed", "Unbalanced"
    fairness_index: float      # 0..100
    what_this_means: str       # narrative summary for assessor


class AnalyseResponse(BaseModel):
    inline_highlight: List[dict]
    sentences: List[SentenceReport]
    summary: SummaryBlock


# ---------- helpers ----------

def softmax(logits: torch.Tensor):
    e = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)


def run_bias_model(text: str):
    # run ONNX model through ORT wrapper
    encoded = bias_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = bias_model(**{k: v.numpy() for k, v in encoded.items()})
        # ORTModelForSequenceClassification returns logits as numpy already
        logits = torch.tensor(outputs.logits)

    probs = softmax(logits)[0].tolist()  # [biase_prob, neutral_prob] (check order)
    # We assume index 0 = "BIASED"
    biased_p = probs[0]
    return float(biased_p)


def run_sentiment_model(text: str):
    results = sentiment_pipe(text)[0]  # list of dicts label/score
    # pick best
    best = max(results, key=lambda x: x["score"])
    label = best["label"].lower()
    conf = float(best["score"])

    # convert to polarity
    if label == "positive":
        polarity = conf
    elif label == "negative":
        polarity = -conf
    else:
        polarity = 0.0

    # normalize polarity slightly toward neutral to avoid hysterics:
    # We shift - reduce magnitude of extreme sentiment a bit
    polarity *= 0.9

    return label, polarity, conf


def calibrate_bias(raw_bias_prob: float, polarity: float):
    """
    We don't want to scream 'HIGH BIAS' for normal teacher tone.
    Rules:
    - If tone is mild (|polarity| < 0.4), damp bias by 30%
    - Hard floor: can't go <0, cap at 1
    """
    bias_adj = raw_bias_prob
    if abs(polarity) < 0.4:
        bias_adj *= 0.7

    bias_adj = max(0.0, min(1.0, bias_adj))
    return bias_adj


def classify_bias_level(bias_adj: float):
    if bias_adj >= 0.7:
        return "high"
    elif bias_adj >= 0.4:
        return "medium"
    else:
        return "low"


def compute_fairness_label(polarity: float, bias_level: str):
    """
    map (tone,bias) -> fairness label
    """
    if bias_level == "high" and abs(polarity) > 0.5:
        return "unbalanced"
    if bias_level == "high":
        return "slightly skewed"
    if bias_level == "medium" and abs(polarity) > 0.6:
        return "slightly skewed"
    return "balanced"


def fairness_color(label: str):
    if label == "balanced":
        return "hsl(150,70%,40%)"      # green
    if label == "slightly skewed":
        return "hsl(45,85%,55%)"       # amber
    return "hsl(0,80%,55%)"            # red


def tone_color(polarity: float):
    # map -1..1 -> red→yellow→green
    # We'll just reuse same mapping as before: hue 0..120
    hue = (polarity + 1) * 60.0  # -1=>0(red),0=>60(yellow),1=>120(green)
    return f"hsl({hue:.0f},80%,50%)"


def bias_color(level: str):
    if level == "low":
        return "hsl(150,70%,40%)"   # green
    if level == "medium":
        return "hsl(45,85%,55%)"    # yellow
    return "hsl(0,80%,55%)"         # red


def generate_note(tone_label: str, polarity: float, bias_level: str, fairness_label: str):
    # contextual heuristic
    if bias_level == "high" and tone_label == "positive":
        return "Possible over-praise — consider adding evidence, not only approval."
    if bias_level == "high" and tone_label == "negative":
        return "Possible harsh tone — could feel personal or discouraging."
    if bias_level == "high" and fairness_label == "unbalanced":
        return "Tone may sound subjective or absolute. Suggest softening language."
    if bias_level == "low" and tone_label == "negative":
        return "Critical but likely constructive."
    if bias_level == "low" and tone_label == "positive":
        return "Supportive tone; seems professionally encouraging."
    return "Tone appears generally professional."


def overall_tone_label(avg_polarity: float):
    if avg_polarity > 0.3:
        return "Positive"
    if avg_polarity < -0.3:
        return "Negative"
    return "Neutral"


def overall_bias_level(sent_bias_levels: List[str]):
    # We'll call it High if >30% of sentences are "high"
    # Medium if most are "medium"
    highs = sum(1 for b in sent_bias_levels if b == "high")
    meds = sum(1 for b in sent_bias_levels if b == "medium")
    total = len(sent_bias_levels) or 1

    high_rate = highs / total
    med_rate = meds / total

    if high_rate > 0.3:
        return "High"
    if med_rate > 0.3 or high_rate > 0.1:
        return "Medium"
    return "Low"


def compute_bias_rate(sent_bias_levels: List[str]):
    # % of sentences medium or high
    flagged = sum(1 for b in sent_bias_levels if b in ("medium", "high"))
    total = len(sent_bias_levels) or 1
    return flagged / total


def fairness_score(fair_labels: List[str]):
    """
    Convert per-sentence fairness_label -> numeric score 0..1:
    balanced = 1.0
    slightly skewed = 0.6
    unbalanced = 0.2
    then average.
    """
    vals = []
    for f in fair_labels:
        if f == "balanced":
            vals.append(1.0)
        elif f == "slightly skewed":
            vals.append(0.6)
        else:
            vals.append(0.2)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def fairness_overall_label(avg_fairness: float):
    if avg_fairness >= 0.8:
        return "Balanced"
    if avg_fairness >= 0.6:
        return "Slightly Skewed"
    return "Unbalanced"


def narrative_summary(tone_lbl: str, bias_lvl: str, fairness_lbl: str):
    # human-facing summary string
    if bias_lvl == "High":
        if tone_lbl == "Positive":
            return "High bias detected — feedback may sound overly flattering or subjective."
        if tone_lbl == "Negative":
            return "High bias detected — tone may feel harsh or personal."
        return "High bias detected — consider making statements more evidence-based."
    if fairness_lbl == "Balanced":
        return "Tone and bias appear balanced — feedback likely professional and fair."
    if tone_lbl == "Positive":
        return "Tone leans positive — ensure praise is supported with specifics."
    if tone_lbl == "Negative":
        return "Tone is critical — ensure critique stays specific and actionable."
    return "Mostly acceptable tone. Watch for moments of subjectivity."


# ---------- route ----------

@app.post("/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest):

    text = req.text.strip()
    if not text:
        return AnalyseResponse(
            inline_highlight=[],
            sentences=[],
            summary=SummaryBlock(
                overall_tone="Neutral",
                avg_polarity=0.0,
                avg_bias_level="Low",
                bias_rate=0.0,
                fairness_overall="Balanced",
                fairness_index=100.0,
                what_this_means="No text provided."
            )
        )

    # split to sentences
    raw_sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]

    per_sentence = []

    for s in raw_sentences:
        # run models
        tone_label, polarity, conf = run_sentiment_model(s)
        raw_bias = run_bias_model(s)
        bias_adj = calibrate_bias(raw_bias, polarity)
        bias_level = classify_bias_level(bias_adj)

        fair_label = compute_fairness_label(polarity, bias_level)

        note = generate_note(tone_label, polarity, bias_level, fair_label)

        # choose colors for frontend:
        tc = tone_color(polarity)
        bc = bias_color(bias_level)
        fc = fairness_color(fair_label)

        per_sentence.append({
            "sentence": s,
            "tone_label": tone_label,
            "tone_score": polarity,
            "tone_confidence": conf,
            "bias_class": bias_level,
            "bias_raw": bias_adj,
            "fairness_label": fair_label,
            "note": note,
            "toneColor": tc,
            "biasColor": bc,
            "fairColor": fc,
        })

    # ---- compute summary ----
    polarities = [p["tone_score"] for p in per_sentence]
    bias_lvls = [p["bias_class"] for p in per_sentence]
    fair_lvls = [p["fairness_label"] for p in per_sentence]

    avg_pol = float(np.mean(polarities)) if polarities else 0.0
    tone_lbl_overall = overall_tone_label(avg_pol)

    bias_rate = compute_bias_rate(bias_lvls)
    bias_lvl_overall = overall_bias_level(bias_lvls)

    fair_idx = fairness_score(fair_lvls)  # 0..1
    fair_label_overall = fairness_overall_label(fair_idx)

    summary_text = narrative_summary(
        tone_lbl_overall,
        bias_lvl_overall,
        fair_label_overall
    )

    summary_block = {
        "overall_tone": tone_lbl_overall,
        "avg_polarity": avg_pol,
        "avg_bias_level": bias_lvl_overall,
        "bias_rate": round(bias_rate * 100, 1),        # %
        "fairness_overall": fair_label_overall,
        "fairness_index": round(fair_idx * 100, 1),     # %
        "what_this_means": summary_text,
    }

    # ---- inline highlight data for Grammarly-style UX ----
    # We’ll give the frontend spans with suggested highlight color & tooltip.
    inline_highlight = []
    for p in per_sentence:
        tooltip_lines = [
            f"Tone: {p['tone_label']} ({p['tone_score']:.2f})",
            f"Bias: {p['bias_class']}",
            f"Balance: {p['fairness_label']}",
            p["note"]
        ]
        inline_highlight.append({
            "text": p["sentence"],
            "toneColor": p["toneColor"],
            "biasColor": p["biasColor"],
            "fairColor": p["fairColor"],
            "tooltip": " · ".join(tooltip_lines)
        })

    # package response
    sentence_reports = [SentenceReport(**p) for p in per_sentence]
    summary_model = SummaryBlock(**summary_block)

    return {
        "inline_highlight": inline_highlight,
        "sentences": sentence_reports,
        "summary": summary_model,
    }


@app.get("/health")
def health():
    return {"ok": True}
