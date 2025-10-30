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
    mode: str = "feedback"   # can be "feedback" or "essay"


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


def calibrate_bias(
    raw_bias_prob: float,
    polarity: float,
    tone_variance: float = 0.0,
    constructive_ratio: float = 0.0,
    context: str = "feedback"
):
    """
    Smarter bias adjustment:
    - Reduces false 'high bias' when tone is consistent or constructive
    - Allows stronger praise/critique in feedback context
    - Penalizes erratic tone variance and harsh tone without constructiveness
    """
    bias_adj = raw_bias_prob

    # 1️⃣ Constructive tone = lower bias
    if abs(polarity) > 0.6 and constructive_ratio > 0.3:
        bias_adj *= 0.8

    # 2️⃣ Consistent tone = more fair
    if tone_variance < 0.25:
        bias_adj *= 0.8

    # 3️⃣ Feedback mode = allow stronger tone range
    if context == "feedback":
        bias_adj *= 0.9

    # 4️⃣ Harsh + unhelpful tone = increase bias
    if polarity < -0.4 and constructive_ratio < 0.2:
        bias_adj = min(1.0, bias_adj * 1.2)

    return max(0.0, min(1.0, bias_adj))


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

import re

# --- simple lexicons for bias flavours ---
FAV_WORDS = ["best", "outstanding", "amazing", "brilliant", "natural talent", "flawless", "incredible"]
HOSTILE_WORDS = ["lazy", "disappointing", "poor effort", "nonsense", "terrible", "awful"]
SUBJECTIVE_WORDS = ["i think", "i feel", "i believe", "personally", "my opinion"]
CONSTRUCTIVE_WORDS = ["improve", "consider", "suggest", "try", "could", "should", "recommend", "plan", "develop"]

def bias_flavours(sentence: str):
    """Detect what kind of bias flavour a sentence might have."""
    s = sentence.lower()
    fav = any(w in s for w in FAV_WORDS)
    host = any(w in s for w in HOSTILE_WORDS)
    subj = any(w in s for w in SUBJECTIVE_WORDS)
    return fav, host, subj

def is_balanced_text(text: str):
    """Detect if the overall text shows hedging or balance."""
    t = text.lower()
    hedges = ["however", "although", "while", "on the other hand", "nevertheless", "in contrast", "both", "whereas"]
    return any(h in t for h in hedges)

def compute_fairness_simple(sentences: list[str]):
    """
    Enhanced fairness scoring (v3).
    Now heavily penalizes emotional, subjective, or unconstructive tone.
    Target distribution: professional text ~90, mild bias ~65, strong bias ~40.
    """
    total_penalty = 0
    triggers = []

    for s in sentences:
        s_low = s.lower()

        # --- Harsh / judgmental tone ---
        if any(w in s_low for w in HOSTILE_WORDS):
            total_penalty += 60
            triggers.append("harsh tone")
        if "!" in s_low or "never" in s_low or "always" in s_low:
            total_penalty += 25
            triggers.append("absolutist phrasing")

        # --- Over-flattering / exaggerated praise ---
        if any(w in s_low for w in FAV_WORDS):
            total_penalty += 45
            triggers.append("over-praise")
        if any(x in s_low for x in ["amazing", "incredible", "perfect", "exceptional", "brilliant"]):
            total_penalty += 35
            triggers.append("exaggerated praise")

        # --- Subjective or personal language ---
        if any(w in s_low for w in SUBJECTIVE_WORDS):
            total_penalty += 30
            triggers.append("personal phrasing")
        if "you are " in s_low or "he is " in s_low or "she is " in s_low:
            total_penalty += 20
            triggers.append("personal reference")

        # --- Lack of constructive / actionable language ---
        if not any(w in s_low for w in CONSTRUCTIVE_WORDS):
            total_penalty += 25
            triggers.append("no constructive advice")

        # --- Emotional punctuation or sarcasm hints ---
        if s.count("!") > 1 or "??" in s or "..." in s:
            total_penalty += 20
            triggers.append("emotional punctuation")

        # --- Too short or fragmentary feedback (non-specific) ---
        if len(s.split()) < 5:
            total_penalty += 15
            triggers.append("vague comment")

    # --- scale much harder ---
    # Each strong trigger now cuts ~10–15 fairness points.
    fairness = 100 - (total_penalty / max(1, len(sentences))) * 1.2
    fairness = max(0, min(100, fairness))

    return fairness, triggers

def generate_note(tone_label: str, polarity: float, bias_level: str, fairness_label: str, sentence: str = ""):
    fav, host, subj = bias_flavours(sentence)
    if fav and bias_level == "high":
        return "Strong praise may appear subjective — add evidence or refer to criteria."
    if host and bias_level == "high":
        return "Tone may feel personal or discouraging — focus on the work, not the student."
    if subj:
        return "Contains personal phrasing — consider rewording toward objective criteria."
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

    raw_sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    per_sentence = []

    # ---- pass 1: per-sentence inference ----
    for s in raw_sentences:
        tone_label, polarity, conf = run_sentiment_model(s)
        raw_bias = run_bias_model(s)
        bias_adj = calibrate_bias(
            raw_bias,
            polarity,
            tone_variance=0.0,
            constructive_ratio=0.0,
            context=req.mode
        )
        bias_level = classify_bias_level(bias_adj)
        fair_label = compute_fairness_label(polarity, bias_level)
        note = generate_note(tone_label, polarity, bias_level, fair_label, s)

        per_sentence.append({
            "sentence": s,
            "tone_label": tone_label,
            "tone_score": polarity,
            "tone_confidence": conf,
            "bias_class": bias_level,
            "bias_raw": bias_adj,
            "fairness_label": fair_label,
            "note": note,
            "toneColor": tone_color(polarity),
            "biasColor": bias_color(bias_level),
            "fairColor": fairness_color(fair_label),
        })

    # ---- aggregate diagnostics ----
    polarities = [p["tone_score"] for p in per_sentence]
    bias_lvls = [p["bias_class"] for p in per_sentence]
    text_lower = text.lower()

    avg_pol = float(np.mean(polarities)) if polarities else 0.0
    tone_var = float(np.std(polarities)) if len(polarities) > 1 else 0.0
    constructive_words = ["improve", "consider", "suggest", "try", "could", "should", "recommend", "plan", "next", "develop"]
    constructive_count = sum(any(w in s.lower() for w in constructive_words) for s in raw_sentences)
    constructive_ratio = constructive_count / len(raw_sentences) if raw_sentences else 0.0

    # ---- heuristic rule boost ----
    excessive_praise = any(w in text_lower for w in ["outstanding", "perfect", "best", "amazing", "flawless", "incredible"])
    harsh_words = any(w in text_lower for w in ["lazy", "poor", "nonsense", "terrible", "awful", "disappointing"])
    personal_refs = any(w in text_lower for w in ["i think", "i feel", "i believe", "i don't", "my opinion", "to me"])
    lack_constructive = constructive_ratio < 0.2
    inconsistent_tone = tone_var > 0.3

    rule_boost = 0.0
    if excessive_praise: rule_boost += 0.25
    if harsh_words: rule_boost += 0.25
    if personal_refs: rule_boost += 0.2
    if lack_constructive: rule_boost += 0.15
    if inconsistent_tone: rule_boost += 0.15

    # ---- combine metrics ----
    bias_rate = compute_bias_rate(bias_lvls)
    base_bias_score = np.mean([
        0.2 if b == "low" else 0.5 if b == "medium" else 0.8
        for b in bias_lvls
    ]) if bias_lvls else 0.0

    bias_final = min(1.0, base_bias_score + rule_boost)

    # 🔹 boost bias when tone variance is high (emotional / inconsistent language)
    if tone_var > 0.3 and abs(avg_pol) > 0.4:
        bias_final = min(1.0, bias_final + 0.05)

    # 🔹 detect extreme bias patterns (very cheerlead-y or very harsh, or absolutist language)
    if abs(avg_pol) > 0.65:
        bias_final = min(1.0, bias_final + 0.1)
    if "!" in text or "clearly" in text_lower or "obvious" in text_lower:
        bias_final = min(1.0, bias_final + 0.05)

    # 🔹 de-boost if the writer shows hedging / balance language
    if is_balanced_text(text):
        bias_final = max(0.0, bias_final - 0.1)

    # map bias_final -> label
    if bias_final >= 0.55:
        bias_lvl_overall = "High"
    elif bias_final >= 0.3:
        bias_lvl_overall = "Medium"
    else:
        bias_lvl_overall = "Low"

    # ---- fairness scoring (using the STRONGER version you defined above) ----
    fairness_index, fairness_triggers = compute_fairness_simple(raw_sentences)

    # classify fairness more honestly (harsher thresholds)
    fair_label_overall = (
        "Balanced" if fairness_index >= 70 else
        "Slightly Skewed" if fairness_index >= 50 else
        "Unbalanced"
    )

    # ---- tone label for summary ----
    tone_lbl_overall = overall_tone_label(avg_pol)

    # human summary text
    if fairness_triggers:
        triggers_text = ", ".join(sorted(set(fairness_triggers)))
        summary_text = (
            f"Overall {fair_label_overall.lower()} feedback — flagged for {triggers_text}."
        )
    else:
        summary_text = "Feedback appears professional and balanced."

    # ---- build summary block for UI ----
    summary_block = {
        "overall_tone": tone_lbl_overall,
        "avg_polarity": avg_pol,
        "avg_bias_level": bias_lvl_overall,
        "bias_rate": round(bias_rate * 100, 1),
        "fairness_overall": fair_label_overall,
        "fairness_index": round(fairness_index, 1),
        "what_this_means": summary_text,
    }

    # ---- inline highlights ----
    inline_highlight = []
    for p in per_sentence:
        tooltip_lines = [
            f"Tone: {p['tone_label']} ({p['tone_score']:.2f})",
            f"Bias: {p['bias_class']}",
            f"Balance: {p['fairness_label']}",
            p["note"],
        ]
        inline_highlight.append({
            "text": p["sentence"],
            "toneColor": p["toneColor"],
            "biasColor": p["biasColor"],
            "fairColor": p["fairColor"],
            "tooltip": " · ".join(tooltip_lines),
        })

    # ---- final response ----
    return {
        "inline_highlight": inline_highlight,
        "sentences": [SentenceReport(**p) for p in per_sentence],
        "summary": SummaryBlock(**summary_block),
    }

@app.get("/health")
def health():
    return {"ok": True}
