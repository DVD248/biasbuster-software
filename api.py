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


# ==========================================================
# 🧩 START DEV MODE CONFIG (Safe Developer Sandbox)
# ==========================================================

from fastapi import Request, HTTPException
import json, os

# === Developer Mode Toggle ===
DEV_MODE = True  # Set False to disable /dev endpoints globally

# === Immutable Default Parameters ===
PARAMS_BASE = {
    # Bias calibration
    "BIAS_HIGH_THRESHOLD": 0.55,
    "BIAS_MEDIUM_THRESHOLD": 0.30,
    "VARIANCE_BIAS_BOOST": 0.05,
    "EXTREME_POLARITY_BIAS_BOOST": 0.10,
    "BALANCED_TEXT_REDUCTION": 0.1,
    "FEEDBACK_MODE_BIAS_SCALE": 0.9,
    "CONSTRUCTIVE_BIAS_REDUCTION": 0.8,

    # Fairness scoring
    "HOSTILE_PENALTY": 60,
    "FAV_PENALTY": 45,
    "SUBJECTIVE_PENALTY": 30,
    "NO_CONSTRUCTIVE_PENALTY": 25,
    "FAIRNESS_SCALE": 1.2,
    "FAIR_BALANCED_MIN": 70,
    "FAIR_SLIGHTLY_SKEWED_MIN": 60,
    "FAIRNESS_BIAS_PENALTY_MED": 0,
    "FAIRNESS_BIAS_PENALTY_HIGH": 0,

    # Sentiment & tone
    "POLARITY_SCALE": 0.9,
    "TONE_VARIANCE_LIMIT": 0.30,
    "HARSH_POLARITY_THRESHOLD": -0.4,
    "POSITIVE_POLARITY_THRESHOLD": 0.6,

    # Constructiveness
    "CONSTRUCTIVE_RATIO_MIN": 0.20,
    "INCONSISTENT_TONE_VAR": 0.30,
    "INCONSISTENT_TONE_BIAS_BOOST": 0.15,

    # UI colors
    "COLOR_FAIR_BALANCED": "#00c36b",
    "COLOR_FAIR_SKEWED": "#ffb700",
    "COLOR_FAIR_UNBALANCED": "#ff3838",

    # Tone summary
    "TONE_POSITIVE_MIN": 0.3,
    "TONE_NEGATIVE_MAX": -0.3,
    "BIAS_RATE_HIGH_THRESHOLD": 0.3,

    # --- Developer tuning additions ---
    "TONE_BASE_HUE": 60,               # hue center for tone color
    "RULE_BOOST_MULT": 1.0,            # multiplier for rule-based bias boosts
    "CONSTRUCTIVE_BOOST": 1.0,         # global fairness bias dampening factor
}

# === Per-user temporary overrides (sandboxed) ===
PARAMS_SESSIONS = {}


# Helper: get the IP key (or a fallback ID)
def _get_client_id(request: Request):
    try:
        return request.client.host
    except Exception:
        return "local"


# Get merged parameters (base + user overrides)
def get_user_params(request: Request):
    ip = _get_client_id(request)
    params = PARAMS_BASE.copy()
    if ip in PARAMS_SESSIONS:
        params.update(PARAMS_SESSIONS[ip])
    return params


# Update or create per-user session params
def set_user_params(request: Request, new_params: dict):
    ip = _get_client_id(request)
    session_params = PARAMS_SESSIONS.get(ip, {}).copy()

    for k, v in new_params.items():
        if k not in PARAMS_BASE:
            continue  # ignore unknown keys
        # Accept both numeric and string values (e.g. color codes)
        session_params[k] = v

    PARAMS_SESSIONS[ip] = session_params


# Reset current user's session
def reset_user_params(request: Request):
    ip = _get_client_id(request)
    if ip in PARAMS_SESSIONS:
        del PARAMS_SESSIONS[ip]


# --- Developer API Endpoints ---
if DEV_MODE:

    @app.get("/dev/params")
    def dev_get_params(request: Request):
        """Return current effective parameters for this user (base + overrides)."""
        return get_user_params(request)

    @app.post("/dev/params")
    async def dev_post_params(request: Request):
        """Update in-memory session parameters (non-persistent)."""
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        set_user_params(request, body)
        return {"ok": True, "updated": list(body.keys())}

    @app.post("/dev/reset")
    def dev_reset_params(request: Request):
        """Clear overrides for this user (back to defaults)."""
        reset_user_params(request)
        return {"ok": True}
    

else:
    # If DEV_MODE is disabled, return 404 for these
    @app.get("/dev/{path:path}")
    def dev_disabled(path: str):
        raise HTTPException(status_code=404, detail="Developer mode disabled.")

# === Optional: Save & Load Dev Presets (session-safe) ===
@app.get("/dev/save")
def dev_save_params(request: Request):
    """Save current user's parameter session as a JSON preset."""
    ip = request.client.host
    params = get_user_params(request)
    os.makedirs("dev_configs", exist_ok=True)
    path = f"dev_configs/{ip}.json"
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return {"ok": True, "saved_to": path}

@app.post("/dev/load")
async def dev_load_params(request: Request):
    """Load a saved JSON preset back into current session."""
    try:
        body = await request.json()
        filename = body.get("filename")
        if not filename:
            return {"error": "filename required"}
    except Exception:
        return {"error": "Invalid request body"}

    path = f"dev_configs/{filename}"
    if not os.path.exists(path):
        return {"error": f"Preset {filename} not found"}

    with open(path) as f:
        loaded = json.load(f)

    ip = request.client.host
    PARAMS_SESSIONS[ip] = loaded
    return {"ok": True, "loaded": filename, "params": loaded}

# ==========================================================
# 🧩 END DEV MODE CONFIG (Safe Developer Sandbox)
# ==========================================================

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
    tone_variance: float,
    constructive_ratio: float,
    context: str,
    params: dict
):
    """
    Adjust bias probability using user-tuned parameters.
    """
    bias_adj = raw_bias_prob

    # 1️⃣ Constructive tone reduces bias
    if abs(polarity) > params["POSITIVE_POLARITY_THRESHOLD"] and constructive_ratio > params["CONSTRUCTIVE_RATIO_MIN"]:
        bias_adj *= params.get("CONSTRUCTIVE_BIAS_REDUCTION", 0.8)

    # 2️⃣ Consistent tone reduces bias
    if tone_variance < params["TONE_VARIANCE_LIMIT"]:
        bias_adj *= params.get("BALANCED_TEXT_REDUCTION", 0.9)

    # 3️⃣ Feedback mode scales bias tolerance
    if context == "feedback":
        bias_adj *= params.get("FEEDBACK_MODE_BIAS_SCALE", 0.9)

    # 4️⃣ Harsh + unhelpful tone increases bias
    if polarity < params["HARSH_POLARITY_THRESHOLD"] and constructive_ratio < params["CONSTRUCTIVE_RATIO_MIN"]:
        bias_adj = min(1.0, bias_adj * (1 + params.get("VARIANCE_BIAS_BOOST", 0.1)))

    return max(0.0, min(1.0, bias_adj))


def classify_bias_level(bias_adj: float, params: dict):
    if bias_adj >= params["BIAS_HIGH_THRESHOLD"]:
        return "high"
    elif bias_adj >= params["BIAS_MEDIUM_THRESHOLD"]:
        return "medium"
    else:
        return "low"


def compute_fairness_label(polarity: float, bias_level: str, params: dict):
    pos_thresh = params.get("POSITIVE_POLARITY_THRESHOLD", 0.6)
    if bias_level == "high" and abs(polarity) > pos_thresh * 0.8:
        return "unbalanced"
    if bias_level == "high":
        return "slightly skewed"
    if bias_level == "medium" and abs(polarity) > pos_thresh:
        return "slightly skewed"
    return "balanced"


def fairness_color(label: str, params: dict):
    """Use dynamic colors from params."""
    if label == "balanced":
        return params.get("COLOR_FAIR_BALANCED", "hsl(150,70%,40%)")
    if label == "slightly skewed":
        return params.get("COLOR_FAIR_SKEWED", "hsl(45,85%,55%)")
    return params.get("COLOR_FAIR_UNBALANCED", "hsl(0,80%,55%)")

def bias_color(level: str, params: dict):
    """Use fairness palette dynamically for bias as well (for consistency)."""
    if level == "low":
        return params.get("COLOR_FAIR_BALANCED", "hsl(150,70%,40%)")
    if level == "medium":
        return params.get("COLOR_FAIR_SKEWED", "hsl(45,85%,55%)")
    return params.get("COLOR_FAIR_UNBALANCED", "hsl(0,80%,55%)")


def compute_fairness_simple(sentences: list[str], params: dict):
    """
    Enhanced fairness scoring (Dev Mode aware).
    Uses dynamic penalties & scaling from current params.
    """
    total_penalty = 0
    triggers = []

    for s in sentences:
        s_low = s.lower()

        # Harsh / judgmental tone
        if any(w in s_low for w in HOSTILE_WORDS):
            total_penalty += params["HOSTILE_PENALTY"]
            triggers.append("harsh tone")
        if "!" in s_low or "never" in s_low or "always" in s_low:
            total_penalty += 25
            triggers.append("absolutist phrasing")

        # Over-flattering praise
        if any(w in s_low for w in FAV_WORDS):
            total_penalty += params["FAV_PENALTY"]
            triggers.append("over-praise")

        # Subjectivity
        if any(w in s_low for w in SUBJECTIVE_WORDS):
            total_penalty += params["SUBJECTIVE_PENALTY"]
            triggers.append("personal phrasing")

        # Missing constructive advice
        if not any(w in s_low for w in CONSTRUCTIVE_WORDS):
            total_penalty += params["NO_CONSTRUCTIVE_PENALTY"]
            triggers.append("no constructive advice")

        # Very short or vague
        if len(s.split()) < 5:
            total_penalty += 15
            triggers.append("vague comment")

    fairness = 100 - (total_penalty / max(1, len(sentences))) * params["FAIRNESS_SCALE"]
    fairness = max(0, min(100, fairness))

    return fairness, triggers

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

def tone_color(polarity: float, params: dict = None):
    base = params.get("TONE_BASE_HUE", 60)
    hue = base + (polarity * 60)  # move ±60° from base
    return f"hsl({hue:.0f},80%,50%)"

# ---------- route ----------

from fastapi import Request

@app.post("/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest, request: Request):
    params = get_user_params(request)

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
            context=req.mode,
            params=params
        )
        bias_level = classify_bias_level(bias_adj, params)
        fair_label = compute_fairness_label(polarity, bias_level, params)
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
            "toneColor": tone_color(polarity, params),
            "biasColor": bias_color(bias_level, params),
            "fairColor": fairness_color(fair_label, params),
        })

    # ---- aggregate diagnostics ----
    polarities = [p["tone_score"] for p in per_sentence]
    bias_lvls = [p["bias_class"] for p in per_sentence]
    text_lower = text.lower()

    avg_pol = float(np.mean(polarities)) if polarities else 0.0
    tone_var = float(np.std(polarities)) if len(polarities) > 1 else 0.0
    constructive_words = ["improve","consider","suggest","try","could","should","recommend","plan","next","develop"]
    constructive_count = sum(any(w in s.lower() for w in constructive_words) for s in raw_sentences)
    constructive_ratio = constructive_count / len(raw_sentences) if raw_sentences else 0.0

    # ---- heuristic rule boost ----
    excessive_praise = any(w in text_lower for w in ["outstanding","perfect","best","amazing","flawless","incredible"])
    harsh_words = any(w in text_lower for w in ["lazy","poor","nonsense","terrible","awful","disappointing"])
    personal_refs = any(w in text_lower for w in ["i think","i feel","i believe","i don't","my opinion","to me"])
    lack_constructive = constructive_ratio < params["CONSTRUCTIVE_RATIO_MIN"]
    inconsistent_tone = tone_var > params["INCONSISTENT_TONE_VAR"]

    rule_boost = 0.0
    if excessive_praise: rule_boost += 0.25
    if harsh_words: rule_boost += 0.25
    if personal_refs: rule_boost += 0.2
    if lack_constructive: rule_boost += 0.15
    if inconsistent_tone: rule_boost += 0.15
    rule_boost *= params.get("RULE_BOOST_MULT", 1.0)

    # ---- combine metrics ----
    bias_rate = compute_bias_rate(bias_lvls)
    base_bias_score = np.mean([0.2 if b=="low" else 0.5 if b=="medium" else 0.8 for b in bias_lvls]) if bias_lvls else 0.0
    bias_final = min(1.0, base_bias_score + rule_boost)

    # 🔹 boost bias when tone variance is high (emotional or inconsistent language)
    if tone_var > params["TONE_VARIANCE_LIMIT"] and abs(avg_pol) > params["POSITIVE_POLARITY_THRESHOLD"]:
        bias_final = min(1.0, bias_final + params["VARIANCE_BIAS_BOOST"])

    # 🔹 detect extreme bias patterns
    if abs(avg_pol) > params["POSITIVE_POLARITY_THRESHOLD"]:
        bias_final = min(1.0, bias_final + params["EXTREME_POLARITY_BIAS_BOOST"])
    elif "!" in text or "clearly" in text.lower() or "obvious" in text.lower():
        bias_final = min(1.0, bias_final + 0.05)

    # 🔹 balance correction
    if is_balanced_text(text):
        bias_final = max(0.0, bias_final - params["BALANCED_TEXT_REDUCTION"])

    # ---- classify overall bias ----
    if bias_final >= params["BIAS_HIGH_THRESHOLD"]:
        bias_lvl_overall = "High"
    elif bias_final >= params["BIAS_MEDIUM_THRESHOLD"]:
        bias_lvl_overall = "Medium"
    else:
        bias_lvl_overall = "Low"

    # ---- fairness ----
    fairness_index, fairness_triggers = compute_fairness_simple(raw_sentences, params)
    
    # Boost fairness slightly if text is constructive overall
    if constructive_ratio > params.get("CONSTRUCTIVE_RATIO_MIN", 0.3):
       fairness_index = min(100, fairness_index * params.get("CONSTRUCTIVE_BOOST", 1.0))


    # Apply bias–fairness penalty link
    if bias_lvl_overall == "Medium":
        fairness_index -= params["FAIRNESS_BIAS_PENALTY_MED"]
    elif bias_lvl_overall == "High":
        fairness_index -= params["FAIRNESS_BIAS_PENALTY_HIGH"]
    fairness_index = max(0, min(100, fairness_index))

    fair_label_overall = (
        "Balanced" if fairness_index >= params["FAIR_BALANCED_MIN"] else
        "Slightly Skewed" if fairness_index >= params["FAIR_SLIGHTLY_SKEWED_MIN"] else
        "Unbalanced"
    )

    tone_lbl_overall = overall_tone_label(avg_pol)
    triggers_text = ", ".join(sorted(set(fairness_triggers))) or "no major issues"
    summary_text = (
        f"Overall {fair_label_overall.lower()} feedback — flagged for {triggers_text}."
        if fairness_triggers else
        "Feedback appears professional and balanced."
    )

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
            "biasColor": bias_color(p["bias_class"], params),
            "fairColor": fairness_color(p["fairness_label"], params),
            "tooltip": " · ".join(tooltip_lines),
        })

    return {
        "inline_highlight": inline_highlight,
        "sentences": [SentenceReport(**p) for p in per_sentence],
        "summary": SummaryBlock(**summary_block),
    }

@app.get("/health")
def health():
    return {"ok": True}
