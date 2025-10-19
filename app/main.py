from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import MarianMTModel, MarianTokenizer
from app.utils import detect_lang, LANG_MAP
import torch

app = FastAPI(title="Language Translator")

# Allow all origins (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported languages (all lightweight OPUS-MT targets)
SUPPORTED_LANGS = list(LANG_MAP.keys())

# Cache models for lightweight reuse
MODEL_CACHE = {}

def get_model(src: str, tgt: str):
    """Load model/tokenizer for a source-target pair, cached."""
    key = f"{src}-{tgt}"
    if key not in MODEL_CACHE:
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.to("cpu")
        MODEL_CACHE[key] = (tokenizer, model)
    return MODEL_CACHE[key]

# Request schema
class TranslateRequest(BaseModel):
    text: str
    lang: str  # target language

@app.get("/")
def root():
    return {"status": "ok", "supported_languages": SUPPORTED_LANGS}

@app.post("/translate")
def translate(req: TranslateRequest):
    text = req.text.strip()
    target = req.lang.strip().lower() if req.lang else "en"

    # Validate target language
    if target not in SUPPORTED_LANGS:
        return {"error": f"Unsupported language '{target}'. Supported: {SUPPORTED_LANGS}"}

    # Detect source language
    src = detect_lang(text)
    if src not in SUPPORTED_LANGS:
        src = "en"

    # Skip translation if source = target
    if src == target:
        return {"translated": text}

    # Load appropriate OPUS-MT model
    tokenizer, model = get_model(src, target)

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Generate translation
    with torch.no_grad():
        translated_tokens = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)

    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return {"translated": translated_text.strip()}
