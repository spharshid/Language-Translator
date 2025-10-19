# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import MarianMTModel, MarianTokenizer
from app.utils import detect_lang, LANG_MAP
import torch
import gc
import os
import time
from typing import Tuple

app = FastAPI(title="Language Translator")

# Allow all origins (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported languages (keys of LANG_MAP)
SUPPORTED_LANGS = list(LANG_MAP.keys())

# Max number of models to keep in memory at once (1 for low-RAM environments)
MAX_MODEL_CACHE = int(os.getenv("MAX_MODEL_CACHE", "1"))

# MODEL_CACHE stores { "src-tgt": (tokenizer, model, last_used_timestamp) }
MODEL_CACHE = {}

def free_model_memory(key: str):
    """Delete a cached model to free memory."""
    try:
        tokenizer, model, _ = MODEL_CACHE.pop(key)
        # try to delete references and run GC
        del tokenizer
        del model
        gc.collect()
        time.sleep(0.1)
    except Exception:
        pass

def evict_if_needed():
    """Ensure MODEL_CACHE size <= MAX_MODEL_CACHE by evicting least recently used."""
    while len(MODEL_CACHE) > MAX_MODEL_CACHE:
        # find oldest entry
        oldest_key = min(MODEL_CACHE.items(), key=lambda kv: kv[1][2])[0]
        free_model_memory(oldest_key)

def get_model(src: str, tgt: str) -> Tuple:
    """Load model/tokenizer for a source-target pair, cached. Evicts old models if needed."""
    key = f"{src}-{tgt}"
    if key in MODEL_CACHE:
        tokenizer, model, _ = MODEL_CACHE[key]
        # update last used timestamp
        MODEL_CACHE[key] = (tokenizer, model, time.time())
        return tokenizer, model

    # If cache full, evict LRU
    if len(MODEL_CACHE) >= MAX_MODEL_CACHE:
        evict_if_needed()

    # load model (this may use significant memory)
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.to("cpu")
    MODEL_CACHE[key] = (tokenizer, model, time.time())
    return tokenizer, model

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

    # Skip translation if source == target
    if src == target:
        return {"translated": text}

    try:
        tokenizer, model = get_model(src, target)
    except Exception as e:
        # likely model doesn't exist or download error
        return {"error": f"Failed to load model for {src}->{target}: {str(e)}"}

    # Tokenize and infer
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            translated_tokens = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    except Exception as e:
        # on inference error, try to free memory and return error
        # attempt to free model memory
        key = f"{src}-{target}"
        if key in MODEL_CACHE:
            free_model_memory(key)
        return {"error": f"Inference failed: {str(e)}"}

    # Optionally evict old models to keep memory low
    evict_if_needed()
    return {"translated": translated_text.strip()}
