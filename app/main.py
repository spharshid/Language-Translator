import os
import json
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import MarianMTModel, MarianTokenizer

# Assuming 'app.utils' is in your project structure, kept as is
from app.utils import detect_lang, LANG_MAP

# ✅ Set cache for Hugging Face Spaces
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"

app = FastAPI(title="Language Translator 🚀")

# Allow all origins (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_LANGS = list(LANG_MAP.keys())
MODEL_CACHE = {}


def get_model(src: str, tgt: str):
    """Load and cache the model/tokenizer for source-target language pair."""
    key = f"{src}-{tgt}"
    if key not in MODEL_CACHE:
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        # Using a context manager for potential device placement if needed, but sticking to original 'cpu'
        model.to("cpu")
        MODEL_CACHE[key] = (tokenizer, model)
    return MODEL_CACHE[key]


class TranslateRequest(BaseModel):
    text: str
    lang: str  # target language code


@app.get("/")
def root():
    return {"status": "ok", "supported_languages": SUPPORTED_LANGS, "endpoint": "/translate"}


# 🔁 Recursively translate JSON string values
def translate_json(obj, translate_func):
    if isinstance(obj, str):
        return translate_func(obj)
    elif isinstance(obj, list):
        return [translate_json(item, translate_func) for item in obj]
    elif isinstance(obj, dict):
        return {
            key: translate_json(value, translate_func)
            for key, value in obj.items()
        }
    else:
        return obj


@app.post("/translate")
def translate(req: TranslateRequest):
    text = req.text.strip()
    # Default target to 'en' if req.lang is None or empty after strip, though Pydantic should handle typing
    target = req.lang.strip().lower() if req.lang else "en"

    if target not in SUPPORTED_LANGS:
        return {"error": f"Unsupported language '{target}'. Supported: {SUPPORTED_LANGS}"}

    # Detect source language
    src = detect_lang(text)
    if src not in SUPPORTED_LANGS:
        src = "en"

    # Skip translation if source and target are same
    if src == target:
        return {"translated": text}

    tokenizer, model = get_model(src, target)

    def do_translate(single_text: str) -> str:
        inputs = tokenizer(single_text, return_tensors="pt", padding=True)
        with torch.no_grad():
            tokens = model.generate(
                **inputs, max_length=128, num_beams=4, early_stopping=True
            )
        return tokenizer.decode(tokens[0], skip_special_tokens=True).strip()

    # 🧠 Try parsing input as JSON
    try:
        json_obj = json.loads(text)
        translated_obj = translate_json(json_obj, do_translate)
        return {"translated": json.dumps(translated_obj, indent=2, ensure_ascii=False)}
    except json.JSONDecodeError:
        # Normal text
        translated_text = do_translate(text)
        return {"translated": translated_text}