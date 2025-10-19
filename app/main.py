from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import MarianMTModel, MarianTokenizer
from app.utils import detect_lang, LANG_MAP
import torch

app = FastAPI(
    title="Language Translator",
    description="🚀 FastAPI backend running on HuggingFace Spaces using Helsinki-NLP/opus-mt models.",
    version="1.0.0"
)

# CORS (HuggingFace allows public API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_LANGS = list(LANG_MAP.keys())
MODEL_CACHE = {}

def get_model(src: str, tgt: str):
    key = f"{src}-{tgt}"
    if key not in MODEL_CACHE:
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.to("cpu")
        MODEL_CACHE[key] = (tokenizer, model)
    return MODEL_CACHE[key]

class TranslateRequest(BaseModel):
    text: str
    lang: str

@app.get("/")
def root():
    return {"message": "✅ HuggingFace Space is live!", "supported_languages": SUPPORTED_LANGS, "endpoint": "/translate"}

@app.post("/translate")
def translate(req: TranslateRequest):
    text = req.text.strip()
    target = req.lang.strip().lower() if req.lang else "en"

    if target not in SUPPORTED_LANGS:
        return {"error": f"Unsupported language '{target}'. Supported: {SUPPORTED_LANGS}"}

    src = detect_lang(text)
    if src not in SUPPORTED_LANGS:
        src = "en"

    if src == target:
        return {"translated": text}

    tokenizer, model = get_model(src, target)
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    with torch.no_grad():
        translated_tokens = model.generate(**inputs, max_length=128)

    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return {"translated": translated_text.strip()}
