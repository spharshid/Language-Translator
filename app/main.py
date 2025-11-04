from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import MarianMTModel, MarianTokenizer
from app.utils import detect_lang, LANG_MAP
import torch
import json

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
    """Load and cache the model/tokenizer for source-target language pair."""
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
    return {"status": "ok", "supported_languages": SUPPORTED_LANGS}


# 🔁 Recursively translate JSON string values
def translate_json(obj, translate_func):
    if isinstance(obj, str):
        return translate_func(obj)
    elif isinstance(obj, list):
        return [translate_json(item, translate_func) for item in obj]
    elif isinstance(obj, dict):
        return {key: translate_json(value, translate_func) for key, value in obj.items()}
    else:
        return obj


@app.post("/translate")
def translate(req: TranslateRequest):
    text = req.text.strip()
    target = req.lang.strip().lower() if req.lang else "en"

    if target not in SUPPORTED_LANGS:
        return {"error": f"Unsupported language '{target}'. Supported: {SUPPORTED_LANGS}"}

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
            tokens = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
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
