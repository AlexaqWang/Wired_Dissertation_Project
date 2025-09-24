# =========================
# Step 0: Install deps
# =========================
!pip install -U transformers accelerate bitsandbytes sentencepiece pillow pandas sentence-transformers open_clip_torch --quiet

import os
import re
import math
import json
import hashlib
import random
import torch
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
from typing import List, Tuple, Dict, Any, Iterable, Optional

from transformers import AutoProcessor, AutoModelForImageTextToText
from sentence_transformers import SentenceTransformer, util as sbert_util

import torch.nn.functional as F

# =========================
# Reproducibility
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# Step 1: Mount Google Drive
# =========================
from google.colab import drive
drive.mount('/content/drive')

# Roots
image_root = "/content/drive/MyDrive/Wired_framing_matched_articles"
out_csv_path = "/content/drive/MyDrive/wired_llava_caption.csv"  # dedicated output for VLM auxiliary pipeline
assert os.path.exists(image_root), f"Image root not found: {image_root}"

# =========================
# Step 2: Load Models (LLaVA, SBERT, optional CLIP)
# =========================
llava_model_id = "llava-hf/llava-1.5-7b-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda" and torch.cuda.is_bf16_supported():
    llava_dtype = torch.bfloat16
else:
    llava_dtype = torch.float16 if device == "cuda" else torch.float32

# LLaVA
processor = AutoProcessor.from_pretrained(llava_model_id, use_fast=True)
llava = AutoModelForImageTextToText.from_pretrained(
    llava_model_id,
    torch_dtype=llava_dtype,
    device_map="auto"
)
llava.eval()

# SBERT for semantic alignment
sbert_device = "cuda" if torch.cuda.is_available() else "cpu"
sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=sbert_device)

# Optional: CLIP (denotation-level alignment)
clip_available = False
try:
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-336", pretrained="openai", device=device
    )
    clip_model.eval()
    clip_tokenize = open_clip.tokenize
    clip_dtype = torch.float16 if device == "cuda" else torch.float32
    clip_available = True
    print("[Info] CLIP ViT-L-14-336 loaded.")
except Exception as e:
    print("[Warn] CLIP not available, will skip denotation-level similarity:", e)

# =========================
# Step 3: Prompts (freeform + structured JSON)
# =========================
FREEFORM_PROMPTS = [
    {
        "id": "caption",
        "type": "freeform",
        "text": "Write a concise, neutral caption (<=40 words) that factually describes the image content (scene, composition, key objects/people) without speculation."
    }
]

STRUCTURED_PROMPTS = [
    {
        "id": "analysis",
        "type": "structured",
        "text": (
            "Analyze the image's affective tone (color palette and lighting), possible metaphors or narratives, "
            "and whether it implies that complex social problems are solvable by technology. "
            "Return ONLY valid JSON with the following schema (no extra text): "
            '{'
            '"tone_label": "optimistic|authoritative|critical|neutral", '
            '"tone_conf": 0.0, '
            '"metaphor_tags": ["tag1","tag2"], '
            '"solutionism_label": "yes|no|uncertain", '
            '"solutionism_conf": 0.0, '
            '"rationale": "one concise sentence explaining your decision"'
            '}'
        )
    }
]

# stable id for prompts set (for reproducibility)
prompts_all_text = "||".join([p["text"] for p in FREEFORM_PROMPTS + STRUCTURED_PROMPTS])
prompts_hash = hashlib.md5(prompts_all_text.encode()).hexdigest()[:8]

# =========================
# Step 4: Helpers (path parsing, article context, image io, batching)
# =========================
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

insert_pat = re.compile(r"^insert[_\- ]?(\d+)?$", re.IGNORECASE)
hero_pat = re.compile(r"^hero$", re.IGNORECASE)

def parse_year_and_title(abs_path: str, root_dir: str) -> Tuple[str, str]:
    """
    Expect path like: root/<year>/<title>/<imagefile>
    Returns (year, title). Falls back to ('unknown', parent_name or stem).
    """
    rel = os.path.relpath(abs_path, root_dir)
    parts = rel.split(os.sep)
    if len(parts) >= 3:
        year = parts[0]
        title = parts[-2]
    elif len(parts) == 2:
        year = parts[0]
        title = os.path.splitext(parts[1])[0]
    else:
        year = "unknown"
        title = os.path.splitext(os.path.basename(abs_path))[0]
    return year, title

def classify_image(fname: str) -> Tuple[str, Any]:
    """
    Classify image as 'hero' or 'insert' (with index) or 'other'.
    Accepted names:
      - hero.jpg
      - insert.jpg / insert1.jpg / insert_1.jpg / insert-1.jpg
    """
    stem = os.path.splitext(fname)[0]
    if hero_pat.match(stem):
        return "hero", 0
    m = insert_pat.match(stem)
    if m:
        idx = m.group(1)
        return "insert", int(idx) if idx else 0
    return "other", None

def load_and_resize_image(path: str, max_side: int = 768) -> Image.Image:
    """
    Load image as RGB and resize so that the longer side <= max_side.
    Keeps aspect ratio. If smaller than max_side, keep original size.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return img

def iter_image_files(root: str) -> Iterable[Tuple[str, str, str]]:
    """
    Yield (abs_path, rel_path, fname) for images under root.
    """
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                fpath = os.path.join(dirpath, fname)
                relpath = os.path.relpath(fpath, root).replace("\\", "/")
                yield fpath, relpath, fname

def iter_batches(lst: List[Any], batch_size: int) -> Iterable[List[Any]]:
    """
    Simple list batching iterator.
    """
    n = len(lst)
    for i in range(0, n, batch_size):
        yield lst[i:i+batch_size]

def make_article_key(rel_path: str, year: str, title: str) -> str:
    """
    Prefer 'year/folder' using first two levels of rel_path; fallback to 'year/title'
    """
    parts = rel_path.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return f"{year}/{title}"

def read_textfile_safe(path: str, max_chars: int = 8000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            s = f.read()
            return s[:max_chars]
    except Exception:
        return ""

def load_article_context(abs_img_path: str) -> Tuple[str, str, str]:
    """
    Try to load (title, dek, text_ref) from meta.json and/or local .txt in the same folder.
    text_ref is truncated for robustness.
    """
    d = os.path.dirname(abs_img_path)
    meta_path = os.path.join(d, "meta.json")
    title, dek, text_ref = "", "", ""

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
                meta = json.load(f)
            title = str(meta.get("title", "") or "")
            dek = str(meta.get("dek", "") or "")
            tf = meta.get("text_file", "")
            if tf:
                tf_path = tf if os.path.isabs(tf) else os.path.join(d, tf)
                if os.path.exists(tf_path):
                    text_ref = read_textfile_safe(tf_path, 20000)
        except Exception:
            pass

    if not text_ref:
        # fallback to common text file names
        for cand in ["article.txt", "text.txt", "body.txt"]:
            p = os.path.join(d, cand)
            if os.path.exists(p):
                text_ref = read_textfile_safe(p, 20000)
                break

    # final fallback
    if not text_ref:
        text_ref = (title + " " + dek).strip()

    # truncate very long text for embedding stability
    if text_ref and len(text_ref) > 4000:
        text_ref = text_ref[:4000]

    return title, dek, text_ref

def cosine_similarity(a_vec: torch.Tensor, b_vec: torch.Tensor) -> float:
    return float(sbert_util.cos_sim(a_vec, b_vec).item())

# =========================
# Step 5: Build image task list with caching
# =========================
# Cache to skip already-answered (image_relpath, prompt_id)
cache = set()
if os.path.exists(out_csv_path):
    try:
        old_df = pd.read_csv(out_csv_path)
        for _, row in old_df.iterrows():
            cache.add((str(row.get("image_relpath", "")), str(row.get("prompt_id", ""))))
        print(f"[Cache] Loaded {len(cache)} existing (image_relpath, prompt_id) pairs.")
    except Exception as e:
        print(f"[Warn] Failed to read existing CSV for cache: {e}")

# Build per-image task metadata
image_tasks: List[Dict[str, Any]] = []
for abs_path, rel_path, fname in iter_image_files(image_root):
    year, title = parse_year_and_title(abs_path, image_root)
    img_type, img_index = classify_image(fname)
    image_tasks.append({
        "abs_path": abs_path,
        "rel_path": rel_path,
        "fname": fname,
        "year": year,
        "title": title,
        "image_type": img_type,
        "image_index": img_index
    })

print(f"[Info] Discovered {len(image_tasks)} images under {image_root}.")

# materialize (image, prompt) samples: 2 prompts per image (freeform + structured)
def make_samples_for_image(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    samples = []
    for plist in (FREEFORM_PROMPTS, STRUCTURED_PROMPTS):
        for p in plist:
            k = (task["rel_path"], p["id"])
            if k in cache:
                continue
            s = dict(task)
            s["prompt_id"] = p["id"]
            s["prompt_type"] = p["type"]
            s["question"] = p["text"]
            samples.append(s)
    return samples

all_samples: List[Dict[str, Any]] = []
for t in image_tasks:
    all_samples.extend(make_samples_for_image(t))

print(f"[Info] Pending samples for inference: {len(all_samples)}")

# =========================
# Step 6: Batched VLM inference + alignment metrics
# =========================
MAX_SIDE = 768
BATCH_SIZE = 8
MAX_NEW_TOKENS = 256

records = []
model_id_used = llava_model_id
now_ts = datetime.now().isoformat(timespec="seconds")

# caches to avoid redundant work
article_ctx_cache: Dict[str, Dict[str, Any]] = {}   # article_key -> {"title":..., "dek":..., "text_ref":..., "text_ref_emb":...}
clip_sim_cache: Dict[str, Optional[float]] = {}     # image_relpath -> clip_sim

def ensure_article_context(s: Dict[str, Any]) -> Dict[str, Any]:
    ak = make_article_key(s["rel_path"], s["year"], s["title"])
    if ak not in article_ctx_cache:
        title, dek, text_ref = load_article_context(s["abs_path"])
        # pre-embed text_ref for SBERT similarity
        text_ref_emb = None
        if text_ref:
            text_ref_emb = sbert.encode(text_ref, convert_to_tensor=True, normalize_embeddings=True)
        article_ctx_cache[ak] = {
            "title": title,
            "dek": dek,
            "text_ref": text_ref,
            "text_ref_emb": text_ref_emb
        }
    return article_ctx_cache[ak]

def compute_clip_sim(img: Image.Image, s: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[float]:
    if not clip_available:
        return None
    rel = s["rel_path"]
    if rel in clip_sim_cache:
        return clip_sim_cache[rel]
    text = (ctx.get("title", "") + ". " + ctx.get("dek", "")).strip() or ctx.get("title", "")
    if not text:
        clip_sim_cache[rel] = None
        return None
    try:
        with torch.no_grad():
            image_in = clip_preprocess(img).unsqueeze(0).to(device, dtype=clip_dtype)
            text_in = clip_tokenize([text]).to(device)
            image_feat = clip_model.encode_image(image_in)
            text_feat = clip_model.encode_text(text_in)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            sim = float((image_feat @ text_feat.T).squeeze().item())
        clip_sim_cache[rel] = sim
        return sim
    except Exception:
        clip_sim_cache[rel] = None
        return None

def build_analysis_text(parsed: Dict[str, Any]) -> str:
    tone = str(parsed.get("tone_label", ""))
    mt = parsed.get("metaphor_tags", [])
    if isinstance(mt, list):
        mt = ", ".join([str(x) for x in mt])
    sol = str(parsed.get("solutionism_label", ""))
    rat = str(parsed.get("rationale", ""))
    return f"tone={tone}; metaphors={mt}; solutionism={sol}; rationale={rat}"

total_batches = math.ceil(len(all_samples) / BATCH_SIZE)
processed = 0

for batch_id, batch in enumerate(iter_batches(all_samples, BATCH_SIZE), 1):
    images: List[Image.Image] = []
    texts: List[str] = []
    payload: List[Dict[str, Any]] = []

    # prepare images, prompts, and precompute contexts
    for s in batch:
        ctx = ensure_article_context(s)
        try:
            img = load_and_resize_image(s["abs_path"], MAX_SIDE)
        except Exception as e:
            err_ans = f"[ERROR] cannot open image: {e}"
            ak = make_article_key(s["rel_path"], s["year"], s["title"])
            records.append({
                "status": "error",
                "error": str(e),
                "year": s["year"],
                "title": s["title"],
                "article_key": ak,
                "image_type": s["image_type"],
                "image_index": s["image_index"],
                "image_name": s["fname"],
                "image_relpath": s["rel_path"],
                "prompt_id": s["prompt_id"],
                "prompt_type": s["prompt_type"],
                "question": s["question"],
                "answer": err_ans,
                "answer_structured": "",
                "parse_ok": False,
                "tone_label": "",
                "tone_conf": np.nan,
                "metaphor_tags": "",
                "solutionism_label": "",
                "solutionism_conf": np.nan,
                "rationale": "",
                "sem_sim_sbert": np.nan,
                "clip_sim": np.nan,
                "model_id": model_id_used,
                "decode": "greedy",
                "prompts_hash": prompts_hash,
                "seed": SEED,
                "ts": now_ts
            })
            continue

        # build chat prompt
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": s["question"]}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        images.append(img)
        texts.append(prompt)
        payload.append(s)

    if not images:
        continue

    # tokenize and move to device
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True).to(llava.device, dtype=llava_dtype)

    with torch.no_grad():
        output = llava.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            do_sample=False  # deterministic decoding
        )

    decoded = processor.batch_decode(output, skip_special_tokens=True)

    # post-process outputs and collect records
    for s, txt, img in zip(payload, decoded, images):
        if "ASSISTANT:" in txt:
            txt = txt.split("ASSISTANT:", 1)[-1].strip()

        ctx = ensure_article_context(s)
        ak = make_article_key(s["rel_path"], s["year"], s["title"])

        # try to parse structured JSON if prompt_type == "structured"
        parse_ok = False
        parsed = {}
        answer_structured = ""
        tone_label = ""
        tone_conf = np.nan
        metaphor_tags = ""
        solutionism_label = ""
        solutionism_conf = np.nan
        rationale = ""

        if s["prompt_type"] == "structured":
            raw = txt.strip()
            # attempt to isolate JSON substring if model added extra text
            json_candidate = raw
            first_brace = raw.find("{")
            last_brace = raw.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_candidate = raw[first_brace:last_brace+1]
            try:
                parsed = json.loads(json_candidate)
                answer_structured = json.dumps(parsed, ensure_ascii=False)
                tone_label = str(parsed.get("tone_label", ""))
                try:
                    tone_conf = float(parsed.get("tone_conf", np.nan))
                except Exception:
                    tone_conf = np.nan
                mt = parsed.get("metaphor_tags", [])
                if isinstance(mt, list):
                    metaphor_tags = ";".join([str(x) for x in mt])
                else:
                    metaphor_tags = str(mt)
                solutionism_label = str(parsed.get("solutionism_label", ""))
                try:
                    solutionism_conf = float(parsed.get("solutionism_conf", np.nan))
                except Exception:
                    solutionism_conf = np.nan
                rationale = str(parsed.get("rationale", ""))
                parse_ok = True
            except Exception:
                parse_ok = False

        # build text for semantic similarity
        if s["prompt_type"] == "structured" and parse_ok:
            answer_for_sim = build_analysis_text(parsed)
            answer_to_save = answer_structured
        else:
            answer_for_sim = txt
            answer_to_save = txt

        # SBERT semantic alignment: text_ref vs answer_for_sim
        sem_sim = np.nan
        try:
            if ctx.get("text_ref") and answer_for_sim:
                # pre-embedded text_ref
                tref_emb = ctx.get("text_ref_emb", None)
                if tref_emb is None:
                    tref_emb = sbert.encode(ctx["text_ref"], convert_to_tensor=True, normalize_embeddings=True)
                    article_ctx_cache[ak]["text_ref_emb"] = tref_emb
                ans_emb = sbert.encode(answer_for_sim, convert_to_tensor=True, normalize_embeddings=True)
                sem_sim = cosine_similarity(tref_emb, ans_emb)
        except Exception:
            sem_sim = np.nan

        # CLIP similarity (denotation-level): image vs (title+dek)
        clip_sim = np.nan
        try:
            cs = compute_clip_sim(img, s, ctx)
            if cs is not None:
                clip_sim = float(cs)
        except Exception:
            clip_sim = np.nan

        records.append({
            "status": "ok",
            "error": "",
            "year": s["year"],
            "title": s["title"],
            "article_key": ak,
            "image_type": s["image_type"],
            "image_index": s["image_index"],
            "image_name": s["fname"],
            "image_relpath": s["rel_path"],
            "prompt_id": s["prompt_id"],
            "prompt_type": s["prompt_type"],
            "question": s["question"],
            "answer": answer_to_save,
            "answer_structured": answer_structured,
            "parse_ok": bool(parse_ok),
            "tone_label": tone_label,
            "tone_conf": tone_conf,
            "metaphor_tags": metaphor_tags,
            "solutionism_label": solutionism_label,
            "solutionism_conf": solutionism_conf,
            "rationale": rationale,
            "text_ref_present": bool(ctx.get("text_ref")),
            "sem_sim_sbert": sem_sim,
            "clip_sim": clip_sim,
            "title_ctx": ctx.get("title", ""),
            "dek_ctx": ctx.get("dek", ""),
            "model_id": model_id_used,
            "decode": "greedy",
            "prompts_hash": prompts_hash,
            "seed": SEED,
            "ts": datetime.now().isoformat(timespec="seconds")
        })

    processed += len(payload)
    print(f"[Batch {batch_id}/{total_batches}] processed {processed}/{len(all_samples)} samples.")

# =========================
# Step 7: Append to CSV with de-dup
# =========================
new_df = pd.DataFrame(records)

# ensure consistent column order for readability
cols_preferred = [
    "status","error","year","title","article_key","image_type","image_index","image_name","image_relpath",
    "prompt_id","prompt_type","question","answer","answer_structured","parse_ok",
    "tone_label","tone_conf","metaphor_tags","solutionism_label","solutionism_conf","rationale",
    "text_ref_present","sem_sim_sbert","clip_sim","title_ctx","dek_ctx",
    "model_id","decode","prompts_hash","seed","ts"
]
for c in cols_preferred:
    if c not in new_df.columns:
        new_df[c] = np.nan
new_df = new_df[cols_preferred]

if os.path.exists(out_csv_path):
    try:
        old_df = pd.read_csv(out_csv_path)
    except Exception as e:
        print(f"[Warn] Failed to read existing CSV, creating a new one: {e}")
        old_df = pd.DataFrame(columns=new_df.columns)
    merged = pd.concat([old_df, new_df], ignore_index=True)
else:
    merged = new_df

# De-dup by keys (image + prompt_id); keep the latest
dedup_keys = ["image_relpath", "prompt_id"]
merged = merged.drop_duplicates(subset=dedup_keys, keep="last")

merged.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
print(f"\n[Done] Appended {len(new_df)} rows. Final rows: {len(merged)}")
print(f"Saved to: {out_csv_path}")
