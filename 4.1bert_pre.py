import pandas as pd
import re
import unicodedata
from bs4 import BeautifulSoup
from tqdm import tqdm
import os

tqdm.pandas(desc="Cleaning text")

INPUT_PATH = "/content/drive/MyDrive/framing_profile_merged.csv"
OUTPUT_CSV = "/content/drive/MyDrive/bertopic_modeling/bertopic_cleaned_articles.csv"
OUTPUT_PARQUET = "/content/drive/MyDrive/bertopic_modeling/bertopic_cleaned_articles.parquet"

FIELDS = ["article_id", "year", "title", "dek", "text", "publish_date", "source_url", "category", "subcategory", "tags"]

MIN_CHARS = 50           # filter out very short docs
MAX_CHARS = 120000       # avoid extremely long docs (topic mixing risk)
ENABLE_CHUNKING = False  # set True if your docs are very long
CHUNK_MAX_CHARS = 6000   # character budget per chunk
MIN_TOKENS = 30          # min token count filter for embeddings

# ---------------- Regex ----------------
URL_RE = re.compile(r"(http|https)://\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
USER_RE = re.compile(r"@\w+")

# ---------------- Helpers ----------------
def html_to_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return BeautifulSoup(s, "lxml").get_text(separator=" ")

def normalize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def replace_special_tokens(s: str) -> str:
    s = URL_RE.sub("[URL]", s)
    s = EMAIL_RE.sub("[EMAIL]", s)
    s = USER_RE.sub("[USER]", s)
    return s

def light_clean_for_embedding(s: str) -> str:
    s = html_to_text(s)
    s = replace_special_tokens(s)
    s = normalize_unicode(s)
    s = "".join(ch for ch in s if ch.isprintable())
    return s

def normalize_for_ctfidf(s: str) -> str:
    s = light_clean_for_embedding(s)
    s = s.lower()
    s = re.sub(r"[^\w\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fuse_fields(row: pd.Series) -> str:
    parts = []
    title = str(row.get("title", "") or "").strip()
    dek = str(row.get("dek", "") or "").strip()
    body = str(row.get("text", "") or "")
    if title:
        parts.append(title)
        parts.append(title)
    if dek:
        parts.append(dek)
    if body:
        parts.append(body)
    return "\n\n".join([p for p in parts if p]).strip()

def chunk_text(s: str, max_chars: int) -> list:
    if len(s) <= max_chars:
        return [s]
    paras = re.split(r"\n{2,}", s)
    chunks, buff = [], ""
    for p in paras:
        if not p.strip():
            continue
        if len(buff) + len(p) + 2 <= max_chars:
            buff = (buff + "\n\n" + p).strip() if buff else p
        else:
            if buff:
                chunks.append(buff)
            if len(p) <= max_chars:
                buff = p
            else:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
                buff = ""
    if buff:
        chunks.append(buff)
    return chunks

# ---------------- Load ----------------
df_raw = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

df = pd.DataFrame()
for col in FIELDS:
    df[col] = df_raw[col] if col in df_raw.columns else ""

df = df.dropna(subset=["text"])
df = df[df["text"].astype(str).str.strip().ne("")]

if "publish_date" in df.columns:
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")

# ---------------- Cleaning ----------------
df["full_text"] = df.progress_apply(fuse_fields, axis=1)
df["n_chars_raw"] = df["full_text"].str.len()
df = df[(df["n_chars_raw"] >= MIN_CHARS) & (df["n_chars_raw"] <= MAX_CHARS)]

df["clean_text_embed"] = df["full_text"].progress_apply(light_clean_for_embedding)
df["clean_text_tfidf"] = df["full_text"].progress_apply(normalize_for_ctfidf)

df = df[df["clean_text_embed"].str.len() >= MIN_CHARS]

if ENABLE_CHUNKING:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        chunks_embed = chunk_text(row["clean_text_embed"], CHUNK_MAX_CHARS)
        chunks_tfidf = chunk_text(row["clean_text_tfidf"], CHUNK_MAX_CHARS)
        n = min(len(chunks_embed), len(chunks_tfidf))
        for i in range(n):
            new_row = row.copy()
            new_row["chunk_id"] = i
            new_row["clean_text_embed"] = chunks_embed[i]
            new_row["clean_text_tfidf"] = chunks_tfidf[i]
            rows.append(new_row)
    df = pd.DataFrame(rows)
else:
    df["chunk_id"] = -1

norm_for_hash = (
    df["clean_text_embed"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
df["hash_key"] = norm_for_hash.map(hash)
df = df.drop_duplicates(subset=["hash_key"])

if "title" in df.columns and "publish_date" in df.columns:
    df = df.drop_duplicates(subset=["title", "publish_date"], keep="first").copy()

df["len_tokens"] = df["clean_text_embed"].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
df = df[df["len_tokens"] >= MIN_TOKENS].copy()

# ---------------- Save ----------------
keep_cols = [
    "article_id", "year", "publish_date", "source_url", "title", "dek",
    "category", "subcategory", "tags", "chunk_id",
    "clean_text_embed", "clean_text_tfidf", "n_chars_raw", "len_tokens"
]
existing_cols = [c for c in keep_cols if c in df.columns]
df_out = df[existing_cols].copy()

df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
try:
    df_out.to_parquet(OUTPUT_PARQUET, index=False)
except Exception:
    pass

print("Created BERTopic-ready texts.")
print(f"Docs: {len(df_out)}")
print(f"Saved CSV: {OUTPUT_CSV}")
print(f"Saved Parquet: {OUTPUT_PARQUET}")
