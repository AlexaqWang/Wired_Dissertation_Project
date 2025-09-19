# === BERTopic: Stage 1 (Connectivity line A, EOM), keep post-processing unchanged ===
# Code and comments in English only.

# === Step 0: Paths & Imports ===
import os, time, json, ast, re, itertools, math, random
import numpy as np
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from tqdm.auto import tqdm
from pathlib import Path

# Optional: torch for device selection
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

tqdm.pandas(desc="Combining text for BERTopic")

# I/O paths (adjust to your environment)
INPUT_PATH  = "/content/drive/MyDrive/bertopic_modeling/bertopic_cleaned_articles.csv"
OUTPUT_ROOT = "/content/drive/MyDrive/bertopic_modeling/output_stage1_A"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Reproducibility
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# === High-level knobs (shared) ===
FILTER_MIN_TOKENS = 8                  # filter out very short docs
USE_TOPIC_MERGE = False                # keep False for initial clustering quality check
TOPIC_MERGE_TARGET = "auto"

APPLY_OUTLIER_REDUCTION = False        # keep False initially (unchanged post-processing)
OUTLIER_THRESHOLD = 0.02

# Vectorizer knobs
VEC_MIN_DF = 5                         # try 2–5 if topics are very diverse
VEC_NGRAM_MAX = 2                      # try 2–3 if you want more phrase features

# Cohesion estimation
COHESION_SAMPLE_PER_CLUSTER = 50       # sample size for intra-cluster cosine estimation
COHESION_MAX_PAIRWISE = 1000           # cap the number of pairwise comparisons per cluster

# === Step 1: Load data & preprocess ===
def parse_tags_to_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set)):
            return " ".join(map(str, v))
    except Exception:
        pass
    return s

def token_len(s):
    return len(re.findall(r"\w+", s or ""))

def combine_main_text(row):
    """
    Build the text for embeddings without duplicating title/dek.
    Primary source: clean_text_embed (already includes title/dek from the cleaning pipeline).
    Fallback: if clean_text_embed is empty, concatenate title and dek minimally.
    """
    base = str(row.get("clean_text_embed", "") or "").strip()
    if base:
        return base
    parts = []
    if pd.notna(row.get("title")) and row["title"]:
        parts.append(str(row["title"]))
    if pd.notna(row.get("dek")) and row["dek"]:
        parts.append(str(row["dek"]))
    return " ".join(parts).strip()

def combine_with_light_weights(row, has_subcat: bool, has_tags: bool):
    """
    Add light-weight structured hints (subcategory, tags) only.
    Do NOT re-add title/dek to avoid duplication.
    """
    base = row["combined_embed"] or ""
    extras = []
    if has_subcat:
        subc = str(row.get("subcategory", "")).strip()
        if subc:
            extras.append(subc)
    if has_tags:
        tg = str(row.get("tags", "")).strip()
        if tg:
            extras.append(tg)
    return (base + " " + " ".join(extras)).strip() if extras else base

print("Loading CSV...")
df = pd.read_csv(INPUT_PATH, encoding="utf-8", low_memory=False)
if "tags" not in df.columns and "tag" in df.columns:
    df = df.rename(columns={"tag": "tags"})

for col in ["title", "dek", "clean_text_embed", "subcategory", "tags", "category"]:
    if col not in df.columns:
        df[col] = ""

df["tags"] = df["tags"].astype(str).map(parse_tags_to_str)

# Keep publish_date as NaT if missing; do not drop rows here
if "publish_date" in df.columns:
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
else:
    df["publish_date"] = pd.NaT

print("Combining fields (no duplication of title/dek)...")
df["combined_embed"] = df.progress_apply(combine_main_text, axis=1)

print("Filtering ultra-short docs...")
df["len_tokens"] = df["combined_embed"].fillna("").map(token_len)
df = df[df["len_tokens"] >= FILTER_MIN_TOKENS].copy()

has_subcat = df["subcategory"].fillna("").astype(str).str.strip().ne("").any()
has_tags   = df["tags"].fillna("").astype(str).str.strip().ne("").any()

df["weighted_text"] = df.progress_apply(
    lambda r: combine_with_light_weights(r, has_subcat, has_tags), axis=1
)
documents = df["weighted_text"].tolist()
print(f"Documents after filtering: {len(documents)}")

# === Step 1.1: Build article_key for alignment (year/article_folder) ===
def safe_slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-_/]", " ", str(s or ""))
    s = re.sub(r"\s+", "-", s.strip())
    return s[:100] if s else "na"

def infer_article_key_from_pathlike(x: str):
    if not x or not isinstance(x, str):
        return None
    try:
        p = Path(x)
        if len(p.parts) >= 2:
            return f"{p.parts[-2]}/{p.parts[-1]}"
    except Exception:
        pass
    tokens = re.split(r"[\\/]+", x)
    if len(tokens) >= 2:
        return f"{tokens[-2]}/{tokens[-1]}"
    return None

def build_article_key(row):
    # Prefer local dir/path-like columns first (adjust names if your CSV uses different columns)
    for col in ["article_dir", "images_dir", "img_dir", "local_dir", "local_path", "output_dir", "path"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            k = infer_article_key_from_pathlike(str(row[col]))
            if k:
                return k

    # Fallback: publish year + slug from url/title
    year = None
    if pd.notna(row.get("publish_date")):
        try:
            year = int(pd.to_datetime(row["publish_date"]).year)
        except Exception:
            year = None
    slug = None
    if "url" in row and pd.notna(row["url"]) and str(row["url"]).strip():
        slug = Path(str(row["url"]).rstrip("/")).name
    if not slug and "slug" in row and pd.notna(row["slug"]):
        slug = str(row["slug"])
    if not slug and "title" in row and pd.notna(row["title"]):
        slug = safe_slug(str(row["title"]))

    if year and slug:
        return f"{year}/{safe_slug(slug)}"
    return None

print("Building article_key for alignment...")
df["article_key"] = df.apply(build_article_key, axis=1)
missing = df["article_key"].isna().sum()
if missing > 0:
    print(f"[WARN] {missing} rows without article_key; please ensure a path-like column exists in INPUT CSV.")
# Keep only rows with a valid article_key for downstream alignment
df = df[df["article_key"].notna()].copy()

# === Step 2: Embeddings & Vectorizer ===
print("Loading embedding model...")
if _TORCH_AVAILABLE:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)

print(f"Encoding documents on device={device}...")
embeddings = embedding_model.encode(
    documents,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, VEC_NGRAM_MAX),
    min_df=VEC_MIN_DF
)

# === Step 3: Helper functions ===
def reassign_noise_by_centroid(topics: np.ndarray,
                               embeddings: np.ndarray,
                               min_sim: float) -> np.ndarray:
    """Reassign -1 points to nearest topic centroid if cosine similarity >= min_sim."""
    topics = np.asarray(topics).copy()
    valid_mask = topics != -1
    unique_topics = np.unique(topics[valid_mask])
    if unique_topics.size == 0:
        return topics

    centroids, topic_ids = [], []
    for t in unique_topics:
        idx = np.where(topics == t)[0]
        if idx.size == 0:
            continue
        c = embeddings[idx].mean(axis=0)
        norm = np.linalg.norm(c) + 1e-12
        c = c / norm
        centroids.append(c)
        topic_ids.append(t)
    if not centroids:
        return topics

    C = np.vstack(centroids)
    topic_ids = np.array(topic_ids)

    noise_idx = np.where(topics == -1)[0]
    if noise_idx.size == 0:
        return topics

    X = embeddings[noise_idx]
    sims = X @ C.T
    best_k = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    reassigned = 0
    for i, idx in enumerate(noise_idx):
        if best_sim[i] >= min_sim:
            topics[idx] = topic_ids[best_k[i]]
            reassigned += 1
    print(f"[Post-assign] Reassigned {reassigned}/{len(noise_idx)} (min_sim={min_sim})")
    return topics

def entropy_from_sizes(sizes):
    total = sum(sizes)
    if total == 0:
        return 0.0
    p = np.array(sizes, dtype=float) / total
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    H_max = np.log(len(p)) if len(p) > 0 else 1.0
    return float(H / H_max) if H_max > 0 else 0.0

def gini_from_sizes(sizes):
    x = np.array(sorted(sizes))
    n = len(x)
    if n == 0:
        return 0.0
    cumx = np.cumsum(x)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(gini)

def hhi_from_sizes(sizes):
    total = sum(sizes)
    if total == 0:
        return 0.0
    p2 = [(s / total) ** 2 for s in sizes if s > 0]
    return float(sum(p2))

def estimate_intra_cluster_cohesion(topics: np.ndarray,
                                    embeddings: np.ndarray,
                                    sample_per_cluster: int = 50,
                                    max_pairwise: int = 1000) -> float:
    """Weighted average of intra-cluster cosine similarity."""
    topics = np.asarray(topics)
    valid = topics != -1
    uniq = [t for t in np.unique(topics[valid]) if t != -1]
    if not uniq:
        return np.nan

    def avg_pairwise_cos(X, max_pairs):
        n = X.shape[0]
        if n < 2:
            return np.nan
        vals = []
        num_pairs = min(max_pairs, n*(n-1)//2)
        for _ in range(num_pairs):
            i, j = np.random.randint(0, n), np.random.randint(0, n-1)
            if j >= i:
                j += 1
            vals.append(float(np.dot(X[i], X[j])))
        return float(np.mean(vals)) if vals else np.nan

    weighted_sum, weight_total = 0.0, 0.0
    for t in uniq:
        idx = np.where(topics == t)[0]
        size = len(idx)
        if size < 2:
            continue
        if size > sample_per_cluster:
            idx = np.random.choice(idx, size=sample_per_cluster, replace=False)
        X = embeddings[idx]
        sim = avg_pairwise_cos(X, max_pairs=max_pairwise)
        if not np.isnan(sim):
            weighted_sum += sim * size
            weight_total += size
    return float(weighted_sum / weight_total) if weight_total > 0 else np.nan

def evaluate_assignment(topics: np.ndarray,
                        probs,
                        embeddings: np.ndarray) -> dict:
    topics = np.asarray(topics)
    total_docs = len(topics)
    noise = int(np.sum(topics == -1))
    noise_ratio = noise / total_docs if total_docs else 0.0

    unique, counts = np.unique(topics[topics != -1], return_counts=True)
    sizes = counts.tolist()
    n_clusters = len(sizes)
    largest_share = (max(sizes) / total_docs) if sizes else 0.0

    size_entropy = entropy_from_sizes(sizes)
    gini = gini_from_sizes(sizes)
    hhi = hhi_from_sizes(sizes)

    if probs is not None and getattr(probs, "shape", None) and probs.shape[0] == total_docs:
        try:
            avg_conf = float(np.max(probs, axis=1).mean())
        except Exception:
            avg_conf = np.nan
    else:
        avg_conf = np.nan

    cohesion = estimate_intra_cluster_cohesion(topics, embeddings,
                                               sample_per_cluster=COHESION_SAMPLE_PER_CLUSTER,
                                               max_pairwise=COHESION_MAX_PAIRWISE)

    return {
        "n_docs": total_docs,
        "noise_ratio": round(noise_ratio, 6),
        "n_clusters": n_clusters,
        "largest_cluster_share": round(largest_share, 6),
        "size_entropy": round(size_entropy, 6),
        "gini": round(gini, 6),
        "hhi": round(hhi, 6),
        "avg_confidence": None if np.isnan(avg_conf) else round(avg_conf, 6),
        "avg_intra_cluster_cohesion": None if np.isnan(cohesion) else round(cohesion, 6),
    }

# Keep a reusable list of output columns (includes article_key for alignment)
DOC_COLS_TO_KEEP = [
    "article_key",
    "publish_date", "title", "dek", "clean_text_embed",
    "combined_embed", "weighted_text",
    "category", "subcategory", "tags",
]
existing_cols = [c for c in DOC_COLS_TO_KEEP if c in df.columns]

# === Step 4: Grid definitions (Connectivity line A) ===
N_NEIGHBORS_GRID      = [60, 90, 120]     # enlarged neighborhoods
MIN_SAMPLES_GRID      = [5, 8, 10]        # relax density threshold
POST_ASSIGN_SIM_GRID  = [None, 0.38, 0.40]  # keep your current post-assign setup (unchanged)

MIN_CLUSTER_SIZE = 15
N_COMPONENTS     = 20
MIN_DIST         = 0.05

# === Step 5: Run grid ===
summary_rows = []

for n_neighbors in N_NEIGHBORS_GRID:
    for min_samples in MIN_SAMPLES_GRID:
        exp_base = f"STRATA_n{n_neighbors}_c{N_COMPONENTS}_mdist{int(MIN_DIST*100):03d}_msize{MIN_CLUSTER_SIZE}_minsamp[{min_samples}]"
        print(f"\n=== Base fit: {exp_base} ===")

        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=N_COMPONENTS,
            min_dist=MIN_DIST,
            densmap=False,
            metric="cosine",
            random_state=RANDOM_STATE
        )

        # Connectivity line: EOM, no epsilon
        hdbscan_model = HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=min_samples,
            metric="euclidean",
            prediction_data=True,
            cluster_selection_method="eom",
            approx_min_span_tree=True
        )

        topic_model = BERTopic(
            embedding_model=None,            # we pass precomputed embeddings
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            language="english",
            verbose=True,
            calculate_probabilities=True,
            top_n_words=10
        )

        t0 = time.time()
        topics_base, probs_base = topic_model.fit_transform(documents, embeddings=embeddings)
        print(f"Fit finished in {round(time.time() - t0, 2)}s")

        # Evaluate base (no post-assign)
        base_metrics = evaluate_assignment(topics_base, probs_base, embeddings)
        base_row = {
            "exp_name": f"{exp_base}_postassign_none",
            "strategy": "A",
            "n_neighbors": n_neighbors,
            "n_components": N_COMPONENTS,
            "min_dist": MIN_DIST,
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "min_samples": min_samples,
            "post_assign_min_sim": None,
            **base_metrics
        }
        summary_rows.append(base_row)

        base_dir = os.path.join(OUTPUT_ROOT, base_row["exp_name"])
        os.makedirs(base_dir, exist_ok=True)

        # === Export base topic info/keywords aligned with model internal state
        topic_info = topic_model.get_topic_info()
        base_topic_info_path = os.path.join(base_dir, "topic_info.csv")
        topic_info.to_csv(base_topic_info_path, index=False)

        # topic_id -> topic_name mapping
        if {"Topic", "Name"}.issubset(set(topic_info.columns)):
            topic_labels = topic_info[["Topic", "Name"]].rename(columns={"Topic": "topic_id", "Name": "topic_name"})
        else:
            # Fallback if schema differs
            topic_labels = topic_info.rename(columns={topic_info.columns[0]: "topic_id"})
            if "topic_name" not in topic_labels.columns:
                topic_labels["topic_name"] = topic_labels["topic_id"].astype(str)
        topic_labels.to_csv(os.path.join(base_dir, "topic_labels.csv"), index=False)

        # Documents with base topics (CSV + Parquet)
        df_out = df[existing_cols].copy()
        df_out["topic_id"] = topics_base
        if probs_base is not None:
            try:
                df_out["probability"] = probs_base.max(axis=1)
            except Exception:
                df_out["probability"] = np.nan
        else:
            df_out["probability"] = np.nan
        df_out["len_tokens"] = df["len_tokens"]

        df_out.to_parquet(os.path.join(base_dir, "document_topics.parquet"), index=False)
        df_out.to_csv(os.path.join(base_dir, "document_topics.csv"), index=False)

        # Export topic->keywords table (unchanged)
        topic_keywords_rows = []
        for topic_id in topic_info["Topic"]:
            if topic_id == -1:
                continue
            words = topic_model.get_topic(topic_id) or []
            for word, weight in words:
                topic_keywords_rows.append({"Topic": topic_id, "Keyword": word, "Weight": weight})
        pd.DataFrame(topic_keywords_rows).to_csv(os.path.join(base_dir, "topic_keywords.csv"), index=False)

        # Save model snapshot (unchanged)
        topic_model.save(os.path.join(base_dir, "bertopic_model"))

        # Post-assign sweep (keep the same post-processing choices; probabilities not exported for post-assign)
        for post_sim in POST_ASSIGN_SIM_GRID:
            if post_sim is None:
                continue

            exp_name = f"{exp_base}_postassign_{int(post_sim*100):02d}"
            print(f"\n--- Post-assign: {exp_name} ---")

            topics = reassign_noise_by_centroid(topics_base, embeddings, min_sim=post_sim)
            probs = probs_base  # keep for metrics only

            # Keep post-processing unchanged: outlier reduction OFF by default
            if APPLY_OUTLIER_REDUCTION:
                topics = topic_model.reduce_outliers(
                    documents=documents,
                    topics=topics,
                    probabilities=probs,
                    strategy="distributions",
                    threshold=OUTLIER_THRESHOLD
                )

            if USE_TOPIC_MERGE:
                topics, _ = topic_model.reduce_topics(documents, topics, nr_topics=TOPIC_MERGE_TARGET)

            metrics = evaluate_assignment(topics, probs, embeddings)
            row = {
                "exp_name": exp_name,
                "strategy": "A",
                "n_neighbors": n_neighbors,
                "n_components": N_COMPONENTS,
                "min_dist": MIN_DIST,
                "min_cluster_size": MIN_CLUSTER_SIZE,
                "min_samples": min_samples,
                "post_assign_min_sim": post_sim,
                **metrics
            }
            summary_rows.append(row)

            exp_dir = os.path.join(OUTPUT_ROOT, exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            # Export documents with post-assigned topics ONLY (CSV + Parquet).
            # Do NOT export topic_info/keywords or save model to avoid inconsistency with internal state.
            df_out = df[existing_cols].copy()
            df_out["topic_id"] = topics
            df_out["probability"] = np.nan  # avoid mismatch with new topics after external reassignment
            df_out["len_tokens"] = df["len_tokens"]

            df_out.to_parquet(os.path.join(exp_dir, "document_topics.parquet"), index=False)
            df_out.to_csv(os.path.join(exp_dir, "document_topics.csv"), index=False)

# === Step 6: Write summary ===
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_ROOT, "experiments_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nAll experiments finished. Summary saved to: {summary_path}")

# === Step 7: Quick guidance (printed) ===
def pretty_guidance(row):
    noise = row["noise_ratio"]
    nclu  = row["n_clusters"]
    lmax  = row["largest_cluster_share"]
    coh   = row["avg_intra_cluster_cohesion"]
    hints = []
    if noise < 0.02:
        hints.append("Noise very low -> consider increasing min_samples or decreasing n_neighbors, or turn off/raise post-assign threshold.")
    elif noise > 0.25:
        hints.append("Noise high -> consider decreasing min_samples or increasing n_neighbors.")
    if lmax < 0.08 and nclu > 60:
        hints.append("Many small clusters -> consider increasing min_cluster_size or slightly reducing n_neighbors.")
    if coh is not None and coh < 0.45:
        hints.append("Low cohesion -> consider stricter min_samples or higher post-assign threshold.")
    return "; ".join(hints) if hints else "Looks balanced. Verify top-words and sample docs manually."

print("\n=== TOP 10 by (low noise × high cohesion) heuristic ===")
if not summary_df.empty:
    tmp = summary_df.copy()
    tmp["coh_score"] = tmp["avg_intra_cluster_cohesion"].fillna(0.0)
    tmp["rank_score"] = (1.0 - tmp["noise_ratio"]) * (0.5 + 0.5 * tmp["coh_score"])
    tmp = tmp.sort_values(["rank_score", "avg_intra_cluster_cohesion"], ascending=[False, False])
    top10 = tmp.head(10)
    for _, r in top10.iterrows():
        print(
            f"{r['exp_name']}: noise={r['noise_ratio']:.3f}, "
            f"n_clusters={r['n_clusters']}, largest_share={r['largest_cluster_share']:.3f}, "
            f"cohesion={r['avg_intra_cluster_cohesion'] if r['avg_intra_cluster_cohesion'] is not None else np.nan:.3f} | "
            f"HINT: {pretty_guidance(r)}"
        )
else:
    print("Summary is empty. Check earlier steps/logs.")




# === Stage 2: Global + Local macro-clustering on topic centroids (Agglomerative) ===
# Code and comments in English only.

import os, re, json, glob
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
ROOTS = [
    "/content/drive/MyDrive/bertopic_modeling/output_stage1_A",
    "/content/drive/MyDrive/bertopic_modeling/output_grid_v2",
]

EXP_BASE = "STRATA_n60_c20_mdist005_msize15_minsamp[8]"
PREFERRED_DOC_SUFFIX = "postassign_40"

# Reporting
TOP_N_TOPICS_PRINT = 10
N_KEYWORDS_PER_TOPIC = 15
N_AGG_KEYWORDS_PER_CLUSTER = 20
N_TITLES_SAMPLE = 8

# ------- Global macro-clustering (Agglomerative only) -------
GLOBAL_K = 12                  # desired number of macro-clusters (set by you)
GLOBAL_K_RANGE_FOR_SCORING = (10, 16)  # only for scoring printouts

# ------- Local re-clustering (optional) -------
ENABLE_LOCAL_RECLUSTERING = True
# re-cluster clusters whose topic-count or doc-count is large
LOCAL_MIN_TOPIC_COUNT = 6          # trigger if a cluster contains >= this many topics
LOCAL_MIN_DOC_COUNT = 600          # or if cluster maps to >= this many documents
LOCAL_K_SEARCH = [2, 3, 4]         # candidate k for local Agglomerative on that cluster
LOCAL_KEEP_IF_SIL_IMPROVES = 0.01  # require at least this silhouette gain to accept split

# encoder / randomness
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ------------------------------------------------------------
# Helpers: directory auto-discovery
# ------------------------------------------------------------
def find_doc_dir(roots, exp_base, preferred_suffix):
    for root in roots:
        preferred = os.path.join(root, f"{exp_base}_{preferred_suffix}")
        if os.path.isfile(os.path.join(preferred, "document_topics.parquet")):
            return preferred
    candidates = []
    for root in roots:
        pattern = os.path.join(root, f"{exp_base}_*")
        for d in glob.glob(pattern):
            if os.path.isfile(os.path.join(d, "document_topics.parquet")):
                candidates.append(d)
    candidates = sorted(candidates, key=lambda p: (0 if "postassign_40" in p else 1, p))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No document_topics.parquet found under any of {roots} for prefix '{exp_base}_*'."
    )

def find_base_dir(roots, exp_base):
    need_files = ["bertopic_model", "topic_keywords.csv", "topic_info.csv"]
    for root in roots:
        ideal = os.path.join(root, f"{exp_base}_postassign_none")
        if all(os.path.exists(os.path.join(ideal, nf)) for nf in need_files):
            return ideal
    candidates = []
    for root in roots:
        pattern = os.path.join(root, f"{exp_base}_*")
        for d in glob.glob(pattern):
            if all(os.path.exists(os.path.join(d, nf)) for nf in need_files):
                candidates.append(d)
    candidates = sorted(candidates, key=lambda p: (0 if "postassign_none" in p else 1, p))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not find a base folder for {exp_base} with {{'bertopic_model','topic_keywords.csv','topic_info.csv'}} "
        f"in any of {roots}."
    )

# ------------------------------------------------------------
# Locate folders robustly
# ------------------------------------------------------------
DIR_DOCS = find_doc_dir(ROOTS, EXP_BASE, PREFERRED_DOC_SUFFIX)
DIR_BASE = find_base_dir(ROOTS, EXP_BASE)
print(f"[INFO] Using DOCS dir: {DIR_DOCS}")
print(f"[INFO] Using BASE dir: {DIR_BASE}")

# ------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------
model = BERTopic.load(os.path.join(DIR_BASE, "bertopic_model"))
topic_info = pd.read_csv(os.path.join(DIR_BASE, "topic_info.csv"))

kw_df = pd.read_csv(os.path.join(DIR_BASE, "topic_keywords.csv")).sort_values(
    ["Topic", "Weight"], ascending=[True, False]
)
kw_df["Topic"] = pd.to_numeric(kw_df["Topic"], errors="coerce").fillna(-1).astype(int)

docs = pd.read_parquet(os.path.join(DIR_DOCS, "document_topics.parquet"))

# Normalize topic column name
CANDIDATES = ["topic", "Topic", "topic_id", "assigned_topic", "topic_label"]
found = next((c for c in CANDIDATES if c in docs.columns), None)
if found is None:
    raise KeyError(
        f"No topic-like column found in document_topics.parquet. "
        f"Available columns: {list(docs.columns)}. Expected one of {CANDIDATES}."
    )
docs[found] = pd.to_numeric(docs[found], errors="coerce").fillna(-1).astype(int)
if found != "topic":
    docs = docs.rename(columns={found: "topic"})

print("[DEBUG] docs.columns:", list(docs.columns))
print("[DEBUG] head:", docs.head(2).to_dict(orient="list"))

def ensure_text(df: pd.DataFrame) -> pd.Series:
    # Prefer weighted_text; else concat title/dek/clean_text_embed
    if "weighted_text" in df.columns:
        return df["weighted_text"].fillna("").astype(str)
    idx = df.index
    title = df["title"].astype(str).fillna("") if "title" in df.columns else pd.Series("", index=idx)
    dek   = df["dek"].astype(str).fillna("") if "dek" in df.columns else pd.Series("", index=idx)
    main  = df["clean_text_embed"].astype(str).fillna("") if "clean_text_embed" in df.columns else pd.Series("", index=idx)
    return (title + " " + dek + " " + main).str.strip()

raw_texts = ensure_text(docs)
texts = (raw_texts
         .str.replace(r"\bnan\b", "", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip())

# ------------------------------------------------------------
# Basic stats & preview
# ------------------------------------------------------------
n_docs = len(docs)
n_noise_docs = int((docs["topic"] == -1).sum())
noise_ratio_docs = n_noise_docs / n_docs if n_docs else 0.0

valid_topics = sorted(int(t) for t in docs["topic"].unique() if t != -1)
n_topics = len(valid_topics)

topic_sizes = (
    docs[docs["topic"] != -1]["topic"]
    .value_counts()
    .rename_axis("Topic")
    .reset_index(name="Size")
    .sort_values("Size", ascending=False)
)

topic2preview = (
    kw_df.groupby("Topic")["Keyword"]
    .apply(lambda s: ", ".join(list(s.head(10)))).to_dict()
)

exp_name = os.path.basename(DIR_DOCS)
print(f"=== Selected Experiment: {exp_name} ===")
print(f"Documents: {n_docs} | Noise docs: {n_noise_docs} | Noise ratio: {noise_ratio_docs:.4f}")
print(f"Number of topics (excluding -1): {n_topics}")

print("\n=== Top topics by document size ===")
for _, row in topic_sizes.head(TOP_N_TOPICS_PRINT).iterrows():
    tid = int(row["Topic"]); size = int(row["Size"])
    words = topic2preview.get(tid, "")
    print(f"Topic {tid:>4} | size={size:>4} | top-words: {words}")

# ------------------------------------------------------------
# Encode docs & build topic centroids (probability-weighted if available)
# ------------------------------------------------------------
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    device = "cpu"

encoder = SentenceTransformer("all-mpnet-base-v2", device=device)
print(f"\nEncoding documents on device={device} ...")
X = encoder.encode(
    texts.tolist(),
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

mask = docs["topic"] != -1
assigned_topics = docs.loc[mask, "topic"].astype(int).to_numpy()
assigned_indices = np.where(mask.to_numpy())[0]
uniq_topics = np.unique(assigned_topics)

# optional probability weighting
prob = None
if "probability" in docs.columns:
    prob = pd.to_numeric(docs.loc[mask, "probability"], errors="coerce").fillna(1.0).to_numpy()
    # clip to [0,1] to avoid weird values
    prob = np.clip(prob, 0.0, 1.0)

topic_centroids = {}
for t in uniq_topics:
    idxs = assigned_indices[assigned_topics == t]
    if idxs.size == 0:
        continue
    if prob is None:
        c = X[idxs].mean(axis=0)
    else:
        # weighted mean with small epsilon to avoid zero-weights
        w = prob[assigned_topics == t]
        w = w + 1e-6
        c = (X[idxs] * w[:, None]).sum(axis=0) / (w.sum() + 1e-6)
    c = c / (np.linalg.norm(c) + 1e-12)
    topic_centroids[int(t)] = c

present_ids = sorted(topic_centroids.keys())
assert len(present_ids) > 0, "No topic centroids found."
T = np.vstack([topic_centroids[t] for t in present_ids])
T = normalize(T, axis=1)  # unit sphere, cosine ~ dot
topicid_to_row = {t: i for i, t in enumerate(present_ids)}

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def eval_labels_cosine(emb: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels != -1
    lab = labels[mask]
    res = {"n_total": len(labels), "n_assigned": int(mask.sum())}
    if lab.size == 0 or len(np.unique(lab)) < 2:
        res.update({"silhouette": -1.0, "n_clusters": 0, "max_cluster_share": 1.0})
        return res
    try:
        sil = float(silhouette_score(emb[mask], lab, metric="cosine"))
    except Exception:
        sil = -1.0
    counts = np.array([(lab == c).sum() for c in np.unique(lab)], dtype=float)
    max_share = counts.max() / len(labels) if len(labels) else 1.0
    res.update({"silhouette": sil, "n_clusters": int(len(np.unique(lab))), "max_cluster_share": float(max_share)})
    return res

def range_penalty(nc: int, low: int, high: int) -> float:
    if nc == 0:
        return 1.0
    if low <= nc <= high:
        return 0.0
    if nc < low:
        return (low - nc) / max(1, low)
    return (nc - high) / max(1, high)

def score_solution(sil, nc, max_share, desired_range):
    # no noise in agglomerative; keep score simple
    return (sil
            - 0.20 * max_share
            - 0.30 * range_penalty(nc, desired_range[0], desired_range[1]))

# ------------------------------------------------------------
# Global Agglomerative clustering (cosine, average)
# ------------------------------------------------------------
def run_agglomerative_k(T: np.ndarray, k: int):
    # sklearn >= 1.2: metric; older: affinity
    try:
        ac = AgglomerativeClustering(n_clusters=k, linkage="average", metric="cosine")
    except TypeError:
        ac = AgglomerativeClustering(n_clusters=k, linkage="average", affinity="cosine")
    labels = ac.fit_predict(T)
    met = eval_labels_cosine(T, labels)
    scr = score_solution(met["silhouette"], met["n_clusters"], met["max_cluster_share"], GLOBAL_K_RANGE_FOR_SCORING)
    return labels, met, scr, ac

labels, met_global, score_global, ac_model = run_agglomerative_k(T, GLOBAL_K)
print(f"\n[GLOBAL|Agglomerative] k={met_global['n_clusters']} | silhouette={met_global['silhouette']:.3f} "
      f"| max_share={met_global['max_cluster_share']:.3f} | score={score_global:.3f}")

# ------------------------------------------------------------
# Local re-clustering on selected big clusters (optional)
# ------------------------------------------------------------
def try_agglomerative_best_k(emb_sub: np.ndarray, k_list: list):
    """Search best k on a sub-embedding with cosine silhouette."""
    cands = []
    for k in k_list:
        if k <= 1 or k > emb_sub.shape[0]:
            continue
        try:
            lab, met, scr, _ = run_agglomerative_k(emb_sub, k)
            cands.append((k, lab, met, scr))
        except Exception:
            continue
    if not cands:
        return None
    # choose by highest silhouette, then score
    cands.sort(key=lambda x: (x[2]["silhouette"], x[3]), reverse=True)
    return cands[0]

def local_recluster_selected(T_all, global_labels, topic_ids, topic_sizes_df, docs_df):
    """
    For clusters that are large in topic-count or doc-count, attempt local re-clustering.
    Accept only if cosine silhouette improves by at least LOCAL_KEEP_IF_SIL_IMPROVES.
    """
    labels_new = global_labels.copy()
    unique_clusters = np.unique(global_labels)
    local_changes = []

    # Precompute mapping: topic -> row index in T_all
    topic_to_row = {t: i for i, t in enumerate(topic_ids)}
    # topic -> doc count
    t_sizes = topic_sizes_df.set_index("Topic")["Size"].to_dict()

    for c in unique_clusters:
        idxs_topic = [i for i, lab in enumerate(global_labels) if lab == c]
        if len(idxs_topic) == 0:
            continue

        # topic-count criterion
        topic_count = len(idxs_topic)

        # doc-count criterion: sum docs mapped to these topics
        topics_in_c = [topic_ids[i] for i in idxs_topic]
        doc_count = int(sum(t_sizes.get(t, 0) for t in topics_in_c))

        if (topic_count < LOCAL_MIN_TOPIC_COUNT) and (doc_count < LOCAL_MIN_DOC_COUNT):
            continue  # skip small clusters

        sub_emb = T_all[idxs_topic]
        base_met = eval_labels_cosine(sub_emb, np.zeros(len(idxs_topic)))  # trivial 1-cluster: silhouette invalid
        # compute current effective silhouette using their current global label (all same label)
        # We instead compute silhouette if we keep them as one cluster -> invalid; set to -inf baseline
        base_sil = -1.0

        # try k in LOCAL_K_SEARCH
        best_local = try_agglomerative_best_k(sub_emb, LOCAL_K_SEARCH)
        if best_local is None:
            continue
        k_best, sub_labels, sub_met, sub_score = best_local

        if (sub_met["silhouette"] - base_sil) >= LOCAL_KEEP_IF_SIL_IMPROVES and k_best >= 2:
            # Remap: keep the largest sub-cluster as old id c, others assign new ids
            uniq_sub = [u for u in np.unique(sub_labels)]
            # order sub-clusters by size desc
            counts = {u: int((sub_labels == u).sum()) for u in uniq_sub}
            uniq_sub_sorted = sorted(uniq_sub, key=lambda u: counts[u], reverse=True)

            # find next new label id
            existing_ids = set(labels_new)
            next_id = (max(existing_ids) + 1) if len(existing_ids) else 0

            keep = uniq_sub_sorted[0]
            mapping = {keep: int(c)}
            for u in uniq_sub_sorted[1:]:
                mapping[u] = next_id
                next_id += 1

            # apply mapping to global labels
            for local_i, global_i in enumerate(idxs_topic):
                labels_new[global_i] = mapping[sub_labels[local_i]]

            local_changes.append({
                "old_cluster": int(c),
                "k_new": int(k_best),
                "silhouette_new": float(sub_met["silhouette"]),
                "topic_count": int(topic_count),
                "doc_count": int(doc_count),
                "created_clusters": [int(mapping[u]) for u in uniq_sub_sorted[1:]],
            })

    return labels_new, local_changes

if ENABLE_LOCAL_RECLUSTERING:
    labels_after_local, local_changes = local_recluster_selected(
        T_all=T,
        global_labels=labels,
        topic_ids=present_ids,
        topic_sizes_df=topic_sizes,
        docs_df=docs
    )
    if local_changes:
        labels = labels_after_local
        print(f"[LOCAL] Applied local re-clustering to {len(local_changes)} global clusters.")
        for chg in local_changes:
            print(f"  - Cluster {chg['old_cluster']} -> k={chg['k_new']}, "
                  f"sil_new={chg['silhouette_new']:.3f}, "
                  f"topic_count={chg['topic_count']}, doc_count={chg['doc_count']}, "
                  f"new_ids={chg['created_clusters']}")
    else:
        print("[LOCAL] No clusters qualified for local re-clustering.")

# ------------------------------------------------------------
# Build mappings and reports
# ------------------------------------------------------------
topic2cluster = {t: int(c) for t, c in zip(present_ids, labels)}
topic_cluster_df = pd.DataFrame({"Topic": present_ids, "Cluster": labels})

# Map document-level clusters via their topic assignment
docs["Cluster"] = docs["topic"].map(lambda x: topic2cluster.get(int(x), np.nan) if x != -1 else np.nan)

print("\n=== Global+Local macro-clusters on topic centroids ===")
print("Topic-level cluster sizes (including -1 if any):")
print(pd.Series(labels).value_counts().sort_index())

print("\nDocument-level cluster sizes:")
print(docs["Cluster"].value_counts(dropna=False).sort_index())

# Aggregate keywords per cluster (size-weighted by topic doc count)
t_sizes = topic_sizes.set_index("Topic")["Size"].to_dict()
kw_df["W"] = kw_df["Weight"] * kw_df["Topic"].map(t_sizes).fillna(0.0)

agg_kw = defaultdict(Counter)
for t, c in topic2cluster.items():
    if c == -1:
        continue
    sub = kw_df[kw_df["Topic"] == t]
    for w, val in zip(sub["Keyword"], sub["W"]):
        agg_kw[int(c)][w] += float(val)

print("\nAggregated keywords per Cluster:")
for c in sorted(k for k in set(labels) if k != -1):
    words = ", ".join([w for w, _ in agg_kw[c].most_common(N_AGG_KEYWORDS_PER_CLUSTER)])
    print(f"Cluster {c}: {words}")

print("\nSample titles per Cluster:")
if "title" not in docs.columns:
    docs["title"] = ""
for c in sorted([c for c in docs["Cluster"].dropna().unique()]):
    sub = docs[docs["Cluster"] == c]
    k = min(N_TITLES_SAMPLE, len(sub))
    sample = sub.sample(k, random_state=RANDOM_SEED) if k > 0 else sub.head(0)
    titles = [str(t)[:120] for t in sample["title"].fillna("").tolist()]
    print(f"\nCluster {int(c)} | n_docs={len(sub)} | n_topics={int((topic_cluster_df['Cluster']==c).sum())}")
    for t in titles:
        print(" -", t)

n_topic_noise = int((topic_cluster_df["Cluster"] == -1).sum())
print(f"\nTopic-level noise topics: {n_topic_noise} out of {len(topic_cluster_df)}")

# ------------------------------------------------------------
# Save artifacts
# ------------------------------------------------------------
map_path = os.path.join(DIR_DOCS, "topic_clusters_global_agglocal.csv")
docs_path = os.path.join(DIR_DOCS, "document_topic_clusters_global_agglocal.parquet")
meta_path = os.path.join(DIR_DOCS, "topic_clusters_global_agglocal_meta.json")

topic_cluster_df.to_csv(map_path, index=False)
docs.to_parquet(docs_path, index=False)

# Metrics for meta
topic_noise_share = float((topic_cluster_df["Cluster"] == -1).sum()) / len(topic_cluster_df) if len(topic_cluster_df) else 1.0
doc_assigned_share = float(docs["Cluster"].notna().sum()) / len(docs) if len(docs) else 0.0
doc_noise_share = float((docs["Cluster"] == -1.0).sum()) / len(docs) if len(docs) else 1.0

meta = {
    "experiment": os.path.basename(DIR_DOCS),
    "roots": ROOTS,
    "exp_base": EXP_BASE,
    "dirs": {"base": DIR_BASE, "docs": DIR_DOCS},
    "n_docs": int(n_docs),
    "noise_ratio_docs_input": float(noise_ratio_docs),
    "n_topics": int(n_topics),
    "macro": {
        "method": "agglomerative",
        "global_k": int(GLOBAL_K),
        "silhouette_global": float(met_global["silhouette"]),
        "max_cluster_share_global": float(met_global["max_cluster_share"]),
        "score_global": float(score_global),
        "desired_range": GLOBAL_K_RANGE_FOR_SCORING,
    },
    "local": {
        "enabled": ENABLE_LOCAL_RECLUSTERING,
        "min_topic_count": int(LOCAL_MIN_TOPIC_COUNT),
        "min_doc_count": int(LOCAL_MIN_DOC_COUNT),
        "k_search": list(map(int, LOCAL_K_SEARCH)),
        "keep_if_sil_improves": float(LOCAL_KEEP_IF_SIL_IMPROVES),
        "changes": local_changes if ENABLE_LOCAL_RECLUSTERING else [],
    },
    "topic_level_cluster_sizes": {str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().to_dict().items()},
    "document_level_cluster_sizes": {str(k): int(v) for k, v in docs["Cluster"].value_counts(dropna=False).sort_index().to_dict().items()},
    "coverage": {
        "topic_noise_share_after": topic_noise_share,
        "doc_assigned_share_after": doc_assigned_share,
        "doc_noise_share_after": doc_noise_share
    }
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\nSaved mapping to: {map_path}")
print(f"Saved doc-level labels to: {docs_path}")
print(f"Saved meta to: {meta_path}")




# === Export document_topics.csv for CLIP alignment  ===

import os, re, json
from pathlib import Path
import numpy as np
import pandas as pd

# ----------- CONFIG: adjust if needed -----------
OUTPUT_DIR = "/content/drive/MyDrive/bertopic_modeling/output_stage1_A"
DOC_TOPICS_PATH = os.path.join(OUTPUT_DIR, "document_topics.csv")
TOPIC_LABELS_PATH = os.path.join(OUTPUT_DIR, "topic_labels.csv")

# Optional: path to CLIP’s aligned file for a quick overlap check
CLIP_ALIGNED_PATH = "/content/drive/MyDrive/out_clip_align/aligned_image_text.csv"
# -----------------------------------------------

def _safe_slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-_/]", " ", str(s or ""))
    s = re.sub(r"\s+", "-", s.strip())
    return s[:100] if s else "na"

def _infer_article_key_from_pathlike(x: str):
    if not x or not isinstance(x, str):
        return None
    try:
        p = Path(x)
        if len(p.parts) >= 2:
            return f"{p.parts[-2]}/{p.parts[-1]}"
    except Exception:
        pass
    tokens = re.split(r"[\\/]+", x)
    if len(tokens) >= 2:
        return f"{tokens[-2]}/{tokens[-1]}"
    return None

def _build_article_key(row: pd.Series):
    # 1) 尝试从本地路径/目录列推断（这些列名按你之前代码常见值列出来了）
    for col in ["article_dir", "images_dir", "img_dir", "local_dir", "local_path", "output_dir", "path"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            k = _infer_article_key_from_pathlike(str(row[col]))
            if k:
                return k

    # 2) 回退：使用 publish_date 的年份 + url/slug/title 的 slug
    year = None
    if pd.notna(row.get("publish_date")):
        try:
            year = int(pd.to_datetime(row["publish_date"]).year)
        except Exception:
            year = None

    slug = None
    if "url" in row and pd.notna(row["url"]) and str(row["url"]).strip():
        slug = Path(str(row["url"]).rstrip("/")).name
    if not slug and "slug" in row and pd.notna(row["slug"]):
        slug = str(row["slug"])
    if not slug and "title" in row and pd.notna(row["title"]):
        slug = _safe_slug(str(row["title"]))

    if year and slug:
        return f"{year}/{_safe_slug(slug)}"
    return None

def _canonicalize_key(s: str) -> str:
    s = str(s).replace("\\", "/").strip().strip("/")
    s = s.lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9/_]", "", s)
    return s

def export_document_topics(df_source: pd.DataFrame,
                           topics_vector,
                           probs_matrix=None,
                           topic_model_obj=None,
                           out_dir=OUTPUT_DIR,
                           doc_csv_path=DOC_TOPICS_PATH,
                           labels_csv_path=TOPIC_LABELS_PATH,
                           clip_aligned_path=CLIP_ALIGNED_PATH):
    os.makedirs(out_dir, exist_ok=True)

    df = df_source.copy()
    for col in ["title", "dek", "publish_date", "url", "slug", "article_dir", "images_dir", "img_dir", "local_dir", "local_path", "output_dir", "path"]:
        if col not in df.columns:
            df[col] = ""

    topics_arr = np.asarray(topics_vector)
    assert len(topics_arr) == len(df), f"Length mismatch: topics={len(topics_arr)} vs docs={len(df)}"

    if "article_key" not in df.columns:
        df["article_key"] = df.apply(_build_article_key, axis=1)
    missing = int(df["article_key"].isna().sum())
    if missing > 0:
        print(f"[WARN] {missing} rows without article_key; they will be dropped from export.")
    df = df[df["article_key"].notna()].copy()

    out = pd.DataFrame({
        "article_key": df["article_key"].astype(str),
        "topic_id": topics_arr[:len(df)]
    })

    if probs_matrix is not None:
        try:
            probs = np.asarray(probs_matrix)
            if probs.shape[0] == len(df):
                out["probability"] = probs.max(axis=1)
        except Exception:
            pass

    out = out.drop_duplicates(subset=["article_key"])
    out.to_csv(doc_csv_path, index=False)
    print(f"[OK] Saved: {doc_csv_path} | rows={len(out)} | topics(uniq)={out['topic_id'].nunique()}")

    if topic_model_obj is not None:
        try:
            tinfo = topic_model_obj.get_topic_info()
            if {"Topic", "Name"}.issubset(tinfo.columns):
                tlabels = tinfo[["Topic", "Name"]].rename(columns={"Topic": "topic_id", "Name": "topic_name"})
                tlabels.to_csv(labels_csv_path, index=False)
                print(f"[OK] Saved: {labels_csv_path} | label rows={len(tlabels)}")
        except Exception as e:
            print(f"[WARN] Could not export topic_labels: {e}")

    if clip_aligned_path and os.path.isfile(clip_aligned_path):
        try:
            clip_df = pd.read_csv(clip_aligned_path, usecols=["article_key"])
            a = set(out["article_key"].map(_canonicalize_key))
            b = set(clip_df["article_key"].map(_canonicalize_key))
            inter = len(a & b)
            print(f"[CHECK] article_key overlap with CLIP aligned: {inter} / {len(a)} "
                  f"({inter/max(1,len(a)):.1%} of BERTopic keys)")
        except Exception as e:
            print(f"[WARN] Overlap check failed: {e}")

try:
    export_document_topics(
        df_source=df,
        topics_vector=topics,
        probs_matrix=probs,
        topic_model_obj=topic_model
    )
except NameError as e:
    print("[ERROR] Please pass the correct variables (df/topics/probs/topic_model).", e)
