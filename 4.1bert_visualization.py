# === Visualization for Stage 1 & Stage 2  ===
# Generates:
# 1) Top topics (Stage 1) & Top macro-clusters (Stage 2) bar charts
# 2) Yearly trends (counts + shares) for Stage 1 (top-K topics) & Stage 2 (top-K macro-clusters)
# 3) Topic -> Macro-cluster assignment heatmap
# 4) Per-macro-cluster Top-N keywords bar charts
# 5) Size bar charts for BOTH stages
# 6) Joint-projected intertopic-style comparison (Stage1 topics vs Stage2 macro-centroids)

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Paths & basic config
# -----------------------------
ROOTS = [
    "/content/drive/MyDrive/bertopic_modeling/output_stage1_A",
    "/content/drive/MyDrive/bertopic_modeling/output_grid_v2",
]
EXP_BASE = "STRATA_n60_c20_mdist005_msize15_minsamp[8]"
PREFERRED_DOC_SUFFIX = "postassign_40"
SAVE_SUBDIR = "figs"

# Chart knobs
TOP_K_TOPICS       = 20
TOP_K_MACRO        = 20
TOP_N_KEYWORDS     = 10
TIME_TOP_CLUSTERS  = 8      # for yearly lines (Stage 2)
TIME_TOP_TOPICS    = 8      # for yearly lines (Stage 1)
FIG_DPI            = 150
RANDOM_STATE       = 42

# Yearly share smoothing
SMOOTH_WINDOW      = 3      # moving-average window (1=no smoothing)
YEAR_MIN_FORCE     = 2003   # force x-axis to start from this year; set None to use data min

# -----------------------------
# Helpers: guard & locate dirs
# -----------------------------
def guard_nonempty(df, name):
    ok = df is not None and not getattr(df, "empty", False)
    if not ok:
        print(f"[SKIP] {name}: no data")
    return ok

def find_doc_dir(roots, exp_base, preferred_suffix):
    for root in roots:
        preferred = os.path.join(root, f"{exp_base}_{preferred_suffix}")
        for cand in [
            "document_topic_clusters_global_agglocal.parquet",
            "document_topic_clusters_global.parquet",
            "document_topics.parquet",
        ]:
            if os.path.isfile(os.path.join(preferred, cand)):
                return preferred
    candidates = []
    for root in roots:
        pattern = os.path.join(root, f"{exp_base}_*")
        for d in glob.glob(pattern):
            for cand in [
                "document_topic_clusters_global_agglocal.parquet",
                "document_topic_clusters_global.parquet",
                "document_topics.parquet",
            ]:
                if os.path.isfile(os.path.join(d, cand)):
                    candidates.append(d); break
    candidates = sorted(candidates, key=lambda p: (0 if "postassign_40" in p else 1, p))
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No Stage-2 docs parquet found.")

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
    raise FileNotFoundError("Stage 1 base dir not found (needs topic_info/keywords/model).")

DIR_DOCS = find_doc_dir(ROOTS, EXP_BASE, PREFERRED_DOC_SUFFIX)
DIR_BASE = find_base_dir(ROOTS, EXP_BASE)
print("[INFO] DIR_DOCS:", DIR_DOCS)
print("[INFO] DIR_BASE:", DIR_BASE)

FIG_DIR = os.path.join(DIR_DOCS, SAVE_SUBDIR)
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
topic_info = pd.read_csv(os.path.join(DIR_BASE, "topic_info.csv"))
kw_df = pd.read_csv(os.path.join(DIR_BASE, "topic_keywords.csv")).sort_values(
    ["Topic", "Weight"], ascending=[True, False]
)
kw_df["Topic"] = pd.to_numeric(kw_df["Topic"], errors="coerce").fillna(-1).astype(int)

parquet_path = None
for cand in [
    "document_topic_clusters_global_agglocal.parquet",
    "document_topic_clusters_global.parquet",
    "document_topics.parquet",
]:
    p = os.path.join(DIR_DOCS, cand)
    if os.path.isfile(p):
        parquet_path = p; break
if parquet_path is None:
    raise FileNotFoundError("No document parquet in Stage 2 dir.")
docs = pd.read_parquet(parquet_path)

map_candidates = [
    os.path.join(DIR_DOCS, "topic_clusters_global_agglocal.csv"),
    os.path.join(DIR_DOCS, "topic_clusters_global.csv"),
]
topic_cluster_map_path = next((p for p in map_candidates if os.path.exists(p)), None)
if topic_cluster_map_path is None:
    raise FileNotFoundError("topic_clusters_global*.csv not found.")
t2c_df = pd.read_csv(topic_cluster_map_path)
t2c_df["Topic"] = pd.to_numeric(t2c_df["Topic"], errors="coerce").astype(int)
t2c_df["Cluster"] = pd.to_numeric(t2c_df["Cluster"], errors="coerce").astype(int)

# -----------------------------
# Precompute counts/summaries
# -----------------------------
if "topic" in docs.columns:
    topic_col = "topic"
elif "topic_id" in docs.columns:
    topic_col = "topic_id"
else:
    raise KeyError("No topic/topic_id column in docs parquet.")
docs[topic_col] = pd.to_numeric(docs[topic_col], errors="coerce").fillna(-1).astype(int)

topic_sizes = (
    docs[docs[topic_col] != -1][topic_col]
    .value_counts().rename_axis("Topic").reset_index(name="Size")
    .sort_values("Size", ascending=False)
)

if "Cluster" not in docs.columns:
    raise KeyError("docs missing 'Cluster' column from Stage 2.")
macro_sizes = (
    docs["Cluster"].dropna().astype(int)
    .value_counts().rename_axis("Cluster").reset_index(name="DocCount")
    .sort_values("DocCount", ascending=False)
)

topic_preview = (
    kw_df.groupby("Topic")["Keyword"]
    .apply(lambda s: ", ".join(list(s.head(5))))
    .to_dict()
)

t_size_map = dict(zip(topic_sizes["Topic"].astype(int), topic_sizes["Size"].astype(int)))
kw_df["W"] = kw_df["Weight"] * kw_df["Topic"].map(t_size_map).fillna(0.0)

from collections import defaultdict, Counter
agg_kw = defaultdict(Counter)
for _, row in t2c_df.iterrows():
    t = int(row["Topic"]); c = int(row["Cluster"])
    sub = kw_df[kw_df["Topic"] == t]
    for w, val in zip(sub["Keyword"], sub["W"]):
        agg_kw[c][w] += float(val)

# -----------------------------
# Small plotting utils
# -----------------------------
def plot_bar_with_labels(names, values, title, xlabel, outpath, annotate_right=True, figsize=(10,6)):
    if not names or not values:
        print(f"[SKIP] {title}: empty input"); return
    fig, ax = plt.subplots(figsize=figsize, dpi=FIG_DPI)
    y = np.arange(len(names))
    ax.barh(y, values)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel); ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    if annotate_right:
        for i, v in enumerate(values):
            ax.text(v, i, f" {int(v)}", va="center", ha="left", fontsize=9)
    plt.tight_layout(); plt.savefig(outpath, dpi=FIG_DPI); plt.close(fig)
    print("[SAVED]", outpath)

def ensure_text_for_embeddings(df: pd.DataFrame) -> pd.Series:
    if "weighted_text" in df.columns:
        return df["weighted_text"].fillna("").astype(str)
    parts = []
    parts.append(df["title"].astype(str).fillna("") if "title" in df.columns else "")
    parts.append(df["dek"].astype(str).fillna("") if "dek" in df.columns else "")
    parts.append(df["clean_text_embed"].astype(str).fillna("") if "clean_text_embed" in df.columns else "")
    t = (parts[0] + " " + parts[1] + " " + parts[2]) if parts else pd.Series([""]*len(df))
    return t.str.replace(r"\bnan\b","",regex=True).str.replace(r"\s+"," ",regex=True).str.strip()

# -----------------------------
# 1) Top bars
# -----------------------------
if guard_nonempty(topic_sizes, "topic_sizes"):
    top_topics = topic_sizes.head(TOP_K_TOPICS).copy()
    top_topics["label"] = top_topics.apply(
        lambda r: f"Topic {int(r['Topic'])}  |  {topic_preview.get(int(r['Topic']), '')}", axis=1
    )
    plot_bar_with_labels(
        names=top_topics["label"].tolist(),
        values=top_topics["Size"].tolist(),
        title=f"Top-{TOP_K_TOPICS} Topics by Size (Stage 1)",
        xlabel="Document count",
        outpath=os.path.join(FIG_DIR, f"bar_top{TOP_K_TOPICS}_topics_stage1.png"),
    )

if guard_nonempty(macro_sizes, "macro_sizes"):
    macro_kw_preview = {
        int(c): ", ".join([w for w, _ in agg_kw[int(c)].most_common(3)])
        for c in macro_sizes["Cluster"].astype(int).tolist()
    }
    top_macro = macro_sizes.head(TOP_K_MACRO).copy()
    top_macro["label"] = top_macro.apply(
        lambda r: f"Cluster {int(r['Cluster'])}  |  {macro_kw_preview.get(int(r['Cluster']), '')}", axis=1
    )
    plot_bar_with_labels(
        names=top_macro["label"].tolist(),
        values=top_macro["DocCount"].tolist(),
        title=f"Top-{TOP_K_MACRO} Macro-Clusters by Size (Stage 2)",
        xlabel="Document count",
        outpath=os.path.join(FIG_DIR, f"bar_top{TOP_K_MACRO}_macro_stage2.png"),
    )

# -----------------------------
# 2) Yearly trends: counts & shares (no filtering)
# -----------------------------
def _full_year_index(series_years):
    if series_years.isna().all():
        return None
    y_min = int(series_years.min()) if YEAR_MIN_FORCE is None else int(YEAR_MIN_FORCE)
    y_max = int(series_years.max())
    return range(y_min, y_max+1)

def _yearly_counts(df, group_col):
    years = pd.to_datetime(df["publish_date"], errors="coerce").dt.year
    df = df.assign(year=years)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    full_idx = _full_year_index(df["year"])
    if full_idx is None:
        return pd.DataFrame()
    # per-group counts
    df = df.dropna(subset=[group_col])
    if group_col == "Cluster":
        df[group_col] = df[group_col].astype(int)
    else:
        df = df[df[group_col] != -1].copy()
        df[group_col] = df[group_col].astype(int)
    gp = df.groupby(["year", group_col]).size().reset_index(name="count")
    # pivot and reindex
    P = gp.pivot(index="year", columns=group_col, values="count").fillna(0.0)
    P = P.reindex(full_idx, fill_value=0.0)
    # totals by year
    totals = P.sum(axis=1).replace(0, np.nan)  # 避免除零
    shares = P.div(totals, axis=0).fillna(0.0)
    return P, shares

def _plot_share_lines(df_s, title, outpath, legend_fmt=str, smooth_window=SMOOTH_WINDOW):
    if df_s is None or df_s.empty:
        print(f"[SKIP] {title}: empty"); return
    years = df_s.index.tolist()
    fig, ax = plt.subplots(figsize=(12,6), dpi=FIG_DPI)
    for col in df_s.columns:
        y = df_s[col].values
        if smooth_window and smooth_window > 1:
            y = pd.Series(y).rolling(smooth_window, min_periods=1, center=True).mean().values
        ax.plot(years, y, marker="o", label=legend_fmt(col))
    ax.set_title(title)
    ax.set_xlabel("Year"); ax.set_ylabel("Share of yearly articles")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _ : f"{int(round(v*100))}%"))
    ax.legend(ncol=4, fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout(); plt.savefig(outpath, dpi=FIG_DPI); plt.close(fig)
    print("[SAVED]", outpath)

def _plot_count_lines(df_c, title, outpath, legend_fmt=str, smooth_window=SMOOTH_WINDOW):
    if df_c is None or df_c.empty:
        print(f"[SKIP] {title}: empty"); return
    years = df_c.index.tolist()
    fig, ax = plt.subplots(figsize=(12,6), dpi=FIG_DPI)
    for col in df_c.columns:
        y = df_c[col].values
        if smooth_window and smooth_window > 1:
            y = pd.Series(y).rolling(smooth_window, min_periods=1, center=True).mean().values
        ax.plot(years, y, marker="o", label=legend_fmt(col))
    ax.set_title(title)
    ax.set_xlabel("Year"); ax.set_ylabel("Document count")
    ax.legend(ncol=4, fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout(); plt.savefig(outpath, dpi=FIG_DPI); plt.close(fig)
    print("[SAVED]", outpath)

# Stage 2 yearly (top K by overall size)
if guard_nonempty(macro_sizes, "macro_sizes for yearly"):
    P2, S2 = _yearly_counts(docs, "Cluster")
    if P2 is not None and not P2.empty:
        top_clusters_for_time = macro_sizes["Cluster"].astype(int).head(TIME_TOP_CLUSTERS).tolist()
        P2 = P2[top_clusters_for_time]
        S2 = S2[top_clusters_for_time]
        _plot_count_lines(P2, "Yearly Count by Macro-Cluster (Stage 2, smoothed)",
                          os.path.join(FIG_DIR, f"trend_yearly_count_stage2_top{TIME_TOP_CLUSTERS}.png"),
                          legend_fmt=lambda c: f"{c}", smooth_window=SMOOTH_WINDOW)
        _plot_share_lines(S2, "Yearly Share by Macro-Cluster (Stage 2, smoothed)",
                          os.path.join(FIG_DIR, f"trend_yearly_share_stage2_top{TIME_TOP_CLUSTERS}.png"),
                          legend_fmt=lambda c: f"{c}", smooth_window=SMOOTH_WINDOW)

# Stage 1 yearly (top K topics by overall size)
if guard_nonempty(topic_sizes, "topic_sizes for yearly"):
    P1, S1 = _yearly_counts(docs.rename(columns={topic_col:"TopicTmp"}), "TopicTmp")
    if P1 is not None and not P1.empty:
        top_topics_for_time = topic_sizes["Topic"].astype(int).head(TIME_TOP_TOPICS).tolist()
        keep = [c for c in top_topics_for_time if c in P1.columns]
        if keep:
            P1 = P1[keep]; S1 = S1[keep]
            _plot_count_lines(P1, "Yearly Count by Topic (Stage 1, smoothed)",
                              os.path.join(FIG_DIR, f"trend_yearly_count_stage1_top{TIME_TOP_TOPICS}.png"),
                              legend_fmt=lambda t: f"Topic {t}", smooth_window=SMOOTH_WINDOW)
            _plot_share_lines(S1, "Yearly Share by Topic (Stage 1, smoothed)",
                              os.path.join(FIG_DIR, f"trend_yearly_share_stage1_top{TIME_TOP_TOPICS}.png"),
                              legend_fmt=lambda t: f"Topic {t}", smooth_window=SMOOTH_WINDOW)

# -----------------------------
# 3) Topic → Macro heatmap
# -----------------------------
def plot_topic2cluster_heatmap(topic_sizes_df, map_df, outpath, figsize=(12,10)):
    if not guard_nonempty(topic_sizes_df, "topic_sizes for heatmap"): return
    if not guard_nonempty(map_df, "t2c map for heatmap"): return
    topics = sorted(topic_sizes_df["Topic"].astype(int).unique().tolist())
    clusters = sorted(map_df["Cluster"].astype(int).unique().tolist())
    if not topics or not clusters:
        print("[SKIP] heatmap: no topics or clusters"); return
    t2c_map = dict(zip(map_df["Topic"].astype(int), map_df["Cluster"].astype(int)))
    size_map = dict(zip(topic_sizes_df["Topic"].astype(int), topic_sizes_df["Size"].astype(int)))
    M = np.zeros((len(topics), len(clusters)), dtype=float)
    for i, t in enumerate(topics):
        c = t2c_map.get(t, None)
        if c is not None and c in clusters:
            j = clusters.index(c); M[i, j] = size_map.get(t, 0)
    fig, ax = plt.subplots(figsize=figsize, dpi=FIG_DPI)
    if (M > 0).any():
        vmin, vmax = np.quantile(M[M > 0], [0.05, 0.95])
        if vmin == vmax: vmin, vmax = 0.0, float(M.max())
        im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_title("Topic → Macro-Cluster Matrix (value = doc count)")
    ax.set_xlabel("Macro-Cluster ID"); ax.set_ylabel("Topic ID")
    ax.set_xticks(range(len(clusters))); ax.set_xticklabels([str(c) for c in clusters])
    ax.set_yticks(range(len(topics)));   ax.set_yticklabels([str(t) for t in topics])
    cbar = fig.colorbar(im, ax=ax, shrink=0.8); cbar.set_label("Document count")
    plt.tight_layout(); plt.savefig(outpath, dpi=FIG_DPI); plt.close(fig)
    print("[SAVED]", outpath)

plot_topic2cluster_heatmap(topic_sizes, t2c_df, os.path.join(FIG_DIR, "heatmap_topic2macro.png"))

# -----------------------------
# 4) Macro keywords bars
# -----------------------------
def plot_macro_keywords(agg_kw_dict, topn, outdir):
    if not agg_kw_dict: print("[SKIP] macro keywords: empty"); return
    for c in sorted(agg_kw_dict.keys()):
        items = agg_kw_dict[c].most_common(topn)
        if not items: continue
        words, vals = zip(*items)
        fig, ax = plt.subplots(figsize=(9, 5), dpi=FIG_DPI)
        y = np.arange(len(words))
        ax.barh(y, vals)
        ax.set_yticks(y); ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel("Weighted importance (∑ keyword_weight × topic_size)")
        ax.set_title(f"Cluster {c} — Top-{topn} Keywords")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.0f}", va="center", ha="left", fontsize=9)
        plt.tight_layout()
        outpath = os.path.join(outdir, f"bar_keywords_cluster{c}_top{topn}.png")
        plt.savefig(outpath, dpi=FIG_DPI); plt.close(fig)
        print("[SAVED]", outpath)

plot_macro_keywords(agg_kw, TOP_N_KEYWORDS, FIG_DIR)

# -----------------------------
# 5) Size bars (both stages)
# -----------------------------
def plot_size_bar(df_ids_counts, id_col, count_col, title, outpath, figsize=(11,6)):
    if not guard_nonempty(df_ids_counts, title): return
    ms = df_ids_counts.copy()
    labels = [f"{id_col[0].upper()}{int(i)}" for i in ms[id_col]]
    vals = ms[count_col].tolist()
    fig, ax = plt.subplots(figsize=figsize, dpi=FIG_DPI)
    x = np.arange(len(labels))
    ax.bar(x, vals)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Document count"); ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.tight_layout(); plt.savefig(outpath, dpi=FIG_DPI); plt.close(fig)
    print("[SAVED]", outpath)

plot_size_bar(
    df_ids_counts=topic_sizes.rename(columns={"Topic":"id","Size":"count"}),
    id_col="id", count_col="count",
    title="Topic Sizes (Stage 1)",
    outpath=os.path.join(FIG_DIR, "bar_all_topics_stage1.png"),
)

plot_size_bar(
    df_ids_counts=macro_sizes.rename(columns={"Cluster":"id","DocCount":"count"}),
    id_col="id", count_col="count",
    title="Macro-Cluster Sizes (Stage 2)",
    outpath=os.path.join(FIG_DIR, "bar_all_macro_stage2.png"),
)

# -----------------------------
# 6) Joint intertopic-style comparison
# -----------------------------
def compute_topic_centroids(docs_df, topic_col_name):
    texts = ensure_text_for_embeddings(docs_df)
    from sentence_transformers import SentenceTransformer
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    encoder = SentenceTransformer("all-mpnet-base-v2", device=device)
    X = encoder.encode(texts.tolist(), show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    mask = docs_df[topic_col_name] != -1
    topics = docs_df.loc[mask, topic_col_name].astype(int).to_numpy()
    idxs   = np.where(mask.to_numpy())[0]
    uniq   = np.unique(topics)

    centroids = {}
    for t in uniq:
        m = idxs[topics == t]
        if m.size == 0: continue
        c = X[m].mean(axis=0)
        n = np.linalg.norm(c) + 1e-12
        centroids[int(t)] = c / n
    return centroids

def build_macro_centroids_from_topics(topic_centroids: dict, t2c_map_df: pd.DataFrame, topic_sizes_df: pd.DataFrame):
    t2c = dict(zip(t2c_map_df["Topic"].astype(int), t2c_map_df["Cluster"].astype(int)))
    size_map = dict(zip(topic_sizes_df["Topic"].astype(int), topic_sizes_df["Size"].astype(int)))
    bucket = {}
    for t, v in topic_centroids.items():
        c = t2c.get(t, None)
        if c is None: continue
        w = float(size_map.get(t, 0))
        if w <= 0: continue
        if c not in bucket: bucket[c] = []
        bucket[c].append((v, w))
    macro_centroids = {}
    for c, arr in bucket.items():
        vecs = np.vstack([v for v, _ in arr])
        ws   = np.array([w for _, w in arr])[:, None]
        m = (vecs * ws).sum(axis=0) / (ws.sum(axis=0) + 1e-12)
        m = m / (np.linalg.norm(m) + 1e-12)
        macro_centroids[int(c)] = m
    return macro_centroids

def joint_project(vectors_A, vectors_B, random_state=RANDOM_STATE):
    M = np.vstack([vectors_A, vectors_B])
    try:
        from umap import UMAP
        um = UMAP(n_neighbors=15, n_components=2, min_dist=0.05,
                  metric="cosine", random_state=random_state)
        Z = um.fit_transform(M)
    except Exception:
        from sklearn.decomposition import PCA
        Z = PCA(n_components=2, random_state=random_state).fit_transform(M)
    ZA = Z[:len(vectors_A)]
    ZB = Z[len(vectors_A):]
    return ZA, ZB

def plot_joint_intertopic(stage1_centroids, stage2_centroids, topic_sizes_df, macro_sizes_df, outdir):
    if not stage1_centroids or not stage2_centroids:
        print("[SKIP] joint intertopic: missing centroids"); return
    t_ids = sorted(stage1_centroids.keys())
    c_ids = sorted(stage2_centroids.keys())
    A = [stage1_centroids[t] for t in t_ids]
    B = [stage2_centroids[c] for c in c_ids]
    ZA, ZB = joint_project(A, B, random_state=RANDOM_STATE)

    t_size_map = dict(zip(topic_sizes_df["Topic"].astype(int), topic_sizes_df["Size"].astype(int)))
    a_sizes = np.array([t_size_map.get(t, 0) for t in t_ids], dtype=float)
    a_bubble = 40 + 360 * (np.sqrt(a_sizes) / (np.sqrt(a_sizes).max() + 1e-9))

    b_size_map = dict(zip(macro_sizes_df["Cluster"].astype(int), macro_sizes_df["DocCount"].astype(int)))
    b_sizes = np.array([b_size_map.get(c, 0) for c in c_ids], dtype=float)
    b_bubble = 70 + 430 * (np.sqrt(b_sizes) / (np.sqrt(b_sizes).max() + 1e-9))

    x_min = min(ZA[:,0].min(), ZB[:,0].min()); x_max = max(ZA[:,0].max(), ZB[:,0].max())
    y_min = min(ZA[:,1].min(), ZB[:,1].min()); y_max = max(ZA[:,1].max(), ZB[:,1].max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), dpi=FIG_DPI, sharex=True, sharey=True)
    ax = axes[0]
    ax.scatter(ZA[:,0], ZA[:,1], s=a_bubble, alpha=0.55, edgecolor="k", linewidth=0.4, color="#90A4AE")
    ax.axhline(0, color="#B0BEC5", lw=0.7); ax.axvline(0, color="#B0BEC5", lw=0.7)
    ax.set_title("Intertopic-style — Stage 1 Topics"); ax.set_xlabel("D1"); ax.set_ylabel("D2")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    cmap = plt.cm.get_cmap("tab20", max(20, len(c_ids)))
    colors = [cmap(i % cmap.N) for i in range(len(c_ids))]
    ax2 = axes[1]
    for i, c in enumerate(c_ids):
        ax2.scatter(ZB[i,0], ZB[i,1], s=b_bubble[i], alpha=0.75, edgecolor="k", linewidth=0.6, color=colors[i])
        ax2.text(ZB[i,0], ZB[i,1], f"C{c}", ha="center", va="center", fontsize=9, color="k")
    ax2.axhline(0, color="#B0BEC5", lw=0.7); ax2.axvline(0, color="#B0BEC5", lw=0.7)
    ax2.set_title("Intertopic-style — Stage 2 Macro-Centroids"); ax2.set_xlabel("D1"); ax2.set_ylabel("D2")
    ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max)

    plt.tight_layout()
    outpath = os.path.join(outdir, "intertopic_stage1_vs_stage2_joint.png")
    plt.savefig(outpath, dpi=FIG_DPI); plt.close(fig)
    print("[SAVED]", outpath)

try:
    topic_centroids = compute_topic_centroids(docs, topic_col)
    macro_centroids = build_macro_centroids_from_topics(topic_centroids, t2c_df, topic_sizes)
    plot_joint_intertopic(topic_centroids, macro_centroids, topic_sizes, macro_sizes, FIG_DIR)
except Exception as e:
    print(f"[SKIP] joint intertopic due to error: {e}")

print("\nAll figures saved to:", FIG_DIR)


# ============================================================
# Topic/Cluster × Framing pipeline
# ============================================================

# ---------- 0) Mount Drive ----------
from google.colab import drive
drive.mount('/content/drive')

# ---------- 1) PATHS (EDIT HERE) ----------
FRAMING_CSV_PATH   = "/content/drive/MyDrive/framing_profile_merged.csv"
DOC_TOPICS_PATH    = "/content/drive/MyDrive/bertopic_modeling/output_stage1_A/STRATA_n60_c20_mdist005_msize15_minsamp[8]_postassign_40/document_topics.parquet"
TOPIC_INFO_PATH    = "/content/drive/MyDrive/bertopic_modeling/output_stage1_A/STRATA_n60_c20_mdist005_msize15_minsamp[8]_postassign_none/topic_info.csv"
TOPIC_KW_PATH      = "/content/drive/MyDrive/bertopic_modeling/output_stage1_A/STRATA_n60_c20_mdist005_msize15_minsamp[8]_postassign_none/topic_keywords.csv"
TOPIC2CLUSTER_PATH = "/content/drive/MyDrive/bertopic_modeling/output_stage1_A/STRATA_n60_c20_mdist005_msize15_minsamp[8]_postassign_40/topic_clusters_global_agglocal.csv"
OUTPUT_DIR         = "/content/drive/MyDrive/figures_results"

MAKE_HEATMAPS     = False
MAKE_COMPACT_FIGS = True

DO_BOOTSTRAP = False
N_BOOT = 1000
BOOT_RANDOM_SEED = 42

# ---------- 2) Imports ----------
import os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt, textwrap
from pathlib import Path
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ---------- 3) Helpers ----------
def safe_slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-_/]", " ", str(s or ""))
    s = re.sub(r"\s+", "-", s.strip())
    return s[:100] if s else "na"

def key_from_title(row):
    year = None
    if "publish_date" in row and pd.notna(row["publish_date"]):
        try:
            year = int(pd.to_datetime(row["publish_date"], errors="coerce").year)
        except Exception:
            year = None
    title = str(row.get("title","")).strip()
    if not title or year is None: return None
    return f"{year}/{safe_slug(title)}"

def first_present(df, *cands):
    m = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in df.columns: return c
        lc = c.lower()
        if lc in m: return m[lc]
    return None

def annotate_heat(ax, mat, row_labels, col_labels, fmt_effect="{:+.2f}", fmt_share="{:.2f}"):
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = mat[i, j]
            if np.isnan(v): t = ""
            else:
                t = fmt_effect.format(v) if col_labels[j].endswith("effect") else fmt_share.format(v)
            ax.text(j, i, t, ha="center", va="center", fontsize=9)

# ---------- 4) Load BERTopic doc-topic assignments ----------
if DOC_TOPICS_PATH.endswith(".parquet"):
    docs = pd.read_parquet(DOC_TOPICS_PATH)
else:
    docs = pd.read_csv(DOC_TOPICS_PATH)

cand_topic_cols = ["topic_id", "topic", "assigned_topic", "topic_label", "Topic"]
topic_col = next((c for c in cand_topic_cols if c in docs.columns), None)
if topic_col is None:
    raise KeyError(f"No topic column in {DOC_TOPICS_PATH}. Expected one of {cand_topic_cols}.")
if topic_col != "topic_id":
    docs = docs.rename(columns={topic_col: "topic_id"})
if "article_key" not in docs.columns:
    raise KeyError("`article_key` missing in document topics file.")
docs = docs.dropna(subset=["article_key","topic_id"]).copy()
docs["topic_id"] = pd.to_numeric(docs["topic_id"], errors="coerce").fillna(-1).astype(int)

# ---------- 5) Load framing profile ----------
prof = pd.read_csv(FRAMING_CSV_PATH)
prof["article_key"] = prof.apply(key_from_title, axis=1)

COL_N_FRAMING = first_present(prof, "n_framing","framing_count","num_framing")
COL_N_COUNTER = first_present(prof, "n_counter","counter_count","num_counter","n_counter_framing")
COL_N_VALID   = first_present(prof, "n_valid","valid_count","num_valid","n_total_valid")
assert all([COL_N_FRAMING, COL_N_COUNTER, COL_N_VALID]), "Missing framing/counter/valid columns in profile."

prof = prof.dropna(subset=["article_key"]).copy()

# ---------- 6) Topic-level aggregation ----------
merged = docs[["article_key","topic_id"]].drop_duplicates().merge(
    prof[["article_key", COL_N_FRAMING, COL_N_COUNTER, COL_N_VALID]],
    on="article_key", how="inner"
)
merged = merged[merged["topic_id"] != -1].copy()

topic_grp = merged.groupby("topic_id").agg(
    n_docs=("article_key", "nunique"),
    total_sent=(COL_N_VALID, "sum"),
    framing_sent=(COL_N_FRAMING, "sum"),
    counter_sent=(COL_N_COUNTER, "sum")
).reset_index()
topic_grp = topic_grp[topic_grp["total_sent"]>0].copy()
topic_grp["framing_share"]  = topic_grp["framing_sent"]/topic_grp["total_sent"]
topic_grp["counter_share"]  = topic_grp["counter_sent"]/topic_grp["total_sent"]
baseline_topic = topic_grp["framing_sent"].sum()/topic_grp["total_sent"].sum()
topic_grp["framing_effect"] = topic_grp["framing_share"] - baseline_topic

tinfo = pd.read_csv(TOPIC_INFO_PATH)  # columns: Topic, Name
tinfo["Topic"] = pd.to_numeric(tinfo["Topic"], errors="coerce").fillna(-1).astype(int)
lab_map = dict(zip(tinfo["Topic"], tinfo["Name"].astype(str)))
topic_grp["topic_label"] = topic_grp["topic_id"].map(lab_map).fillna(topic_grp["topic_id"].astype(str))
topic_grp["topic_label"] = topic_grp.apply(lambda r: f'{r["topic_id"]} | {str(r["topic_label"])[:45]}', axis=1)
topic_grp = topic_grp.sort_values("framing_share", ascending=False).reset_index(drop=True)
topic_grp.to_csv(os.path.join(OUTPUT_DIR, "topic_framing_with_labels.csv"), index=False)

# ---------- 7) Cluster-level aggregation ----------
t2c = pd.read_csv(TOPIC2CLUSTER_PATH)
t2c["Topic"]   = pd.to_numeric(t2c["Topic"], errors="coerce").astype("Int64")
t2c["Cluster"] = pd.to_numeric(t2c["Cluster"], errors="coerce").astype("Int64")

tg = topic_grp.merge(t2c.rename(columns={"Topic":"topic_id"}), on="topic_id", how="left")

cluster_grp = tg.groupby("Cluster", dropna=True).agg(
    n_topics=("topic_id","nunique"),
    n_docs=("n_docs","sum"),
    total_sent=("total_sent","sum"),
    framing_sent=("framing_sent","sum"),
    counter_sent=("counter_sent","sum")
).reset_index()
cluster_grp = cluster_grp[cluster_grp["total_sent"]>0].copy()
cluster_grp["framing_share"]  = cluster_grp["framing_sent"]/cluster_grp["total_sent"]
cluster_grp["counter_share"]  = cluster_grp["counter_sent"]/cluster_grp["total_sent"]
baseline_cluster = cluster_grp["framing_sent"].sum()/cluster_grp["total_sent"].sum()
cluster_grp["framing_effect"] = cluster_grp["framing_share"] - baseline_cluster

kw_df = pd.read_csv(TOPIC_KW_PATH)
kw_df["Topic"] = pd.to_numeric(kw_df["Topic"], errors="coerce").fillna(-1).astype(int)
size_map = dict(zip(topic_grp["topic_id"], topic_grp["n_docs"]))
kw_df["W"] = kw_df["Weight"] * kw_df["Topic"].map(size_map).fillna(0.0)
kw_df = kw_df.merge(t2c.rename(columns={"Topic":"Topic"}), on="Topic", how="left")
clu_kw = (kw_df.dropna(subset=["Cluster"])
              .groupby("Cluster")["Keyword"]
              .apply(lambda s: ", ".join(pd.Series(s).value_counts().index[:8]))
              .reset_index()
              .rename(columns={"Keyword":"keywords"}))
cluster_grp = cluster_grp.merge(clu_kw, on="Cluster", how="left")
cluster_grp = cluster_grp.sort_values("framing_share", ascending=False).reset_index(drop=True)
cluster_grp.to_csv(os.path.join(OUTPUT_DIR, "cluster_framing_shares.csv"), index=False)
clu_kw.to_csv(os.path.join(OUTPUT_DIR, "cluster_keywords.csv"), index=False)

# ---------- 8) (optional) heatmaps ----------
if MAKE_HEATMAPS:
    row_labels = topic_grp["topic_label"].tolist()
    col_labels = ["framing_share","counter_share","framing_effect"]
    mat = topic_grp[col_labels].to_numpy()
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35*len(row_labels))))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels))); ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels))); ax.set_yticklabels(row_labels)
    annotate_heat(ax, mat, row_labels, col_labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Per-topic shares of framing vs counter-framing (Stage-1)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_topic_by_framing_heatmap.png"), dpi=300)
    plt.close(fig)

# ---------- 9) (optional) bootstrap ----------
# ... (same as before; enable DO_BOOTSTRAP if needed)

# ---------- 10) Publication-ready figures ----------
SAVE_DPI = 300
FIG_DIR  = OUTPUT_DIR
MAKE_TOPIC_FIGS = False   # Set to True if topic-level plots are needed
N_BOOT = 2000             # Bootstrap iterations (Figure B)
BOOT_RANDOM_SEED = 42

# Clean, consistent paper style (no seaborn dependency)
plt.rcParams.update({
    "figure.dpi": 100,
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------- helpers ----------
def _short_label(cid, kws, k=3, width=36):
    import textwrap
    topk = ", ".join((kws or "").split(", ")[:k])
    s = f"{int(cid)} | {topk}"
    return textwrap.shorten(s, width=width, placeholder="…")

def _bootstrap_effect_ci(df, n_boot=2000, seed=42):
    """Bootstrap CI for cluster-level framing_effect.
    Resample at article level within each cluster.
    """
    rng = np.random.default_rng(seed)

    # Attach cluster id to merged
    global merged, t2c, COL_N_VALID, COL_N_FRAMING
    m = merged.merge(
        t2c.rename(columns={"Topic": "topic_id"}),
        on="topic_id", how="left"
    ).dropna(subset=["Cluster"]).copy()

    # Aggregate per (article_key, Cluster)
    art = (m.groupby(["article_key", "Cluster"], as_index=False)
             .agg(total_sent=(COL_N_VALID, "sum"),
                  framing_sent=(COL_N_FRAMING, "sum")))

    # Ensure numeric dtypes
    art["total_sent"]   = pd.to_numeric(art["total_sent"], errors="coerce")
    art["framing_sent"] = pd.to_numeric(art["framing_sent"], errors="coerce")
    art = art.replace([np.inf, -np.inf], np.nan).dropna(subset=["total_sent", "framing_sent"])
    art = art[art["total_sent"] > 0].copy()

    # Global baseline
    base = art["framing_sent"].sum() / art["total_sent"].sum()

    rows = []
    clusters = sorted(art["Cluster"].unique())
    for c in clusters:
        sub = art.loc[art["Cluster"] == c, ["framing_sent", "total_sent"]].to_numpy(dtype=float)
        if len(sub) < 30:
            # Too few articles: return NaN CI to avoid over-interpretation
            rows.append((c, np.nan, np.nan))
            continue

        vals = []
        idx = np.arange(len(sub))
        for _ in range(n_boot):
            ss = sub[rng.choice(idx, size=len(idx), replace=True)]
            f_share = ss[:, 0].sum() / ss[:, 1].sum()  # framing_sent / total_sent
            vals.append(f_share - base)

        lo, hi = np.percentile(vals, [2.5, 97.5])
        rows.append((c, lo, hi))

    return pd.DataFrame(rows, columns=["Cluster", "ci_lo", "ci_hi"])


def _save_caption_stats_cluster(cluster_grp, path_csv):
    """Export key stats for captions/text: baseline, per-cluster n_docs, framing_share, effect, CI (if available)."""
    out = cluster_grp[["Cluster","n_docs","framing_share","counter_share","framing_effect","keywords"]].copy()
    out.to_csv(path_csv, index=False)

# ---------- Figure A: Stacked bars (Top clusters by coverage) ----------
# Take the top 8 macro-clusters by coverage (n_docs), then order by framing_share for display
sel = cluster_grp.sort_values("n_docs", ascending=False).head(8).copy()
sel = sel.sort_values("framing_share", ascending=False)
labels = [_short_label(c, k, k=2) for c,k in zip(sel["Cluster"], sel["keywords"])]

fig = plt.figure(figsize=(3.6, 2.8))
gs  = fig.add_gridspec(1, 1, left=0.14, right=0.82, bottom=0.28, top=0.86)
ax  = fig.add_subplot(gs[0, 0])

w = 0.65
xpos = np.arange(len(sel))
f   = sel["framing_share"].to_numpy()
cnt = sel["counter_share"].to_numpy()
other = np.clip(1.0 - f - cnt, 0.0, 1.0)

b1 = ax.bar(xpos, f,   width=w, label="Framing")
b2 = ax.bar(xpos, cnt, width=w, bottom=f, label="Counter-framing")
b3 = ax.bar(xpos, other, width=w, bottom=f+cnt, label="Other")

# Axes
ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=28, ha="right")
ax.set_ylim(0, 1);   ax.set_ylabel("Share")

# Annotate sample size per cluster above bars
for i, n in enumerate(sel["n_docs"].to_numpy()):
    ax.text(xpos[i], 1.02, f"n={int(n)}", ha="center", va="bottom", fontsize=7)

# Place legend on the right to avoid occlusion
handles, lab = ax.get_legend_handles_labels()
fig.legend(handles, lab, loc="center left", bbox_to_anchor=(0.84, 0.57), frameon=False)

fig.suptitle("Framing vs. counter by macro-topic (top clusters by coverage)")
fig.savefig(os.path.join(FIG_DIR, "figA_cluster_stacked_topcoverage.png"),
            dpi=SAVE_DPI, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "figA_cluster_stacked_topcoverage.pdf"),
            bbox_inches="tight")
plt.close(fig)

# ---------- Figure B: Framing Effect (relative to baseline) + 95% CI ----------
# Compute bootstrap CI
ci_df = _bootstrap_effect_ci(cluster_grp, n_boot=N_BOOT, seed=BOOT_RANDOM_SEED)
eff = cluster_grp.merge(ci_df, on="Cluster", how="left")

# Select top 12 by |effect| to highlight the strongest deviations from baseline
eff["abs_effect"] = eff["framing_effect"].abs()
sel_b = (eff.sort_values(["abs_effect","n_docs"], ascending=[False, False])
            .head(12)
            .sort_values("framing_effect", ascending=True))
ylabels = [_short_label(c, k, k=2) for c,k in zip(sel_b["Cluster"], sel_b["keywords"])]

fig = plt.figure(figsize=(3.6, 3.0))
gs  = fig.add_gridspec(1, 1, left=0.22, right=0.96, bottom=0.14, top=0.88)
ax  = fig.add_subplot(gs[0, 0])

ypos = np.arange(len(sel_b))
ax.axvline(0.0, linestyle="--", linewidth=1, alpha=0.7)  # Baseline

ax.barh(ypos, sel_b["framing_effect"], height=0.6)
# Draw error bars if CI is available
if sel_b[["ci_lo","ci_hi"]].notna().all().all():
    ax.errorbar(x=(sel_b["ci_lo"]+sel_b["ci_hi"])/2.0, y=ypos,
                xerr=(sel_b["ci_hi"]-sel_b["ci_lo"])/2.0,
                linestyle="none", capsize=2, linewidth=1)

ax.set_yticks(ypos); ax.set_yticklabels(ylabels)
ax.set_xlabel("Framing effect (share − overall baseline)")
fig.suptitle("Technosolutionist framing effect by macro-topic (with 95% CI)")
fig.savefig(os.path.join(FIG_DIR, "figB_cluster_effect_ci.png"),
            dpi=SAVE_DPI, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "figB_cluster_effect_ci.pdf"),
            bbox_inches="tight")
plt.close(fig)

# ---------- Figure C: Cluster × metric heatmap ----------
heat_cols = ["framing_share","counter_share","framing_effect"]
mat = cluster_grp[heat_cols].to_numpy()
rlabels = [ _short_label(c, k, k=3, width=40) for c,k in zip(cluster_grp["Cluster"], cluster_grp["keywords"]) ]

fig, ax = plt.subplots(figsize=(4.2, max(3.0, 0.35*len(rlabels))))
im = ax.imshow(mat, aspect="auto")
ax.set_xticks(np.arange(len(heat_cols))); ax.set_xticklabels(heat_cols, rotation=20, ha="right")
ax.set_yticks(np.arange(len(rlabels))); ax.set_yticklabels(rlabels)
# Print numbers inside cells (two decimals; signed for effect)
for i in range(len(rlabels)):
    for j in range(len(heat_cols)):
        ax.text(j, i, f"{mat[i,j]:+.2f}" if heat_cols[j].endswith("effect") else f"{mat[i,j]:.2f}",
                ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Macro-topic × framing metrics (share/effect)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figC_cluster_heatmap.png"), dpi=SAVE_DPI)
fig.savefig(os.path.join(FIG_DIR, "figC_cluster_heatmap.pdf"))
plt.close(fig)

# ---------- (Optional) Topic-level figures ----------
if MAKE_TOPIC_FIGS:
    # Topic-level effect bars: top 20 by |effect|
    topic_sel = (topic_grp.assign(abs_effect=lambda d: d["framing_effect"].abs())
                           .sort_values(["abs_effect","n_docs"], ascending=[False, False])
                           .head(20)
                           .sort_values("framing_effect"))
    fig = plt.figure(figsize=(3.6, 3.6))
    gs  = fig.add_gridspec(1, 1, left=0.24, right=0.98, bottom=0.12, top=0.9)
    ax  = fig.add_subplot(gs[0, 0])
    ypos = np.arange(len(topic_sel))
    ax.axvline(0.0, linestyle="--", linewidth=1, alpha=0.7)
    ax.barh(ypos, topic_sel["framing_effect"], height=0.6)
    ax.set_yticks(ypos); ax.set_yticklabels(topic_sel["topic_label"])
    ax.set_xlabel("Framing effect (share − baseline)")
    fig.suptitle("Technosolutionist framing effect by topic (top |effect|)")
    fig.savefig(os.path.join(FIG_DIR, "figD_topic_effect_top20.png"),
                dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "figD_topic_effect_top20.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    # Topic heatmap (same three metrics)
    tmat = topic_grp[["framing_share","counter_share","framing_effect"]].to_numpy()
    trlabels = topic_grp["topic_label"].tolist()
    fig, ax = plt.subplots(figsize=(4.4, max(3.0, 0.30*len(trlabels))))
    im = ax.imshow(tmat, aspect="auto")
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(["framing_share","counter_share","framing_effect"],
                                                    rotation=20, ha="right")
    ax.set_yticks(np.arange(len(trlabels))); ax.set_yticklabels(trlabels)
    for i in range(len(trlabels)):
        for j in range(3):
            ax.text(j, i, f"{tmat[i,j]:+.2f}" if j==2 else f"{tmat[i,j]:.2f}",
                    ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Topic × framing metrics (share/effect)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figE_topic_heatmap.png"), dpi=SAVE_DPI)
    fig.savefig(os.path.join(FIG_DIR, "figE_topic_heatmap.pdf"))
    plt.close(fig)

# ---------- Export stats for captions/main text ----------
_save_caption_stats_cluster(cluster_grp, os.path.join(FIG_DIR, "table_cluster_caption_stats.csv"))

print("Figures saved to:", FIG_DIR)

