# Colab-ready script for RQ3 visualizations
# Author: you
# Notes:
# - English-only code/comments
# - Matplotlib only; no seaborn; one chart per figure; no explicit colors
# - Safe to run even if some CSVs are missing (will skip gracefully)

# =========================
# 0) Imports & config
# =========================
import os
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# If running in Colab, uncomment to mount Drive:
# from google.colab import drive
# drive.mount('/content/drive')

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# CHANGE THIS to your Stage-1 out_dir (cfg.out_dir), usually ends with /clip_align
BASE_DIR = "/content/drive/MyDrive/bertopic_modeling/output_stage1_A/STRATA_n90_c20_mdist005_msize15_minsamp[8]_postassign_40/clip_align"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

FIG_DIR = os.path.join(BASE_DIR, "figs_rq3")
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

# default thresholds for alignment rate
TAU_MAIN = 0.30
TAU_ALT = [0.20, 0.25]  # optional sensitivity in appendix

RNG = np.random.default_rng(42)

# =========================
# 1) Utils
# =========================
def ensure_df(path):
    if path and os.path.isfile(path):
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext == ".parquet":
            return pd.read_parquet(path)
    return None

def savefig(path, dpi=160):
    Path(Path(path).parent).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")

def parse_year_from_publish_date(x):
    if pd.isna(x):
        return np.nan
    # try YYYY first
    m = re.search(r"(19|20)\d{2}", str(x))
    return int(m.group(0)) if m else np.nan

def parse_year_from_key(k):
    if pd.isna(k):
        return np.nan
    m = re.search(r"(19|20)\d{2}", str(k))
    return int(m.group(0)) if m else np.nan

def bootstrap_rate_ci(binary_values, B=2000, alpha=0.05, rng=RNG):
    arr = np.asarray(binary_values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, (np.nan, np.nan)
    idx = rng.integers(0, n, size=(B, n))
    boots = arr[idx].mean(axis=1)
    boots.sort()
    mean = float(arr.mean())
    lo = float(boots[int((alpha/2)*B)])
    hi = float(boots[int((1-alpha/2)*B)])
    return mean, (lo, hi)

def topk_by_size(series_counts, k):
    order = series_counts.sort_values(ascending=False).index.tolist()
    return order[:k]

# =========================
# 2) Load available CSVs
# =========================
p_topic_article = os.path.join(BASE_DIR, "topic_article_clusters.csv")         # per-article, per-topic, has it_sim_mean
p_macro_article = os.path.join(BASE_DIR, "macro_article_clusters.csv")         # per-article, per-macro, has it_sim_mean
p_topic_tension = os.path.join(BASE_DIR, "topic_tension_stats.csv")            # per-article tension (topic-local)
p_macro_tension = os.path.join(BASE_DIR, "macro_tension_stats.csv")
p_topic_rate_tau30 = os.path.join(BASE_DIR, "topic_alignment_rate_tau30.csv")
p_macro_rate_tau30 = os.path.join(BASE_DIR, "macro_alignment_rate_tau30.csv")
p_yearly_align = os.path.join(BASE_DIR, "yearly_alignment.csv")
p_aligned = os.path.join(BASE_DIR, "aligned_image_text.csv")                   # article_key + image_cluster + topic_id (+ maybe publish_date, title, dek)
# Optional: textual framing effect by topic (provide your own export if available)
p_framing_effect_topic = os.path.join(BASE_DIR, "framing_effect_by_topic.csv") # expected columns: topic_id, framing_effect

df_topic_article = ensure_df(p_topic_article)
df_macro_article = ensure_df(p_macro_article)
df_topic_tension = ensure_df(p_topic_tension)
df_macro_tension = ensure_df(p_macro_tension)
df_topic_rate30 = ensure_df(p_topic_rate_tau30)
df_macro_rate30 = ensure_df(p_macro_rate_tau30)
df_yearly = ensure_df(p_yearly_align)
df_aligned = ensure_df(p_aligned)
df_fe = ensure_df(p_framing_effect_topic)

# =========================
# 3) Figure A: similarity distribution (article-level)
# =========================
def fig_similarity_distribution():
    # try topic_article first; fallback to macro_article; fallback to aligned (if it has per-article it_sim_mean)
    src = None
    if df_topic_article is not None and "it_sim_mean" in df_topic_article.columns:
        src = df_topic_article.copy()
        sim = pd.to_numeric(src["it_sim_mean"], errors="coerce").dropna().values
    elif df_macro_article is not None and "it_sim_mean" in df_macro_article.columns:
        src = df_macro_article.copy()
        sim = pd.to_numeric(src["it_sim_mean"], errors="coerce").dropna().values
    elif df_aligned is not None and "it_sim_mean" in df_aligned.columns:
        sim = pd.to_numeric(df_aligned["it_sim_mean"], errors="coerce").dropna().values
    else:
        print("[Skip] Similarity distribution: no it_sim_mean found.")
        return

    if sim.size == 0:
        print("[Skip] Similarity distribution: empty similarity.")
        return

    plt.figure(figsize=(8,5))
    # histogram
    bins = 40
    plt.hist(sim, bins=bins, density=True, alpha=0.6)
    # KDE overlay
    try:
        kde = gaussian_kde(sim)
        xs = np.linspace(sim.min(), sim.max(), 200)
        plt.plot(xs, kde(xs))
    except Exception:
        pass
    plt.axvline(TAU_MAIN, linestyle="--")
    plt.xlabel("Article-level image–text similarity")
    plt.ylabel("Density")
    plt.title("Distribution of article-level image–text similarity")
    savefig(os.path.join(FIG_DIR, "fig_5_3a_similarity_distribution.png"))

# =========================
# 4) Figure B: yearly NMI/AMI timeseries
# =========================
def fig_yearly_nmi_ami():
    if df_yearly is None or not set(["year","nmi","ami"]).issubset(df_yearly.columns):
        print("[Skip] Yearly NMI/AMI: yearly_alignment.csv missing or invalid.")
        return
    d = df_yearly.copy()
    d = d.sort_values("year")
    plt.figure(figsize=(9,5))
    plt.plot(d["year"].values, d["nmi"].values, label="NMI")
    plt.plot(d["year"].values, d["ami"].values, label="AMI")
    plt.xlabel("Year")
    plt.ylabel("Score")
    plt.title("Yearly image–text alignment (NMI/AMI)")
    plt.legend()
    savefig(os.path.join(FIG_DIR, "fig_5_3b_yearly_nmi_ami.png"))

# =========================
# 5) Figure C: topic-level alignment rate @ tau=0.30 (with CI)
# =========================
def compute_topic_alignment_from_articles(tau=TAU_MAIN):
    if df_topic_article is None or not set(["topic_id","it_sim_mean"]).issubset(df_topic_article.columns):
        return None
    d = df_topic_article.copy()
    d["topic_id"] = pd.to_numeric(d["topic_id"], errors="coerce")
    d["it_sim_mean"] = pd.to_numeric(d["it_sim_mean"], errors="coerce")
    d = d.dropna(subset=["topic_id","it_sim_mean"])
    d["aligned"] = (d["it_sim_mean"] >= float(tau)).astype(float)

    rows = []
    for tid, g in d.groupby("topic_id"):
        mean, (lo, hi) = bootstrap_rate_ci(g["aligned"].values, B=2000, alpha=0.05, rng=RNG)
        rows.append({"topic_id": int(tid), f"rate_tau{int(tau*100)}": mean, "ci_lo": lo, "ci_hi": hi, "n": len(g)})
    out = pd.DataFrame(rows)
    return out.sort_values(f"rate_tau{int(tau*100)}", ascending=False)

def fig_topic_alignment_rate():
    d = compute_topic_alignment_from_articles(TAU_MAIN)
    if d is None or d.empty:
        print("[Skip] Topic alignment rate: topic_article_clusters.csv missing or invalid.")
        return
    topN = min(30, len(d))
    dd = d.head(topN)
    y = np.arange(len(dd))
    plt.figure(figsize=(10, max(6, 0.35*len(dd))))
    plt.barh(y, dd[f"rate_tau{int(TAU_MAIN*100)}"].values)
    # error bars: draw thin lines manually
    for i, (lo, hi) in enumerate(zip(dd["ci_lo"].values, dd["ci_hi"].values)):
        plt.plot([lo, hi], [i, i])
    plt.yticks(y, [f"t{int(t)} (n={int(n)})" for t, n in zip(dd["topic_id"].values, dd["n"].values)])
    plt.xlabel(f"Alignment rate @ tau={TAU_MAIN:.2f}")
    plt.title("Topic-level alignment rate (with 95% CI)")
    savefig(os.path.join(FIG_DIR, "fig_5_3c_topic_alignment_rate_tau30.png"))

# =========================
# 6) Figure D: macro-level alignment rate @ tau=0.30 (with CI)
# =========================
def compute_macro_alignment_from_articles(tau=TAU_MAIN):
    if df_macro_article is None or not set(["macro_id","it_sim_mean"]).issubset(df_macro_article.columns):
        return None
    d = df_macro_article.copy()
    d["macro_id"] = pd.to_numeric(d["macro_id"], errors="coerce")
    d["it_sim_mean"] = pd.to_numeric(d["it_sim_mean"], errors="coerce")
    d = d.dropna(subset=["macro_id","it_sim_mean"])
    d["aligned"] = (d["it_sim_mean"] >= float(tau)).astype(float)

    rows = []
    for mid, g in d.groupby("macro_id"):
        mean, (lo, hi) = bootstrap_rate_ci(g["aligned"].values, B=2000, alpha=0.05, rng=RNG)
        rows.append({"macro_id": int(mid), f"rate_tau{int(tau*100)}": mean, "ci_lo": lo, "ci_hi": hi, "n": len(g)})
    out = pd.DataFrame(rows)
    return out.sort_values(f"rate_tau{int(tau*100)}", ascending=False)

def fig_macro_alignment_rate():
    d = compute_macro_alignment_from_articles(TAU_MAIN)
    if d is None or d.empty:
        print("[Skip] Macro alignment rate: macro_article_clusters.csv missing or invalid.")
        return
    plt.figure(figsize=(10, 6))
    y = np.arange(len(d))
    plt.barh(y, d[f"rate_tau{int(TAU_MAIN*100)}"].values)
    for i, (lo, hi) in enumerate(zip(d["ci_lo"].values, d["ci_hi"].values)):
        plt.plot([lo, hi], [i, i])
    plt.yticks(y, [f"M{int(m)} (n={int(n)})" for m, n in zip(d["macro_id"].values, d["n"].values)])
    plt.xlabel(f"Alignment rate @ tau={TAU_MAIN:.2f}")
    plt.title("Macro-level alignment rate (with 95% CI)")
    savefig(os.path.join(FIG_DIR, "fig_5_3d_macro_alignment_rate_tau30.png"))

# =========================
# 7) Figure E: topic-wise tension distributions (boxplot)
# =========================
def fig_topic_tension_distributions(top_k=20, min_n=20):
    src = None
    if df_topic_tension is not None and set(["topic_id","tension"]).issubset(df_topic_tension.columns):
        src = df_topic_tension.copy()
    elif df_topic_article is not None and set(["topic_id","it_sim_mean"]).issubset(df_topic_article.columns):
        # fallback: approximate tension = 1 - it_sim_mean (not identical to sim_ic_doc tension)
        tmp = df_topic_article.copy()
        tmp["topic_id"] = pd.to_numeric(tmp["topic_id"], errors="coerce")
        tmp["tension"] = 1.0 - pd.to_numeric(tmp["it_sim_mean"], errors="coerce")
        src = tmp[["topic_id","tension"]]
    else:
        print("[Skip] Topic tension distributions: no suitable source.")
        return

    src = src.dropna(subset=["topic_id","tension"])
    sizes = src.groupby("topic_id")["tension"].size()
    keep = [tid for tid, n in sizes.items() if n >= min_n]
    if not keep:
        print("[Skip] Topic tension: not enough data per topic.")
        return
    top = topk_by_size(sizes.loc[keep], k=min(top_k, len(keep)))

    data = [src.loc[src["topic_id"]==tid, "tension"].values for tid in top]
    labels = [f"t{int(tid)} (n={int(sizes[tid])})" for tid in top]

    plt.figure(figsize=(10, max(6, 0.35*len(top))))
    plt.boxplot(data, vert=False, showfliers=False)
    plt.yticks(np.arange(1, len(labels)+1), labels)
    plt.xlabel("Tension score (higher = less aligned)")
    plt.title("Topic-wise tension distributions")
    savefig(os.path.join(FIG_DIR, "fig_5_3e_topic_tension_boxplot.png"))

# =========================
# 8) Figure F: alignment vs textual framing_effect (scatter)
# =========================
def fig_alignment_vs_framing_effect():
    if df_fe is None or "topic_id" not in df_fe.columns or "framing_effect" not in df_fe.columns:
        print("[Skip] Alignment vs framing_effect: framing_effect_by_topic.csv not found.")
        return
    ar = compute_topic_alignment_from_articles(TAU_MAIN)
    if ar is None or ar.empty:
        print("[Skip] Alignment vs framing_effect: topic alignment unavailable.")
        return
    d = ar.merge(df_fe[["topic_id","framing_effect"]], on="topic_id", how="inner")
    if d.empty:
        print("[Skip] Alignment vs framing_effect: no overlap.")
        return

    x = d[f"rate_tau{int(TAU_MAIN*100)}"].values
    y = pd.to_numeric(d["framing_effect"], errors="coerce").values

    plt.figure(figsize=(7,6))
    plt.scatter(x, y)
    plt.xlabel(f"Alignment rate @ tau={TAU_MAIN:.2f}")
    plt.ylabel("Textual framing effect")
    plt.title("Coupling between alignment and textual solutionism")
    # annotate a few extreme points
    idx = np.argsort(y)[-5:]
    for i in idx:
        plt.annotate(f"t{int(d.iloc[i]['topic_id'])}", (x[i], y[i]))
    savefig(os.path.join(FIG_DIR, "fig_5_3f_alignment_vs_framing_effect.png"))

# =========================
# 9) Figure G: quadrant view + export cases (CSV)
# =========================
def fig_quadrant_and_cases():
    if df_fe is None or "topic_id" not in df_fe.columns or "framing_effect" not in df_fe.columns:
        print("[Skip] Quadrant: framing_effect_by_topic.csv not found.")
        return
    ar = compute_topic_alignment_from_articles(TAU_MAIN)
    if ar is None or ar.empty:
        print("[Skip] Quadrant: topic alignment unavailable.")
        return
    d = ar.merge(df_fe[["topic_id","framing_effect"]], on="topic_id", how="inner").dropna()
    if d.empty:
        print("[Skip] Quadrant: no overlap.")
        return

    x = d[f"rate_tau{int(TAU_MAIN*100)}"].values
    y = pd.to_numeric(d["framing_effect"], errors="coerce").values
    x_med = np.nanmedian(x)
    y_med = np.nanmedian(y)

    plt.figure(figsize=(7,6))
    plt.scatter(x, y)
    plt.axvline(x_med, linestyle="--")
    plt.axhline(y_med, linestyle="--")
    plt.xlabel(f"Alignment rate @ tau={TAU_MAIN:.2f}")
    plt.ylabel("Textual framing effect")
    plt.title("Quadrant view: High/Low Alignment × Positive/Negative Effect")
    savefig(os.path.join(FIG_DIR, "fig_5_3g_quadrant.png"))

    # Export topic ids per quadrant
    quad_rows = []
    for _, r in d.iterrows():
        quad = ("H" if r[f"rate_tau{int(TAU_MAIN*100)}"]>=x_med else "L") + ("P" if r["framing_effect"]>=y_med else "N")
        quad_rows.append({"topic_id": int(r["topic_id"]), "alignment": float(r[f"rate_tau{int(TAU_MAIN*100)}"]), "framing_effect": float(r["framing_effect"]), "quadrant": quad})
    qdf = pd.DataFrame(quad_rows).sort_values(["quadrant","alignment"], ascending=[True, False])
    qpath = os.path.join(FIG_DIR, "quadrant_topics.csv")
    qdf.to_csv(qpath, index=False)
    print(f"[Saved] {qpath}")

# =========================
# 10) Figure H: image_cluster × topic_id co-occurrence heatmap
# =========================
def fig_cooccurrence_heatmap(top_topics=20, top_clusters=20, normalize=True):
    if df_aligned is None or not set(["image_cluster","topic_id"]).issubset(df_aligned.columns):
        print("[Skip] Heatmap: aligned_image_text.csv missing or invalid.")
        return
    d = df_aligned[["image_cluster","topic_id"]].copy()
    d["image_cluster"] = pd.to_numeric(d["image_cluster"], errors="coerce")
    d["topic_id"] = pd.to_numeric(d["topic_id"], errors="coerce")
    d = d.dropna().astype({"image_cluster": int, "topic_id": int})

    # focus top topics and clusters by frequency
    t_counts = d["topic_id"].value_counts()
    c_counts = d["image_cluster"].value_counts()
    keep_t = topk_by_size(t_counts, top_topics)
    keep_c = topk_by_size(c_counts, top_clusters)
    sub = d[d["topic_id"].isin(keep_t) & d["image_cluster"].isin(keep_c)]
    if sub.empty:
        print("[Skip] Heatmap: empty after filtering.")
        return
    # pivot
    M = sub.pivot_table(index="topic_id", columns="image_cluster", values="topic_id", aggfunc="count", fill_value=0).astype(float)
    if normalize:
        M = M / (M.values.sum() + 1e-9)

    plt.figure(figsize=(10, 7))
    plt.imshow(M.values, aspect="auto")
    plt.colorbar()
    plt.xticks(np.arange(M.shape[1]), M.columns.tolist(), rotation=90)
    plt.yticks(np.arange(M.shape[0]), M.index.tolist())
    plt.xlabel("Image cluster id")
    plt.ylabel("Topic id")
    plt.title("Co-occurrence heatmap: image clusters × topics")
    savefig(os.path.join(FIG_DIR, "fig_5_3i_cooccurrence_heatmap.png"))

# =========================
# 11) Figure K: yearly alignment by topic (top-k topics)
# =========================
def ensure_year_column(df, key_col="article_key", publish_col="publish_date"):
    y = None
    if publish_col in df.columns:
        y = df[publish_col].apply(parse_year_from_publish_date)
    if y is None or y.isna().all():
        if key_col in df.columns:
            y = df[key_col].apply(parse_year_from_key)
    return y

def fig_yearly_alignment_by_topic(top_k_topics=6, tau=TAU_MAIN, min_year=2003, max_year=2024):
    if df_topic_article is None or not set(["topic_id","it_sim_mean"]).issubset(df_topic_article.columns):
        print("[Skip] Yearly alignment by topic: topic_article_clusters.csv missing or invalid.")
        return
    d = df_topic_article.copy()
    d["year"] = ensure_year_column(d, key_col="article_key", publish_col="publish_date")
    d["topic_id"] = pd.to_numeric(d["topic_id"], errors="coerce")
    d["it_sim_mean"] = pd.to_numeric(d["it_sim_mean"], errors="coerce")
    d = d.dropna(subset=["topic_id","it_sim_mean","year"])
    d = d[(d["year"]>=min_year) & (d["year"]<=max_year)]
    d["aligned"] = (d["it_sim_mean"] >= float(tau)).astype(float)

    # pick top-k topics by total count
    top_topics = topk_by_size(d["topic_id"].value_counts(), top_k_topics)
    d = d[d["topic_id"].isin(top_topics)]

    plt.figure(figsize=(10,6))
    for tid, g in d.groupby("topic_id"):
        yy = sorted(g["year"].unique())
        rates = []
        for y in yy:
            gy = g[g["year"]==y]["aligned"].values
            if len(gy)==0:
                rates.append(np.nan)
            else:
                rates.append(float(np.nanmean(gy)))
        plt.plot(yy, rates, label=f"t{int(tid)}")
    plt.xlabel("Year")
    plt.ylabel(f"Alignment rate @ tau={tau:.2f}")
    plt.title("Yearly alignment by topic (top-k topics)")
    plt.legend()
    savefig(os.path.join(FIG_DIR, "fig_5_3k_yearly_alignment_by_topic.png"))

# =========================
# 12) Appendix table: highest-tension articles (export CSV)
# =========================
def table_high_tension_articles(top_n=50):
    # source: topic_tension_stats.csv preferred; fallback to topic_article_clusters.csv approximating tension
    if df_topic_tension is not None and set(["article_key","topic_id","tension"]).issubset(df_topic_tension.columns):
        t = df_topic_tension.copy()
        t["tension"] = pd.to_numeric(t["tension"], errors="coerce")
        t = t.dropna(subset=["article_key","topic_id","tension"])
    elif df_topic_article is not None and set(["article_key","topic_id","it_sim_mean"]).issubset(df_topic_article.columns):
        t = df_topic_article.copy()
        t["tension"] = 1.0 - pd.to_numeric(t["it_sim_mean"], errors="coerce")
        t = t.dropna(subset=["article_key","topic_id","tension"])
    else:
        print("[Skip] High-tension table: no suitable source.")
        return

    # enrich with title/dek/year if available (from aligned or topic_article)
    enrich = None
    if df_aligned is not None:
        enrich = df_aligned.copy()
    elif df_topic_article is not None:
        enrich = df_topic_article.copy()

    if enrich is not None:
        cols = [c for c in ["article_key","title","dek","publish_date"] if c in enrich.columns]
        if cols:
            t = t.merge(enrich[cols].drop_duplicates("article_key"), on="article_key", how="left")

    t["year"] = ensure_year_column(t, key_col="article_key", publish_col="publish_date")
    t = t.sort_values("tension", ascending=False).head(top_n)
    outp = os.path.join(FIG_DIR, "table_high_tension_articles.csv")
    t.to_csv(outp, index=False)
    print(f"[Saved] {outp}")

# =========================
# 13) Run all
# =========================
fig_similarity_distribution()
fig_yearly_nmi_ami()
fig_topic_alignment_rate()
fig_macro_alignment_rate()
fig_topic_tension_distributions(top_k=20, min_n=20)
fig_alignment_vs_framing_effect()
fig_quadrant_and_cases()
fig_cooccurrence_heatmap(top_topics=20, top_clusters=20, normalize=True)
fig_yearly_alignment_by_topic(top_k_topics=6, tau=TAU_MAIN, min_year=2003, max_year=2024)
table_high_tension_articles(top_n=50)

# Optional: sensitivity for tau=0.20 and 0.25 (appendix)
for tau in [0.20, 0.25]:
    TAU_TMP = tau
    def _tmp_topic_rate():
        d = compute_topic_alignment_from_articles(TAU_TMP)
        if d is None or d.empty:
            return
        y = np.arange(len(d))
        plt.figure(figsize=(10, 6))
        plt.barh(y, d[f"rate_tau{int(TAU_TMP*100)}"].values)
        for i, (lo, hi) in enumerate(zip(d["ci_lo"].values, d["ci_hi"].values)):
            plt.plot([lo, hi], [i, i])
        plt.yticks(y, [f"t{int(t)}" for t in d["topic_id"].values])
        plt.xlabel(f"Alignment rate @ tau={TAU_TMP:.2f}")
        plt.title(f"Topic-level alignment rate (95% CI) @ tau={TAU_TMP:.2f}")
        savefig(os.path.join(FIG_DIR, f"fig_topic_alignment_rate_tau{int(TAU_TMP*100)}.png"))
    _tmp_topic_rate()
print("[Done]")
