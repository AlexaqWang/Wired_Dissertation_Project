# -*- coding: utf-8 -*-
import os
import sys
import json
import hashlib
import joblib
import nltk
import gc
import math
import random
import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
import lightgbm as lgb  # for early_stopping callbacks

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import CalibratedClassifierCV

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =============================
# Colab paths & basic config
# =============================

BASE_DIR = '/content/drive/MyDrive/4framingmarker'

# Input files
LABELED_FILE = os.path.join(BASE_DIR, "labeled_sentences_manually.csv")  # cols: sentence, level1_label, [level2_label], [article_id]
UNLABELED_FILE = os.path.join(BASE_DIR, "article_metadata_summary.csv")  # cols: title, dek, text, [article_id], [category]

# Output base directories
DIR_MODELS = os.path.join(BASE_DIR, "level1_label_results", "models")
DIR_PRED = os.path.join(BASE_DIR, "level1_label_results", "predictions")
DIR_PRED_ROOT = os.path.join(BASE_DIR, "predictions")
for d in [DIR_MODELS, DIR_PRED, DIR_PRED_ROOT]:
    os.makedirs(d, exist_ok=True)

# SBERT
MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 32
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sentence splitting
NLTK_LANG = "english"

# Classifiers (baselines)
CLASSIFIERS = ["logreg_en", "linearsvm", "lgbm"]

# Cross-validation folds
N_FOLDS = 5

# Embedding cache
EMB_CACHE_DIR = os.path.join(BASE_DIR, "embedding_cache")
os.makedirs(EMB_CACHE_DIR, exist_ok=True)
FORCE_REENCODE = False

# Pseudo-labeling constraints (one round)
PSEUDO_POS_TH = 0.80
PSEUDO_NEG_TH = 0.20
PSEUDO_MAX_PER_ARTICLE = 100
PSEUDO_MAX_PER_CLASS_L1 = 100_000
PSEUDO_MAX_PER_CLASS_L2 = 20_000

# L1 建议兼顾：宏平均更稳
L1_DECISION_OBJECTIVE = "macro_f1"

# L2 要抓 framing，建议对正类友好或宏平均
L2_DECISION_OBJECTIVE = "macro_f1"

# L2 binary second-round filter threshold (0 -> disabled)
L2_SECOND_ROUND_THRESHOLD = 0.0

L1_DECISION_THRESHOLD_OVERRIDE = 0.50   # 建议 0.50（或 0.48~0.52 微调）
L2_DECISION_THRESHOLD_OVERRIDE = 0.52   # 建议 0.52（或 0.50~0.53）

warnings.filterwarnings("ignore")
tqdm.pandas()

# Ensure NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download('punkt', quiet=True)
# Some environments require punkt_tab
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

random.seed(SEED)
np.random.seed(SEED)

# =============================
# Utilities
# =============================

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_sbert(model_name: str, device: str) -> SentenceTransformer:
    print(f"[Info] Loading SBERT: {model_name} on {device}")
    return SentenceTransformer(model_name, device=device)

def _hash_items(texts: List[str]) -> str:
    h = hashlib.md5()
    for t in texts:
        h.update(t.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def encode_with_cache(
    sbert: SentenceTransformer,
    texts: List[str],
    tag: str,
    batch_size: int = 32,
    device: str = "cpu",
    force: bool = False
) -> np.ndarray:
    cache_path = os.path.join(EMB_CACHE_DIR, f"{tag}_{_hash_items(texts)}.npy")
    if (not force) and os.path.exists(cache_path):
        emb = np.load(cache_path)
        print(f"[Cache] Loaded embeddings from {cache_path} shape={emb.shape}")
        return emb.astype(np.float32, copy=False)
    emb = sbert.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, device=device)
    emb = emb.astype(np.float32, copy=False)
    np.save(cache_path, emb)
    print(f"[Cache] Saved embeddings to {cache_path} shape={emb.shape}")
    return emb

def ensure_columns(df: pd.DataFrame, cols: List[str], msg: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{msg} missing columns: {missing}")

def normalize_unlabeled_df(df_u: pd.DataFrame) -> pd.DataFrame:
    """Unify column names and keep required fields; ensure 'category' exists."""
    df_u = df_u.copy()
    for wrong in ["cateogry", "category "]:
        if wrong in df_u.columns and "category" not in df_u.columns:
            df_u = df_u.rename(columns={wrong: "category"})
    if "title" not in df_u.columns:
        df_u["title"] = ""
    else:
        df_u["title"] = df_u["title"].fillna("")
    if "dek" not in df_u.columns:
        df_u["dek"] = ""
    else:
        df_u["dek"] = df_u["dek"].fillna("")
    if "text" not in df_u.columns:
        df_u["text"] = ""
    df_u = df_u.dropna(subset=["text"])
    if "category" not in df_u.columns:
        df_u["category"] = None
    return df_u

def sentence_split_article(row: pd.Series) -> List[Dict]:
    """Sentence tokenization across title/dek/text; propagate category."""
    parts = []
    for sec in ["title", "dek", "text"]:
        txt = str(row.get(sec, "")).strip()
        if txt:
            sents = nltk.sent_tokenize(txt, language=NLTK_LANG)
            for sent in sents:
                s = sent.strip()
                if s:
                    parts.append({
                        "article_id": row.get("article_id", row.name),
                        "year": row.get("year", None),
                        "category": row.get("category", None),
                        "section": sec,
                        "sentence": s
                    })
    return parts

def _is_clean_sentence(s: str) -> bool:
    """Simple filter for very short or non-linguistic strings."""
    s = str(s).strip()
    if len(s) < 5:
        return False
    import re
    letters = len(re.findall(r"[A-Za-z0-9,.!?;:]", s))
    return letters / max(1, len(s)) >= 0.2

def subdirs_for(clf_name: str) -> Dict[str, str]:
    model_dir = os.path.join(DIR_MODELS, clf_name)
    pred_dir = os.path.join(DIR_PRED_ROOT, clf_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    return {
        "models": model_dir,
        "pred": pred_dir,
        "l1_sup_model": os.path.join(model_dir, "l1_supervised.pkl"),
        "l1_semi_model": os.path.join(model_dir, "l1_semi_supervised.pkl"),
        "l2_sup_model": os.path.join(model_dir, "l2_l1eq1_binary_supervised.pkl"),
        "l2_semi_model": os.path.join(model_dir, "l2_l1eq1_binary_semi.pkl"),
        "l1_sup_thr_json": os.path.join(model_dir, "l1_sup_threshold.json"),
        "l1_semi_thr_json": os.path.join(model_dir, "l1_semi_threshold.json"),
        "l2_sup_thr_json": os.path.join(model_dir, "l2_sup_threshold.json"),
        "l2_semi_thr_json": os.path.join(model_dir, "l2_semi_threshold.json"),
        "l1_split_json": os.path.join(model_dir, "l1_fixed_group_split.json"),
        "l2_split_json": os.path.join(model_dir, "l2_fixed_group_split.json"),
        "l1_sup_pred_csv": os.path.join(pred_dir, "l1_sentence_predictions_supervised.csv"),
        "l1_semi_pred_csv": os.path.join(pred_dir, "l1_sentence_predictions_semi.csv"),
        "l2_sup_pred_csv": os.path.join(pred_dir, "l2_l1eq1_binary_supervised.csv"),
        "l2_semi_pred_csv": os.path.join(pred_dir, "l2_l1eq1_binary_semi.csv"),
        "final_3class_csv": os.path.join(pred_dir, "final_3class_labels.csv"),
        "plot_l1_sup_hist": os.path.join(pred_dir, "l1_sup_prob1_hist.png"),
        "plot_l1_sup_pr":   os.path.join(pred_dir, "l1_sup_pr_curve.png"),
        "plot_l1_semi_hist":os.path.join(pred_dir, "l1_semi_prob1_hist.png"),
        "plot_l1_semi_pr":  os.path.join(pred_dir, "l1_semi_pr_curve.png"),
        "plot_l2_sup_hist": os.path.join(pred_dir, "l2_l1eq1_prob1_hist_sup.png"),
        "plot_l2_semi_hist":os.path.join(pred_dir, "l2_l1eq1_prob1_hist_semi.png"),
        "summary_json":     os.path.join(pred_dir, "summary_metrics.json"),
        "eval_json":        os.path.join(pred_dir, "evaluation_three_models.json"),
    }

# Classifier factory
def build_classifier(name: str):
    if name == "logreg_en":
        base = LogisticRegression(
            solver="saga", penalty="elasticnet", l1_ratio=0.3,
            C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED
        )
    elif name == "linearsvm":
        base = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)
    elif name == "lgbm":
        base = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=64,
            min_data_in_leaf=40,
            min_gain_to_split=0.0,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1
        )
    else:
        raise ValueError("Unsupported classifier. Use 'logreg_en' | 'linearsvm' | 'lgbm'.")
    return base

def _fit_with_optional_early_stopping(clf, X_tr, y_tr, X_val=None, y_val=None):
    """Fit with early stopping for LGBM when validation is provided; otherwise standard fit."""
    if isinstance(clf, LGBMClassifier) and X_val is not None and y_val is not None:
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
    else:
        clf.fit(X_tr, y_tr)

def _fit_with_inner_val_if_needed(clf, X_tr, y_tr, random_state: int = SEED):
    """For LGBM, split out an inner validation set to avoid leakage; otherwise fit directly."""
    if isinstance(clf, LGBMClassifier):
        X_tr_in, X_val_in, y_tr_in, y_val_in = train_test_split(
            X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=random_state
        )
        _fit_with_optional_early_stopping(clf, X_tr_in, y_tr_in, X_val_in, y_val_in)
    else:
        clf.fit(X_tr, y_tr)
    return clf

def maybe_calibrate(clf, X: np.ndarray, y: List, method: Optional[str] = None):
    """Wrap with CalibratedClassifierCV only when method is specified."""
    if method is None:
        clf.fit(X, y)
        return clf, None
    calibrator = CalibratedClassifierCV(clf, method=method, cv=5)
    calibrator.fit(X, y)
    return calibrator, calibrator

def plot_histogram(probs: np.ndarray, threshold: float, title: str, out_path: str):
    plt.figure(figsize=(8, 5))
    plt.hist(probs, bins=50, edgecolor="black")
    if threshold is not None:
        plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold:.2f}")
        plt.legend()
    plt.title(title)
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_pr_curve(y_true: np.ndarray, prob1: np.ndarray, title: str, out_path: str):
    precision, recall, _ = precision_recall_curve(y_true, prob1)
    ap = average_precision_score(y_true, prob1)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def tune_threshold(y_true: np.ndarray, prob1: np.ndarray, objective: str = "class0_f1") -> Tuple[float, Dict]:
    best_t = 0.5
    best_score = -1.0
    records = []
    for t in np.linspace(0.0, 1.0, 201):
        y_pred = (prob1 >= t).astype(int)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
        f1_0 = float(per_class_f1[0]) if len(per_class_f1) == 2 else 0.0
        f1_1 = float(per_class_f1[1]) if len(per_class_f1) == 2 else 0.0

        if objective == "class0_f1":
            score = f1_0
        elif objective == "class1_f1":
            score = f1_1
        else:  # "macro_f1"
            score = macro_f1

        records.append((float(t), float(macro_f1), float(f1_0)))
        if score > best_score:
            best_score = score
            best_t = t
    return float(best_t), {"grid": records, "best_score": float(best_score)}


def cap_per_article(df: pd.DataFrame, per_article: int, prob_col: str) -> pd.DataFrame:
    if "article_id" not in df.columns or per_article is None:
        return df
    return (df.sort_values(prob_col, ascending=False)
              .groupby("article_id", group_keys=False)
              .head(per_article))

def symmetric_sample(df_pos: pd.DataFrame, df_neg_pool: pd.DataFrame,
                     max_per_class: Optional[int], seed: int=SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = min(len(df_pos), len(df_neg_pool))
    if max_per_class is not None:
        n = min(n, max_per_class)
    if n == 0:
        return df_pos.head(0), df_neg_pool.head(0)
    df_pos_s = df_pos.sample(n=n, random_state=seed) if len(df_pos) > n else df_pos
    df_neg_s = df_neg_pool.sample(n=n, random_state=seed) if len(df_neg_pool) > n else df_neg_pool
    return df_pos_s, df_neg_s

def effective_n_splits(y: List[int], groups: List, desired: int) -> int:
    """Compute robust number of splits ensuring both classes have enough groups."""
    y = np.asarray(y)
    g = np.asarray(groups)
    total_groups = len(np.unique(g))
    pos_groups = len(np.unique(g[y == 1])) if (y == 1).any() else 0
    neg_groups = len(np.unique(g[y == 0])) if (y == 0).any() else 0
    upper = min(desired, total_groups, pos_groups, neg_groups)
    return max(2, upper)

# ---- L2 label mapping ----
def _map_l2_to_binary(series: pd.Series) -> pd.Series:
    """
    Map original level2_label to binary:
      framing(1/2/3/4) -> 1
      counter_framing(-1) -> 0
      irrelevant(0 or invalid) -> NaN (dropped)
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s.isin([1, 2, 3, 4])] = 1
    out[s == -1] = 0
    return out

def has_l2_supervision(df: pd.DataFrame) -> bool:
    """
    Conditions to enable L2 training:
    - level2_label exists
    - within L1==1 subset, both framing(1/2/3/4) and counter(-1) exist
    - ignore level2=0 (irrelevant)
    """
    if "level2_label" not in df.columns:
        return False
    df2 = df.copy()
    df2["level1_label"] = pd.to_numeric(df2["level1_label"], errors="coerce").fillna(0).astype(int)
    sub = df2[df2["level1_label"] == 1]["level2_label"]
    if sub.empty:
        return False
    y_bin = _map_l2_to_binary(sub).dropna()
    if y_bin.empty:
        return False
    n_pos = int((y_bin == 1).sum())
    n_neg = int((y_bin == 0).sum())
    return n_pos > 0 and n_neg > 0

# =============================
# L1: Supervised + Semi-supervised
# =============================

def l1_train_supervised(
    sbert: SentenceTransformer,
    df_labeled: pd.DataFrame,
    classifier_name: str,
    calibration: Optional[str] = None,
    objective: str = L1_DECISION_OBJECTIVE
) -> Tuple[object, dict, str, float, np.ndarray, np.ndarray, set, set]:
    ensure_columns(df_labeled, ["sentence", "level1_label"], "L1 labeled")
    df = df_labeled.copy()
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["level1_label"] = pd.to_numeric(df["level1_label"], errors="coerce").fillna(0).astype(int)
    if "article_id" not in df.columns:
        df["article_id"] = df.index // 5

    c0, c1 = int((df["level1_label"] == 0).sum()), int((df["level1_label"] == 1).sum())
    print(f"[L1][{classifier_name}] Labeled class distribution: 0={c0}, 1={c1}")

    X_text = df["sentence"].tolist()
    y = df["level1_label"].tolist()
    groups = df["article_id"].tolist()

    X_emb = encode_with_cache(sbert, X_text, tag="l1_labeled_sentences", batch_size=BATCH_SIZE, device=DEVICE, force=FORCE_REENCODE)

    n_splits_eff = effective_n_splits(y, groups, N_FOLDS)
    if n_splits_eff < 2:
        raise RuntimeError("Not enough samples per class/group for StratifiedGroupKFold in L1 supervised.")

    sgkf = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=SEED)
    cv_results = {"precision_macro": [], "recall_macro": [], "f1_macro": []}

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X_emb, y, groups)):
        base = build_classifier(classifier_name)
        X_tr, X_te = X_emb[tr_idx], X_emb[te_idx]
        y_tr = np.array(y)[tr_idx]
        y_te = np.array(y)[te_idx]

        if calibration is None:
            # Use inner validation only for LGBM to avoid leakage
            _fit_with_inner_val_if_needed(base, X_tr, y_tr, random_state=SEED + fold)
            proba = base.predict_proba(X_te)[:, 1]
        else:
            model_c = CalibratedClassifierCV(base, method=calibration, cv=5)
            model_c.fit(X_tr, y_tr)
            proba = model_c.predict_proba(X_te)[:, 1]

        y_pred = (proba >= 0.5).astype(int)
        cv_results["precision_macro"].append(precision_score(y_te, y_pred, average="macro", zero_division=0))
        cv_results["recall_macro"].append(recall_score(y_te, y_pred, average="macro", zero_division=0))
        cv_results["f1_macro"].append(f1_score(y_te, y_pred, average="macro", zero_division=0))
        print(f"[L1][{classifier_name}][Fold {fold}] f1_macro={cv_results['f1_macro'][-1]:.4f}")

    print(f"[L1][{classifier_name}] CV Results (StratifiedGroupKFold, macro):",
          {k: float(np.mean(v)) for k, v in cv_results.items()})

    # Fixed group split (80/20)
    uniq_groups = pd.Series(groups).drop_duplicates().sample(frac=1.0, random_state=SEED).tolist()
    cut = int(len(uniq_groups) * 0.8)
    tr_groups = set(uniq_groups[:cut])
    te_groups = set(uniq_groups[cut:])
    tr_mask = pd.Series(groups).isin(tr_groups).values
    te_mask = pd.Series(groups).isin(te_groups).values

    X_tr, y_tr = X_emb[tr_mask], np.array(y)[tr_mask]
    X_te, y_te = X_emb[te_mask], np.array(y)[te_mask]

    base = build_classifier(classifier_name)
    if calibration is None:
        _fit_with_inner_val_if_needed(base, X_tr, y_tr, random_state=SEED)
        model = base
    else:
        model, _ = maybe_calibrate(base, X_tr, y_tr, method=calibration)

    prob_te = model.predict_proba(X_te)[:, 1]
    best_t, info = tune_threshold(y_te, prob_te, objective=objective)
    y_pred = (prob_te >= best_t).astype(int)
    report = classification_report(y_te, y_pred)
    print(f"[L1][{classifier_name}] Tuned threshold (objective={objective}): {best_t:.3f}  best_score={info['best_score']:.4f}")
    print(f"[L1][{classifier_name}] Holdout report (with tuned threshold):\n{report}")

    return model, cv_results, report, float(best_t), X_te, y_te, tr_groups, te_groups

def _apply_clean_and_dedup(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    df = df[df["sentence"].astype(str).map(_is_clean_sentence)]
    if "article_id" in df.columns:
        df = df.drop_duplicates(subset=["article_id", "sentence"])
    else:
        df = df.drop_duplicates(subset=["sentence"])
    if prob_col not in df.columns:
        df[prob_col] = 0.5
    return df

def _adaptive_quantile_cuts(df: pd.DataFrame, prob_col: str,
                            pos_th: float, neg_th: float,
                            max_per_class: Optional[int]) -> Tuple[float, float]:
    if max_per_class is None:
        return pos_th, neg_th
    pos_mask_raw = df[prob_col] >= pos_th
    neg_mask_raw = df[prob_col] <= neg_th
    n_pos_raw = int(pos_mask_raw.sum())
    n_neg_raw = int(neg_mask_raw.sum())

    pos_cut = pos_th
    neg_cut = neg_th
    if n_pos_raw > max_per_class and n_pos_raw > 0:
        keep_ratio = max_per_class / n_pos_raw
        q = 1.0 - keep_ratio
        pos_cut = max(pos_th, float(df.loc[pos_mask_raw, prob_col].quantile(q)))
    if n_neg_raw > max_per_class and n_neg_raw > 0:
        keep_ratio = max_per_class / n_neg_raw
        q = keep_ratio
        neg_cut = min(neg_th, float(df.loc[neg_mask_raw, prob_col].quantile(q)))
    return pos_cut, neg_cut

def l1_predict_and_pseudo_label(
    model,
    sbert: SentenceTransformer,
    df_unlab_sent: pd.DataFrame,
    pos_th: float = PSEUDO_POS_TH,
    neg_th: float = PSEUDO_NEG_TH,
    export_csv: Optional[str] = None,
    hist_path: Optional[str] = None,
    pr_path: Optional[str] = None,
    labeled_ref: Optional[pd.DataFrame] = None,
    decision_threshold: float = 0.5,
    max_per_article: int = PSEUDO_MAX_PER_ARTICLE,
    max_per_class: Optional[int] = PSEUDO_MAX_PER_CLASS_L1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_columns(df_unlab_sent, ["sentence"], "Unlabeled sentence split")
    df = df_unlab_sent.copy()
    X_text = df["sentence"].astype(str).str.strip().tolist()
    X_emb = encode_with_cache(sbert, X_text, tag="l1_unlabeled_sentences", batch_size=BATCH_SIZE, device=DEVICE, force=FORCE_REENCODE)

    proba = model.predict_proba(X_emb)
    df["l1_prob0"] = proba[:, 0]
    df["l1_prob1"] = proba[:, 1]
    df["l1_label"] = (df["l1_prob1"] >= decision_threshold).astype(int)

    if export_csv:
        df.to_csv(export_csv, index=False, encoding="utf-8-sig")
        print(f"[L1] Saved sentence-level predictions to {export_csv}")

    if hist_path:
        plot_histogram(df["l1_prob1"].values, pos_th, "L1 prob_1 distribution", hist_path)

    if pr_path and labeled_ref is not None and "level1_label" in labeled_ref.columns:
        X_ref = labeled_ref["sentence"].astype(str).str.strip().tolist()
        X_ref_emb = encode_with_cache(sbert, X_ref, tag="l1_ref_labeled_sentences", batch_size=BATCH_SIZE, device=DEVICE, force=FORCE_REENCODE)
        ref_prob1 = model.predict_proba(X_ref_emb)[:, 1]
        y_true = labeled_ref["level1_label"].astype(int).values
        plot_pr_curve(y_true, ref_prob1, "L1 PR curve (on labeled set projection)", pr_path)

    df = _apply_clean_and_dedup(df, "l1_prob1")

    pos_cut, neg_cut = _adaptive_quantile_cuts(df, "l1_prob1", pos_th, neg_th, max_per_class)
    pos_mask = df["l1_prob1"].values >= pos_cut
    neg_mask = df["l1_prob1"].values <= neg_cut

    df_pos = df[pos_mask].copy()
    df_neg_pool = df[neg_mask].copy()

    df_pos = cap_per_article(df_pos, max_per_article, prob_col="l1_prob1")
    df_neg_pool = cap_per_article(df_neg_pool, max_per_article, prob_col="l1_prob1")

    df_pos_s, df_neg_s = symmetric_sample(df_pos, df_neg_pool, max_per_class=max_per_class, seed=SEED)

    df_pos_s["level1_label"] = 1
    df_neg_s["level1_label"] = 0

    df_pseudo = pd.concat([df_pos_s, df_neg_s], ignore_index=True)
    print(f"[L1] Pseudo positives={len(df_pos_s)}, pseudo negatives kept={len(df_neg_s)}, total={len(df_pseudo)}")

    df_pseudo_train = df_pseudo[["sentence", "level1_label", "article_id"]].copy()
    if "article_id" not in df_pseudo_train.columns:
        df_pseudo_train["article_id"] = -1

    return df, df_pseudo_train

def l1_train_semi_supervised(
    sbert: SentenceTransformer,
    df_labeled: pd.DataFrame,
    df_pseudo_full: pd.DataFrame,
    classifier_name: str,
    calibration: Optional[str] = None,
    objective: str = L1_DECISION_OBJECTIVE,
    labeled_holdout_X: np.ndarray = None,
    labeled_holdout_y: np.ndarray = None,
    allowed_train_groups: Optional[set] = None
) -> Tuple[object, dict, str, float, Dict]:
    ensure_columns(df_labeled, ["sentence", "level1_label"], "L1 labeled")
    ensure_columns(df_pseudo_full, ["sentence", "level1_label", "article_id"], "L1 pseudo")

    df = df_labeled.copy()
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["level1_label"] = pd.to_numeric(df["level1_label"], errors="coerce").fillna(0).astype(int)
    if "article_id" not in df.columns:
        df["article_id"] = df.index // 5

    if allowed_train_groups is not None:
        df = df[df["article_id"].isin(allowed_train_groups)].copy()

    df_pseudo = df_pseudo_full.copy()
    df_pseudo["sentence"] = df_pseudo["sentence"].astype(str).str.strip()
    df_pseudo["level1_label"] = pd.to_numeric(df_pseudo["level1_label"], errors="coerce").fillna(0).astype(int)

    df_combined = pd.concat(
        [df[["sentence", "level1_label", "article_id"]],
         df_pseudo[["sentence", "level1_label", "article_id"]]],
        ignore_index=True
    )

    X_texts = df_combined["sentence"].tolist()
    y = df_combined["level1_label"].tolist()
    groups = df_combined["article_id"].tolist()

    X_emb = encode_with_cache(sbert, X_texts, tag="l1_semi_combined_sentences", batch_size=BATCH_SIZE, device=DEVICE, force=FORCE_REENCODE)

    n_splits_eff = effective_n_splits(y, groups, N_FOLDS)
    if n_splits_eff < 2:
        raise RuntimeError("Not enough samples per class/group for StratifiedGroupKFold in L1 semi.")

    sgkf = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=SEED)
    cv_results = {"precision_macro": [], "recall_macro": [], "f1_macro": []}

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X_emb, y, groups)):
        base = build_classifier(classifier_name)
        X_tr, X_te = X_emb[tr_idx], X_emb[te_idx]
        y_tr = np.array(y)[tr_idx]
        y_te = np.array(y)[te_idx]

        if calibration is None:
            _fit_with_inner_val_if_needed(base, X_tr, y_tr, random_state=SEED + fold)
            proba = base.predict_proba(X_te)[:, 1]
        else:
            model_c = CalibratedClassifierCV(base, method=calibration, cv=5)
            model_c.fit(X_tr, y_tr)
            proba = model_c.predict_proba(X_te)[:, 1]

        y_pred = (proba >= 0.5).astype(int)
        cv_results["precision_macro"].append(precision_score(y_te, y_pred, average="macro", zero_division=0))
        cv_results["recall_macro"].append(recall_score(y_te, y_pred, average="macro", zero_division=0))
        cv_results["f1_macro"].append(f1_score(y_te, y_pred, average="macro", zero_division=0))
        print(f"[L1-semi][{classifier_name}][Fold {fold}] f1_macro={cv_results['f1_macro'][-1]:.4f}")

    print(f"[L1-semi][{classifier_name}] CV Results (StratifiedGroupKFold, macro):",
          {k: float(np.mean(v)) for k, v in cv_results.items()})

    base = build_classifier(classifier_name)
    if calibration is None:
        model = _fit_with_inner_val_if_needed(base, X_emb, np.array(y), random_state=SEED)
    else:
        model, _ = maybe_calibrate(base, X_emb, np.array(y), method=calibration)

    prob_holdout = model.predict_proba(labeled_holdout_X)[:, 1]
    best_t, info = tune_threshold(labeled_holdout_y, prob_holdout, objective=objective)
    y_pred = (prob_holdout >= best_t).astype(int)
    report = classification_report(labeled_holdout_y, y_pred)
    print(f"[L1-semi][{classifier_name}] Tuned threshold on TRUE holdout (objective={objective}): {best_t:.3f}  best_score={info['best_score']:.4f}")
    print(f"[L1-semi][{classifier_name}] Holdout report on TRUE labeled set:\n{report}")
    print("[Note] Do not compare scores on pseudo-labeled training set with true holdout.")

    metrics = {}
    rep_lines = report.strip().splitlines()
    for line in rep_lines:
        if line.strip().lower().startswith("macro avg"):
            parts = [p for p in line.split(" ") if p]
            if len(parts) >= 4:
                try:
                    metrics["macro_precision"] = float(parts[1])
                    metrics["macro_recall"] = float(parts[2])
                    metrics["macro_f1"] = float(parts[3])
                except:
                    pass
            break

    return model, cv_results, report, float(best_t), metrics

# =============================
# L2: Binary (on L1==1)
# =============================

def l2_train_supervised(
    sbert: SentenceTransformer,
    df_labeled: pd.DataFrame,
    classifier_name: str,
    calibration: Optional[str] = None,
) -> Tuple[object, dict, str, float, np.ndarray, np.ndarray, set, set]:
    ensure_columns(df_labeled, ["sentence", "level1_label", "level2_label"], "L2 labeled")
    df = df_labeled.copy()

    df = df[pd.to_numeric(df["level1_label"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if len(df) == 0:
        raise RuntimeError("No level1==1 samples for L2 supervised training.")

    df["sentence"] = df["sentence"].astype(str).str.strip()

    y_bin = _map_l2_to_binary(df["level2_label"])
    keep = ~y_bin.isna()
    df = df.loc[keep].copy()
    df["is_true_framing"] = y_bin.loc[keep].astype(int)

    if "article_id" not in df.columns:
        df["article_id"] = df.index // 5

    c0, c1 = int((df["is_true_framing"] == 0).sum()), int((df["is_true_framing"] == 1).sum())
    print(f"[L2-binary][{classifier_name}] Labeled class distribution: counter(0)={c0}, framing(1)={c1}")

    X_texts = df["sentence"].tolist()
    y = df["is_true_framing"].tolist()
    groups = df["article_id"].tolist()

    X_emb = encode_with_cache(sbert, X_texts, tag="l2_labeled_sentences_l1eq1_binary", batch_size=BATCH_SIZE, device=DEVICE, force=FORCE_REENCODE)

    n_splits_eff = effective_n_splits(y, groups, N_FOLDS)
    if n_splits_eff < 2:
        raise RuntimeError("Not enough samples per class/group for StratifiedGroupKFold in L2 supervised.")

    sgkf = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=SEED)
    cv_results = {"precision_macro": [], "recall_macro": [], "f1_macro": []}

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X_emb, y, groups)):
        base = build_classifier(classifier_name)
        X_tr, X_te = X_emb[tr_idx], X_emb[te_idx]
        y_tr = np.array(y)[tr_idx]
        y_te = np.array(y)[te_idx]

        if calibration is None:
            _fit_with_inner_val_if_needed(base, X_tr, y_tr, random_state=SEED + fold)
            proba = base.predict_proba(X_te)[:, 1]
        else:
            model_c = CalibratedClassifierCV(base, method=calibration, cv=5)
            model_c.fit(X_tr, y_tr)
            proba = model_c.predict_proba(X_te)[:, 1]

        y_pred = (proba >= 0.5).astype(int)
        cv_results["precision_macro"].append(precision_score(y_te, y_pred, average="macro", zero_division=0))
        cv_results["recall_macro"].append(recall_score(y_te, y_pred, average="macro", zero_division=0))
        cv_results["f1_macro"].append(f1_score(y_te, y_pred, average="macro", zero_division=0))
        print(f"[L2-binary][{classifier_name}][Fold {fold}] f1_macro={cv_results['f1_macro'][-1]:.4f}")

    print(f"[L2-binary][{classifier_name}] CV Results (StratifiedGroupKFold, macro):",
          {k: float(np.mean(v)) for k, v in cv_results.items()})

    # Fixed L2 groups split
    uniq_groups2 = pd.Series(groups).drop_duplicates().sample(frac=1.0, random_state=SEED).tolist()
    cut2 = int(len(uniq_groups2) * 0.8)
    tr_groups2 = set(uniq_groups2[:cut2])
    te_groups2 = set(uniq_groups2[cut2:])
    tr_mask2 = pd.Series(groups).isin(tr_groups2).values
    te_mask2 = pd.Series(groups).isin(te_groups2).values

    X_tr, y_tr = X_emb[tr_mask2], np.array(y)[tr_mask2]
    X_te, y_te = X_emb[te_mask2], np.array(y)[te_mask2]

    base = build_classifier(classifier_name)
    if calibration is None:
        _fit_with_inner_val_if_needed(base, X_tr, y_tr, random_state=SEED)
        model = base
    else:
        model, _ = maybe_calibrate(base, X_tr, y_tr, method=calibration)

    prob_te = model.predict_proba(X_te)[:, 1]
    l2_sup_th, info = tune_threshold(y_te, prob_te, objective=L2_DECISION_OBJECTIVE)

    y_pred = (prob_te >= l2_sup_th).astype(int)
    report = classification_report(y_te, y_pred)
    print(f"[L2-binary][{classifier_name}] Tuned threshold: {l2_sup_th:.3f} best_score={info['best_score']:.4f}")
    print(f"[L2-binary][{classifier_name}] Holdout report:\n{report}")

    return model, cv_results, report, float(l2_sup_th), X_te, y_te, tr_groups2, te_groups2

def l2_build_pseudo_from_l1eq1(
    model_l2,
    sbert: SentenceTransformer,
    df_l1eq1_unlab_sent: pd.DataFrame,
    pos_th: float = PSEUDO_POS_TH,
    neg_th: Optional[float] = None,
    max_per_article: int = PSEUDO_MAX_PER_ARTICLE,
    max_per_class: Optional[int] = PSEUDO_MAX_PER_CLASS_L2,
    hist_path: Optional[str] = None
) -> pd.DataFrame:
    ensure_columns(df_l1eq1_unlab_sent, ["sentence", "article_id"], "L2 unlabeled sentences (L1==1)")
    df = df_l1eq1_unlab_sent.copy()

    X_texts = df["sentence"].astype(str).str.strip().tolist()
    X_emb = encode_with_cache(sbert, X_texts, tag="l2_unlabeled_l1eq1_sentences", batch_size=BATCH_SIZE, device=DEVICE, force=FORCE_REENCODE)

    prob1 = model_l2.predict_proba(X_emb)[:, 1]
    df["l2_prob1"] = prob1
    df["l2_pred_bin"] = (prob1 >= 0.5).astype(int)

    if hist_path:
        plot_histogram(df["l2_prob1"].values, pos_th, "L2 prob_1 distribution (from supervised)", hist_path)

    if neg_th is None:
        neg_th = 1.0 - pos_th

    df = _apply_clean_and_dedup(df, "l2_prob1")

    pos_cut, neg_cut = _adaptive_quantile_cuts(df, "l2_prob1", pos_th, neg_th, max_per_class)
    pos_mask = df["l2_prob1"] >= pos_cut
    neg_mask = df["l2_prob1"] <= neg_cut

    df_pos = df[pos_mask].copy()
    df_neg_pool = df[neg_mask].copy()

    df_pos = cap_per_article(df_pos, max_per_article, prob_col="l2_prob1")
    df_neg_pool = cap_per_article(df_neg_pool, max_per_article, prob_col="l2_prob1")

    df_pos_s, df_neg_s = symmetric_sample(df_pos, df_neg_pool, max_per_class=max_per_class, seed=SEED)

    df_pos_out = pd.DataFrame({
        "text": df_pos_s["sentence"].astype(str).tolist(),
        "is_true_framing": np.ones(len(df_pos_s), dtype=int).tolist(),
        "prob1": df_pos_s["l2_prob1"].tolist(),
        "article_id": df_pos_s["article_id"].tolist(),
    })
    df_neg_out = pd.DataFrame({
        "text": df_neg_s["sentence"].astype(str).tolist(),
        "is_true_framing": np.zeros(len(df_neg_s), dtype=int).tolist(),
        "prob1": df_neg_s["l2_prob1"].tolist(),
        "article_id": df_neg_s["article_id"].tolist(),
    })

    pseudo = pd.concat([df_pos_out, df_neg_out], ignore_index=True)
    print(f"[L2-binary] Pseudo positives={len(df_pos_out)}, pseudo negatives kept={len(df_neg_out)}, total={len(pseudo)}")

    return pseudo

def l2_train_semi(
    sbert: SentenceTransformer,
    df_labeled: pd.DataFrame,
    df_pseudo: pd.DataFrame,
    classifier_name: str,
    calibration: Optional[str] = None,
    labeled_holdout_X: np.ndarray = None,
    labeled_holdout_y: np.ndarray = None,
    allowed_train_groups: Optional[set] = None,
) -> Tuple[object, dict, str, Dict, float]:
    ensure_columns(df_labeled, ["sentence", "level1_label", "level2_label"], "L2 labeled")
    ensure_columns(df_pseudo, ["text", "is_true_framing", "article_id"], "L2 pseudo")

    df_lab = df_labeled[pd.to_numeric(df_labeled["level1_label"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if "article_id" not in df_lab.columns:
        df_lab["article_id"] = df_lab.index // 5

    if allowed_train_groups is not None:
        df_lab = df_lab[df_lab["article_id"].isin(allowed_train_groups)].copy()

    df_lab["sentence"] = df_lab["sentence"].astype(str).str.strip()

    y_bin = _map_l2_to_binary(df_lab["level2_label"])
    keep = ~y_bin.isna()
    df_lab = df_lab.loc[keep].copy()
    if len(df_lab) == 0:
        raise RuntimeError("No valid (L1==1 & L2 in {-1,1,2,3,4}) samples for L2 semi training.")
    df_lab["is_true_framing"] = y_bin.loc[keep].astype(int)

    df_combined = pd.concat([
        pd.DataFrame({"text": df_lab["sentence"], "is_true_framing": df_lab["is_true_framing"], "article_id": df_lab["article_id"]}),
        df_pseudo[["text", "is_true_framing", "article_id"]]
    ], ignore_index=True)

    X_texts = df_combined["text"].astype(str).str.strip().tolist()
    y = df_combined["is_true_framing"].astype(int).tolist()
    groups = df_combined["article_id"].tolist()

    X_emb = encode_with_cache(sbert, X_texts, tag="l2_semi_combined_sentences_binary", batch_size=BATCH_SIZE, device=DEVICE, force=FORCE_REENCODE)

    n_splits_eff = effective_n_splits(y, groups, N_FOLDS)
    if n_splits_eff < 2:
        raise RuntimeError("Not enough samples per class/group for StratifiedGroupKFold in L2 semi.")

    sgkf = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=SEED)
    cv_results = {"precision_macro": [], "recall_macro": [], "f1_macro": []}

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X_emb, y, groups)):
        base = build_classifier(classifier_name)
        X_tr, X_te = X_emb[tr_idx], X_emb[te_idx]
        y_tr = np.array(y)[tr_idx]
        y_te = np.array(y)[te_idx]

        if calibration is None:
            _fit_with_inner_val_if_needed(base, X_tr, y_tr, random_state=SEED + fold)
            proba = base.predict_proba(X_te)[:, 1]
        else:
            model_c = CalibratedClassifierCV(base, method=calibration, cv=5)
            model_c.fit(X_tr, y_tr)
            proba = model_c.predict_proba(X_te)[:, 1]

        y_pred = (proba >= 0.5).astype(int)
        cv_results["precision_macro"].append(precision_score(y_te, y_pred, average="macro", zero_division=0))
        cv_results["recall_macro"].append(recall_score(y_te, y_pred, average="macro", zero_division=0))
        cv_results["f1_macro"].append(f1_score(y_te, y_pred, average="macro", zero_division=0))
        print(f"[L2-semi-binary][{classifier_name}][Fold {fold}] f1_macro={cv_results['f1_macro'][-1]:.4f}")

    print(f"[L2-semi-binary][{classifier_name}] CV Results (StratifiedGroupKFold, macro):",
          {k: float(np.mean(v)) for k, v in cv_results.items()})

    base = build_classifier(classifier_name)
    if calibration is None:
        model = _fit_with_inner_val_if_needed(base, X_emb, np.array(y), random_state=SEED)
    else:
        model, _ = maybe_calibrate(base, X_emb, np.array(y), method=calibration)

    prob_holdout = model.predict_proba(labeled_holdout_X)[:, 1]
    l2_semi_th, info = tune_threshold(labeled_holdout_y, prob_holdout, objective=L2_DECISION_OBJECTIVE)

    y_pred = (prob_holdout >= l2_semi_th).astype(int)
    report = classification_report(labeled_holdout_y, y_pred)
    print(f"[L2-semi-binary][{classifier_name}] Tuned threshold on TRUE holdout: {l2_semi_th:.3f} best_score={info['best_score']:.4f}")
    print(f"[L2-semi-binary][{classifier_name}] Holdout report on TRUE labeled set:\n{report}")
    print("[Note] Do not compare pseudo-labeled training scores with true validation.")

    metrics = {}
    rep_lines = report.strip().splitlines()
    for line in rep_lines:
        if line.strip().lower().startswith("macro avg"):
            parts = [p for p in line.split(" ") if p]
            if len(parts) >= 4:
                try:
                    metrics["macro_precision"] = float(parts[1])
                    metrics["macro_recall"] = float(parts[2])
                    metrics["macro_f1"] = float(parts[3])
                except:
                    pass
            break

    return model, cv_results, report, metrics, float(l2_semi_th)

# =============================
# IO helpers & pipeline
# =============================

def load_labeled_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["sentence", "level1_label"])
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["level1_label"] = pd.to_numeric(df["level1_label"], errors="coerce").fillna(0).astype(int)
    if "level2_label" in df.columns:
        df["level2_label"] = pd.to_numeric(df["level2_label"], errors="coerce").fillna(0).astype(int)
    if "cateogry" in df.columns and "category" not in df.columns:
        df = df.rename(columns={"cateogry": "category"})
    return df

def split_unlabeled_sentences(unlabeled_path: str) -> pd.DataFrame:
    df_u = pd.read_csv(unlabeled_path)
    df_u = normalize_unlabeled_df(df_u)
    rows = []
    for _, row in tqdm(df_u.iterrows(), total=len(df_u), desc="Splitting unlabeled into sentences"):
        rows.extend(sentence_split_article(row))
    df_sent = pd.DataFrame(rows)
    return df_sent

# =============================
# Evaluation helper (standalone)
# =============================

def evaluate_three_models(
    sbert: SentenceTransformer,
    df_labeled: pd.DataFrame,
    classifier_names: List[str] = None
) -> Dict:
    if classifier_names is None:
        classifier_names = CLASSIFIERS

    results = {}

    df = df_labeled.copy()
    if "article_id" not in df.columns:
        df["article_id"] = df.index // 5
    groups_all = df["article_id"].tolist()

    X_text_all = df["sentence"].astype(str).str.strip().tolist()
    y_l1_all = pd.to_numeric(df["level1_label"], errors="coerce").fillna(0).astype(int).values
    X_emb_all = encode_with_cache(sbert, X_text_all, tag="eval_l1_labeled_sentences", batch_size=BATCH_SIZE, device=DEVICE, force=FORCE_REENCODE)

    l2_valid = has_l2_supervision(df)
    if l2_valid:
        df_l2_all = df[df["level1_label"] == 1].copy()
        if "article_id" not in df_l2_all.columns:
            df_l2_all["article_id"] = df_l2_all.index // 5

        y_l2_series = _map_l2_to_binary(df_l2_all["level2_label"])
        keep_l2 = ~y_l2_series.isna()
        df_l2_all = df_l2_all.loc[keep_l2].copy()
        y_l2_all = y_l2_series.loc[keep_l2].astype(int).values

        groups_l2_all = df_l2_all["article_id"].tolist()
        X_text_l2_all = df_l2_all["sentence"].astype(str).str.strip().tolist()
        X_emb_l2_all = encode_with_cache(sbert, X_text_l2_all, tag="eval_l2_labeled_sentences", batch_size=BATCH_SIZE, device=DEVICE, force=FORCE_REENCODE)

    for clf_name in classifier_names:
        paths = subdirs_for(clf_name)
        entry = {"l1": {}, "l2": {}}

        # Load fixed group split for L1 if exists
        if os.path.exists(paths["l1_split_json"]):
            splits = json.load(open(paths["l1_split_json"], "r"))
            valid_groups = set(splits["valid_groups"])
            te_mask_l1 = pd.Series(groups_all).isin(valid_groups).values
        else:
            uniq_groups = pd.Series(groups_all).drop_duplicates().sample(frac=1.0, random_state=SEED).tolist()
            cut = int(len(uniq_groups) * 0.8)
            te_groups = set(uniq_groups[cut:])
            te_mask_l1 = pd.Series(groups_all).isin(te_groups).values

        X_te_l1, y_te_l1 = X_emb_all[te_mask_l1], y_l1_all[te_mask_l1]

        # Load L1 model and threshold
        if os.path.exists(paths["l1_semi_model"]):
            l1_model = joblib.load(paths["l1_semi_model"])
            th_path = paths["l1_semi_thr_json"]
        else:
            l1_model = joblib.load(paths["l1_sup_model"])
            th_path = paths["l1_sup_thr_json"]
        with open(th_path, "r") as f:
            l1_th = float(json.load(f)["threshold"])

        prob = l1_model.predict_proba(X_te_l1)[:, 1]
        y_pred = (prob >= l1_th).astype(int)
        rep = classification_report(y_te_l1, y_pred, output_dict=True, zero_division=0)
        entry["l1"]["threshold"] = l1_th
        entry["l1"]["macro_precision"] = rep["macro avg"]["precision"]
        entry["l1"]["macro_recall"] = rep["macro avg"]["recall"]
        entry["l1"]["macro_f1"] = rep["macro avg"]["f1-score"]

        # L2 section
        if l2_valid:
            if os.path.exists(paths["l2_split_json"]):
                splits2 = json.load(open(paths["l2_split_json"], "r"))
                valid_groups2 = set(splits2["valid_groups"])
                te_mask_l2 = pd.Series(groups_l2_all).isin(valid_groups2).values
            else:
                uniq_groups2 = pd.Series(groups_l2_all).drop_duplicates().sample(frac=1.0, random_state=SEED).tolist()
                cut2 = int(len(uniq_groups2) * 0.8)
                te_groups2 = set(uniq_groups2[cut2:])
                te_mask_l2 = pd.Series(groups_l2_all).isin(te_groups2).values

            X_te_l2, y_te_l2 = X_emb_l2_all[te_mask_l2], y_l2_all[te_mask_l2]

            use_semi = os.path.exists(paths["l2_semi_model"]) and os.path.exists(paths["l2_semi_thr_json"])
            if use_semi:
                l2_model = joblib.load(paths["l2_semi_model"])
                with open(paths["l2_semi_thr_json"], "r") as f:
                    l2_th = float(json.load(f)["threshold"])
            else:
                l2_model = joblib.load(paths["l2_sup_model"])
                with open(paths["l2_sup_thr_json"], "r") as f:
                    l2_th = float(json.load(f)["threshold"])

            prob2 = l2_model.predict_proba(X_te_l2)[:, 1]
            y_pred2 = (prob2 >= l2_th).astype(int)
            rep2 = classification_report(y_te_l2, y_pred2, output_dict=True, zero_division=0)
            entry["l2"]["threshold"] = l2_th
            entry["l2"]["macro_precision"] = rep2["macro avg"]["precision"]
            entry["l2"]["macro_recall"] = rep2["macro avg"]["recall"]
            entry["l2"]["macro_f1"] = rep2["macro avg"]["f1-score"]
        else:
            entry["l2"]["skipped"] = True

        results[clf_name] = entry

        with open(paths["eval_json"], "w") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
        print(f"[Eval] Saved {clf_name} evaluation to {paths['eval_json']}")

    overall_eval_path = os.path.join(DIR_PRED_ROOT, "overall_evaluation_three_models.json")
    with open(overall_eval_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Eval] Overall evaluation saved to {overall_eval_path}")

    return results

# =============================
# Main
# =============================

def _safe_int_list(it):
    out = []
    for x in it:
        try:
            out.append(int(x))
        except:
            out.append(-1)
    return out

def main():
    set_global_seed(SEED)
    print(f"[Info] Using device: {DEVICE}")

    # Probability calibration policy:
    # Only calibrate LinearSVC with sigmoid (Platt); no external calibration for logreg or LGBM.
    def calib_for(clf_name: str) -> Optional[str]:
        if clf_name == "linearsvm":
            return "sigmoid"
        return None

    # Load SBERT
    sbert = get_sbert(MODEL_NAME, DEVICE)

    # Load labeled & unlabeled
    df_labeled = load_labeled_df(LABELED_FILE)
    print(f"[Data] Labeled rows: {len(df_labeled)}  (with L2 rows: {(df_labeled['level1_label']==1).sum() if 'level1_label' in df_labeled.columns else 0})")

    print("[Step] Splitting unlabeled articles into sentences ...")
    df_unlab_sent = split_unlabeled_sentences(UNLABELED_FILE)
    print(f"[Data] Unlabeled sentences (raw): {len(df_unlab_sent)}")

    # Leakage guard
    if "article_id" in df_labeled.columns and "article_id" in df_unlab_sent.columns:
        labeled_article_ids = pd.to_numeric(df_labeled["article_id"], errors="coerce").dropna().astype(int).unique().tolist()
        before = len(df_unlab_sent)
        df_unlab_sent = df_unlab_sent[~df_unlab_sent["article_id"].isin(labeled_article_ids)].copy()
        print(f"[Leakage-Guard] Removed {before - len(df_unlab_sent)} unlabeled sentences from overlapping articles.")
    labeled_sents = set(df_labeled["sentence"].astype(str).str.strip().tolist())
    before2 = len(df_unlab_sent)
    df_unlab_sent = df_unlab_sent[~df_unlab_sent["sentence"].astype(str).str.strip().isin(labeled_sents)].copy()
    print(f"[Leakage-Guard] Removed {before2 - len(df_unlab_sent)} unlabeled sentences duplicated in labeled set.")

    print(f"[Data] Unlabeled sentences (after guard): {len(df_unlab_sent)}")

    overall_summary = {}

    for clf_name in CLASSIFIERS:
        print("\n" + "="*70)
        print(f"==== PIPELINE for classifier: {clf_name} ====")
        print("="*70)

        paths = subdirs_for(clf_name)
        calibration = calib_for(clf_name)

        # ===== L1 Supervised =====
        print(f"\n===== L1: Supervised training [{clf_name}] =====")
        l1_sup_model, l1_cv, l1_sup_report, l1_sup_th, l1_hold_X, l1_hold_y, l1_tr_groups, l1_te_groups = l1_train_supervised(
            sbert=sbert,
            df_labeled=df_labeled,
            classifier_name=clf_name,
            calibration=calibration,
            objective=L1_DECISION_OBJECTIVE,
        )
        joblib.dump(l1_sup_model, paths["l1_sup_model"])
        with open(paths["l1_sup_thr_json"], "w") as f:
            json.dump({"threshold": l1_sup_th}, f)
        with open(paths["l1_split_json"], "w") as f:
            json.dump({"train_groups": _safe_int_list(l1_tr_groups),
                       "valid_groups": _safe_int_list(l1_te_groups)}, f)
        print(f"[Save] L1 supervised model: {paths['l1_sup_model']}")
        print(f"[Save] L1 supervised tuned threshold: {l1_sup_th:.3f} -> {paths['l1_sup_thr_json']}")

        # Predict on unlabeled
        print(f"[L1] Predicting (supervised) on unlabeled sentences [{clf_name}] ...")
        l1_sup_pred_df, df_l1_pseudo = l1_predict_and_pseudo_label(
            model=l1_sup_model,
            sbert=sbert,
            df_unlab_sent=df_unlab_sent,
            pos_th=PSEUDO_POS_TH,
            neg_th=PSEUDO_NEG_TH,
            export_csv=paths["l1_sup_pred_csv"],
            hist_path=paths["plot_l1_sup_hist"],
            pr_path=paths["plot_l1_sup_pr"],
            labeled_ref=df_labeled[["sentence", "level1_label"]].dropna(),
            decision_threshold=l1_sup_th,
            max_per_article=PSEUDO_MAX_PER_ARTICLE,
            max_per_class=PSEUDO_MAX_PER_CLASS_L1
        )
        print(f"[L1-supervised] Saved predictions: {paths['l1_sup_pred_csv']}")

        # ===== L1 Semi-supervised =====
        print(f"\n===== L1: Semi-supervised training (one round, strict) [{clf_name}] =====")
        l1_semi_model, l1_semi_cv, l1_semi_report, l1_semi_th, l1_semi_metrics = l1_train_semi_supervised(
            sbert=sbert,
            df_labeled=df_labeled,
            df_pseudo_full=df_l1_pseudo,
            classifier_name=clf_name,
            calibration=calibration,
            objective=L1_DECISION_OBJECTIVE,
            labeled_holdout_X=l1_hold_X,
            labeled_holdout_y=l1_hold_y,
            allowed_train_groups=set(l1_tr_groups)
        )
        joblib.dump(l1_semi_model, paths["l1_semi_model"])
        with open(paths["l1_semi_thr_json"], "w") as f:
            json.dump({"threshold": l1_semi_th}, f)
        print(f"[Save] L1 semi-supervised model: {paths['l1_semi_model']}")
        print(f"[Save] L1 semi-supervised tuned threshold: {l1_semi_th:.3f} -> {paths['l1_semi_thr_json']}")

        print(f"[L1] Predicting (semi-supervised) on unlabeled sentences [{clf_name}] ...")
        l1_semi_pred_df, _ = l1_predict_and_pseudo_label(
            model=l1_semi_model,
            sbert=sbert,
            df_unlab_sent=df_unlab_sent,
            pos_th=PSEUDO_POS_TH,
            neg_th=PSEUDO_NEG_TH,
            export_csv=paths["l1_semi_pred_csv"],
            hist_path=paths["plot_l1_semi_hist"],
            pr_path=paths["plot_l1_semi_pr"],
            labeled_ref=df_labeled[["sentence", "level1_label"]].dropna(),
            decision_threshold=l1_semi_th,
            max_per_article=PSEUDO_MAX_PER_ARTICLE,
            max_per_class=PSEUDO_MAX_PER_CLASS_L1
        )
        print(f"[L1-semi] Saved predictions: {paths['l1_semi_pred_csv']}")

        # ===== L2 =====
        l2_summary = {}
        if has_l2_supervision(df_labeled):
            print(f"\n===== L2: Supervised training (L1==1) [{clf_name}] =====")
            l2_sup_model, l2_cv, l2_sup_report, l2_sup_th, l2_hold_X, l2_hold_y, l2_tr_groups, l2_te_groups = l2_train_supervised(
                sbert=sbert,
                df_labeled=df_labeled,
                classifier_name=clf_name,
                calibration=calibration,
            )
            joblib.dump(l2_sup_model, paths["l2_sup_model"])
            with open(paths["l2_sup_thr_json"], "w") as f:
                json.dump({"threshold": l2_sup_th}, f)
            with open(paths["l2_split_json"], "w") as f:
                json.dump({"train_groups": _safe_int_list(l2_tr_groups),
                           "valid_groups": _safe_int_list(l2_te_groups)}, f)
            print(f"[Save] L2 supervised model: {paths['l2_sup_model']}")
            print(f"[Save] L2 supervised tuned threshold: {l2_sup_th:.3f} -> {paths['l2_sup_thr_json']}")

            print(f"[L2] Selecting unlabeled sentences with L1(semi)==1 [{clf_name}] ...")
            df_l1eq1 = l1_semi_pred_df[l1_semi_pred_df["l1_label"] == 1].copy()
            if len(df_l1eq1) == 0:
                print("[L2] No unlabeled sentences with L1==1. Skipping L2.")
                final_df = l1_semi_pred_df.copy()
                final_df["final_label_3class"] = np.where(final_df["l1_label"] == 1, 1, 0)
                final_df.to_csv(paths["final_3class_csv"], index=False, encoding="utf-8-sig")
                print(f"[Final] Saved 3-class mapping to {paths['final_3class_csv']}")
            else:
                X_texts_ref = df_l1eq1["sentence"].astype(str).str.strip().tolist()
                X_emb_ref = encode_with_cache(
                    sbert, X_texts_ref, tag="l2_ref_unlab_l1eq1_sentences",
                    batch_size=BATCH_SIZE, device=DEVICE, force=FORCE_REENCODE
                )
                proba_ref = l2_sup_model.predict_proba(X_emb_ref)[:, 1]
                l2_pred_bin_sup = (proba_ref >= l2_sup_th).astype(int)

                df_l2_sup_out = df_l1eq1.copy()
                df_l2_sup_out["l2_prob1_sup"] = proba_ref
                df_l2_sup_out["l2_pred_sup"] = l2_pred_bin_sup
                if L2_SECOND_ROUND_THRESHOLD > 0.0:
                    df_l2_sup_out = df_l2_sup_out[df_l2_sup_out["l2_prob1_sup"] >= L2_SECOND_ROUND_THRESHOLD].copy()
                plot_histogram(df_l2_sup_out["l2_prob1_sup"].values, L2_SECOND_ROUND_THRESHOLD,
                               "L2 prob_1 distribution (supervised)", paths["plot_l2_sup_hist"])
                df_l2_sup_out.to_csv(paths["l2_sup_pred_csv"], index=False, encoding="utf-8-sig")
                print(f"[L2] Saved supervised L2 predictions: {paths['l2_sup_pred_csv']}")

                print(f"[L2] Building pseudo labels (binary) from supervised L2 [{clf_name}] ...")
                l2_pseudo_hist_path = os.path.join(paths["pred"], "l2_pseudo_prob1_hist.png")
                l2_pseudo = l2_build_pseudo_from_l1eq1(
                    model_l2=l2_sup_model,
                    sbert=sbert,
                    df_l1eq1_unlab_sent=df_l1eq1,
                    pos_th=PSEUDO_POS_TH,
                    neg_th=PSEUDO_NEG_TH,
                    max_per_article=PSEUDO_MAX_PER_ARTICLE,
                    max_per_class=PSEUDO_MAX_PER_CLASS_L2,
                    hist_path=l2_pseudo_hist_path
                )

                print(f"\n===== L2: Semi-supervised training (binary, one round) [{clf_name}] =====")
                l2_semi_model, l2_semi_cv, l2_semi_report, l2_semi_metrics, l2_semi_th = l2_train_semi(
                    sbert=sbert,
                    df_labeled=df_labeled,
                    df_pseudo=l2_pseudo,
                    classifier_name=clf_name,
                    calibration=calibration,
                    labeled_holdout_X=l2_hold_X,
                    labeled_holdout_y=l2_hold_y,
                    allowed_train_groups=set(l2_tr_groups)
                )
                joblib.dump(l2_semi_model, paths["l2_semi_model"])
                with open(paths["l2_semi_thr_json"], "w") as f:
                    json.dump({"threshold": l2_semi_th}, f)
                print(f"[Save] L2 semi-supervised model: {paths['l2_semi_model']}")
                print(f"[Save] L2 semi-supervised tuned threshold: {l2_semi_th:.3f} -> {paths['l2_semi_thr_json']}")

                proba_semi = l2_semi_model.predict_proba(X_emb_ref)[:, 1]
                l2_pred_bin_semi = (proba_semi >= l2_semi_th).astype(int)
                df_l2_semi_out = df_l1eq1.copy()
                df_l2_semi_out["l2_prob1_semi"] = proba_semi
                df_l2_semi_out["l2_pred_semi"] = l2_pred_bin_semi
                if L2_SECOND_ROUND_THRESHOLD > 0.0:
                    df_l2_semi_out = df_l2_semi_out[df_l2_semi_out["l2_prob1_semi"] >= L2_SECOND_ROUND_THRESHOLD].copy()
                plot_histogram(df_l2_semi_out["l2_prob1_semi"].values, L2_SECOND_ROUND_THRESHOLD,
                               "L2 prob_1 distribution (semi)", paths["plot_l2_semi_hist"])
                df_l2_semi_out.to_csv(paths["l2_semi_pred_csv"], index=False, encoding="utf-8-sig")
                print(f"[L2] Saved semi-supervised L2 predictions: {paths['l2_semi_pred_csv']}")

                final_df = l1_semi_pred_df.copy()
                final_df["final_label_3class"] = 0
                merge_keys = ["sentence"]
                if "article_id" in final_df.columns and "article_id" in df_l2_semi_out.columns:
                    merge_keys = ["article_id", "sentence"]
                final_df = final_df.merge(
                    df_l2_semi_out[merge_keys + ["l2_pred_semi"]],
                    on=merge_keys,
                    how="left"
                )
                idx_l1_1 = final_df["l1_label"] == 1
                final_df.loc[idx_l1_1 & (final_df["l2_pred_semi"] == 1), "final_label_3class"] = 1  # framing
                final_df.loc[idx_l1_1 & (final_df["l2_pred_semi"] == 0), "final_label_3class"] = 2  # counter
                final_df.to_csv(paths["final_3class_csv"], index=False, encoding="utf-8-sig")
                print(f"[Final] Saved 3-class mapping to {paths['final_3class_csv']}")

                l2_summary = {
                    "sup_report": l2_sup_report,
                    "sup_threshold": l2_sup_th,
                    "semi_report": l2_semi_report,
                    "semi_threshold": l2_semi_th,
                    "semi_metrics_true_holdout": l2_semi_metrics
                }
        else:
            print("[Info] No valid L2 supervision (need both framing and counter under L1==1). Skipping L2 for this classifier.")

        # Summary and persist
        summary = {
            "classifier": clf_name,
            "l1": {
                "supervised": {"threshold": l1_sup_th, "report": l1_sup_report},
                "semi_supervised": {
                    "threshold": l1_semi_th,
                    "report_true_holdout": l1_semi_report,
                    "metrics_true_holdout": l1_semi_metrics,
                    "note": "Do not compare pseudo-labeled training scores with true validation."
                }
            },
            "l2": l2_summary
        }
        with open(paths["summary_json"], "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        overall_summary[clf_name] = summary

    with open(os.path.join(DIR_PRED_ROOT, "overall_summary.json"), "w") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print("\n[Done] Pipeline finished successfully for all classifiers.")
    print(f"[Summary saved] {os.path.join(DIR_PRED_ROOT, 'overall_summary.json')}")

# =============================
# Entrypoint
# =============================

if __name__ == "__main__":
    """
    Usage:
    - Train & infer (pipeline): python script.py
    - Standalone evaluation:   python script.py --eval
    """
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        set_global_seed(SEED)
        sbert = get_sbert(MODEL_NAME, DEVICE)
        df_labeled = load_labeled_df(LABELED_FILE)
        _ = evaluate_three_models(sbert, df_labeled, CLASSIFIERS)
    else:
        main()
