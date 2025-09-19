# -*- coding: utf-8 -*-
"""
Unified evaluation visualizations (L1/L2 + final 3-class):
- PR curve (precision–recall curve)
- Probability histogram
- Reliability diagram (calibration / reliability diagram)
- Thresholded confusion matrix
- Bootstrap 95% CI (AP and F1)
- Final 3-class (none=0, framing=1, counter=2) confusion matrix + report

Output directory strictly follows:
  /content/drive/MyDrive/4framingmarker/predictions/<clf>/...

Dependencies:
  pip install pandas numpy matplotlib scikit-learn sentence-transformers joblib

Last updated: 2025-09
"""

import os, json, random, warnings
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score
)
from sklearn.calibration import calibration_curve
import joblib
import torch
from sentence_transformers import SentenceTransformer

# -----------------------
# Global parameters
# -----------------------
BASE_DIR = "/content/drive/MyDrive/4framingmarker"
LABELED_FILE = os.path.join(BASE_DIR, "labeled_sentences_manually.csv")
EMB_CACHE_DIR = os.path.join(BASE_DIR, "embedding_cache")
PRED_ROOT = os.path.join(BASE_DIR, "predictions")
MODELS_ROOT = os.path.join(BASE_DIR, "level1_label_results", "models")

# The same three classifiers as the training script
CLASSIFIERS = ["logreg_en", "linearsvm", "lgbm"]

MODEL_NAME = "all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_BOOT = 1000
CI = (2.5, 97.5)

os.makedirs(EMB_CACHE_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
warnings.filterwarnings("ignore")


# -----------------------
# Utility functions
# -----------------------
def load_labeled_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["sentence", "level1_label"])
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["level1_label"] = pd.to_numeric(df["level1_label"], errors="coerce").fillna(0).astype(int)
    if "level2_label" in df.columns:
        df["level2_label"] = pd.to_numeric(df["level2_label"], errors="coerce").fillna(0).astype(int)
    if "article_id" not in df.columns:
        # Group-wise evaluation needs a group; if missing, construct a weak proxy
        df["article_id"] = np.arange(len(df)) // 5
    return df


def sbert_encode_cached(sbert: SentenceTransformer, texts, tag: str) -> np.ndarray:
    import hashlib
    def _h(it):
        h = hashlib.md5()
        for t in it:
            h.update(str(t).encode("utf-8", errors="ignore"))
        return h.hexdigest()
    cache = os.path.join(EMB_CACHE_DIR, f"{tag}_{_h(texts)}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    emb = sbert.encode(list(texts), batch_size=32, convert_to_numpy=True,
                       show_progress_bar=True, device=DEVICE)
    emb = emb.astype(np.float32, copy=False)
    np.save(cache, emb)
    return emb


def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def bootstrap_ci_metric(y_true, scores, threshold: Optional[float] = None,
                        metric: str = "AP", n_boot: int = N_BOOT, seed: int = SEED) -> Dict:
    """
    metric: "AP" for average precision; "F1" for thresholded macro F1 (positive=1)
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    vals = []
    for _ in range(n_boot):
        m = rng.choice(idx, size=len(idx), replace=True)
        yt = np.asarray(y_true)[m]
        sc = np.asarray(scores)[m]
        if metric.upper() == "AP":
            v = average_precision_score(yt, sc)
        else:
            pred = (sc >= (0.5 if threshold is None else threshold)).astype(int)
            v = f1_score(yt, pred, average="macro", zero_division=0)
        vals.append(v)
    lo, hi = np.percentile(vals, CI)
    return {"mean": float(np.mean(vals)), "lo": float(lo), "hi": float(hi)}


def plot_pr(y_true, prob1, title, out_path):
    p, r, _ = precision_recall_curve(y_true, prob1)
    ap = average_precision_score(y_true, prob1)
    plt.figure(figsize=(6, 5))
    plt.plot(r, p, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def plot_hist(prob1, thr, title, out_path):
    plt.figure(figsize=(6, 4))
    plt.hist(prob1, bins=50, edgecolor="black")
    if thr is not None:
        plt.axvline(thr, linestyle="--", label=f"thr={thr:.2f}")
        plt.legend()
    plt.xlabel("Probability (positive class)"); plt.ylabel("Count"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def plot_reliability(y_true, prob1, title, out_path, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, prob1, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(5.5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def plot_cm(cm, labels, title, out_path):
    plt.figure(figsize=(5.2, 4.6))
    im = plt.imshow(cm, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > 0.6 * vmax else "black", fontsize=9)
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def map_l2_to_binary(series: pd.Series) -> pd.Series:
    # framing 1/2/3/4 -> 1; counter -1 -> 0; else NaN
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s.isin([1, 2, 3, 4])] = 1
    out[s == -1] = 0
    return out


# -----------------------
# Main procedure: regenerate all visualizations by classifier
# -----------------------
def main():
    # 1) Read labeled set & SBERT
    df = load_labeled_df(LABELED_FILE)
    print(f"[Info] Using device: {DEVICE}")
    sbert = SentenceTransformer(MODEL_NAME, device=DEVICE)

    X_all = sbert_encode_cached(sbert, df["sentence"], tag="reval_all_labeled")

    # 2) Evaluate per classifier
    for clf in CLASSIFIERS:
        print(f"\n=== Re-eval visuals for: {clf} ===")
        model_dir = os.path.join(MODELS_ROOT, clf)
        pred_dir = ensure_dir(os.path.join(PRED_ROOT, clf))

        # 2.1 Load L1 model and threshold (prefer semi)
        l1_model_path = os.path.join(model_dir, "l1_semi_supervised.pkl")
        l1_thr_path = os.path.join(model_dir, "l1_semi_threshold.json")
        l1_split_path = os.path.join(model_dir, "l1_fixed_group_split.json")
        if not os.path.exists(l1_model_path):
            l1_model_path = os.path.join(model_dir, "l1_supervised.pkl")
            l1_thr_path = os.path.join(model_dir, "l1_sup_threshold.json")
        if not (os.path.exists(l1_model_path) and os.path.exists(l1_thr_path)):
            print(f"[Skip {clf}] L1 model or threshold missing"); continue

        l1_model = joblib.load(l1_model_path)
        l1_thr = float(json.load(open(l1_thr_path))["threshold"])

        # 2.2 L1 evaluation split (reuse fixed groups if possible)
        groups = df["article_id"].tolist()
        if os.path.exists(l1_split_path):
            valid_groups = set(json.load(open(l1_split_path))["valid_groups"])
            te_mask = pd.Series(groups).isin(valid_groups).values
        else:
            uniq = pd.Series(groups).drop_duplicates().sample(frac=1.0, random_state=SEED).tolist()
            cut = int(len(uniq) * 0.8); te_groups = set(uniq[cut:])
            te_mask = pd.Series(groups).isin(te_groups).values

        y_l1 = df["level1_label"].values
        X_te_l1 = X_all[te_mask]; y_te_l1 = y_l1[te_mask]
        prob1_l1 = l1_model.predict_proba(X_te_l1)[:, 1]
        y_pred_l1 = (prob1_l1 >= l1_thr).astype(int)

        # 2.3 L1 visualizations
        plot_pr(y_te_l1, prob1_l1, "L1 Precision–Recall (holdout)",
                os.path.join(pred_dir, "reval_l1_pr_curve.png"))
        plot_hist(prob1_l1, l1_thr, "L1 Probability Histogram (holdout)",
                os.path.join(pred_dir, "reval_l1_prob_hist.png"))
        plot_reliability(y_te_l1, prob1_l1, "L1 Reliability Diagram (holdout)",
                os.path.join(pred_dir, "reval_l1_reliability.png"))
        cm_l1 = confusion_matrix(y_te_l1, y_pred_l1, labels=[0, 1])
        plot_cm(cm_l1, ["none(0)", "framing_or_counter(1)"],
                "L1 Confusion Matrix (thresholded)",
                os.path.join(pred_dir, "reval_l1_confusion.png"))

        # 2.4 L1 CI (AP and F1)
        ci_ap_l1 = bootstrap_ci_metric(y_te_l1, prob1_l1, metric="AP")
        ci_f1_l1 = bootstrap_ci_metric(y_te_l1, prob1_l1, threshold=l1_thr, metric="F1")
        save_json({"AP": ci_ap_l1, "F1_macro": ci_f1_l1},
                  os.path.join(pred_dir, "reval_l1_bootstrap_ci.json"))

        # 2.5 If L2 supervision exists: load L2 model/threshold and evaluate
        has_l2 = ("level2_label" in df.columns) and (df["level1_label"] == 1).any()
        if has_l2:
            # Use only supervised samples where L1==1
            df_l2 = df[df["level1_label"] == 1].copy()
            y_l2_true = map_l2_to_binary(df_l2["level2_label"])
            keep = ~y_l2_true.isna()
            df_l2 = df_l2.loc[keep].copy()

            if len(df_l2) >= 10 and y_l2_true.loc[keep].nunique() == 2:
                # align embeddings
                X_l2 = sbert_encode_cached(sbert, df_l2["sentence"], tag="reval_l2_labeled")
                groups_l2 = df_l2["article_id"].tolist()
                y_l2 = y_l2_true.loc[keep].astype(int).values

                # Load L2 model (prefer semi)
                l2_model_path = os.path.join(model_dir, "l2_l1eq1_binary_semi.pkl")
                l2_thr_path = os.path.join(model_dir, "l2_semi_threshold.json")
                l2_split_path = os.path.join(model_dir, "l2_fixed_group_split.json")
                if not os.path.exists(l2_model_path):
                    l2_model_path = os.path.join(model_dir, "l2_l1eq1_binary_supervised.pkl")
                    l2_thr_path = os.path.join(model_dir, "l2_sup_threshold.json")
                if not (os.path.exists(l2_model_path) and os.path.exists(l2_thr_path)):
                    print(f"[Warn {clf}] L2 model or threshold missing, skip L2 visualizations")
                else:
                    l2_model = joblib.load(l2_model_path)
                    l2_thr = float(json.load(open(l2_thr_path))["threshold"])
                    # L2 holdout split
                    if os.path.exists(l2_split_path):
                        valid2 = set(json.load(open(l2_split_path))["valid_groups"])
                        te_mask2 = pd.Series(groups_l2).isin(valid2).values
                    else:
                        uniq2 = pd.Series(groups_l2).drop_duplicates().sample(frac=1.0, random_state=SEED).tolist()
                        cut2 = int(len(uniq2) * 0.8); te2 = set(uniq2[cut2:])
                        te_mask2 = pd.Series(groups_l2).isin(te2).values

                    X_te_l2 = X_l2[te_mask2]; y_te_l2 = y_l2[te_mask2]
                    prob1_l2 = l2_model.predict_proba(X_te_l2)[:, 1]
                    y_pred_l2 = (prob1_l2 >= l2_thr).astype(int)

                    # L2 visualizations
                    plot_pr(y_te_l2, prob1_l2, "L2 Precision–Recall (holdout, within L1==1)",
                            os.path.join(pred_dir, "reval_l2_pr_curve.png"))
                    plot_hist(prob1_l2, l2_thr, "L2 Probability Histogram (holdout)",
                              os.path.join(pred_dir, "reval_l2_prob_hist.png"))
                    plot_reliability(y_te_l2, prob1_l2, "L2 Reliability Diagram (holdout)",
                                     os.path.join(pred_dir, "reval_l2_reliability.png"))
                    cm_l2 = confusion_matrix(y_te_l2, y_pred_l2, labels=[0, 1])
                    plot_cm(cm_l2, ["counter(0)", "framing(1)"],
                            "L2 Confusion Matrix (thresholded)",
                            os.path.join(pred_dir, "reval_l2_confusion.png"))

                    # L2 CI
                    ci_ap_l2 = bootstrap_ci_metric(y_te_l2, prob1_l2, metric="AP")
                    ci_f1_l2 = bootstrap_ci_metric(y_te_l2, prob1_l2, threshold=l2_thr, metric="F1")
                    save_json({"AP": ci_ap_l2, "F1_macro": ci_f1_l2},
                              os.path.join(pred_dir, "reval_l2_bootstrap_ci.json"))

                    # 2.6 Final 3-class evaluation (on available labeled samples)
                    # Rules: if L1_pred=0 => none(0);
                    #        if L1_pred=1 and L2_pred=1 => framing(1);
                    #        if L1_pred=1 and L2_pred=0 => counter(2)
                    df_3 = df_l2.copy()  # here L1==1 and L2 is valid
                    # L1 probabilities on the full set; align to the subset
                    idx3 = df.index.get_indexer(df_3.index)
                    prob1_l1_all = l1_model.predict_proba(X_all)[:, 1]
                    prob1_l1_3 = prob1_l1_all[idx3]
                    ypred_l1_3 = (prob1_l1_3 >= l1_thr).astype(int)

                    # For fairness, recompute L2 probabilities on df_3
                    X_l2_all3 = sbert_encode_cached(sbert, df_3["sentence"], tag="reval_l2_for_3class_allsubset")
                    prob1_l2_3 = l2_model.predict_proba(X_l2_all3)[:, 1]
                    ypred_l2_3 = (prob1_l2_3 >= l2_thr).astype(int)

                    y_pred_3 = np.zeros(len(df_3), dtype=int)
                    y_pred_3[ypred_l1_3 == 0] = 0
                    y_pred_3[(ypred_l1_3 == 1) & (ypred_l2_3 == 1)] = 1
                    y_pred_3[(ypred_l1_3 == 1) & (ypred_l2_3 == 0)] = 2

                    # Ground-truth 3-class: derived from L1 and L2 labels
                    y_true_3 = np.zeros(len(df_3), dtype=int)
                    y_true_3[df_3["level1_label"].values == 0] = 0
                    y_true_3[df_3["level1_label"].values == 1] = np.where(
                        map_l2_to_binary(df_3["level2_label"]).astype(int).values == 1, 1, 2
                    )

                    labels3 = ["none(0)", "framing(1)", "counter(2)"]
                    cm3 = confusion_matrix(y_true_3, y_pred_3, labels=[0, 1, 2])
                    plot_cm(cm3, labels3, "Final 3-class Confusion Matrix (thresholded cascade)",
                            os.path.join(pred_dir, "reval_3class_confusion.png"))
                    rep3 = classification_report(y_true_3, y_pred_3, labels=[0, 1, 2],
                                                 target_names=labels3, zero_division=0, digits=4)
                    with open(os.path.join(pred_dir, "reval_3class_report.txt"), "w") as f:
                        f.write(rep3)
            else:
                print(f"[Info {clf}] Insufficient L2 supervised samples, skip L2/3-class visualizations")
        else:
            print(f"[Info {clf}] No L2 supervision, skip L2/3-class visualizations")

    print("\n[Done] All visualizations regenerated (see predictions/<clf>/reval_*)")


if __name__ == "__main__":
    main()
