import pandas as pd

# --- 路径 ---
profile_path = r"C:\Users\VV\PycharmProjects\WiredMultimodal\framing_profile_by_article.csv"
meta_path    = r"C:\Users\VV\PycharmProjects\WiredMultimodal\4framing marker\wired_framing_matchedall.csv"
out_path     = r"C:\Users\VV\PycharmProjects\WiredMultimodal\framing_profile_merged.csv"

# --- 读数据 ---
df_prof = pd.read_csv(profile_path)
df_meta = pd.read_csv(meta_path)

# --- 统一键类型（很关键，否则 merge 会对不上）---
df_prof["article_id"] = df_prof["article_id"].astype(str).str.strip()
df_meta["article_id"] = df_meta["article_id"].astype(str).str.strip()

# --- 如 meta 可能存在同一 article_id 多行（例如多图），可选择是否去重 ---
# 若一文一行：取消下一行注释以去重（保留首行）
# df_meta = df_meta.drop_duplicates(subset=["article_id"], keep="first")

# --- 合并策略 ---
# how="left"：以 meta 为主，尽量不丢文章；how="inner"：只保留两边都有的
MERGE_HOW = "left"   # 可改为 "inner"
df_merged = df_meta.merge(df_prof, on="article_id", how=MERGE_HOW)

# --- 基本质量报告 ---
n_meta = df_meta["article_id"].nunique()
n_prof = df_prof["article_id"].nunique()
n_merge = df_merged["article_id"].nunique()
n_missing_profile = df_merged["article_id"].isna().sum() if "article_id" not in df_prof.columns else (df_merged["n_valid"].isna().sum())

print(f"[INFO] unique articles in meta:     {n_meta}")
print(f"[INFO] unique articles in profile:  {n_prof}")
print(f"[INFO] unique articles after merge: {n_merge}")
print(f"[INFO] rows without profile (NaN):  {n_missing_profile}")

# --- 可选：对缺失 profile 的文章做填充（谨慎）
# df_merged[["n_none","n_counter","n_framing","n_valid","framing_ratio","counter_ratio","net_framing_score"]] = \
#     df_merged[["n_none","n_counter","n_framing","n_valid","framing_ratio","counter_ratio","net_framing_score"]].fillna(0)

# --- 保存 ---
df_merged.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"[OK] merged CSV saved to: {out_path}")
