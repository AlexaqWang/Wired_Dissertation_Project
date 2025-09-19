# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Fixed main categories
MAIN_CATEGORIES = {
    "culture", "security", "gear", "business", "politics",
    "science", "the big story", "ideas", "photo"
}

# Root directory containing year folders
DATA_ROOT = r"C:\Users\VV\PycharmProjects\WiredMultimodal\1dataset\wired_sitemap_articles"

all_records = []

# Scan year folders
for year in tqdm(os.listdir(DATA_ROOT), desc="Scanning years"):
    year_path = os.path.join(DATA_ROOT, year)
    if not os.path.isdir(year_path) or not year.isdigit():
        continue

    for article_folder in os.listdir(year_path):
        article_path = os.path.join(year_path, article_folder)
        if not os.path.isdir(article_path):
            continue

        meta_path = os.path.join(article_path, "meta.json")
        if not os.path.exists(meta_path):
            continue

        try:
            with open(meta_path, "r", encoding="utf-8-sig") as f:
                meta = json.load(f)
        except Exception:
            # Skip unreadable or malformed JSON
            continue

        # Normalize category
        category = meta.get("category", None)
        if isinstance(category, list):
            category = ", ".join(map(str, category))
        category_str = str(category).strip().lower() if category else "unknown"
        final_category = category_str if category_str in MAIN_CATEGORIES else "unknown"

        all_records.append({
            "year": year,
            "article_id": article_folder,
            "category": final_category,
            "original_category": category_str,
            "source_url": meta.get("source_url", "")
        })

# Save the flat table
output_csv = os.path.join(DATA_ROOT, "..", "all_articles_with_category.csv")
df = pd.DataFrame(all_records)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"Saved article-category table: {output_csv} (rows={len(df)})")

# Category summary
summary = df["category"].value_counts()
summary_csv = os.path.join(DATA_ROOT, "..", "category_summary.csv")
summary.to_csv(summary_csv, header=["count"], encoding="utf-8-sig")
print(f"Saved category summary: {summary_csv}")

# Pie chart (optional)
plt.figure(figsize=(10, 10))
plt.pie(
    summary.values,
    labels=summary.index,
    autopct="%1.1f%%",
    startangle=140,
    wedgeprops={"edgecolor": "white"}
)
plt.title("Distribution of Article Categories")
plt.axis("equal")
plt.tight_layout()
pie_path = os.path.join(DATA_ROOT, "..", "category_pie_chart.png")
plt.savefig(pie_path, dpi=300)
plt.show()
print(f"Pie chart saved to: {pie_path}")

# ================= Stacked Bar Chart: Top Categories by Year =================
# Ensure year is integer and sorted
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df = df.dropna(subset=["year"]).copy()
df["year"] = df["year"].astype(int)

# Aggregate counts by (year, category)
yearly_counts = df.groupby(["year", "category"]).size().unstack(fill_value=0)

# Keep top-10 categories by overall frequency (includes 'unknown' if it is in top-10)
top_categories = (
    yearly_counts.sum()
    .sort_values(ascending=False)
    .head(10)
    .index
)
yearly_counts = yearly_counts[top_categories]

# Sort rows by year
yearly_counts = yearly_counts.sort_index()

# Plot stacked bar chart
plt.figure(figsize=(12, 6))
ax = yearly_counts.plot(kind="bar", stacked=True, width=0.85)
plt.title("Top Categories by Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Categories", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

bar_chart_path = os.path.join(DATA_ROOT, "..", "tag_by_year_stacked_bar_chart.png")
plt.savefig(bar_chart_path, dpi=300)
plt.show()
print(f"Stacked bar chart saved to: {bar_chart_path}")
