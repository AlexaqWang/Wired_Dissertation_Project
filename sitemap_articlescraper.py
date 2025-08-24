import pandas as pd
from tqdm import tqdm
from datascraping.article_scraper import WiredArticleScraper
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load CSV file
csv_path = "wired_sitemap_article_links.csv"
df = pd.read_csv(csv_path)
df.columns = [c.strip().lower() for c in df.columns]
df = df.dropna(subset=["url"])
df = df.drop_duplicates(subset=["url"])

output_dir = r"/1dataset/wired_sitemap_articles"

tasks = [
    {"url": row["url"], "tag": row["tag"] if "tag" in row else None}
    for _, row in df.iterrows()
]

skipped = []
failed = []

def scrape_task(task):
    url = task["url"]
    tag = task["tag"]
    try:
        scraper = WiredArticleScraper(url, base_dir=output_dir, tag=tag)
        result = scraper.run()
        if result == "skipped":
            return ("skipped", url, tag)
        elif result == "failed":
            return ("failed", url, tag)
        else:
            return ("success", url, tag)
    except Exception:
        return ("failed", url, tag)

# Execute and classify results
results = []
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(scrape_task, task) for task in tasks]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping articles"):
        results.append(future.result())

# Record by cateogry
for status, url, tag in results:
    if status == "skipped":
        skipped.append({"url": url, "tag": tag})
    elif status == "failed":
        failed.append({"url": url, "tag": tag})

# Save CSV files
pd.DataFrame(skipped).to_csv("skipped_articles.csv", index=False)
pd.DataFrame(failed).to_csv("../failed_articles.csv", index=False)
