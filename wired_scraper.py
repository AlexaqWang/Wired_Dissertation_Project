import pandas as pd
from datascraping.article_scraper2 import WiredArticleScraper
from tqdm import tqdm

# === 参数设置 ===
csv_path = "../article_links.csv"
base_output_dir = r"/0datascraping"
max_articles = 10

# === 加载链接 ===
df = pd.read_csv(csv_path)
print(df.columns)  # 查看实际列名
urls = df["url"].dropna().unique().tolist()  # 改为实际列名

if max_articles:
    urls = urls[:max_articles]

# === 批量抓取 ===
for url in tqdm(urls, desc="Scraping articles"):
    try:
        scraper = WiredArticleScraper(url, base_dir=base_output_dir)
        scraper.run()
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
