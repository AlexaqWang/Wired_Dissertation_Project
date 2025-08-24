import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import time
import gzip
from io import BytesIO

# General function: extract all <loc> entries from a cateogry or cateogry index
def get_loc_links(xml_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(xml_url, headers=headers)
    response.raise_for_status()

    content_type = response.headers.get('Content-Type', '')
    if 'gzip' in content_type or xml_url.endswith('.gz'):
        with gzip.open(BytesIO(response.content), 'rt', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'xml')
    else:
        soup = BeautifulSoup(response.content, 'xml')

    locs = [tag.text.strip() for tag in soup.find_all("loc")]
    print(f"[DEBUG] {xml_url} - {len(locs)} <loc> tags found")
    return locs

# Step 1: extract all sub-sitemaps from cateogry index
def get_all_sub_sitemaps(index_urls):
    sub_sitemaps = []
    for index_url in index_urls:
        print(f"\n[INFO] Fetching sub-sitemaps from: {index_url}")
        links = get_loc_links(index_url)
        print(f"[DEBUG] Found {len(links)} sub-sitemaps in {index_url}")
        sub_sitemaps += links
    return sub_sitemaps

# Step 2: extract article links with source cateogry (tagURL), up to 100
def extract_all_article_links(sub_sitemaps, max_links=100, delay=0.5):
    all_records = []
    for sitemap_url in tqdm(sub_sitemaps, desc="Parsing article sitemaps"):
        if len(all_records) >= max_links:
            break
        try:
            links = get_loc_links(sitemap_url)
            print(f"[DEBUG] {sitemap_url} - {len(links)} article links")
            for link in links:
                all_records.append({"URL": link, "tagURL": sitemap_url})
                if len(all_records) >= max_links:
                    break
            time.sleep(delay)
        except Exception as e:
            print(f"[!] Error parsing {sitemap_url}: {e}")
    return all_records

# Main cateogry index list (can be extended)
sitemap_index_urls = [
    "https://www.wired.com/sitemap.xml",
    "https://www.wired.com/sitemap.xml1"
]

# Main program execution
if __name__ == "__main__":
    print("Step 1: Collecting all sub-cateogry URLs...")
    sub_sitemap_urls = get_all_sub_sitemaps(sitemap_index_urls)
    print(f"[INFO] Total sub-sitemaps found: {len(sub_sitemap_urls)}")
    print(f"[DEBUG] Sample sub-sitemaps: {sub_sitemap_urls[:3]}")

    print("\nStep 2: Extracting up to 100 article links with tagURL...")
    article_records = extract_all_article_links(sub_sitemap_urls, max_links=100)
    print(f"[INFO] Total article links extracted: {len(article_records)}")

    if article_records:
        df = pd.DataFrame(article_records)
        print("[DEBUG] First few records:\n", df.head())
        df.to_csv("wired_sample_100_links_with_tag.csv", index=False)
        print("Sample links with tagURL saved to wired_sample_100_links_with_tag.csv")
    else:
        print("No article links found. CSV not saved.")
