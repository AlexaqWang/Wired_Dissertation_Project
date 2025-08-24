import requests
from bs4 import BeautifulSoup
import csv
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

headers = {'User-Agent': 'Mozilla/5.0'}

# Load tag page links
def load_tag_urls(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip().startswith('https://')]

# Extract all article links from the current page
def extract_links(soup):
    links = []
    for a in soup.find_all('a', class_="SummaryItemHedLink-civMjp ejgyuy summary-item-tracking__hed-link summary-item__hed-link"):
        href = a.get('href')
        if href and '/story/' in href:
            full_url = 'https://www.wired.com' + href.split('?redirectURL=')[-1]
            links.append(full_url)
    return links

# Crawl all pages of a given tag
def crawl_all_pages(base_url, tag_name, max_try_pages=500):
    all_links = set()
    empty_count = 0

    for page in range(1, max_try_pages + 1):
        url = f"{base_url}?page={page}"
        try:
            res = requests.get(url, headers=headers, timeout=8)
            if res.status_code == 404:
                print(f"[404] Not found: {url}")
                break
            elif res.status_code != 200:
                print(f"[!] Status {res.status_code} at {url}")
                break
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] {url} - {e}")
            break

        soup = BeautifulSoup(res.text, 'html.parser')
        links = extract_links(soup)

        print(f"[INFO] {url} --> {len(links)} articles")

        if not links:
            empty_count += 1
            if empty_count >= 2:
                print(f"[STOP] Two empty pages. Stop at page {page}.")
                break
        else:
            empty_count = 0
            all_links.update(links)

        time.sleep(0.1)  # Can be increased to avoid IP blocking

    return [(tag_name, link) for link in all_links]

# Save results to CSV (append mode)
def save_to_csv(file_path, data):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Tag', 'Article Link'])
        for tag, link in sorted(data):
            writer.writerow([tag, link])

# Process a single tag in parallel
def process_tag(url):
    tag_name = url.rstrip('/').split('/')[-1]
    print(f"\nCrawling tag: {tag_name} ({url})")
    tag_links = crawl_all_pages(url, tag_name)
    print(f"{tag_name} --> {len(tag_links)} articles")
    save_to_csv("../cateogry/wired_category_csv/wired_tag_articles.csv", tag_links)
    return tag_name, len(tag_links)

# Main entry
if __name__ == "__main__":
    tag_urls = load_tag_urls("../cateogry/wired_category_csv/wired_tag_urls.txt")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_tag, url) for url in tag_urls]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Crawling all tag pages"):
            try:
                tag, count = f.result()
            except Exception as e:
                print(f"[!] Error in thread: {e}")
