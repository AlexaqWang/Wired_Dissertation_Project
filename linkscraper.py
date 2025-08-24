import requests
from bs4 import BeautifulSoup
import csv
import time
from tqdm import tqdm

headers = {'User-Agent': 'Mozilla/5.0'}

# Wired cateogry structure (main cateogry → list of subcategory URLs)
category_structure = {
    "culture": [
        "https://www.wired.com/category/culture/digital-culture/",
        "https://www.wired.com/tag/video-games/",
        "https://www.wired.com/category/culture/movies/",
        "https://www.wired.com/category/culture/books/",
    ],
    "security": [
        "https://www.wired.com/category/security/cyberattacks-hacks/",
        "https://www.wired.com/category/security/privacy/",
        "https://www.wired.com/category/security/security-advice/",
        "https://www.wired.com/category/security/security-news/",
    ],
    "politics": [
        "https://www.wired.com/category/politics/policy",
        "https://www.wired.com/category/politics/disinformation",
        "https://www.wired.com/category/politics/extremism/",
        "https://www.wired.com/category/politics/global-elections",
        "https://www.wired.com/category/politics/politics-news",
    ],
    "big-story": [
        "https://www.wired.com/the-big-interview/",
    ],
    "business": [
        "https://www.wired.com/category/artificial-intelligence/",
        "https://www.wired.com/category/business/big-tech/",
        "https://www.wired.com/category/business/tech-culture/",
        "https://www.wired.com/category/business/social-media/",
        "https://www.wired.com/category/business/policy-net-neutrality/",
        "https://www.wired.com/category/business/blockchain-cryptocurrency/",
        "https://www.wired.com/category/business/computers-software/",
        "https://www.wired.com/category/business/retail-and-logistics/",
        "https://www.wired.com/category/business/transportation/",
        "https://www.wired.com/category/business/energy/",
    ],
    "science": [
        "https://www.wired.com/category/science/health-medicine/",
        "https://www.wired.com/tag/environment",
        "https://www.wired.com/category/science/environment-climate-change/",
        "https://www.wired.com/tag/energy/",
        "https://www.wired.com/category/science/space/",
        "https://www.wired.com/category/science/physics-math/",
        "https://www.wired.com/category/science/biotech-genetic-engineering/",
        "https://www.wired.com/category/science/psychology-neuroscience/",
    ],
}

# Extract article links
def extract_links(soup):
    links = []
    for a in soup.find_all('a', class_="SummaryItemHedLink-civMjp ejgyuy summary-item-tracking__hed-link summary-item__hed-link"):
        href = a.get('href')
        if href and '/story/' in href:
            full_url = 'https://www.wired.com' + href.split('?redirectURL=')[-1]
            links.append(full_url)
    return links

# Crawl all article links from a subcategory
def crawl_all_pages(base_url, max_try_pages=500):
    all_links = set()
    empty_count = 0
    for page in range(1, max_try_pages + 1):
        url = f"{base_url}?page={page}"
        res = requests.get(url, headers=headers)

        if res.status_code != 200:
            print(f"[!] Request failed at {url} (Status {res.status_code})")
            break

        soup = BeautifulSoup(res.text, 'html.parser')
        links = extract_links(soup)

        print(f"[INFO] {url} --> {len(links)} articles")

        if not links:
            empty_count += 1
            if empty_count >= 2:
                print(f"[STOP] Two empty pages encountered. Stopping at page {page}.")
                break
        else:
            empty_count = 0
            all_links.update(links)

        time.sleep(1)

    return all_links

# Main routine: iterate all main categories and subcategory URLs
tagged_links = set()
all_tasks = [(main_cat, url) for main_cat, urls in category_structure.items() for url in urls]

for main_cat, subcat_url in tqdm(all_tasks, desc="Crawling subcategories"):
    subcat_name = subcat_url.rstrip('/').split('/')[-1] or main_cat
    links = crawl_all_pages(subcat_url)
    for link in links:
        tagged_links.add((main_cat, subcat_name, link))

# Save result to CSV file
with open('wired_category_csv/wired_tagged_links_with_subcategory.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['main_category', 'subcategory', 'article_link'])
    for row in sorted(tagged_links):
        writer.writerow(row)

print(f"\nFinished. Total unique articles saved: {len(tagged_links)}")
