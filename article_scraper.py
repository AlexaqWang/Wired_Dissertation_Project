import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from pathlib import Path
from tqdm import tqdm
import re
import os


def sanitize_filename(name, max_length=100):
    name = re.sub(r'[<>:"/\\|?*\']', '_', name)
    name = re.sub(r'[“”‘’—–,，…]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip(' _')
    return name[:max_length]


def extract_image_src(tag):
    for attr in ["src", "data-src", "srcset"]:
        src = tag.get(attr)
        if src:
            if "," in src:
                src = src.split(",")[-1].strip().split()[0]
            return src
    return None


class WiredArticleScraper:
    def __init__(self, url, base_dir="wired_articles", tag=None):
        self.url = url
        self.tag = tag
        self.headers = {
            "User-Agent": "Mozilla/5.0"
        }
        self.base_dir = Path(base_dir)
        self.soup = None
        self.title = "N/A"
        self.authors = []
        self.publish_date = "N/A"
        self.dek = "N/A"
        self.category = "N/A"
        self.text_content = ""
        self.hero_image_url = None
        self.insert_images = []
        self.article_dir = None
        self.text_file = None
        self.hero_image_path = None

    def fetch_page(self):
        response = requests.get(self.url, headers=self.headers)
        response.raise_for_status()
        self.soup = BeautifulSoup(response.content, "html.parser")

    def parse_metadata(self):
        title_tag = self.soup.find("h1", {"data-testid": "ContentHeaderHed"})
        self.title = title_tag.get_text(strip=True) if title_tag else "N/A"

        author_wrapper = self.soup.find("span", class_=lambda c: c and c.startswith("BylineNamesWrapper"))
        if author_wrapper:
            author_tags = author_wrapper.find_all("span", {"data-testid": "BylineName"})
            self.authors = [a.get_text(strip=True) for a in author_tags]

        time_tag = self.soup.find("time", {"data-testid": "ContentHeaderPublishDate"})
        self.publish_date = time_tag.get_text(strip=True) if time_tag else "N/A"

        dek_tag = self.soup.find("div", attrs={"class": lambda x: x and "ContentHeaderDek" in x})
        self.dek = dek_tag.get_text(strip=True) if dek_tag else "N/A"

        for selector in [
            'a[data-testid="ContentHeaderRubric"] span',
            'span[class*="RubricName"]',
            'div[class*="ContentHeaderRubric"] span',
            'a.rubric__link span'
        ]:
            rubric_tag = self.soup.select_one(selector)
            if rubric_tag:
                self.category = rubric_tag.get_text(strip=True)
                break

    def parse_text_and_images(self):
        container = self.soup.select_one("div.body__inner-container") or \
                    self.soup.select_one("article.article__chunks")
        if not container:
            return

        paragraphs = []
        seen = set()
        img_index = 1

        for tag in container.find_all(["p", "img", "picture"]):
            if tag.name == "p":
                text = tag.get_text(strip=True)
                if text:
                    paragraphs.append(text)
            elif tag.name in ["img", "source"]:
                src = extract_image_src(tag)
                if not src:
                    continue
                if src.startswith("//"):
                    src = "https:" + src
                if "media.wired.com/photos" in src and src not in seen:
                    seen.add(src)
                    full_url = urljoin("https://media.wired.com", src)
                    filename = f"insert_{img_index}.jpg"
                    self.insert_images.append({"url": full_url, "filename": filename})
                    paragraphs.append(f"[{filename}]")
                    img_index += 1

        self.text_content = "\n".join(paragraphs)

    def extract_hero_image(self):
        for tag in self.soup.select("img, picture source"):
            src = extract_image_src(tag)
            if not src:
                continue
            if src.startswith("//"):
                src = "https:" + src
            if "media.wired.com/photos" in src:
                self.hero_image_url = urljoin("https://media.wired.com", src)
                break

    def save_article(self):
        safe_title = sanitize_filename(self.title.replace(" ", "_"))

        # Extract year (default unknown)
        year = None
        if self.publish_date and re.search(r"\d{4}", self.publish_date):
            year = re.search(r"\d{4}", self.publish_date).group()
            year_int = int(year)
            if not (1993 <= year_int <= 2024):
                print(f"[!] Skipping {self.url} — year {year} out of range.")
                return
        else:
            print(f"[!] Skipping {self.url} — cannot determine year.")
            return

        # Build directory path
        if self.tag:
            self.article_dir = self.base_dir / sanitize_filename(self.tag) / year / safe_title
        else:
            self.article_dir = self.base_dir / year / safe_title

        self.article_dir.mkdir(parents=True, exist_ok=True)

        self.text_file = self.article_dir / "article.txt"
        with open(self.text_file, "w", encoding="utf-8") as f:
            f.write(self.text_content)

    def download_image(self, url, filename):
        filepath = self.article_dir / filename
        if filepath.exists():
            return filename
        try:
            img_resp = requests.get(url, stream=True, headers=self.headers, timeout=10)
            if img_resp.status_code == 200 and "image" in img_resp.headers.get("Content-Type", ""):
                with open(filepath, "wb") as f:
                    for chunk in img_resp.iter_content(1024):
                        f.write(chunk)
                return filename
        except Exception as e:
            print(f"[!] Error downloading image {url}: {e}")
        return None

    def download_images(self):
        if self.hero_image_url:
            hero_name = "hero.jpg"
            if self.download_image(self.hero_image_url, hero_name):
                self.hero_image_path = hero_name

        for image in self.insert_images:
            self.download_image(image["url"], image["filename"])

    def save_metadata(self):
        meta = {
            "title": self.title,
            "author": self.authors,
            "publish_date": self.publish_date,
            "dek": self.dek,
            "cateogry": self.category,
            "hero_image": self.hero_image_path,
            "image_url": self.hero_image_url,
            "insert_images": self.insert_images,
            "source_url": self.url,
            "text_file": self.text_file.name if self.text_file else None,
        }

        with open(self.article_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def run(self):
        try:
            self.fetch_page()
            self.parse_metadata()
            self.parse_text_and_images()
            self.extract_hero_image()

            # Add logic: if no image found, skip this article
            if not self.hero_image_url and not self.insert_images:
                print(f"[!] Skipping {self.url} — no images found.")
                return

            self.save_article()
            self.download_images()
            self.save_metadata()
        except Exception as e:
            print(f"[!] Failed to scrape {self.url}: {e}")
