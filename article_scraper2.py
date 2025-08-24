import os
import json
import re
import time
import random
import logging
import concurrent.futures
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wired_full_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WiredFullUpdater")


class WiredArticleUpdater:
    def __init__(self, article_path):
        self.article_path = Path(article_path)
        self.meta_file = self.article_path / "meta.json"
        self.text_file = self.article_path / "article.txt"
        self.url = None
        self.original_title = None
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        self.retries = 3
        self.timeout = 20
        self.soup = None
        self.title = "N/A"
        self.text_content = ""
        self.original_metadata = {}
        self.new_metadata = {}
        self.backup_time = time.strftime("%Y%m%d%H%M%S")

    def load_existing_metadata(self):
        """Load existing metadata"""
        if not self.meta_file.exists():
            logger.error(f"Meta file not found: {self.meta_file}")
            return False

        try:
            with open(self.meta_file, 'mode', encoding='utf-8') as f:
                self.original_metadata = json.load(f)
                self.url = self.original_metadata.get('source_url', '')
                self.original_title = self.original_metadata.get('title', '')
                return True
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return False

    def fetch_page(self):
        """Fetch page and parse with BeautifulSoup"""
        if not self.url:
            logger.error("No URL found for article")
            return False

        for attempt in range(self.retries):
            try:
                response = requests.get(self.url, headers=self.headers, timeout=self.timeout)
                response.raise_for_status()

                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type:
                    logger.warning(f"Unexpected content type for {self.url}: {content_type}")
                    return False

                self.soup = BeautifulSoup(response.content, "html.parser")
                return True
            except requests.exceptions.RequestException as e:
                if attempt < self.retries - 1:
                    wait = 2 ** attempt + random.uniform(1, 3)
                    logger.warning(f"Retry {attempt + 1}/{self.retries} for {self.url} in {wait:.1f}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"Failed to fetch {self.url}: {e}")
                    return False
        return False

    def parse_title(self):
        """Parse article title"""
        try:
            title_tag = self.soup.find("h1", {"data-testid": "ContentHeaderHed"})
            if not title_tag:
                title_tag = self.soup.find("h1", class_="content-header__hed")
            self.title = title_tag.get_text(strip=True) if title_tag else self.original_title
            return True
        except Exception as e:
            logger.error(f"Error parsing title: {e}")
            self.title = self.original_title
            return False

    def extract_paragraph_text(self, element):
        """Recursively extract paragraph text preserving whitespace"""
        paragraph_text = ""
        for child in element.children:
            if isinstance(child, str):
                paragraph_text += child
            elif child.name in ["em", "strong", "span", "a", "i", "b"]:
                inner_text = self.extract_paragraph_text(child)
                paragraph_text += inner_text
            elif child.name == "br":
                paragraph_text += "\n"

        paragraph_text = re.sub(r'[ \t]+', ' ', paragraph_text)
        paragraph_text = re.sub(r'\n\s*\n', '\n\n', paragraph_text)
        return paragraph_text.strip()

    def parse_text_content(self):
        """Parse main content and preserve original structure (ignore images)"""
        try:
            container = self.soup.select_one("div.body__inner-container")
            if not container:
                container = self.soup.select_one("article.article__chunks")
            if not container:
                container = self.soup.find("div", class_="article-body-component")
            if not container:
                logger.warning(f"No content container found for {self.url}")
                return False

            content_parts = []

            for element in container.find_all(recursive=False):
                if element.name == "p":
                    paragraph_text = self.extract_paragraph_text(element)
                    if paragraph_text:
                        content_parts.append(paragraph_text)
                elif element.name in ["h2", "h3", "h4"]:
                    header_text = element.get_text().strip()
                    if header_text:
                        content_parts.append(f"\n\n### {header_text} ###\n")
                elif element.name == "blockquote":
                    quote_text = element.get_text().strip()
                    if quote_text:
                        content_parts.append(f"\n> {quote_text}\n")
                elif element.name == "ul":
                    for li in element.find_all("li", recursive=False):
                        item_text = li.get_text().strip()
                        if item_text:
                            content_parts.append(f"{item_text}")
                    content_parts.append("")

            full_text = "\n\n".join(content_parts)
            full_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', full_text)
            full_text = re.sub(r'(\w)([.,!?;:])(\w)', r'\1\2 \3', full_text)
            full_text = re.sub(r'\n{3,}', '\n\n', full_text)

            self.text_content = full_text
            return True
        except Exception as e:
            logger.error(f"Error parsing content: {e}")
            return False

    def create_backup(self):
        """Create backup of original files"""
        try:
            if self.text_file.exists():
                backup_text = self.text_file.with_name(f"article_original_{self.backup_time}.txt")
                self.text_file.rename(backup_text)

            if self.meta_file.exists():
                backup_meta = self.meta_file.with_name(f"meta_original_{self.backup_time}.json")
                self.meta_file.rename(backup_meta)

            return True
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False

    def restore_backup(self):
        """Restore original files from backup"""
        try:
            backup_text = self.text_file.with_name(f"article_original_{self.backup_time}.txt")
            if backup_text.exists():
                backup_text.rename(self.text_file)

            backup_meta = self.meta_file.with_name(f"meta_original_{self.backup_time}.json")
            if backup_meta.exists():
                backup_meta.rename(self.meta_file)

            return True
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False

    def update_files(self):
        """Update text and metadata files"""
        try:
            with open(self.text_file, 'w', encoding='utf-8') as f:
                f.write(self.text_content)

            self.new_metadata = self.original_metadata.copy()
            self.new_metadata["title"] = self.title
            with open(self.meta_file, 'w', encoding='utf-8') as f:
                json.dump(self.new_metadata, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            logger.error(f"Error updating files: {e}")
            return False

    def run(self):
        """Run full update process"""
        try:
            if not self.load_existing_metadata():
                return False, "Failed to load metadata"

            logger.info(f"Processing: {self.article_path.name}")

            if not self.create_backup():
                return False, "Backup failed"

            if not self.fetch_page():
                self.restore_backup()
                return False, "Failed to fetch page"

            if not self.parse_title():
                self.restore_backup()
                return False, "Failed to parse title"

            if not self.parse_text_content():
                self.restore_backup()
                return False, "Failed to parse content"

            if not self.update_files():
                self.restore_backup()
                return False, "Failed to update files"

            logger.info(f"Successfully updated {self.article_path.name}")
            return True, "Success"
        except Exception as e:
            logger.exception(f"Failed to update {self.article_path.name}: {e}")
            try:
                self.restore_backup()
            except:
                pass
            return False, str(e)


def get_all_article_paths(root_dir):
    """Get all article directories containing meta.json"""
    root_path = Path(root_dir)
    article_paths = []
    for year_dir in root_path.iterdir():
        if not year_dir.is_dir():
            continue
        for article_dir in year_dir.iterdir():
            if article_dir.is_dir():
                meta_file = article_dir / "meta.json"
                if meta_file.exists():
                    article_paths.append(article_dir)
    return article_paths


def process_article(article_path):
    """Process single article (for parallel execution)"""
    updater = WiredArticleUpdater(article_path)
    success, message = updater.run()
    return {
        "path": str(article_path),
        "success": success,
        "message": message
    }


def update_full_dataset(dataset_root, max_workers=5):
    """Update full 1dataset"""
    start_time = time.time()
    logger.info("Starting full 1dataset update")

    article_paths = get_all_article_paths(dataset_root)
    total = len(article_paths)

    if total == 0:
        logger.error("No articles found in 1dataset")
        return 0, 0

    logger.info(f"Found {total} articles to update")
    pbar = tqdm(total=total, desc="Updating articles", unit="article")

    success_count = 0
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(process_article, path): path
            for path in article_paths
        }

        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
                if result["success"]:
                    success_count += 1
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results.append({
                    "path": str(path),
                    "success": False,
                    "message": str(e)
                })
            finally:
                pbar.update(1)

    pbar.close()

    elapsed = time.time() - start_time
    fail_count = total - success_count

    report = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
        "elapsed_seconds": round(elapsed, 2),
        "total_articles": total,
        "success_count": success_count,
        "fail_count": fail_count,
        "results": results
    }

    with open("../wired_update_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Update completed in {elapsed:.2f} seconds")
    logger.info(f"Success: {success_count}, Failed: {fail_count}")

    return success_count, fail_count


if __name__ == "__main__":
    DATASET_ROOT = r"C:\Users\VV\PycharmProjects\WiredMultimodal\dataset\wired_sitemap_articles"
    MAX_WORKERS = 5

    print("=" * 60)
    print(f"Starting FULL UPDATE of Wired 1dataset")
    print(f"Dataset location: {DATASET_ROOT}")
    print(f"Using {MAX_WORKERS} parallel workers")
    print("=" * 60)

    success, fail = update_full_dataset(DATASET_ROOT, max_workers=MAX_WORKERS)

    print("\n" + "=" * 60)
    print(f"UPDATE SUMMARY")
    print(f"Total articles: {success + fail}")
    print(f"Successfully updated: {success}")
    print(f"Failed: {fail}")
    print(f"Detailed report saved to: wired_update_report.json")
    print("=" * 60)

    if fail > 0:
        print("\nFailed articles (first 10):")
        try:
            with open("../wired_update_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)
                fails = [r for r in report["results"] if not r["success"]]
                for i, fail in enumerate(fails[:10]):
                    print(f"{i + 1}. {fail['path']}")
                    print(f"   Reason: {fail['message']}")
        except:
            print("Could not load detailed failure report")
