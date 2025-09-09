import os
import json
from tqdm import tqdm

def load_meta(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_meta(json_path, data):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def get_all_meta_paths(root_dir):
    meta_paths = {}
    for category_dir in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category_dir)
        if not os.path.isdir(category_path):
            continue
        for article_folder in os.listdir(category_path):
            article_path = os.path.join(category_path, article_folder)
            meta_path = os.path.join(article_path, 'meta.json')
            if os.path.isfile(meta_path):
                try:
                    meta = load_meta(meta_path)
                    url = meta.get("source_url", "").strip()
                    if url:
                        meta_paths[url] = meta_path
                except Exception as e:
                    print(f"[ERROR] Loading {meta_path}: {e}")
    return meta_paths

def get_all_sitemap_paths(root_dir):
    meta_paths = {}
    for year_folder in os.listdir(root_dir):
        year_path = os.path.join(root_dir, year_folder)
        if not os.path.isdir(year_path):
            continue
        for article_folder in os.listdir(year_path):
            article_path = os.path.join(year_path, article_folder)
            meta_path = os.path.join(article_path, 'meta.json')
            if os.path.isfile(meta_path):
                try:
                    meta = load_meta(meta_path)
                    url = meta.get("source_url", "").strip()
                    if url:
                        meta_paths[url] = meta_path
                except Exception as e:
                    print(f"[ERROR] Loading {meta_path}: {e}")
    return meta_paths

def merge_meta(category_meta, sitemap_meta):
    updated = False
    for key, value in category_meta.items():
        if key not in sitemap_meta:
            sitemap_meta[key] = value
            updated = True
    return updated, sitemap_meta

# 路径
category_root = r"C:\Users\VV\PycharmProjects\WiredMultimodal\dataset\wired_category_articles"
sitemap_root = r"C:\Users\VV\PycharmProjects\WiredMultimodal\dataset\wired_sitemap_articles"


# 获取所有 meta 路径
category_meta_map = get_all_meta_paths(category_root)
sitemap_meta_map = get_all_sitemap_paths(sitemap_root)

# 合并
match_count = 0
update_count = 0

for url in tqdm(sitemap_meta_map, desc="Merging meta"):
    if url in category_meta_map:
        category_meta = load_meta(category_meta_map[url])
        sitemap_meta = load_meta(sitemap_meta_map[url])
        updated, merged_meta = merge_meta(category_meta, sitemap_meta)
        if updated:
            save_meta(sitemap_meta_map[url], merged_meta)
            update_count += 1
        match_count += 1

print(f"\nMatched articles: {match_count}")
print(f"Updated sitemap articles: {update_count}")
print(f"Total category articles: {len(category_meta_map)}")
print(f"Total sitemap articles: {len(sitemap_meta_map)}")
