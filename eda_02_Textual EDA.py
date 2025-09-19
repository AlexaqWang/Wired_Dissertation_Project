# -*- coding: utf-8 -*-
# Textual EDA: token frequency, n-grams, tag co-occurrence
import pandas as pd
import numpy as np
import re
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt

# Load
df = pd.read_csv("wired_articles.csv")
texts = df['text'].fillna("").astype(str)

# Simple cleaner
def clean_text(s):
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

cleaned = texts.apply(clean_text)

# 1) Unigram frequency
tokens = []
for t in cleaned:
    tokens.extend(t.split())
unigram_counts = Counter(tokens)
top_uni = unigram_counts.most_common(50)
print("Top 50 unigrams:", top_uni)

# 2) N-grams (bigrams and trigrams)
def ngrams(words, n):
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

bigram_counts, trigram_counts = Counter(), Counter()
for t in cleaned:
    ws = t.split()
    bigram_counts.update(ngrams(ws, 2))
    trigram_counts.update(ngrams(ws, 3))

print("Top 30 bigrams:", bigram_counts.most_common(30))
print("Top 30 trigrams:", trigram_counts.most_common(30))

# 3) Tags co-occurrence matrix (if tags column exists like "ai;privacy;security")
def split_tags(x):
    if pd.isna(x) or not isinstance(x, str) or x.strip()=="":
        return []
    # Support separators: ';' or ','
    parts = re.split(r"[;,]\s*", x.strip())
    return [p for p in parts if p]

df['tag_list'] = df['tags'].apply(split_tags) if 'tags' in df.columns else [[]]

# Build co-occurrence counts
pair_counts = Counter()
for tags in df['tag_list']:
    uniq = sorted(set([t.lower() for t in tags]))
    for a, b in combinations(uniq, 2):
        pair_counts[(a, b)] += 1

# Show top pairs
print("Top 50 tag pairs:", pair_counts.most_common(50))

# Optional: convert to an edge list CSV for network tools (e.g., Gephi)
edges = pd.DataFrame(
    [(a, b, c) for (a, b), c in pair_counts.items()],
    columns=['source','target','weight']
)
edges.to_csv("tag_cooccurrence_edges.csv", index=False)
print("Saved tag co-occurrence edges to tag_cooccurrence_edges.csv")
