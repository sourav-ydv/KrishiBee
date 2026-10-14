import json
import math
import os
import re
from collections import Counter

KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "knowledge")

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "the", "is", "are", "of", "to", "in", "on", "for", "and", "or", "what", "when",
    "how", "should", "i", "my", "do", "does", "can", "me", "it", "at", "be", "with", "crop",
    "wheat", "rice", "maize", "chickpea", "mustard", "soybean", "millet", "pearl", "pigeon", "pea",
}

HINDI_TERMS = {
    "sichai": "irrigation", "sinchai": "irrigation", "sinchaai": "irrigation", "sichayi": "irrigation",
    "सिंचाई": "irrigation", "pani": "water", "paani": "water", "पानी": "water",
    "kab": "when timing", "kb": "when timing", "कब": "when timing", "samay": "timing", "समय": "timing",
    "din": "days", "दिन": "days",
    "pehli": "first", "pahli": "first", "पहली": "first", "pehla": "first", "पहला": "first",
    "kitna": "how much amount", "kitni": "how much amount", "कितना": "how much amount", "कितनी": "how much amount",
    "matra": "amount", "मात्रा": "amount",
    "bowai": "sowing", "buwai": "sowing", "bovai": "sowing", "बुवाई": "sowing", "बुआई": "sowing",
    "khad": "fertilizer", "खाद": "fertilizer", "urvarak": "fertilizer", "उर्वरक": "fertilizer",
    "urea": "urea fertilizer nitrogen", "yuriya": "urea fertilizer nitrogen", "यूरिया": "urea fertilizer nitrogen",
    "nirai": "weeding weed", "nindai": "weeding weed", "निराई": "weeding weed",
    "kharpatwar": "weed", "खरपतवार": "weed",
    "gehrai": "depth sowing", "gahrai": "depth sowing", "गहराई": "depth sowing",
    "katai": "harvest", "कटाई": "harvest",
}


def expand_query(query):
    raw = re.findall(r"[a-z0-9]+|[\u0900-\u097f]+", query.lower())
    extra = [HINDI_TERMS[t] for t in raw if t in HINDI_TERMS]
    return " ".join([query] + extra)


def tokenize(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class KnowledgeBase:
    def __init__(self, entries):
        self.entries = entries
        self.docs = [tokenize(f"{e['topic']} {e['text']}") for e in entries]
        self.tf = [Counter(d) for d in self.docs]
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = sum(self.doc_len) / len(self.docs) if self.docs else 0
        self.df = Counter()
        for d in self.docs:
            for term in set(d):
                self.df[term] += 1

    def score(self, query_tokens, idx, k1=1.5, b=0.75):
        n = len(self.docs)
        total = 0.0
        for term in query_tokens:
            f = self.tf[idx].get(term, 0)
            if f == 0:
                continue
            idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
            total += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self.doc_len[idx] / self.avg_len))
        return total

    def search(self, query, crop=None, top_k=3):
        candidates = [
            i for i, e in enumerate(self.entries)
            if crop is None or e["crop"].lower() in (crop.lower(), "general")
        ]
        if not candidates:
            return []
        tokens = tokenize(expand_query(query))
        if not tokens:
            return []
        scored = sorted(((self.score(tokens, i), i) for i in candidates), reverse=True)
        return [self.entries[i] for s, i in scored if s > 0][:top_k]


def load_entries():
    entries = []
    if os.path.isdir(KB_DIR):
        for name in sorted(os.listdir(KB_DIR)):
            if name.endswith(".json"):
                with open(os.path.join(KB_DIR, name), encoding="utf-8") as f:
                    entries.extend(
                        e for e in json.load(f)
                        if {"crop", "topic", "text", "source_title", "source_url"} <= e.keys()
                    )
    return entries


kb = KnowledgeBase(load_entries())