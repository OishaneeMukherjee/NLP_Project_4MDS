import math
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

# ---- TF-IDF + Cosine similarity (query vs sentences/docs) ----
def tfidf_cosine_similarity(query: str, docs: list[str]) -> np.ndarray:
    if not docs:
        return np.array([])
    tfidf = TfidfVectorizer()
    vecs = tfidf.fit_transform([query] + docs)
    sims = sk_cosine(vecs[0:1], vecs[1:]).flatten()
    return sims

# ---- PMI (simple bigram PMI using a window size of 1) ----
def pmi(word1: str, word2: str, tokens: list[str]) -> float:
    if not tokens or word1 == "" or word2 == "":
        return 0.0
    total = len(tokens)
    if total < 2:
        return 0.0
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    bigram_counts = Counter(bigrams)
    token_counts = Counter(tokens)

    p_w1 = token_counts[word1] / total if token_counts[word1] else 0
    p_w2 = token_counts[word2] / total if token_counts[word2] else 0
    p_w1w2 = bigram_counts[(word1, word2)] / (total - 1) if bigram_counts[(word1, word2)] else 0

    if p_w1 == 0 or p_w2 == 0 or p_w1w2 == 0:
        return 0.0
    return float(math.log2(p_w1w2 / (p_w1 * p_w2)))

# ---- Word similarity via embeddings (graceful fallbacks) ----
# Primary: spaCy vectors (if available); spaCy small has limited vectors but returns something
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

def word_similarity(word1: str, word2: str):
    """
    Attempts spaCy vector similarity. If unavailable, returns None.
    """
    if not _nlp:
        return None
    t1 = _nlp(word1)
    t2 = _nlp(word2)
    try:
        sim = float(t1.similarity(t2))
        return sim
    except Exception:
        return None
