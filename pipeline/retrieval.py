import re
from collections import defaultdict
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def split_sentences(text: str) -> List[str]:
    # Simple splitter for UI speed; you can swap with nltk.sent_tokenize
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]

# ----- Inverted Index -----
def build_inverted_index(text: str) -> Tuple[dict, List[str]]:
    sentences = split_sentences(text)
    index = defaultdict(list)
    for i, s in enumerate(sentences):
        for w in re.findall(r"[A-Za-z]+", s.lower()):
            if i not in index[w]:
                index[w].append(i)
    return index, sentences

def search_in_index(query: str, index: dict, sentences: List[str]) -> List[Tuple[int, str]]:
    hits = set()
    for w in re.findall(r"[A-Za-z]+", query.lower()):
        for sid in index.get(w, []):
            hits.add(sid)
    return sorted([(i, sentences[i]) for i in hits], key=lambda x: x[0])

# ----- TF-IDF Scoring for Retrieval -----
def score_sentences(query: str, sentences: List[str]) -> List[Tuple[int, float]]:
    if not sentences or not query.strip():
        return []
    vect = TfidfVectorizer()
    X = vect.fit_transform([query] + sentences)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()
    ranked = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)
    return ranked

# ----- Extractive QA: pick best sentence -----
def answer_question(question: str, sentences: List[str]) -> Tuple[str, float, int]:
    if not sentences or not question.strip():
        return "", 0.0, -1
    ranked = score_sentences(question, sentences)
    best_idx, best_score = ranked[0]
    return sentences[best_idx], float(best_score), int(best_idx)

# ----- IR Evaluation (toy) -----
def evaluate_ir(query: str, sentences: List[str], top_k: int = 5) -> dict:
    """
    Very simple evaluation using term overlap as 'relevance' proxy.
    Relevant if a sentence contains any query term (bag-of-words heuristic).
    """
    query_terms = set(re.findall(r"[A-Za-z]+", query.lower()))
    ranked = score_sentences(query, sentences)
    top = ranked[:top_k]
    retrieved_ids = [i for i, _ in top]
    relevant_ids = [i for i, s in enumerate(sentences)
                    if any(t in re.findall(r"[A-Za-z]+", s.lower()) for t in query_terms)]
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    true_pos = len(retrieved_set & relevant_set)
    precision = true_pos / max(len(retrieved_set), 1)
    recall = true_pos / max(len(relevant_set), 1)
    return {
        "top_k": top_k,
        "retrieved": retrieved_ids,
        "relevant": relevant_ids[:50],
        "precision": round(precision, 3),
        "recall": round(recall, 3),
    }
