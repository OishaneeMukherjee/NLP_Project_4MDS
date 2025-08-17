from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

def extract_keywords(text: str, top_n: int = 10):
    """
    Extract top keywords using TF-IDF on the single document.
    """
    if not text.strip():
        return []
    tfidf = TfidfVectorizer(max_features=200, ngram_range=(1,2))
    X = tfidf.fit_transform([text])
    feats = tfidf.get_feature_names_out()
    scores = X.toarray()[0]
    ranked = sorted(zip(feats, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

def analyze_sentiment(text: str):
    """
    TextBlob sentiment: polarity (-1..1), subjectivity (0..1).
    """
    blob = TextBlob(text)
    s = blob.sentiment
    return float(s.polarity), float(s.subjectivity)
