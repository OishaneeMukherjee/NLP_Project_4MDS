from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(corpus, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer
