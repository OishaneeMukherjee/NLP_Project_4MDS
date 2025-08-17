import re
import nltk
from nltk.corpus import stopwords

# Ensure corpora are available (silent on re-runs)
for pkg in ["stopwords", "punkt", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))

def sent_tokenize(text: str):
    return nltk.sent_tokenize(text)

def word_tokenize(text: str):
    return nltk.word_tokenize(text)

def clean_text(text: str) -> str:
    """
    Basic cleaning: remove non-letters, lowercase, remove stopwords.
    """
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)
