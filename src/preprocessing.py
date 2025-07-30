import re
import spacy
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower()
    return " ".join(text.split())

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
