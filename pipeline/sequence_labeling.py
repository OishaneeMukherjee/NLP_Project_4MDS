import spacy
from nltk.wsd import lesk

# Load spaCy small English model once (fast)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Graceful hint; app will show a message if features requiring it are used
    nlp = None

def pos_and_ner(text: str):
    """
    Returns POS tags and Named Entities using spaCy.
    """
    if not nlp:
        return [], []
    doc = nlp(text)
    pos_tags = [(t.text, t.pos_) for t in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return pos_tags, entities

def word_sense_disambiguation(sentence: str, target: str):
    """
    Classic Lesk algorithm for WSD using NLTK.
    Provide a sentence and a target word.
    """
    if not sentence or not target:
        return "Provide sentence and word."
    sense = lesk(sentence.split(), target)
    if sense:
        return f"{sense.name()} — {sense.definition()}"
    return "No sense found."

def simple_discourse_cohesion(sentences):
    """
    Pedagogical 'cohesion' cue: returns pairs of adjacent sentences that
    share content words (overlap > 2). Not full coreference!
    """
    if not sentences:
        return []
    overlaps = []
    for i in range(len(sentences)-1):
        a = set(w.lower() for w in sentences[i].split() if w.isalpha() and len(w) > 2)
        b = set(w.lower() for w in sentences[i+1].split() if w.isalpha() and len(w) > 2)
        inter = a & b
        if len(inter) >= 3:
            overlaps.append((i, i+1, sorted(list(inter))[:6]))
    return overlaps
