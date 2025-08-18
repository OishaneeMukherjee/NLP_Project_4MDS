import re
import matplotlib.pyplot as plt
from collections import Counter

import streamlit as st
import wikipedia
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.wsd import lesk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

# ---------------------------
# NLTK setup
# ---------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

STOP_WORDS = set(stopwords.words("english"))

# ---------------------------
# Utilities
# ---------------------------

def clean_wiki_markup(text: str) -> str:
    """Remove common Wikipedia markup (headings, refs like [1], templates)."""
    if not text:
        return ""
    t = text

    # remove references like [1], [2], [note 3]
    t = re.sub(r"\[\s*\d+\s*\]", " ", t)
    t = re.sub(r"\[\s*note\s*\d+\s*\]", " ", t, flags=re.I)

    # remove file/template curly braces crudely {{...}}
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)

    # strip heading markup '== Heading ==' lines
    t = re.sub(r"(?m)^\s*={2,6}\s*.+?\s*={2,6}\s*$", " ", t)

    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def parse_sections_from_content(content: str):
    """
    Parse sections using Wikipedia-style headings:
    == H2 ==, === H3 ===, etc.
    Returns a list of (level, title, text).
    """
    if not content:
        return []

    pattern = re.compile(r"(?m)^(={2,6})\s*(.+?)\s*\1\s*$")
    matches = list(pattern.finditer(content))

    sections = []
    if not matches:
        # no headings found – treat whole thing as Introduction
        sections.append((2, "Introduction", content.strip()))
        return sections

    # Add an implicit intro if there is text before first heading
    start0 = 0
    if matches[0].start() > 0:
        intro_text = content[:matches[0].start()].strip()
        if intro_text:
            sections.append((2, "Introduction", intro_text))

    # Walk headings
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()

        sections.append((level, title, body))

    return sections


def get_sections(page, content: str):
    """
    Prefer wikipedia.Page.sections (titles) + page.section(title) to fetch text.
    Fallback to regex parsing on raw content for completeness.
    """
    out = []
    try:
        titles = getattr(page, "sections", None) or []
        if titles:
            # Top-level titles; fetch text for each
            for title in titles:
                sec_text = page.section(title) or ""
                if sec_text.strip():
                    out.append((2, title, sec_text.strip()))
        # If nothing came back, fall back to regex parsing
        if not out:
            out = parse_sections_from_content(content)
    except Exception:
        out = parse_sections_from_content(content)

    # Deduplicate by title while preserving order
    seen = set()
    uniq = []
    for lvl, t, txt in out:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append((lvl, t, txt))
    return uniq


def sent_clean(s: str) -> str:
    s = s.strip()
    # drop leftover equals heading lines or junk
    if re.match(r"^=+", s):
        return ""
    return s


def extractive_summary_tfidf_mmr(
    text: str,
    ratio: float = 0.18,
    min_sentences: int = 5,
    max_sentences: int = 12,
    mmr_lambda: float = 0.65,
):
    """
    Extractive summary using sentence-level TF-IDF with MMR to reduce redundancy.
    - ratio: fraction of sentences to keep
    - min/max_sentences: bounds for output length
    - mmr_lambda: tradeoff between relevance and diversity (0-1)
    """
    if not text or not text.strip():
        return ""

    # 1) Clean markup for better sentence splitting & features
    cleaned = clean_wiki_markup(text)

    # 2) Sentence segmentation + basic filtering
    sentences = [sent_clean(s) for s in nltk.sent_tokenize(cleaned)]
    sentences = [s for s in sentences if s and len(s.split()) >= 8]  # drop tiny/noisy

    if not sentences:
        return ""

    # 3) TF-IDF on sentences (use unigrams+bigrams; ignore very common/rare)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=2,
    )
    try:
        tfidf = vectorizer.fit_transform(sentences)  # [n_sent, n_terms]
    except ValueError:
        # fallback when text is very short
        return " ".join(sentences[:max(min_sentences, 3)])

    # 4) Base scores = sum of TF-IDF weights per sentence
    base_scores = tfidf.sum(axis=1).A1  # flatten to 1D array

    # 5) Target sentence count
    k = max(min_sentences, min(max_sentences, int(len(sentences) * ratio)))
    if len(sentences) <= k:
        return " ".join(sentences)

    # 6) MMR selection to avoid redundancy
    sim_matrix = cosine_similarity(tfidf)  # sentence-to-sentence similarity
    selected = []
    candidates = list(range(len(sentences)))
    # pick the highest scoring sentence first
    first = max(candidates, key=lambda i: base_scores[i])
    selected.append(first)
    candidates.remove(first)

    while len(selected) < k and candidates:
        def mmr_score(i):
            relevance = base_scores[i]
            diversity = max(sim_matrix[i][j] for j in selected) if selected else 0.0
            return mmr_lambda * relevance - (1 - mmr_lambda) * diversity

        nxt = max(candidates, key=mmr_score)
        selected.append(nxt)
        candidates.remove(nxt)

    # 7) Restore original order for readability
    selected.sort()
    summary = " ".join(sentences[i] for i in selected)
    return summary


def page_stats(content: str):
    tokens = nltk.word_tokenize(content)
    words = [w.lower() for w in tokens if w.isalpha()]
    unique_words = set(words)
    read_time_min = max(1, round(len(words) / 200))  # ~200 wpm
    return len(words), len(unique_words), read_time_min, words


def draw_wordcloud(words):
    if not words:
        return
    wc = WordCloud(width=900, height=450, background_color="white").generate(" ".join(words))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def top_term_freq(words, k=20):
    freq = Counter(words)
    return freq.most_common(k)


def keyword_extraction_tfidf(content: str, k: int = 15):
    vec = TfidfVectorizer(stop_words="english", max_features=k)
    X = vec.fit_transform([content])
    return list(vec.get_feature_names_out())


def perform_wsd(sentence: str, word: str):
    sense = lesk(nltk.word_tokenize(sentence), word)
    if sense:
        return {
            "word": word,
            "synset": sense.name(),
            "definition": sense.definition(),
            "examples": sense.examples(),
        }
    return None

def perform_pos_tagging(text: str, num_sentences: int = 5):
    """Return POS tags for first N sentences of text."""
    sentences = nltk.sent_tokenize(text)
    tagged_sentences = []
    for sent in sentences[:num_sentences]:
        tokens = nltk.word_tokenize(sent)
        pos_tags = nltk.pos_tag(tokens)
        tagged_sentences.append((sent, pos_tags))
    return tagged_sentences


def perform_ner(text: str, num_sentences: int = 5):
    """Return NER chunks for first N sentences of text."""
    sentences = nltk.sent_tokenize(text)
    ner_results = []
    for sent in sentences[:num_sentences]:
        tokens = nltk.word_tokenize(sent)
        pos_tags = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(pos_tags, binary=False)
        ner_results.append((sent, chunks))
    return ner_results
# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Wikipedia NLP Analyzer", layout="wide")
st.title("📘 Wikipedia Page Analyzer")

wiki_url = st.text_input("Enter Wikipedia URL (e.g., https://en.wikipedia.org/wiki/Photosynthesis):", "")

if wiki_url:
    try:
        # Extract title from URL (last part; handle underscores)
        raw_title = wiki_url.strip().split("/")[-1]
        title = re.sub(r"_", " ", raw_title)
        page = wikipedia.page(title)
        raw_content = page.content or ""
        content = raw_content.strip()

        # 1) Preview
        st.header("👀 Preview (first 1000 chars)")
        st.write(content[:1000] + ("..." if len(content) > 1000 else ""))

        # 2) Section Breakdown (fixed)
        st.header("📑 Section Breakdown")
        sections = get_sections(page, content)
        if sections:
            for lvl, sec_title, sec_text in sections:
                # Indent by level for visual hierarchy
                indent = " " * max(0, lvl - 2)
                st.subheader(f"{indent}{sec_title}")
                snippet = sec_text[:600] + ("..." if len(sec_text) > 600 else "")
                st.write(snippet)
        else:
            st.info("No sections detected. Showing introduction only.")
            st.write(content[:1200] + ("..." if len(content) > 1200 else ""))

        # 3) Extractive Summary (TF-IDF + MMR) — fixed & stronger
        st.header("📝 Extractive Summary (TF-IDF)")
        col1, col2 = st.columns(2)
        with col1:
            ratio = st.slider("Summary ratio (fraction of sentences)", 0.05, 0.40, 0.18, 0.01)
        with col2:
            mmr_lambda = st.slider("MMR λ (relevance ↔ diversity)", 0.10, 0.95, 0.65, 0.05)

        if st.button("Generate Summary"):
            summary = extractive_summary_tfidf_mmr(
                text=content,
                ratio=ratio,
                mmr_lambda=mmr_lambda,
            )
            if not summary:
                st.warning("Could not generate a summary (content may be too short).")
            else:
                st.write(summary)

        # 4) Page Stats
        st.header("📊 Page Statistics")
        wc, uw, rt, words = page_stats(clean_wiki_markup(content))
        st.write(f"**Word Count:** {wc}")
        st.write(f"**Unique Words:** {uw}")
        st.write(f"**Estimated Read Time:** {rt} minute(s)")

        # 5) Basic Visualization
        st.header("📈 Basic Visualization")
        st.subheader("Top 20 Most Frequent Words")
        common = top_term_freq([w for w in words if w not in STOP_WORDS], k=20)
        st.table(common)

        st.subheader("Word Cloud")
        draw_wordcloud([w for w in words if w not in STOP_WORDS])

        # 6) Keyword Extraction (TF-IDF)
        st.header("🔑 Keyword Extraction (TF-IDF)")
        kw = keyword_extraction_tfidf(clean_wiki_markup(content), k=15)
        st.write(", ".join(kw) if kw else "No keywords extracted.")

        # 7) Word Sense Disambiguation (user-provided)
        st.header("🔍 Word Sense Disambiguation (WSD)")
        user_sentence = st.text_input("Enter a sentence from the page (or any sentence):", "")
        user_word = st.text_input("Target word to disambiguate (must appear in the sentence):", "")
        if st.button("Run WSD"):
            if not user_sentence or not user_word:
                st.warning("Please provide both a sentence and a target word.")
            else:
                res = perform_wsd(user_sentence, user_word)
                if res:
                    st.write(f"**Word:** {res['word']}")
                    st.write(f"**Sense (Synset):** {res['synset']}")
                    st.write(f"**Definition:** {res['definition']}")
                    st.write(f"**Examples:** {res['examples']}")
                else:
                    st.info("No sense could be determined. Try a longer sentence containing more context.")
        # 8) POS Tagging
        st.header("📝 Part-of-Speech (POS) Tagging")
        num_sent_pos = st.slider("Number of sentences to analyze (POS)", 1, 10, 3)
        pos_results = perform_pos_tagging(clean_wiki_markup(content), num_sentences=num_sent_pos)
        for sent, tags in pos_results:
            st.write(f"**Sentence:** {sent}")
            st.write(tags)

        # 9) Named Entity Recognition (NER)
        st.header("🏷️ Named Entity Recognition (NER)")
        num_sent_ner = st.slider("Number of sentences to analyze (NER)", 1, 10, 3)
        ner_results = perform_ner(clean_wiki_markup(content), num_sentences=num_sent_ner)
        for sent, tree in ner_results:
            st.write(f"**Sentence:** {sent}")
            st.text(tree.pformat())

    except Exception as e:
        st.error(f"Error fetching or processing page: {e}")
else:
    st.info("Paste a Wikipedia URL to begin.")
