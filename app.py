# app.py (Python 3.13 compatible; no gensim/scipy)

import math
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import plotly.express as px
import streamlit as st
import wikipedia
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.wsd import lesk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from wordcloud import WordCloud

# Try to import optional metric card styling (won't break if missing)
try:
    from streamlit_extras.metric_cards import style_metric_cards
except Exception:
    def style_metric_cards():
        pass

# ---------------------------
# spaCy setup for POS/NER (with safe auto-download)
# ---------------------------
import importlib
try:
    import spacy  # noqa
    try:
        nlp_spacy = spacy.load("en_core_web_sm")
    except OSError:
        # Auto-download the small English model if not available
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
        nlp_spacy = spacy.load("en_core_web_sm")
except Exception:
    nlp_spacy = None  # If spaCy truly unavailable, we handle gracefully later

# ---------------------------
# NLTK setup - Fixed initialization (quiet & robust)
# ---------------------------
def download_nltk_data():
    """Download required NLTK data with proper error handling."""
    need = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/wordnet", "wordnet"),
        ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
        ("corpora/words", "words"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/universal_tagset", "universal_tagset"),
    ]
    for path, pkg in need:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

download_nltk_data()
STOP_WORDS = set(stopwords.words("english"))

# ---------------------------
# Utilities
# ---------------------------
@st.cache_data(ttl=3600)
def get_wiki_content(title):
    """Cached function to fetch Wikipedia content."""
    try:
        page = wikipedia.page(title, auto_suggest=False, redirect=True)
        return page, page.content
    except wikipedia.DisambiguationError as e:
        st.error(f"Disambiguation needed: {e.options[:5]}...")
        return None, None
    except wikipedia.PageError:
        st.error("Page not found - check the URL")
        return None, None
    except Exception as e:
        st.error(f"Error fetching page: {str(e)}")
        return None, None

def clean_wiki_markup(text: str) -> str:
    """Remove common Wikipedia markup (headings, refs like [1], templates)."""
    if not text:
        return ""
    t = text
    t = re.sub(r"\[\s*\d+\s*\]", " ", t)
    t = re.sub(r"\[\s*note\s*\d+\s*\]", " ", t, flags=re.I)
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)
    t = re.sub(r"(?m)^\s*={2,6}\s*.+?\s*={2,6}\s*$", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def parse_sections_from_content(content: str):
    """Parse sections using Wikipedia-style headings."""
    if not content:
        return []
    pattern = re.compile(r"(?m)^(={2,6})\s*(.+?)\s*\1\s*$")
    matches = list(pattern.finditer(content))
    sections = []
    if not matches:
        sections.append((2, "Introduction", content.strip()))
        return sections
    if matches[0].start() > 0:
        intro_text = content[:matches[0].start()].strip()
        if intro_text:
            sections.append((2, "Introduction", intro_text))
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        sections.append((level, title, body))
    seen = set()
    return [(lvl, t, txt) for lvl, t, txt in sections if not (t.lower() in seen or seen.add(t.lower()))]

def get_sections(page, content: str):
    """Extract sections with fallback to regex parsing."""
    try:
        titles = getattr(page, "sections", None) or []
        if titles:
            return [(2, title, (page.section(title) or "").strip()) for title in titles if page.section(title)]
        return parse_sections_from_content(content)
    except Exception:
        return parse_sections_from_content(content)

def sent_clean(s: str) -> str:
    s = s.strip()
    return "" if re.match(r"^=+", s) else s

@st.cache_data(show_spinner=False)
def extractive_summary_tfidf_mmr(text: str, ratio: float = 0.18, min_sentences: int = 5, max_sentences: int = 12, mmr_lambda: float = 0.65):
    """Extractive summary using sentence-level TF-IDF with MMR."""
    cleaned = clean_wiki_markup(text)
    sentences = [sent_clean(s) for s in nltk.sent_tokenize(cleaned)]
    sentences = [s for s in sentences if s and len(s.split()) >= 8]
    if not sentences:
        return ""
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.85, min_df=2)
        tfidf = vectorizer.fit_transform(sentences)
        base_scores = tfidf.sum(axis=1).A1
        k = max(min_sentences, min(max_sentences, int(len(sentences) * ratio)))
        if len(sentences) <= k:
            return " ".join(sentences)
        sim_matrix = cosine_similarity(tfidf)
        selected = []
        candidates = list(range(len(sentences)))
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
        selected.sort()
        return " ".join(sentences[i] for i in selected)
    except ValueError:
        return " ".join(sentences[:max(min_sentences, 3)])

def page_stats(content: str):
    """Calculate basic page statistics."""
    tokens = nltk.word_tokenize(content)
    words = [w.lower() for w in tokens if w.isalpha()]
    unique_words = set(words)
    read_time_min = max(1, round(len(words) / 200))
    return len(words), len(unique_words), read_time_min, words

def draw_wordcloud(words):
    """Generate and display word cloud."""
    if not words:
        return
    wc = WordCloud(width=900, height=450, background_color="white").generate(" ".join(words))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)

@st.cache_data(show_spinner=False)
def keyword_extraction_tfidf(content: str, k: int = 15):
    """Extract keywords using TF-IDF."""
    vec = TfidfVectorizer(stop_words="english", max_features=k)
    _ = vec.fit_transform([content])
    return list(vec.get_feature_names_out())

def perform_wsd(sentence: str, word: str):
    """Perform Word Sense Disambiguation (Lesk)."""
    sense = lesk(nltk.word_tokenize(sentence), word)
    if sense:
        return {
            "word": word,
            "synset": sense.name(),
            "definition": sense.definition(),
            "examples": sense.examples(),
        }
    return None

def calculate_flesch_score(text):
    """Calculate readability score (Flesch Reading Ease)."""
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return 0.0
    words = [w for s in sentences for w in nltk.word_tokenize(s) if w.isalpha()]
    if not words:
        return 0.0
    syllable_count = 0
    for word in words:
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word and word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        syllable_count += count
    return round(206.835 - 1.015*(len(words)/len(sentences)) - 84.6*(syllable_count/len(words)), 1)

def create_timeline_chart():
    """Create development timeline visualization."""
    df = pd.DataFrame([
        dict(Task="Research & Planning", Start='2025-08-17', End='2025-08-18', Resource="Planning"),
        dict(Task="NLP Implementation", Start='2025-08-19', End='2025-08-22', Resource="Development"),
        dict(Task="UI Development", Start='2025-08-23', End='2025-08-25', Resource="Development"),
    ])

    fig = px.timeline(df, x_start="Start", x_end="End", y="Task", color="Resource",
                      title="Project Development Timeline (August 17-25, 2025)")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=400)
    return fig


# >>> ROUGE & metrics helpers
def _tokenize_words(text: str):
    return [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]

def _ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)]))

def _prec_recall_f1(overlap, pred, gold):
    prec = overlap / pred if pred > 0 else 0.0
    rec  = overlap / gold if gold > 0 else 0.0
    f1 = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)

def rouge_n(hypothesis: str, reference: str, n: int = 1):
    hyp_tokens = _tokenize_words(hypothesis)
    ref_tokens = _tokenize_words(reference)
    hyp_ngrams = Counter(_ngrams(hyp_tokens, n))
    ref_ngrams = Counter(_ngrams(ref_tokens, n))
    overlap = sum((hyp_ngrams & ref_ngrams).values())  # true multiset overlap
    return _prec_recall_f1(overlap, sum(hyp_ngrams.values()), sum(ref_ngrams.values()))


def _lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def rouge_l(hypothesis: str, reference: str):
    hyp_tokens = _tokenize_words(hypothesis)
    ref_tokens = _tokenize_words(reference)
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    return _prec_recall_f1(lcs, len(hyp_tokens), len(ref_tokens))

def summary_redundancy_ratio(text: str):
    """Bigram redundancy proxy: repeated bigrams / total bigrams"""
    toks = _tokenize_words(text)
    bigs = _ngrams(toks, 2)
    if not bigs:
        return 0.0
    total = len(bigs)
    uniq = len(set(bigs))
    return round(1 - (uniq/total), 4)

def calc_cosine_similarity(text1, text2):
    vec = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vec[0:1], vec[1:2])[0][0]

def calc_perplexity(text, smoothing=1e-8):
    """Calculate perplexity based on unigram LM with Laplace smoothing."""
    tokens = [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    total = sum(freq.values())
    # convert to probabilities with smoothing
    probs = {w: (freq[w] + smoothing) / (total + smoothing * len(freq)) for w in freq}
    # average log probability
    log_prob = 0.0
    for w in tokens:
        log_prob += math.log(probs.get(w, smoothing), 2)
    entropy = -log_prob / len(tokens)
    return round(2 ** entropy, 4)


# ---------------------------
# Information Retrieval
# ---------------------------
@st.cache_data(show_spinner=False)
def build_sentence_index(content):
    """Create a search index of sentences with TF-IDF vectors."""
    cleaned = clean_wiki_markup(content)
    sentences = [sent_clean(s) for s in nltk.sent_tokenize(cleaned)]
    sentences = [s for s in sentences if s and len(s.split()) >= 5]
    if not sentences:
        return None, None, None
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return sentences, tfidf_matrix, vectorizer

def information_retrieval_search(query, sentences, tfidf_matrix, vectorizer, top_k=5):
    """Search for relevant sentences using TF-IDF cosine similarity."""
    if not sentences:
        return []
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append({
                "sentence": sentences[idx],
                "similarity": float(similarities[idx]),
                "index": int(idx),
            })
    return results

# ---------------------------
# Co-occurrence Embeddings (replacement for Word2Vec)
# ---------------------------
@st.cache_data(show_spinner=False)
def build_cooccurrence_embeddings(content, window_size=4, min_freq=2, max_vocab=5000):
    """
    Build simple word co-occurrence vectors:
      1) tokenize text into words
      2) build a co-occurrence matrix within a sliding window
      3) return a term->vector mapping (rows of the co-occurrence matrix)
    """
    # Tokenize sentences & words
    sents = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(content)]
    tokens = []
    for s in sents:
        toks = [w.lower() for w in s if w.isalpha() and w.lower() not in STOP_WORDS]
        if toks:
            tokens.extend(toks)

    if len(tokens) < 100:
        return {}, None  # not enough context

    # Build vocabulary by frequency
    freq = Counter(tokens)
    vocab = [w for w, c in freq.most_common(max_vocab) if c >= min_freq]
    if len(vocab) < 10:
        return {}, None

    index = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    # Use an efficient sparse-like build (but with numpy since V is capped)
    mat = np.zeros((V, V), dtype=np.float32)

    # Sliding window co-occurrence
    for i, w in enumerate(tokens):
        if w not in index:
            continue
        wi = index[w]
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        for j in range(start, end):
            if j == i:
                continue
            c = tokens[j]
            if c in index:
                cj = index[c]
                mat[wi, cj] += 1.0

    # Row-normalize (to mitigate frequency effects)
    row_sums = mat.sum(axis=1, keepdims=True) + 1e-9
    mat = mat / row_sums

    # Term -> vector
    term_vectors = {w: mat[index[w]] for w in vocab}
    return term_vectors, vocab

def most_similar_terms(term_vectors, target, topn=10):
    """Find most similar terms to target using cosine similarity."""
    target = target.lower()
    if target not in term_vectors:
        return []
    tv = term_vectors[target].reshape(1, -1)
    sims = []
    for w, v in term_vectors.items():
        if w == target:
            continue
        sim = float(cosine_similarity(tv, v.reshape(1, -1))[0][0])
        sims.append((w, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:topn]

def visualize_term_embeddings(term_vectors, words):
    """PCA 2D projection of selected word vectors."""
    valid = [w for w in words if w in term_vectors]
    if len(valid) < 2:
        return None
    X = np.stack([term_vectors[w] for w in valid], axis=0)
    pca = PCA(n_components=2)
    pts = pca.fit_transform(X)
    df = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "word": valid})
    plt.figure(figsize=(10, 8))
    plt.scatter(df["x"], df["y"], alpha=0.7)
    for _, r in df.iterrows():
        plt.annotate(r["word"], (r["x"], r["y"]), fontsize=12)
    plt.title("Co-occurrence Embeddings (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    return plt

# ---------------------------
# Methodology & Diagrams
# ---------------------------
def show_methodology_diagrams():
    """Display methodology diagrams for the project."""
    st.subheader("Methodology & System Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Text Processing Pipeline
        ```
        Wikipedia Content
              ↓
        Markup Cleaning
              ↓
        Sentence Tokenization
              ↓
        TF-IDF Vectorization
              ↓
        MMR-based Selection → Summary
              ↓
        Evaluation (ROUGE, Cosine)
        ```
        """)
    with col2:
        st.markdown("""
        ### Embeddings (Co-occurrence)
        ```
        Tokenized Text
              ↓
        Sliding Window Co-occurrence
              ↓
        Row-normalized Term Vectors
              ↓
        Similarity & PCA Visualization
        ```
        """)
    st.markdown("""
    ### Information Retrieval Process
    ```
    User Query → TF-IDF Vectorization
                      ↓
    Cosine Similarity with Sentence Vectors
                      ↓
    Ranked Results → Top-K Retrieval
    ```
    """)

def diversity_score(text):
    """Simple diversity: unique bigrams / total bigrams"""
    toks = _tokenize_words(text)
    bigs = _ngrams(toks, 2)
    if not bigs:
        return 0.0
    return round(len(set(bigs)) / len(bigs), 4)


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Wikipedia NLP Analyzer", layout="wide", page_icon="📚")

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container { background: #f8f9fa }
    .sidebar .sidebar-content { background: #ffffff; border-right: 1px solid #eee }
    h1, h2, h3 { color: #2c3e50; }
    .stMetric {
        background-color: #ffffff; border-radius: 10px; padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricValue"] { color: black !important; }
    [data-testid="stMetricLabel"] { color: black !important; }
    .custom-text-area { border: 2px solid #4CAF50; border-radius: 5px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# Main title
st.title("📚 Wikipedia NLP Analyzer")

# Problem definition section
with st.expander("📌 Problem Definition & Methodology", expanded=False):
    st.markdown("""
    ### The Challenge: Information Overload
    Wikipedia is a vast repository of knowledge, but its dense articles can be challenging to digest and analyze quickly. This tool aims to solve this by providing automated NLP-driven insights.

    **Key Goals:**
    1.  **Summarization:** Condense long articles into key sentences using TF-IDF with MMR.
    2.  **Linguistic Analysis:** Deconstruct text into its grammatical and structural components.
    3.  **Knowledge Extraction:** Identify important keywords and named entities.
    4.  **Information Retrieval:** Enable semantic search within articles.
    5.  **Embeddings:** Learn simple word vectors via co-occurrence (Python 3.13 friendly).

    **Workflow:**
    - **Input:** A public Wikipedia URL.
    - **Process:** Fetch → Clean Markup → Analyze (TF-IDF, POS/NER, Co-occurrence) → Visualize.
    - **Output:** An interactive dashboard with summaries, stats, and visualizations.
    """)
    show_methodology_diagrams()

# Main content input
wiki_url = st.text_input("Enter a Wikipedia URL to start", "https://en.wikipedia.org/wiki/Natural_language_processing")

if wiki_url:
    raw_title = wiki_url.strip().split("/")[-1]
    title = re.sub(r"_", " ", raw_title)

    with st.spinner(f"Analyzing '{title}'..."):
        page, raw_content = get_wiki_content(title)

    if page and raw_content:
        content = clean_wiki_markup(raw_content)

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard", "📜 Content Explorer", "🔍 Information Retrieval",
            "🔤 Embeddings (Co-occurrence)", "🔬 Advanced Analysis", "🔄 Project Timeline"
        ])

        with tab1:
            st.header(f"Analysis Dashboard for: *{page.title}*")

            # --- METRICS ---
            st.subheader("Key Metrics")
            wc, uw, rt, words = page_stats(content)
            sections = get_sections(page, raw_content)
            readability_score = calculate_flesch_score(content)

            cols = st.columns(5)
            cols[0].metric("Word Count", f"{wc:,}")
            cols[1].metric("Unique Words", f"{uw:,}")
            cols[2].metric("Read Time (mins)", f"~{rt}")
            cols[3].metric("Sections", len(sections))
            cols[4].metric("Readability Score", f"{readability_score}",
                           help="Flesch Reading Ease: 90-100 (Very Easy), 60-70 (Plain English), 0-30 (Very Confusing)")
            style_metric_cards()

            st.markdown("---")

            # --- SUMMARY & KEYWORDS ---
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📝 Extractive Summary")
                ratio = st.slider("Summary Length (Ratio of sentences)", 0.05, 0.40, 0.15, 0.01, key="ratio")
                summary = extractive_summary_tfidf_mmr(text=content, ratio=ratio)
                st.text_area("Generated Summary", summary, height=250)
                st.session_state["generated_summary"] = summary

            with col2:
                st.subheader("🔑 Keywords")
                keywords = keyword_extraction_tfidf(content)
                st.multiselect("Top Keywords (via TF-IDF)", options=keywords, default=keywords)
                st.session_state["generated_keywords"] = keywords

                st.subheader("☁️ Word Cloud")
                filtered_words = [w for w in words if w.lower() not in STOP_WORDS and len(w) > 2]
                if filtered_words:
                    draw_wordcloud(filtered_words)
                else:
                    st.warning("Not enough words to generate a word cloud.")

        with tab2:
            st.header("Content Explorer")
            st.subheader("📑 Section Breakdown")
            sections = get_sections(page, raw_content)
            if sections:
                for lvl, sec_title, sec_text in sections:
                    with st.expander(f"{sec_title}"):
                        st.write(sec_text)
            else:
                st.info("No sections detected. Showing full content.")
                st.write(content)

        with tab3:
            st.header("🔍 Information Retrieval")
            st.markdown("""
            **Methodology**: This module uses TF-IDF vectorization and cosine similarity to retrieve 
            the most relevant sentences from the article based on your query.
            """)
            sentences, tfidf_matrix, vectorizer = build_sentence_index(content)
            if sentences:
                query = st.text_input("Enter your search query:", "machine learning")
                top_k = st.slider("Number of results to show:", 3, 10, 5)
                if st.button("Search"):
                    with st.spinner("Searching for relevant content..."):
                        results = information_retrieval_search(query, sentences, tfidf_matrix, vectorizer, top_k)
                    if results:
                        st.subheader(f"Top {len(results)} Results")
                        for i, result in enumerate(results):
                            with st.expander(f"Result #{i+1} (Similarity: {result['similarity']:.3f})"):
                                st.write(result['sentence'])
                        st.subheader("Retrieval Evaluation")
                        avg_similarity = float(np.mean([r['similarity'] for r in results]))
                        st.metric("Average Similarity Score", f"{avg_similarity:.3f}")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        similarities = [r['similarity'] for r in results]
                        ax.bar(range(1, len(similarities)+1), similarities)
                        ax.set_xlabel("Result Rank")
                        ax.set_ylabel("Cosine Similarity")
                        ax.set_title("Similarity Scores by Result Rank")
                        st.pyplot(fig)
                    else:
                        st.warning("No relevant results found. Try a different query.")
            else:
                st.warning("Not enough content to build a search index.")

        with tab4:
            st.header("🔤 Embeddings (Co-occurrence)")
            st.markdown("""
            **Methodology**: A simple, Python-3.13-friendly alternative to Word2Vec.
            We build a word co-occurrence matrix using a sliding window, normalize rows to get vectors, 
            then use cosine similarity + PCA to explore related words and visualize them.
            """)
            with st.spinner("Building co-occurrence embeddings..."):
                term_vectors, vocab = build_cooccurrence_embeddings(content, window_size=4, min_freq=2, max_vocab=4000)

            if term_vectors:
                st.success(f"Vocabulary size: {len(term_vectors):,}")
                word = st.text_input("Enter a word to find similar words:", "language")
                if word:
                    sims = most_similar_terms(term_vectors, word, topn=10)
                    if sims:
                        st.subheader(f"Words similar to '{word}'")
                        df_sims = pd.DataFrame(sims, columns=["Word", "Similarity"])
                        st.dataframe(df_sims.style.format({"Similarity": "{:.3f}"}), use_container_width=True)

                        st.subheader("Embedding Space Visualization")
                        words_to_plot = [word] + [w for w, _ in sims[:5]]
                        fig = visualize_term_embeddings(term_vectors, words_to_plot)
                        if fig:
                            st.pyplot(fig)
                        else:
                            st.warning("Not enough valid words to plot.")
                    else:
                        st.warning(f"'{word}' not in vocabulary or no similar terms found. Try another word.")
            else:
                st.warning("Not enough content to build embeddings (article may be too short).")

        with tab5:
            st.header("Advanced Linguistic Analysis")

            # Choose text source
            analysis_option = st.radio(
                "Choose text to analyze:",
                ["Use Wikipedia article content", "Enter custom text"],
                horizontal=True
            )

            if analysis_option == "Use Wikipedia article content":
                text_to_analyze = " ".join(nltk.sent_tokenize(content)[:3])
                st.info(f"Analyzing first few sentences from: {page.title}")
            else:
                text_to_analyze = st.text_area(
                    "Enter your own text to analyze:",
                    value="Apple Inc. is planning to open a new store in Paris next month. Elon Musk announced the new Tesla model yesterday.",
                    height=100,
                    help="Enter any text you want to analyze for POS tagging and NER"
                )

            if text_to_analyze:
                st.markdown(f"**Text to analyze:** {text_to_analyze}")

                nlp_tab1, nlp_tab2 = st.tabs(["Part-of-Speech (POS) Tagging", "Named Entity Recognition (NER)"])

                with nlp_tab1:
                    st.subheader("Grammatical Components (POS)")
                    if nlp_spacy is not None:
                        doc = nlp_spacy(text_to_analyze)
                        rows = [(t.text, t.pos_, t.tag_) for t in doc]
                        df_tags = pd.DataFrame(rows, columns=["Token", "POS", "Tag"])
                        st.dataframe(df_tags, use_container_width=True)
                    else:
                        st.warning("spaCy not available. Install spaCy to enable fast POS tagging.")

                with nlp_tab2:
                    st.subheader("Identified Entities (NER)")
                    if nlp_spacy is not None:
                        doc = nlp_spacy(text_to_analyze)
                        ents = [(ent.text, ent.label_) for ent in doc.ents]
                        if ents:
                            df_entities = pd.DataFrame(ents, columns=["Entity", "Type"])
                            st.dataframe(df_entities, use_container_width=True)
                        else:
                            st.info("No named entities found in this text.")
                    else:
                        st.warning("spaCy not available. Install spaCy to enable NER.")

            else:
                st.warning("Please enter text to analyze.")

            # Evaluation & Metrics section
            st.markdown("---")
            with st.expander("📈 Evaluation & Metrics (ROUGE, Keywords, Ratios)", expanded=False):
                gen_sum = st.session_state.get("generated_summary", "")
                st.text_area("Generated Summary (for reference)", gen_sum, height=120, disabled=True)

                ref_summary = st.text_area(
                    "Paste a human-written *reference* summary (optional, for ROUGE):",
                    height=150,
                    placeholder="Paste gold summary here to compute ROUGE-1/2/L..."
                )

                st.subheader("Evaluation Metrics")
                if gen_sum:
                    # --- Improved Evaluation Metrics ---
                    perp = calc_perplexity(gen_sum)
                    st.metric("Perplexity (unigram LM)", perp, help="Lower is better; <100 usually indicates good fluency")

                    if ref_summary.strip():
                        cos = calc_cosine_similarity(gen_sum, ref_summary)
                        r1 = rouge_n(gen_sum, ref_summary, n=1)
                        r2 = rouge_n(gen_sum, ref_summary, n=2)
                        rl = rouge_l(gen_sum, ref_summary)
                        st.metric("Cosine Similarity", round(cos, 4), help="Higher is better; closer to 1 means more overlap")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("ROUGE-1 F1", r1[2])
                        c2.metric("ROUGE-2 F1", r2[2])
                        c3.metric("ROUGE-L F1", rl[2])

                    # Redundancy & Diversity
                    redun = summary_redundancy_ratio(gen_sum)
                    div = diversity_score(gen_sum)
                    c4, c5 = st.columns(2)
                    c4.metric("Redundancy (bigrams)", redun, help="Lower = less repetition")
                    c5.metric("Diversity Score", div, help="Higher = more diverse summary")


                

                st.caption("Use this section for quantitative results (ROUGE/F1), interpretation, and presentation.")

        with tab6:
            st.header("Project Timeline & Details")
            st.plotly_chart(create_timeline_chart(), use_container_width=True)

            st.subheader("Challenges & Solutions")
            st.markdown("""
            - **Challenge:** Inconsistent Wikipedia markup.
              - **Solution:** Multi-stage regex cleaning pipeline.
            - **Challenge:** Summarizer produced repetitive sentences.
              - **Solution:** Implemented Maximal Marginal Relevance (MMR) to promote diversity.
            - **Challenge:** Slow performance on large articles.
              - **Solution:** Cached expensive computations with `@st.cache_data`.
            - **Challenge:** Word2Vec/Scipy incompatibility on Python 3.13.
              - **Solution:** Replaced with co-occurrence-based embeddings (no SciPy/Gensim).
            """)

            st.subheader("Limitations")
            st.warning("""
            - Optimized for **English** articles.
            - Extractive summary may be less coherent than human-written abstractive summaries.
            - Co-occurrence embeddings are local to the article and vocabulary is limited.
            - TF-IDF retrieval is lexical; it may miss deep semantics vs. transformer-based models.
            """)

            st.subheader("Tool & Library Justifications")
            st.markdown("""
            - **NLTK**: Tokenization and classic NLP utilities.
            - **spaCy**: Fast, production-grade POS/NER.
            - **Scikit-learn**: TF-IDF, cosine similarity, PCA.
            - **Streamlit**: Rapid interactive dashboarding.
            - **Plotly/Matplotlib**: Rich visualizations.
            """)

        # --- SIDEBAR ---
        st.sidebar.header("Download Center")
        summary = st.session_state.get("generated_summary", "")
        keywords = st.session_state.get("generated_keywords", [])
        st.sidebar.download_button(
            label="📥 Export Analysis (JSON)",
            data=json.dumps({
                "title": page.title,
                "url": wiki_url,
                "summary": summary,
                "keywords": keywords,
                "stats": {
                    "word_count": wc,
                    "unique_words": uw,
                    "read_time_minutes": rt,
                    "readability_score": readability_score,
                    "sections": len(sections)
                }
            }, indent=2),
            file_name=f"analysis_{title.replace(' ', '_')}.json",
            mime="application/json"
        )
        st.sidebar.markdown("---")
        st.sidebar.info("This app uses NLTK, spaCy, and scikit-learn for NLP tasks. UI built with Streamlit.")

else:
    st.info("Please enter a Wikipedia URL above to begin the analysis.")
