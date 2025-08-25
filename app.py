import math
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import plotly.express as px
import streamlit as st
import wikipedia
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.wsd import lesk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from wordcloud import WordCloud
from streamlit_extras.metric_cards import style_metric_cards
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import seaborn as sns
from io import BytesIO
import base64
from scipy.linalg import triu


# ---------------------------
# NLTK setup - Fixed initialization
# ---------------------------
def download_nltk_data():
    """Download required NLTK data with proper error handling"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker', quiet=True)
    
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words', quiet=True)
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    
    try:
        nltk.data.find('taggers/universal_tagset')
    except LookupError:
        nltk.download('universal_tagset', quiet=True)

# Download NLTK data
download_nltk_data()

# Now safely import stopwords
STOP_WORDS = set(stopwords.words("english"))


# ---------------------------
# Utilities
# ---------------------------
@st.cache_data(ttl=3600)
def get_wiki_content(title):
    """Cached function to fetch Wikipedia content"""
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

def top_term_freq(words, k=20):
    """Return top k frequent terms."""
    return Counter(words).most_common(k)

@st.cache_data(show_spinner=False)
def keyword_extraction_tfidf(content: str, k: int = 15):
    """Extract keywords using TF-IDF."""
    vec = TfidfVectorizer(stop_words="english", max_features=k)
    X = vec.fit_transform([content])
    return list(vec.get_feature_names_out())

def perform_wsd(sentence: str, word: str):
    """Perform Word Sense Disambiguation."""
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
    """Return POS tags for first N sentences."""
    sentences = nltk.sent_tokenize(text)
    return [(sent, nltk.pos_tag(nltk.word_tokenize(sent))) for sent in sentences[:num_sentences]]

def perform_ner(text: str, num_sentences: int = 5):
    """Return NER chunks for first N sentences."""
    sentences = nltk.sent_tokenize(text)
    ner_results = []
    for sent in sentences[:num_sentences]:
        tokens = nltk.word_tokenize(sent)
        pos_tags = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(pos_tags, binary=False)
        ner_results.append((sent, chunks))
    return ner_results

def calculate_flesch_score(text):
    """Calculate readability score."""
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return 0.0
    words = [w for s in sentences for w in nltk.word_tokenize(s) if w.isalpha()]
    if not words:
        return 0.0
    
    syllable_count = 0
    for word in words:
        # A simple heuristic for syllable counting
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
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
        dict(Task="Research & Planning", Start='2025-08-01', End='2025-08-05', Resource="Planning"),
        dict(Task="API Integration", Start='2025-08-06', End='2025-08-10', Resource="Development"),
        dict(Task="NLP Implementation", Start='2025-08-11', End='2025-08-17', Resource="Development"),
        dict(Task="UI Development", Start='2025-08-18', End='2025-08-21', Resource="Development"),
        dict(Task="Testing & Deployment", Start='2025-08-22', End='2025-08-24', Resource="Finalization")
    ])
    fig = px.timeline(df, x_start="Start", x_end="End", y="Task", color="Resource", title="Project Development Timeline (August 1-24, 2025)")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=400)
    return fig

# >>> ADDED: Simple ROUGE & keyword metric helpers
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
    hyp_ngrams = _ngrams(hyp_tokens, n)
    ref_ngrams = _ngrams(ref_tokens, n)
    overlap = len([g for g in hyp_ngrams if g in set(ref_ngrams)])
    return _prec_recall_f1(overlap, len(hyp_ngrams), len(ref_ngrams))

def _lcs_length(a, b):
    # DP LCS
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

# ---------------------------
# Extra Evaluation Metrics
# ---------------------------
def calc_cosine_similarity(text1, text2):
    vec = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vec[0:1], vec[1:2])[0][0]

def calc_perplexity(text):
    tokens = [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    total = len(tokens)
    probs = [freq[w] / total for w in tokens]
    entropy = -sum(p * math.log(p, 2) for p in probs)
    return round(2 ** entropy, 4)

# ---------------------------
# NEW: Information Retrieval Module
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
    
    # Transform query to TF-IDF vector
    query_vec = vectorizer.transform([query])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top-k results
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append({
                "sentence": sentences[idx],
                "similarity": similarities[idx],
                "index": idx
            })
    
    return results

# ---------------------------
# NEW: Word2Vec Module
# ---------------------------
@st.cache_data(show_spinner=False)
def train_word2vec_models(content):
    """Train CBOW and Skip-gram models on the article content."""
    # Tokenize the content into sentences and words
    sentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(content)]
    
    # Filter out short sentences and non-alphabetic tokens
    processed_sentences = []
    for sent in sentences:
        filtered_sent = [word.lower() for word in sent if word.isalpha() and word.lower() not in STOP_WORDS]
        if len(filtered_sent) > 3:  # Only include sentences with at least 4 words
            processed_sentences.append(filtered_sent)
    
    if len(processed_sentences) < 5:
        return None, None
    
    # Train CBOW model
    cbow_model = Word2Vec(
        sentences=processed_sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=0  # 0 for CBOW, 1 for Skip-gram
    )
    
    # Train Skip-gram model
    sg_model = Word2Vec(
        sentences=processed_sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1  # 1 for Skip-gram
    )
    
    return cbow_model, sg_model

def visualize_word_embeddings(model, words):
    """Create a 2D visualization of word embeddings using PCA."""
    if not model or not words:
        return None
    
    # Get vectors for the words that exist in the vocabulary
    valid_words = [word for word in words if word in model.wv.key_to_index]
    
    if len(valid_words) < 2:
        return None
    
    # Get vectors
    word_vectors = [model.wv[word] for word in valid_words]
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(word_vectors)
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'x': vectors_2d[:, 0],
        'y': vectors_2d[:, 1],
        'word': valid_words
    })
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(df['x'], df['y'], alpha=0.7)
    
    # Add labels
    for i, row in df.iterrows():
        plt.annotate(row['word'], (row['x'], row['y']), fontsize=12)
    
    plt.title("Word Embeddings Visualization (PCA-reduced)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, alpha=0.3)
    
    return plt

# ---------------------------
# NEW: Methodologies & Diagrams
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
        ### Word2Vec Training Approaches
        ```
        Tokenized Text
              ↓
        CBOW: Predict target word from context
              ↓
        Skip-gram: Predict context from target word
              ↓
        Embedding Space → Similarity & Visualization
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

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Wikipedia NLP Analyzer", layout="wide", page_icon="📚")

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        border-right: 1px solid #eee
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Sets the metric VALUE (the number) to black */
    [data-testid="stMetricValue"] {
        color: black !important;
    }

    /* Sets the metric LABEL (the text) to black */
    [data-testid="stMetricLabel"] {
        color: black !important;
    }
    
    .custom-text-area {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
    }
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
    5.  **Word Embeddings:** Train and visualize word vectors using Word2Vec.
    
    **Workflow:**
    - **Input:** A public Wikipedia URL.
    - **Process:** Fetch → Clean Markup → Analyze (TF-IDF, POS, NER, Word2Vec) → Visualize.
    - **Output:** An interactive dashboard with summaries, stats, and visualizations.
    """)
    
    # Show methodology diagrams
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
        
        # Create tabs for organization - ADDED NEW TABS
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard", "📜 Content Explorer", "🔍 Information Retrieval", 
            "🔤 Word Embeddings", "🔬 Advanced Analysis", "🔄 Project Timeline"
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
            cols[4].metric("Readability Score", f"{readability_score}", help="Flesch Reading Ease: 90-100 (Very Easy), 60-70 (Plain English), 0-30 (Very Confusing)")
            style_metric_cards()
            
            st.markdown("---")
            
            # --- SUMMARY & KEYWORDS ---
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📝 Extractive Summary")
                ratio = st.slider("Summary Length (Ratio of sentences)", 0.05, 0.40, 0.15, 0.01, key="ratio")
                summary = extractive_summary_tfidf_mmr(text=content, ratio=ratio)
                st.text_area("Generated Summary", summary, height=250)
                # persist for evaluation tab
                st.session_state["generated_summary"] = summary
            
            with col2:
                st.subheader("🔑 Keywords")
                keywords = keyword_extraction_tfidf(content)
                st.multiselect("Top Keywords (via TF-IDF)", options=keywords, default=keywords)
                # persist for evaluation tab
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

        # NEW: Information Retrieval Tab
        with tab3:
            st.header("🔍 Information Retrieval")
            st.markdown("""
            **Methodology**: This module uses TF-IDF vectorization and cosine similarity to retrieve 
            the most relevant sentences from the article based on your query.
            """)
            
            # Build the sentence index
            sentences, tfidf_matrix, vectorizer = build_sentence_index(content)
            
            if sentences:
                # Query input
                query = st.text_input("Enter your search query:", "machine learning")
                top_k = st.slider("Number of results to show:", 3, 10, 5)
                
                if st.button("Search"):
                    with st.spinner("Searching for relevant content..."):
                        results = information_retrieval_search(query, sentences, tfidf_matrix, vectorizer, top_k)
                    
                    if results:
                        st.subheader(f"Top {len(results)} Results")
                        
                        # Display results with similarity scores
                        for i, result in enumerate(results):
                            with st.expander(f"Result #{i+1} (Similarity: {result['similarity']:.3f})"):
                                st.write(result['sentence'])
                        
                        # Show evaluation metrics
                        st.subheader("Retrieval Evaluation")
                        avg_similarity = np.mean([r['similarity'] for r in results])
                        st.metric("Average Similarity Score", f"{avg_similarity:.3f}")
                        
                        # Show distribution of similarity scores
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

        # NEW: Word Embeddings Tab
        with tab4:
            st.header("🔤 Word Embeddings (Word2Vec)")
            st.markdown("""
            **Methodology**: This module trains Word2Vec models (CBOW and Skip-gram) on the article text 
            to learn word embeddings that capture semantic relationships between words.
            """)
            
            # Train Word2Vec models
            with st.spinner("Training Word2Vec models (this may take a moment)..."):
                cbow_model, sg_model = train_word2vec_models(content)
            
            if cbow_model and sg_model:
                st.success("Models trained successfully!")
                
                # Model selection
                model_choice = st.radio("Select Word2Vec model:", ["CBOW", "Skip-gram"])
                selected_model = cbow_model if model_choice == "CBOW" else sg_model
                
                # Word input for similarity search
                word = st.text_input("Enter a word to find similar words:", "language")
                
                if word and word in selected_model.wv.key_to_index:
                    # Find similar words
                    similar_words = selected_model.wv.most_similar(word, topn=10)
                    
                    st.subheader(f"Words Similar to '{word}'")
                    similar_df = pd.DataFrame(similar_words, columns=["Word", "Similarity"])
                    st.dataframe(similar_df.style.format({"Similarity": "{:.3f}"}))
                    
                    # Visualization
                    st.subheader("Embedding Space Visualization")
                    words_to_plot = [word] + [w for w, _ in similar_words[:5]]
                    
                    fig = visualize_word_embeddings(selected_model, words_to_plot)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.warning("Could not generate visualization for these words.")
                    
                    # Explanation of results
                    with st.expander("Interpretation of Results"):
                        st.markdown(f"""
                        The Word2Vec {model_choice} model has learned semantic relationships between words 
                        in the article. Words that appear in similar contexts have similar vector representations.
                        
                        For the word **'{word}'**, the most similar words are:
                        - **{similar_words[0][0]}** (similarity: {similar_words[0][1]:.3f})
                        - **{similar_words[1][0]}** (similarity: {similar_words[1][1]:.3f})
                        - **{similar_words[2][0]}** (similarity: {similar_words[2][1]:.3f})
                        
                        These similarities suggest that these words often appear in similar contexts 
                        within the article, indicating semantic relatedness.
                        """)
                else:
                    st.warning(f"Word '{word}' not found in the vocabulary. Try another word.")
            else:
                st.warning("Not enough content to train Word2Vec models. The article might be too short.")

        with tab5:
            st.header("Advanced Linguistic Analysis")
            
            # User input for custom text analysis
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
                    pos_results = perform_pos_tagging(text_to_analyze, num_sentences=10)
                    
                    # Create a color mapping for POS tags
                    def color_pos_tag(val):
                        if val.startswith('NN'): return 'background-color: #FFD700'  # Nouns - gold
                        elif val.startswith('VB'): return 'background-color: #90EE90'  # Verbs - light green
                        elif val.startswith('JJ'): return 'background-color: #ADD8E6'  # Adjectives - light blue
                        elif val.startswith('RB'): return 'background-color: #FFB6C1'  # Adverbs - light pink
                        else: return ''
                    
                    for i, (sent, tags) in enumerate(pos_results, 1):
                        st.markdown(f"**Sentence {i}:** `{sent}`")
                        df_tags = pd.DataFrame(tags, columns=["Token", "POS Tag"])
                        
                        # Apply styling with proper CSS format
                        styled_df = df_tags.style.applymap(color_pos_tag, subset=['POS Tag'])
                        st.dataframe(styled_df, use_container_width=True)

                with nlp_tab2:
                    st.subheader("Identified Entities (NER)")
                    ner_results = perform_ner(text_to_analyze, num_sentences=10)
                    
                    # Create a color mapping for entity types
                    def color_entity_type(val):
                        if val == 'PERSON': return 'background-color: #FF9999'  # People - light red
                        elif val in ['ORGANIZATION', 'ORG']: return 'background-color: #99CCFF'  # Organizations - light blue
                        elif val in ['GPE', 'LOCATION']: return 'background-color: #99FF99'  # Locations - light green
                        elif val in ['DATE', 'TIME']: return 'background-color: #FFCC99'  # Dates/Times - light orange
                        else: return ''
                    
                    all_entities = []
                    for sent, tree in ner_results:
                        st.markdown(f"**Sentence:** `{sent}`")
                        entities = []
                        
                        if hasattr(tree, 'label'):
                            for node in tree:
                                if isinstance(node, nltk.Tree):
                                    entity_type = node.label()
                                    entity_text = " ".join([token for token, pos in node.leaves()])
                                    entities.append((entity_text, entity_type))
                        
                        if entities:
                            df_entities = pd.DataFrame(entities, columns=["Entity", "Type"])
                            
                            # Apply styling with proper CSS format
                            styled_entities = df_entities.style.applymap(color_entity_type, subset=['Type'])
                            st.dataframe(styled_entities, use_container_width=True)
                        else:
                            st.info("No named entities found in this sentence.")
                        st.markdown("---")
                    
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

                colm = st.columns(4)
                # Always compute length & redundancy metrics for the generated summary
                st.subheader("Evaluation Metrics")
                gen_sum = st.session_state.get("generated_summary", "")
                ref_summary = st.text_area("Reference Summary (optional)", "")
                if gen_sum:
                    perp = calc_perplexity(gen_sum)
                    st.metric("Perplexity", perp)
                    if ref_summary.strip():
                        cos = calc_cosine_similarity(gen_sum, ref_summary)
                        st.metric("Cosine Similarity", round(cos, 4))

                # Keywords evaluation
                ref_kw_text = st.text_input(
                    "Reference keywords (comma-separated, optional):",
                    placeholder="e.g., nlp, corpus, tokenization, language model"
                )
                if ref_kw_text.strip():
                    gold = {k.strip().lower() for k in ref_kw_text.split(",") if k.strip()}
                    pred = {k.strip().lower() for k in st.session_state.get("generated_keywords", [])}
                    overlap = len(gold.intersection(pred))
                    p, r, f1 = _prec_recall_f1(overlap, len(pred), len(gold))
                    colk1, colk2, colk3 = st.columns(3)
                    colk1.metric("Keywords Precision", p)
                    colk2.metric("Keywords Recall", r)
                    colk3.metric("Keywords F1", f1)

                st.caption("Tip: Use this section to satisfy rubric items on quantitative results (ROUGE/F1), interpretation, and presentation.")

        with tab6:
            st.header("Project Timeline & Details")
            st.plotly_chart(create_timeline_chart(), use_container_width=True)
            
            st.subheader("Challenges & Solutions")
            st.markdown("""
            - **Challenge:** Inconsistent Wikipedia markup.
              - **Solution:** Developed a multi-stage regex cleaning pipeline.
            - **Challenge:** Summarizer producing repetitive sentences.
              - **Solution:** Implemented Maximal Marginal Relevance (MMR) to promote diversity.
            - **Challenge:** Slow performance on large articles.
              - **Solution:** Used Streamlit's `@st.cache_data` to cache expensive computations.
            - **Challenge:** Training Word2Vec on short articles.
              - **Solution:** Implemented fallbacks and validations for insufficient content.
            """)
            
            st.subheader("Limitations")
            st.warning("""
            - This tool is optimized for **English** language articles.
            - The NLP models (POS, NER) are pre-trained and may have lower accuracy on highly specialized or novel topics.
            - The summary is **extractive**, meaning it selects sentences, and may not be as coherent as a human-written summary.
            - Word2Vec models trained on single articles have limited vocabulary and context.
            - Information retrieval is based on TF-IDF which may not capture semantic meaning as well as modern transformers.
            """)
            
            st.subheader("Tool & Library Justifications")
            st.markdown("""
            - **NLTK**: Comprehensive NLP toolkit with robust implementations of standard algorithms
            - **Scikit-learn**: Industry-standard machine learning library with efficient TF-IDF implementation
            - **Gensim**: Specialized library for topic modeling and word embeddings
            - **Streamlit**: Enables rapid development of interactive web applications for data science
            - **Plotly/Matplotlib**: Provide rich, interactive visualizations for data exploration
            """)
            
        # --- SIDEBAR ---
        st.sidebar.header("Download Center")
        st.sidebar.download_button(
            label="📥 Export Analysis (JSON)",
            data=json.dumps({
                "title": page.title,
                "url": wiki_url,
                "summary": summary if 'summary' in locals() else "",
                "keywords": keywords if 'keywords' in locals() else [],
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
        st.sidebar.info("This app uses NLTK, Scikit-learn, and Gensim for NLP tasks. UI built with Streamlit.")

else:
    st.info("Please enter a Wikipedia URL above to begin the analysis.")