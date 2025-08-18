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
from wordcloud import WordCloud
from streamlit_extras.metric_cards import style_metric_cards

# ---------------------------
# NLTK setup
# ---------------------------
# Use a try-except block for robustness. This checks if data exists before downloading.
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    st.info("Performing first-time NLTK download...")
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("maxent_ne_chunker", quiet=True)
    nltk.download("words", quiet=True)

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
        dict(Task="Research", Start='2025-07-21', End='2025-07-27', Resource="Planning"),
        dict(Task="API & Preprocessing", Start='2025-07-28', End='2025-08-03', Resource="Development"),
        dict(Task="NLP Implementation", Start='2025-08-04', End='2025-08-17', Resource="Development"),
        dict(Task="UI & Deployment", Start='2025-08-18', End='2025-08-24', Resource="Finalization")
    ])
    fig = px.timeline(df, x_start="Start", x_end="End", y="Task", color="Resource", title="Project Development Timeline")
    fig.update_yaxes(autorange="reversed")
    return fig

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
    1.  **Summarization:** Condense long articles into key sentences.
    2.  **Linguistic Analysis:** Deconstruct text into its grammatical and structural components.
    3.  **Knowledge Extraction:** Identify important keywords and named entities.
    
    **Workflow:**
    - **Input:** A public Wikipedia URL.
    - **Process:** Fetch -> Clean Markup -> Analyze (TF-IDF, POS, NER) -> Visualize.
    - **Output:** An interactive dashboard with summaries, stats, and visualizations.
    """)

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

    /* --- NEW RULE ADDED HERE --- */
    /* Sets the metric LABEL (the text) to black */
    [data-testid="stMetricLabel"] {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)


# Main content input
wiki_url = st.text_input("Enter a Wikipedia URL to start", "https://en.wikipedia.org/wiki/Natural_language_processing")

if wiki_url:
    raw_title = wiki_url.strip().split("/")[-1]
    title = re.sub(r"_", " ", raw_title)
    
    with st.spinner(f"Analyzing '{title}'..."):
        page, raw_content = get_wiki_content(title)

    if page and raw_content:
        content = clean_wiki_markup(raw_content)
        
        # Create tabs for organization
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📜 Content Explorer", "🔬 Advanced Analysis", "🔄 Project Timeline"])
        
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
            
            with col2:
                st.subheader("🔑 Keywords")
                keywords = keyword_extraction_tfidf(content)
                st.multiselect("Top Keywords (via TF-IDF)", options=keywords, default=keywords)
                
                st.subheader("☁️ Word Cloud")
                filtered_words = [w for w in words if w.lower() not in STOP_WORDS and len(w) > 2]
                if filtered_words:
                    draw_wordcloud(filtered_words)
                else:
                    st.warning("Not enough words to generate a word cloud.")

        with tab2:
            st.header("Content Explorer")
            st.subheader("📑 Section Breakdown")
            if sections:
                for lvl, sec_title, sec_text in sections:
                    with st.expander(f"{sec_title}"):
                        st.write(sec_text)
            else:
                st.info("No sections detected. Showing full content.")
                st.write(content)

        with tab3:
            st.header("Advanced Linguistic Analysis")
            st.info("Analyze the first few sentences of the article or input your own text.")
            
            text_to_analyze = st.text_area(
                "Text to Analyze", 
                value=" ".join(nltk.sent_tokenize(content)[:3]),
                height=150
            )

            if text_to_analyze:
                nlp_tab1, nlp_tab2 = st.tabs(["Part-of-Speech (POS) Tagging", "Named Entity Recognition (NER)"])
                
                with nlp_tab1:
                    st.subheader("Grammatical Components (POS)")
                    pos_results = perform_pos_tagging(text_to_analyze, num_sentences=10) # Analyze all sentences in area
                    for i, (sent, tags) in enumerate(pos_results, 1):
                        df_tags = pd.DataFrame(tags, columns=["Token", "POS Tag"])
                        st.dataframe(df_tags.T, use_container_width=True)

                with nlp_tab2:
                    st.subheader("Identified Entities (NER)")
                    ner_results = perform_ner(text_to_analyze, num_sentences=10) # Analyze all sentences in area
                    all_entities = []
                    for sent, tree in ner_results:
                        for node in tree:
                            if isinstance(node, nltk.Tree):
                                entity_type = node.label()
                                entity_text = " ".join([token for token, pos in node.leaves()])
                                all_entities.append((entity_text, entity_type))
                    
                    if all_entities:
                        df_entities = pd.DataFrame(all_entities, columns=["Entity", "Type"]).drop_duplicates()
                        st.dataframe(df_entities, use_container_width=True)
                    else:
                        st.info("No named entities found in the provided text.")
            else:
                st.warning("Please enter text to analyze.")

        with tab4:
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
            """)
            
            st.subheader("Limitations")
            st.warning("""
            - This tool is optimized for **English** language articles.
            - The NLP models (POS, NER) are pre-trained and may have lower accuracy on highly specialized or novel topics.
            - The summary is **extractive**, meaning it selects sentences, and may not be as coherent as a human-written summary.
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
        st.sidebar.info("This app uses NLTK and Scikit-learn for NLP tasks. UI built with Streamlit.")

else:
    st.info("Please enter a Wikipedia URL above to begin the analysis.")