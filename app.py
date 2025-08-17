import streamlit as st
import wikipedia
import nltk
import spacy
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.parse import urlparse, unquote
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Summarizer (HuggingFace BART model)
try:
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
except Exception as e:
    st.error(f"Summarization model not available: {e}")
    summarizer = None

    
# Extended Lesk (better WSD)
try:
    from pywsd.lesk import simple_lesk
except:
    st.warning("pywsd not installed. Run: pip install pywsd")

# NLTK downloads
nltk.download('punkt')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm") 
# Load SpaCy model (for semantic similarity)
try:
    nlp = spacy.load("en_core_web_md")
except:
    st.error("SpaCy model not found. Run: python -m spacy download en_core_web_md")


# ---- TF-IDF extractive summarizer (entire page, no chunking) ----
def tfidf_extractive_summary(full_text: str, ratio: float = 0.25, max_chars: int = 4000) -> str:
    """
    Summarize full_text by selecting top-K sentences via sentence-level TF-IDF scores.
    - ratio: fraction of sentences to keep (0.05–0.6 typical)
    - max_chars: hard safety cap for Streamlit rendering
    """
    if not full_text or not full_text.strip():
        return ""

    # 1) Sentence segmentation (fast & robust)
    sentences = nltk.sent_tokenize(full_text)
    if len(sentences) == 0:
        return ""

    # 2) Build sentence-level TF-IDF
    # Each sentence is treated as a "document". Score is sum of tf-idf weights.
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        X = vectorizer.fit_transform(sentences)  # shape: [n_sentences, n_terms]
    except ValueError:
        # happens if text is too small or all stopwords
        return "Content too short to summarize."

    # 3) Score sentences and pick top-K by score
    # Sum TF-IDF weights across terms for each sentence
    scores = X.sum(axis=1)  # sparse matrix -> column vector
    # Convert to flat Python list of floats
    scores = [float(scores[i, 0]) for i in range(scores.shape[0])]

    k = max(1, int(len(sentences) * max(min(ratio, 1.0), 0.01)))
    # indices of top-K sentences by score
    ranked_idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:k]
    # restore original order for readability
    ranked_idx.sort()

    summary = " ".join(sentences[i] for i in ranked_idx).strip()

    # 4) Safety cap for UI stability
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."

    return summary


# # === Extractive (TF-IDF) summarizer helpers ===
# from nltk.stem import WordNetLemmatizer
# import math

# lemmatizer = WordNetLemmatizer()

# def _freq_matrix_spacy(sentences, nlp_obj):
#     freq_matrix = {}
#     stop_words = nlp_obj.Defaults.stop_words

#     for sent in sentences:
#         freq_table = {}
#         # keep only alpha-numeric tokens; lower; lemmatize
#         words = [t.text.lower() for t in sent if t.text.isalnum()]
#         for w in words:
#             w = lemmatizer.lemmatize(w)
#             if w in stop_words or not w:
#                 continue
#             freq_table[w] = freq_table.get(w, 0) + 1

#         # use a short, stable key for the sentence
#         key = sent.text[:15]
#         freq_matrix[key] = freq_table

#     return freq_matrix

# def _tf_matrix_from_freq(freq_matrix):
#     tf_matrix = {}
#     for s_key, ftab in freq_matrix.items():
#         tf_tab = {}
#         denom = max(len(ftab), 1)
#         for w, c in ftab.items():
#             tf_tab[w] = c / denom
#         tf_matrix[s_key] = tf_tab
#     return tf_matrix

# def _sentences_per_word(freq_matrix):
#     spw = {}
#     for _, ftab in freq_matrix.items():
#         for w in ftab.keys():
#             spw[w] = spw.get(w, 0) + 1
#     return spw

# def _idf_matrix_from(freq_matrix, spw, total_sents):
#     idf_matrix = {}
#     for s_key, ftab in freq_matrix.items():
#         idf_tab = {}
#         for w in ftab.keys():
#             # add 1 smoothing to avoid div-by-zero
#             idf_tab[w] = math.log10((total_sents + 1) / float(spw.get(w, 1)))
#         idf_matrix[s_key] = idf_tab
#     return idf_matrix

# def _tfidf_matrix(tf_mat, idf_mat):
#     tfidf = {}
#     for (s_key, tf_tab) in tf_mat.items():
#         idf_tab = idf_mat.get(s_key, {})
#         tfidf_tab = {}
#         for w, tfv in tf_tab.items():
#             tfidf_tab[w] = tfv * idf_tab.get(w, 0.0)
#         tfidf[s_key] = tfidf_tab
#     return tfidf

# def _score_sentences(tfidf):
#     scores = {}
#     for s_key, tab in tfidf.items():
#         if not tab:
#             continue
#         scores[s_key] = sum(tab.values()) / len(tab)
#     return scores

# def _average_score(scores):
#     if not scores:
#         return 0.0
#     return sum(scores.values()) / len(scores)

# def extractive_tfidf_summary(full_text, nlp_obj, ratio=0.25, boost=1.3, max_chars=4000):
#     """
#     ratio: approx fraction of sentences to keep (0<ratio<=1)
#     boost: threshold multiplier (higher = shorter summary)
#     max_chars: final safe cap to avoid UI crashes
#     """
#     if not full_text or not full_text.strip():
#         return ""

    # Process with SpaCy (robust to long docs)
    doc = nlp_obj(full_text)
    sentences = list(doc.sents)
    if not sentences:
        return ""

    freq = _freq_matrix_spacy(sentences, nlp_obj)
    tfm  = _tf_matrix_from_freq(freq)
    spw  = _sentences_per_word(freq)
    idfm = _idf_matrix_from(freq, spw, len(sentences))
    tfidf = _tfidf_matrix(tfm, idfm)
    s_scores = _score_sentences(tfidf)
    avg = _average_score(s_scores)

    # dynamic threshold from ratio + boost
    # sort sentences by score, pick top-k ~ ratio
    k = max(1, int(len(sentences) * min(max(ratio, 0.05), 1.0)))
    ranked = sorted(
        ((i, s, s_scores.get(s.text[:15], 0.0)) for i, s in enumerate(sentences)),
        key=lambda x: x[2],
        reverse=True
    )
    top_idx = set(i for i, _, _ in ranked[:k])

    # keep original order
    selected = [sentences[i].text for i in range(len(sentences)) if i in top_idx]
    summary = " ".join(selected).strip()

    # safe truncate for Streamlit
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."

    return summary



# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(page_title="Wikipedia NLP Project", layout="wide")

st.title("Wikipedia NLP Explorer")
st.write("Analyze any Wikipedia page by pasting its URL below.")

# ---------------------------
# Extract title from Wikipedia URL
# ---------------------------
def extract_title_from_url(url):
    try:
        path = urlparse(url).path  # /wiki/Photosynthesis
        title = path.split("/wiki/")[-1]
        return unquote(title.replace("_", " "))
    except:
        return None

# ---------------------------

# Input field for Wikipedia URL
# ---------------------------
wiki_url = st.text_input("Enter Wikipedia URL:", "")

# Fetch Wikipedia Page
# ---------------------------
if st.button("Fetch Page"):
    title = extract_title_from_url(wiki_url)
    if title:
        try:
            # Normalize title
            query = title.strip()

            # Search for possible matches
            search_results = wikipedia.search(query)
            if not search_results:
                st.error(f"No results found for: {query}")
            else:
                # Prefer exact match if available
                exact_match = next((r for r in search_results if r.lower() == query.lower()), None)

                if exact_match:
                    best_match = exact_match
                else:
                    best_match = search_results[0]  # fallback

                page = wikipedia.page(best_match)

                st.session_state['page_title'] = best_match
                st.session_state['page_content'] = page.content

                st.success(f"Fetched page: {best_match}")
    
    #         # Generate summary using full content
    #         if summarizer:
    #             try:
    #                 sentences = nltk.sent_tokenize(st.session_state['page_content'])
    #                 chunks, current_chunk = [], []
    #                 max_chunk_words = 800  # safe limit for BART

    #                 for sentence in sentences:
    #                     if sum(len(s.split()) for s in current_chunk) + len(sentence.split()) <= max_chunk_words:
    #                         current_chunk.append(sentence)
    #                     else:
    #                         chunks.append(" ".join(current_chunk))
    #                         current_chunk = [sentence]
    #                 if current_chunk:
    #                     chunks.append(" ".join(current_chunk))

    #                 summaries = []
    #                 for chunk in chunks:
    #                     res = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
    #                     summaries.append(res[0]['summary_text'])

    #                 final_summary = " ".join(summaries)

    #                 st.subheader("Generated Summary:")
    #                 st.write(final_summary.strip())

    #             except Exception as e:
    #                 st.error(f"Error during summarization: {e}")
    #         else:
    #             st.warning("Summarizer not initialized.")
        except Exception as e:
            st.error(f"Error fetching page: {e}")
    else:
        st.warning("Please enter a valid Wikipedia URL.")


# ---------------------------
# NLP Tasks (only if content exists)
# ---------------------------
if 'page_content' in st.session_state:
    text = st.session_state['page_content']
    limited_text = text[:5000]  # limit for efficiency

    # Preview
    with st.expander("Preview Article Content"):
        st.write(limited_text[:1000] + "...")
        st.info("Showing first 1000 characters.")


    
# ---------------------------
# Wikipedia Page Summarization
# ---------------------------
with st.expander("Wikipedia Page Summarization"):
    st.write("Generate a fast extractive summary of the entire article using sentence-level TF-IDF.")

    col_a, col_b = st.columns(2)
    with col_a:
        ratio = st.slider(
            "Summary length (fraction of sentences kept)",
            0.05, 0.60, 0.25, 0.05
        )
    with col_b:
        max_chars = st.number_input(
            "Max summary characters (safety cap)",
            min_value=500, max_value=10000, value=4000, step=250
        )

    if st.button("Summarize Article"):
        try:
            summary_text = tfidf_extractive_summary(
                full_text=text,
                ratio=float(ratio),
                max_chars=int(max_chars),
            )
            if not summary_text:
                st.warning("Could not generate a summary (empty or very short content).")
            else:
                st.subheader("Generated Summary (Extractive TF-IDF):")
                st.write(summary_text)
                st.info(f"Original length: {len(text.split())} words → Summary length: {len(summary_text.split())} words")
        except Exception as e:
            st.error(f"Error during summarization: {e}")


    # ---------------------------
    # Keyword Extraction
    # ---------------------------
    with st.expander("Keyword Extraction (TF-IDF)"):
        if st.button("Extract Keywords"):
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform([limited_text])
            scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            st.subheader("Top Keywords:")
            for word, score in sorted_scores[:15]:
                st.write(f"- {word} ({score:.4f})")

    # ---------------------------
    # Semantic Analysis
    # ---------------------------
    with st.expander("Semantic Analysis"):
        st.write("This finds the most semantically related sentence pairs from the Wikipedia page.")

        if st.button("Run Semantic Analysis"):
            doc = nlp(limited_text)
            sentences = list(doc.sents)

            # Compute embeddings
            embeddings = [sent.vector for sent in sentences]

            # Similarity matrix
            sim_matrix = cosine_similarity(embeddings)

            results = []
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    results.append(((sentences[i].text, sentences[j].text), sim_matrix[i][j]))

            # Sort by similarity
            results = sorted(results, key=lambda x: x[1], reverse=True)[:5]

            st.subheader("Top Semantic Similarities:")
            for (sent_pair, score) in results:
                st.markdown(f"**Sentence 1:** {sent_pair[0]}")
                st.markdown(f"**Sentence 2:** {sent_pair[1]}")
                st.write(f"**Similarity Score:** {score:.4f}")
                st.markdown("---")

    # ---------------------------
    # Word Sense Disambiguation
    # ---------------------------
    with st.expander("Word Sense Disambiguation (WSD)"):
        sentence = st.text_input("Enter a sentence from the article:", key="wsd_sentence")
        target_word = st.text_input("Enter the target word:", key="wsd_word")

        if st.button("Disambiguate Word"):
            if sentence and target_word:
                try:
                    if 'simple_lesk' in globals():
                        sense = simple_lesk(sentence, target_word, pos='n')
                    else:
                        sense = lesk(nltk.word_tokenize(sentence), target_word, 'n')

                    if sense:
                        st.subheader("Disambiguation Result:")
                        st.write(f"**Word:** {target_word}")
                        st.write(f"**Sense (Synset):** {sense.name()}")
                        st.write(f"**Definition:** {sense.definition()}")
                        st.write(f"**Examples:** {sense.examples()}")
                    else:
                        st.warning("No sense found. Try another word or sentence.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please provide both sentence and target word.")