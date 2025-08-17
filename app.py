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
            page = wikipedia.page(title)
            st.session_state['page_title'] = title
            st.session_state['page_content'] = page.content

            st.success(f"Fetched page: {title}")

            # Generate summary using full content
            if summarizer:
                try:
                    sentences = nltk.sent_tokenize(st.session_state['page_content'])
                    chunks, current_chunk = [], []
                    max_chunk_words = 800  # safe limit for BART

                    for sentence in sentences:
                        if sum(len(s.split()) for s in current_chunk) + len(sentence.split()) <= max_chunk_words:
                            current_chunk.append(sentence)
                        else:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = [sentence]
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))

                    summaries = []
                    for chunk in chunks:
                        res = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
                        summaries.append(res[0]['summary_text'])

                    final_summary = " ".join(summaries)

                    st.subheader("Generated Summary:")
                    st.write(final_summary.strip())

                except Exception as e:
                    st.error(f"Error during summarization: {e}")
            else:
                st.warning("Summarizer not initialized.")
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


     # Wikipedia Page Summarization
    # ---------------------------
    with st.expander("Wikipedia Page Summarization"):
        st.write("This generates a concise summary of the entire article using an abstractive NLP model.")

        if st.button("Summarize Article"):
            if summarizer:
                try:
                    # Sentence-based chunking for better summaries
                    sentences = nltk.sent_tokenize(text)
                    chunks, current_chunk = [], []
                    max_chunk_words = 800  # safe limit for BART

                    for sentence in sentences:
                        if sum(len(s.split()) for s in current_chunk) + len(sentence.split()) <= max_chunk_words:
                            current_chunk.append(sentence)
                        else:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = [sentence]
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))

                    st.info(f"Article split into {len(chunks)} chunks for summarization.")

                    summaries = []
                    for i, chunk in enumerate(chunks):
                        res = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
                        summaries.append(res[0]['summary_text'])

                    final_summary = " ".join(summaries)

                    st.subheader("Generated Summary:")
                    st.write(final_summary.strip())
                except Exception as e:
                    st.error(f"Error during summarization: {e}")
            else:
                st.warning("Summarizer not initialized.")
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