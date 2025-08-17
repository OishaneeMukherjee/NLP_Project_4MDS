import streamlit as st
import wikipedia
import nltk
from textblob import TextBlob
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extended Lesk (better WSD)
try:
    from pywsd.lesk import simple_lesk
except:
    st.warning("⚠️ pywsd not installed. Run: pip install pywsd")

# NLTK downloads
nltk.download('punkt')
nltk.download('wordnet')

st.set_page_config(page_title="Wikipedia NLP Project", layout="wide")

st.title("📘 Wikipedia NLP Explorer")
st.write("An Interactive IR & NLP Project with Search, Summaries, Keywords, Sentiment, and Word Sense Disambiguation.")

# -----------------------------------
# 🔍 Search Wikipedia Section
# -----------------------------------
st.header("🔍 Search Wikipedia")
query = st.text_input("Enter a topic to search:", "")
if st.button("Search"):
    try:
        search_results = wikipedia.search(query)
        st.success(f"Found {len(search_results)} results.")
        st.write(search_results)
    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------------
# 📄 Fetch Summary Section
# -----------------------------------
st.header("📄 Fetch Wikipedia Summary")
topic = st.text_input("Enter topic for summary:", key="summary_topic")
if st.button("Get Summary"):
    try:
        summary = wikipedia.summary(topic, sentences=5)
        st.subheader("Summary:")
        st.write(summary)
    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------------
# 📝 Keyword Extraction (TF-IDF)
# -----------------------------------
st.header("📝 Keyword Extraction")
doc = st.text_area("Paste text or summary for keyword extraction:")
if st.button("Extract Keywords"):
    if doc.strip():
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform([doc])
            scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            st.subheader("Top Keywords:")
            for word, score in sorted_scores[:10]:
                st.write(f"- {word} ({score:.4f})")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter some text for keyword extraction.")

# -----------------------------------
# 😊 Sentiment Analysis
# -----------------------------------
st.header("😊 Sentiment Analysis")
text_sentiment = st.text_area("Enter text for sentiment analysis:")
if st.button("Analyze Sentiment"):
    if text_sentiment.strip():
        blob = TextBlob(text_sentiment)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        st.subheader("Sentiment Results:")
        st.write(f"**Polarity:** {polarity}")
        st.write(f"**Subjectivity:** {subjectivity}")
        if polarity > 0:
            st.success("Positive Sentiment")
        elif polarity < 0:
            st.error("Negative Sentiment")
        else:
            st.info("Neutral Sentiment")
    else:
        st.warning("Please enter some text for sentiment analysis.")

# -----------------------------------
# 🔗 Word Sense Disambiguation (Improved)
# -----------------------------------
st.header("🔗 Word Sense Disambiguation (WSD)")
sentence = st.text_input("Enter a sentence for WSD:", key="wsd_sentence")
target_word = st.text_input("Enter the target word:", key="wsd_word")

if st.button("Disambiguate Word"):
    if sentence and target_word:
        try:
            # Extended Lesk expects raw sentence (string)
            if 'simple_lesk' in globals():
                sense = simple_lesk(sentence, target_word, pos='n')  # ✅ FIX: pass sentence as string
            else:
                sense = lesk(nltk.word_tokenize(sentence), target_word, 'n')  # fallback

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

