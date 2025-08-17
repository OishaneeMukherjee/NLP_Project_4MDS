import streamlit as st

from pipeline.fetch import fetch_wikipedia_content
from pipeline.preprocess import clean_text, sent_tokenize, word_tokenize
from pipeline.analysis import extract_keywords, analyze_sentiment
from pipeline.sequence_labeling import pos_and_ner, word_sense_disambiguation, simple_discourse_cohesion
from pipeline.vector_semantics import tfidf_cosine_similarity, pmi, word_similarity
from pipeline.retrieval import build_inverted_index, search_in_index, answer_question, evaluate_ir, split_sentences, score_sentences

st.set_page_config(page_title="Wikipedia NLP Analyzer", layout="wide")
st.title("📘 Wikipedia Page Analyzer — Tag, Score, and Search")

with st.sidebar:
    st.header("⚙️ Settings")
    default_topic = st.text_input("Default topic", "Natural Language Processing")
    wiki_sentences = st.slider("Sentences to fetch", 5, 30, 15)
    st.caption("Tip: For very short topics, increase sentence count.")

# Main input
query = st.text_input("🔎 Enter a Wikipedia topic to analyze", value=default_topic)
go = st.button("Fetch & Run Pipeline", type="primary")

if go:
    with st.spinner("Fetching Wikipedia content..."):
        raw_text = fetch_wikipedia_content(query, sentences=wiki_sentences)

    if not raw_text or raw_text.lower().startswith(("error", "page not found", "disambiguation")):
        st.error(raw_text or "Failed to fetch text.")
        st.stop()

    st.subheader("📄 Wikipedia Summary")
    st.write(raw_text)

    # Precompute helpful artifacts
    sentences = split_sentences(raw_text)
    cleaned = clean_text(raw_text)
    tokens = word_tokenize(cleaned)

    # Quick overview cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sentences", len(sentences))
    with col2:
        st.metric("Tokens (cleaned)", len(tokens))
    with col3:
        pol, subj = analyze_sentiment(raw_text)
        st.metric("Polarity", f"{pol:.3f}")
        st.metric("Subjectivity", f"{subj:.3f}")

    st.markdown("---")

    tab2, tab3, tab4, tab0 = st.tabs([
        "Unit 2: Sequence Labeling & Semantics",
        "Unit 3: Vector Semantics & Embeddings",
        "Unit 4: Information Retrieval & QA",
        "Tags & Sentiment (Quick View)"
    ])

    # ---------------------- Unit 2 ----------------------
    with tab2:
        st.header("🧩 POS Tagging & Named Entities")
        posner_btn = st.button("Run POS & NER", key="posner")
        if posner_btn:
            pos_tags, entities = pos_and_ner(raw_text)
            if not pos_tags and not entities:
                st.warning("spaCy model not found. Run: `python -m spacy download en_core_web_sm`")
            else:
                st.write("**Sample POS tags (first 40):**")
                st.write(pos_tags[:40])
                st.write("**Named Entities:**")
                if entities:
                    st.write(entities)
                else:
                    st.info("No entities detected in this short summary.")

        st.subheader("🔎 Word Sense Disambiguation (Lesk)")
        wsd_col1, wsd_col2 = st.columns([2,1])
        with wsd_col1:
            wsd_sent = st.text_input("Sentence for WSD", value=sentences[0] if sentences else "")
        with wsd_col2:
            wsd_word = st.text_input("Target word", value="model")
        if st.button("Disambiguate", key="wsd_btn"):
            sense = word_sense_disambiguation(wsd_sent, wsd_word)
            st.write(f"**Sense:** {sense}")

        st.subheader("🧵 Simple Discourse Cohesion Cue")
        if st.button("Compute overlaps", key="cohesion"):
            links = simple_discourse_cohesion(sentences)
            if not links:
                st.info("No strong overlaps found between adjacent sentences.")
            else:
                for i, j, inter in links:
                    st.write(f"Sent {i} ↔ Sent {j} | overlap: {', '.join(inter)}")
                    st.caption(f"S{i}: {sentences[i]}")
                    st.caption(f"S{j}: {sentences[j]}")

    # ---------------------- Unit 3 ----------------------
    with tab3:
        st.header("📐 TF-IDF & Cosine Similarity")
        if sentences:
            sims = tfidf_cosine_similarity(query, sentences)
            if sims.size:
                best_idx = int(sims.argmax())
                st.write(f"**Best matching sentence (to query)**: S{best_idx} (score={sims[best_idx]:.3f})")
                st.info(sentences[best_idx])

                st.subheader("Top-5 relevant sentences")
                topk = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)[:5]
                for idx, sc in topk:
                    st.write(f"- S{idx} | {sc:.3f} — {sentences[idx]}")
            else:
                st.info("No sentences to score.")
        else:
            st.info("No sentences found in the fetched summary.")

        st.subheader("📊 Pointwise Mutual Information (PMI)")
        pmi_col1, pmi_col2 = st.columns(2)
        with pmi_col1:
            w1 = st.text_input("Word 1", "natural")
        with pmi_col2:
            w2 = st.text_input("Word 2", "language")
        if st.button("Compute PMI", key="pmi_btn"):
            val = pmi(w1.lower(), w2.lower(), tokens)
            st.write(f"PMI({w1}, {w2}) = **{val:.3f}** (window=1 bigram)")

        st.subheader("🔗 Word Similarity (Embeddings)")
        ws_col1, ws_col2 = st.columns(2)
        with ws_col1:
            a = st.text_input("Word A", "model")
        with ws_col2:
            b = st.text_input("Word B", "algorithm")
        if st.button("Compute Similarity", key="ws_btn"):
            sim = word_similarity(a, b)
            if sim is None:
                st.warning("Embedding similarity unavailable. Install spaCy model with vectors: `python -m spacy download en_core_web_sm` (limited) or larger model.")
            else:
                st.write(f"Similarity({a}, {b}) = **{sim:.3f}**")

    # ---------------------- Unit 4 ----------------------
    with tab4:
        st.header("📚 Inverted Index & Search")
        idx, sents = build_inverted_index(raw_text)
        q = st.text_input("Search query (bag-of-words)", "learning models")
        if st.button("Search Index", key="idx_search"):
            results = search_in_index(q, idx, sents)
            if not results:
                st.info("No matches found.")
            else:
                for i, s in results[:20]:
                    st.write(f"- S{i}: {s}")

        st.subheader("🧠 Extractive QA (Sentence Selection)")
        qa_q = st.text_input("Ask a question about the text", "What is natural language processing?")
        if st.button("Get Answer", key="qa_btn"):
            ans, score, sid = answer_question(qa_q, sents)
            st.write(f"**Answer (S{sid}, score={score:.3f})**: {ans}")

        st.subheader("📏 IR Evaluation (Toy Precision/Recall)")
        k = st.slider("Top-K", 1, 10, 5)
        if st.button("Evaluate IR", key="ir_eval"):
            metrics = evaluate_ir(q or query, sents, top_k=k)
            st.write(metrics)

    # ---------------------- Quick View ----------------------
    with tab0:
        st.header("🏷️ Tags & 📊 Sentiment")
        tags = extract_keywords(cleaned, top_n=12)
        if tags:
            st.write("**Top TF-IDF tags:**")
            for w, s in tags:
                st.write(f"- **{w}** → {s:.3f}")
        else:
            st.info("No tags extracted.")

        st.write("**Sentiment:**")
        st.write(f"- Polarity: `{pol:.3f}` (−1 → +1)")
        st.write(f"- Subjectivity: `{subj:.3f}` (0 → 1)")

else:
    st.info("Enter a topic and click **Fetch & Run Pipeline** to start.")
