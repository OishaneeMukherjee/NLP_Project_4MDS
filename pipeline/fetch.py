import wikipedia

def fetch_wikipedia_content(query: str, sentences: int = 15) -> str:
    """
    Fetch a concise Wikipedia summary for the query.
    Falls back with an error message if disambiguation or page errors occur.
    """
    try:
        # Prefer summary for fast demos; avoids huge pages
        text = wikipedia.summary(query, sentences=sentences, auto_suggest=True, redirect=True)
        return text
    except wikipedia.DisambiguationError as e:
        return f"Disambiguation required. Choose one of: {', '.join(e.options[:10])} ..."
    except wikipedia.PageError:
        return "Page not found. Try a different query."
    except Exception as exc:
        return f"Error fetching page: {exc}"
