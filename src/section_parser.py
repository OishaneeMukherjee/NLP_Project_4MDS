from bs4 import BeautifulSoup

def extract_sections_from_html(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    sections = {}
    for tag in soup.find_all(['h1', 'h2', 'h3']):
        section = tag.get_text().strip().lower()
        next_tag = tag.find_next_sibling()
        content = []
        while next_tag and next_tag.name not in ['h1', 'h2', 'h3']:
            content.append(next_tag.get_text(strip=True))
            next_tag = next_tag.find_next_sibling()
        if content:
            sections[section] = " ".join(content)
    return sections
