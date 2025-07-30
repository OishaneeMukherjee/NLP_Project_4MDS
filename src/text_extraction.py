# src/text_extraction.py

from bs4 import BeautifulSoup
import re

def extract_text_from_html(html):
    """Extract plain text from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def extract_sections(html_text):
    """Extract sections like skills, education, experience based on headers."""
    soup = BeautifulSoup(html_text, 'html.parser')
    sections = {}
    headers = soup.find_all(['h1', 'h2', 'h3', 'b', 'strong'])

    for tag in headers:
        section_title = tag.get_text().strip().lower()
        content = []
        next_tag = tag.find_next_sibling()
        while next_tag and next_tag.name not in ['h1', 'h2', 'h3', 'b', 'strong']:
            content.append(next_tag.get_text(strip=True))
            next_tag = next_tag.find_next_sibling()
        if content:
            sections[section_title] = ' '.join(content)

    return sections

def extract_emails(text):
    return re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)

def extract_phone_numbers(text):
    return re.findall(r'\b\d{10}\b', text)

def extract_links(text):
    return re.findall(r'(https?://[^\s]+)', text)
