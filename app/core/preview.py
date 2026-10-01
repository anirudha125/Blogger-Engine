"""
Document preview extraction and normalization for Blogger Engine search results.

Provides deterministic, lightweight boilerplate cleanup and excerpt extraction
from the beginning (chunk 1) of a document without invoking LLMs.
"""

import re
from typing import List, Optional


def sanitize_mojibake(text: str) -> str:
    """Normalize common encoding artifacts and control characters."""
    if not text:
        return ""
    return (
        text.replace("â€™", "'")
            .replace("â€˜", "'")
            .replace("â€œ", '"')
            .replace("â€\x9d", '"')
            .replace("â€", '"')
            .replace("â€”", "—")
            .replace("â€“", "–")
            .replace("â€¦", "…")
            .replace("Â", "")
            .replace("\u00a0", " ")
            .replace("●", " ")
            .replace("•", " ")
    )


# Common web and scraper boilerplate markers to skip or trim from the beginning of technical blogs
BOILERPLATE_SKIP_MARKERS = [
    r"Table of contents?\s*(?:\[[^\]]*\])?",
    r"Back to blog",
    r"Skip to (?:Main )?Content",
    r"Please enable JavaScript\s*[^\.]*\.",
    r"Sign up\s+Sign in",
    r"Share this post",
    r"Permalink\s+Comments\s+Share",
    r"Listen\s+Share",
    r"All guides",
    r"No items found\.",
    r"Read on to understand",
    r"Toggle navigation",
    r"Navigation menu",
]


def clean_leading_boilerplate(text: str, title: Optional[str] = None) -> str:
    """
    Deterministic cleanup of leading navigation, site headers, title repetitions, and boilerplate.
    Does NOT attempt semantic summarization.
    """
    if not text:
        return ""

    t = sanitize_mojibake(text)
    t = re.sub(r"\s+", " ", t).strip()

    # Strip markdown images ![alt](url) and links [text](url) -> text
    t = re.sub(r"!\[.*?\]\(.*?\)", "", t)
    t = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", t)

    # If title is provided and text repeats the title at the very beginning, strip it
    if title:
        clean_title = sanitize_mojibake(title).strip()
        core_title = re.split(r"\s+[|\-—]\s+", clean_title)[0].strip()
        if core_title and len(core_title) > 6:
            # Strip leading title
            pattern = r"^" + re.escape(core_title) + r"[\s:|\-—]*"
            t = re.sub(pattern, "", t, flags=re.IGNORECASE).strip()
            # If title is repeated in the first 300 characters before the body starts, advance to the body
            match = re.search(r"\b" + re.escape(core_title) + r"\b[\s:|\-—]*", t, flags=re.IGNORECASE)
            if match and match.start() < 250:
                after_title = t[match.end():].strip()
                if len(after_title.split()) > 20:
                    t = after_title

    # Skip past known leading navigation markers
    for marker in BOILERPLATE_SKIP_MARKERS:
        m = re.search(marker, t[:300], flags=re.IGNORECASE)
        if m:
            candidate = t[m.end():].strip()
            if len(candidate.split()) > 20:
                t = candidate
                break

    # Strip any leading punctuation or residual divider characters
    t = t.lstrip(" :|-—.,\n\t")
    return t


def split_sentences(text: str) -> List[str]:
    """Split text into sentences while respecting common technical abbreviations."""
    if not text:
        return []

    protected = text
    abbrevs = ["e.g.", "i.e.", "et al.", "vs.", "Fig.", "Dr.", "Prof.", "approx.", "etc."]
    for idx, abbr in enumerate(abbrevs):
        protected = protected.replace(abbr, f"__ABBR_{idx}__")

    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", protected)

    sentences = []
    for s in raw_sentences:
        for idx, abbr in enumerate(abbrevs):
            s = s.replace(f"__ABBR_{idx}__", abbr)
        s = s.strip()
        if s:
            sentences.append(s)
    return sentences


def extract_document_preview(
    first_chunk_content: Optional[str],
    fallback_content: Optional[str] = None,
    title: Optional[str] = None,
    min_words: int = 35,
    max_words: int = 50,
    max_sentences: int = 3
) -> str:
    """
    Extract a short readable preview (first 2-3 sentences, roughly 35-50 words maximum,
    followed by '...' when truncated) from the beginning of the document.

    If the first chunk is unavailable, empty, or unusable, safely falls back to the
    existing retrieved chunk.
    """
    raw_first = (first_chunk_content or "").strip()
    cleaned = clean_leading_boilerplate(raw_first, title)

    # Fallback if first chunk is empty or unusable (< 15 words)
    words = cleaned.split()
    if len(words) < 15 and fallback_content:
        cleaned = clean_leading_boilerplate(fallback_content.strip(), title)
        words = cleaned.split()

    if not words:
        return ""

    sentences = split_sentences(cleaned)
    selected_sentences = []
    current_word_count = 0

    for s in sentences:
        s_words = s.split()
        if not s_words:
            continue
        # If adding this sentence exceeds max_words and we already have at least 1-2 sentences
        if selected_sentences and (current_word_count + len(s_words) > max_words + 5):
            break
        selected_sentences.append(s)
        current_word_count += len(s_words)
        if len(selected_sentences) >= max_sentences or current_word_count >= min_words:
            break

    result = " ".join(selected_sentences).strip()
    result_words = result.split()

    # Bound to approximately 35-50 words max
    if len(result_words) > max_words:
        result = " ".join(result_words[:max_words]).rstrip(".,;:!?\"'()[]") + "..."
    else:
        # If truncated from longer text, append "..."
        if len(words) > len(result_words) and not result.endswith("..."):
            result = result.rstrip(".,;:!?\"'()[]") + "..."

    return result
