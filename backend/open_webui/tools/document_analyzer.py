"""
title: Document Analyzer
description: Analyze documents to extract summaries, named entities, and key dates. Works on any text provided in the conversation.
author: salim4n
version: 0.1.0
license: MIT
"""

import json
import re
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Date extraction patterns
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    # ISO: 2024-03-15, 2024/03/15
    (r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", "%Y-%m-%d"),
    # EU/FR: 15/03/2024, 15-03-2024
    (r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b", "%d-%m-%Y"),
    # Written: March 15, 2024 | 15 March 2024
    (
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
        None,
    ),
    (
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
        None,
    ),
    # Written FR: 15 janvier 2024, mars 2024
    (
        r"\b(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4})\b",
        None,
    ),
    # Month Year: March 2024, Q1 2024
    (
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
        None,
    ),
    (r"\b(Q[1-4]\s+\d{4})\b", None),
    # Year only in context: "in 2024", "since 2019"
    (r"\b(?:in|since|from|by|until|before|after|year)\s+(\d{4})\b", None),
]

_ENTITY_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "URL": r"https?://[^\s<>\"']+",
    "PHONE": r"(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b",
    "MONEY": r"[$€£¥]\s?\d[\d,]*(?:\.\d+)?[BMKbmk]?|\d[\d,]*(?:\.\d+)?[BMKbmk]?\s?(?:USD|EUR|GBP|JPY)\b",
    "PERCENTAGE": r"\b\d+(?:\.\d+)?%",
}


def _extract_dates_from_text(text: str) -> list[dict]:
    """Extract dates and their surrounding context from text."""
    found = []
    seen = set()

    for pattern, _ in _DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            date_str = match.group(1) if match.lastindex else match.group(0)
            if date_str in seen:
                continue
            seen.add(date_str)

            # Get surrounding context (±60 chars)
            start = max(0, match.start() - 60)
            end = min(len(text), match.end() + 60)
            context = text[start:end].strip()
            # Clean up context boundaries to nearest word
            if start > 0:
                context = "..." + context[context.find(" ") + 1 :]
            if end < len(text):
                context = context[: context.rfind(" ")] + "..."

            found.append({"date": date_str, "context": context})

    # Sort by position in text
    return found


def _extract_pattern_entities(text: str) -> dict[str, list[str]]:
    """Extract structured entities using regex patterns."""
    entities = {}
    for entity_type, pattern in _ENTITY_PATTERNS.items():
        matches = list(set(re.findall(pattern, text)))
        if matches:
            entities[entity_type] = matches[:20]  # Limit per type
    return entities


def _extract_capitalized_phrases(text: str) -> list[str]:
    """Extract potential named entities (capitalized multi-word phrases)."""
    # Match sequences of capitalized words (2+ words)
    pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
    matches = re.findall(pattern, text)

    # Filter out common false positives (sentence starters after periods)
    stopwords = {
        "The", "This", "That", "These", "Those", "There", "When", "Where",
        "What", "Which", "How", "After", "Before", "During", "Between",
        "According", "Based", "However", "Therefore", "Furthermore",
        "In", "On", "At", "For", "With", "From", "To", "By",
    }

    filtered = []
    seen = set()
    for match in matches:
        first_word = match.split()[0]
        if first_word not in stopwords and match not in seen:
            seen.add(match)
            filtered.append(match)

    return filtered[:30]  # Limit


def _compute_text_stats(text: str) -> dict:
    """Compute basic text statistics."""
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": len(paragraphs),
        "avg_words_per_sentence": round(len(words) / max(len(sentences), 1), 1),
    }


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class Tools:
    def __init__(self):
        pass

    async def analyze_document(
        self,
        text: str,
        __user__: dict = None,
    ) -> str:
        """
        Perform a full analysis of a document or text: extract a structured summary prompt,
        named entities (people, organizations, places, emails, URLs, monetary values),
        key dates with their context, and text statistics.
        Use this tool when the user asks to analyze, summarize, or extract information from a document or text.

        :param text: The text content to analyze (paste document content or text from the conversation)
        :return: JSON with entities, dates, statistics, and analysis-ready text segments
        """
        if not text or not text.strip():
            return json.dumps({"error": "No text provided"}, ensure_ascii=False)

        text = text.strip()

        # Truncate very long texts
        max_chars = 100_000
        truncated = len(text) > max_chars
        analysis_text = text[:max_chars] if truncated else text

        # Run all extractions
        stats = _compute_text_stats(analysis_text)
        dates = _extract_dates_from_text(analysis_text)
        pattern_entities = _extract_pattern_entities(analysis_text)
        capitalized_phrases = _extract_capitalized_phrases(analysis_text)

        result = {
            "text_statistics": stats,
            "key_dates": dates[:30],
            "entities": {
                "potential_names_and_organizations": capitalized_phrases,
                **pattern_entities,
            },
            "truncated": truncated,
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    async def extract_entities(
        self,
        text: str,
        __user__: dict = None,
    ) -> str:
        """
        Extract named entities from text: people, organizations, places, emails, URLs,
        phone numbers, monetary values, and percentages.
        Use this when the user specifically asks to identify entities, names, or structured data in text.

        :param text: The text to extract entities from
        :return: JSON with categorized entities found in the text
        """
        if not text or not text.strip():
            return json.dumps({"error": "No text provided"}, ensure_ascii=False)

        text = text.strip()[:100_000]

        pattern_entities = _extract_pattern_entities(text)
        capitalized_phrases = _extract_capitalized_phrases(text)

        result = {
            "potential_names_and_organizations": capitalized_phrases,
            **pattern_entities,
            "total_entities_found": sum(len(v) for v in pattern_entities.values())
            + len(capitalized_phrases),
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    async def extract_key_dates(
        self,
        text: str,
        __user__: dict = None,
    ) -> str:
        """
        Extract all dates and temporal references from text, along with their surrounding context.
        Supports ISO dates, European formats, written dates in English and French, quarters, and year references.
        Use this when the user wants to find important dates, deadlines, or temporal information in a document.

        :param text: The text to extract dates from
        :return: JSON with dates found and their context
        """
        if not text or not text.strip():
            return json.dumps({"error": "No text provided"}, ensure_ascii=False)

        text = text.strip()[:100_000]

        dates = _extract_dates_from_text(text)

        result = {
            "dates_found": len(dates),
            "dates": dates[:50],
        }

        return json.dumps(result, ensure_ascii=False, indent=2)
