"""
Deterministic normalization utilities for entity resolution candidate generation.

This module provides pure-Python, deterministic normalization functions for text,
tokens, initials, phonetics (Soundex), and structured attribute values.

Design & Normalization Principles:
1. Immutability: Normalization functions never modify input data; they return fresh,
   canonical derived representations.
2. Unicode & Scripts: Uses Unicode NFKC decomposition/recomposition and Unicode
   casefolding (str.casefold) to handle multi-lingual text, accented characters,
   and script variations reliably.
3. Whitespace & Punctuation: Collapses repeated whitespace and normalizes punctuation
   without blindly deleting alphanumeric tokens from identifiers or organization names.
4. Non-Person & Multi-Type Support: Does not assume all entities are persons.
   Organization names, locations, and codes are normalized without assuming
   a "Firstname Lastname" structure.
5. Missing & Empty Safety: Safely handles None, empty strings, non-string primitives,
   and unusual input without raising unexpected exceptions.
"""

from datetime import date, datetime
import re
import string
from typing import Any, List, Optional
import unicodedata


# Precompiled regex patterns
_WHITESPACE_COLLAPSE = re.compile(r"\s+")
_NON_ALPHANUMERIC_OR_SPACE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(text: Optional[str]) -> str:
    """
    Normalizes textual input deterministically:
    - Safe for None or non-string inputs (returns empty string).
    - Applies Unicode NFKC normalization (recomposes compatible characters).
    - Applies case-folding (str.casefold, more robust than str.lower for international text).
    - Strips leading and trailing whitespace.
    - Collapses repeated whitespace to a single space.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Unicode normalization: NFKC decomposes and standardizes presentation forms
    normalized = unicodedata.normalize("NFKC", text)
    # Casefolding: comprehensive lowercasing for all Unicode scripts
    folded = normalized.casefold()
    # Collapse whitespace
    collapsed = _WHITESPACE_COLLAPSE.sub(" ", folded).strip()
    return collapsed


def strip_punctuation(text: Optional[str], replace_with_space: bool = True) -> str:
    """
    Removes punctuation while preserving alphanumeric characters across all scripts.

    If replace_with_space is True (default), punctuation characters are replaced
    by spaces so token boundaries are preserved (e.g., 'B. Wayne' -> 'b wayne',
    'semi-conductor' -> 'semi conductor').
    """
    if not text:
        return ""

    norm = normalize_text(text)
    if replace_with_space:
        # Replace punctuation characters with a space
        cleaned = "".join(
            " " if (unicodedata.category(c).startswith("P") or c in string.punctuation) else c
            for c in norm
        )
    else:
        # Simply omit punctuation
        cleaned = "".join(
            c for c in norm
            if not (unicodedata.category(c).startswith("P") or c in string.punctuation)
        )

    return _WHITESPACE_COLLAPSE.sub(" ", cleaned).strip()


def tokenize_name(text: Optional[str], min_length: int = 1) -> List[str]:
    """
    Tokenizes a name or text string into distinct, normalized alphanumeric tokens.

    Punctuation is treated as a token separator. Tokens shorter than min_length
    are filtered out. Preserves order of appearance.
    """
    cleaned = strip_punctuation(text, replace_with_space=True)
    if not cleaned:
        return []

    tokens = [tok for tok in cleaned.split(" ") if len(tok) >= min_length]
    return tokens


def extract_initials(text: Optional[str]) -> str:
    """
    Extracts deterministic initials / acronym representations:
    - For multi-token text: first character of each token (e.g. 'International Business Machines' -> 'ibm').
    - For single token that already resembles an acronym (e.g. 'IBM' or 'NASA'):
      returns the normalized token if length >= 2.
    """
    if not text:
        return ""

    raw_norm = normalize_text(text)
    tokens = tokenize_name(text, min_length=1)

    if not tokens:
        return ""

    if len(tokens) == 1:
        single = tokens[0]
        # If the input was an acronym like 'IBM'
        if len(single) >= 2 and (text.strip().isupper() or len(single) <= 5):
            return single
        return single[:1]

    # Multi-token: first character of each token
    return "".join(tok[0] for tok in tokens)


def name_fingerprint(text: Optional[str]) -> str:
    """
    Generates an order-invariant token fingerprint for a name.

    Deduplicates and alphabetically sorts unique name tokens.
    Example: 'Smith, John' -> 'john smith'
             'John Smith'  -> 'john smith'
    """
    tokens = tokenize_name(text, min_length=1)
    if not tokens:
        return ""
    unique_sorted = sorted(set(tokens))
    return " ".join(unique_sorted)


def soundex(token: Optional[str]) -> str:
    """
    Calculates the standard American Soundex code for an individual Latin-alphabet name token.

    Soundex algorithm:
    1. Retains first letter in uppercase.
    2. Maps remaining letters:
       - B, F, P, V -> 1
       - C, G, J, K, Q, S, X, Z -> 2
       - D, T -> 3
       - L -> 4
       - M, N -> 5
       - R -> 6
       - A, E, I, O, U, Y, H, W -> unmapped / ignored
    3. Collapses adjacent identical digits. Letters separated by H or W retain the same digit.
    4. Pads with zeros to length 4 or truncates to length 4.

    Limitations:
    - Specifically designed for English/Latin surnames; inapplicable to non-Latin scripts.
    - If token contains no Latin letters or first character is non-Latin, returns empty string.
    """
    if not token or not isinstance(token, str):
        return ""

    # Decompose Unicode accents (e.g. 'José' -> 'Jose')
    decomposed = unicodedata.normalize("NFKD", token)
    clean_latin = "".join(c for c in decomposed if "A" <= c <= "Z" or "a" <= c <= "z")
    if not clean_latin:
        return ""

    upper = clean_latin.upper()
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    first_letter = upper[0]
    codes = [first_letter]
    prev_code = mapping.get(first_letter, "0")

    for char in upper[1:]:
        code = mapping.get(char, "0")
        if code != "0":
            if code != prev_code:
                codes.append(code)
            prev_code = code
        else:
            if char not in ("H", "W"):
                prev_code = "0"

    result = "".join(codes)[:4]
    return result.ljust(4, "0")


def normalize_attribute_value(val: Any) -> str:
    """
    Converts and normalizes arbitrary attribute values into a deterministic string key:
    - None -> empty string
    - str -> normalize_text(str)
    - date / datetime -> ISO formatted string (e.g. '1980-01-01')
    - int / float -> str(val)
    - list / tuple / set -> comma-separated normalized elements
    - dict -> normalized sorted key:value pairs
    """
    if val is None:
        return ""

    if isinstance(val, (datetime, date)):
        return val.isoformat()

    if isinstance(val, str):
        return normalize_text(val)

    if isinstance(val, (int, float, bool)):
        return str(val).lower()

    if isinstance(val, (list, tuple, set, frozenset)):
        items = [normalize_attribute_value(x) for x in val if x is not None]
        non_empty = [x for x in items if x]
        return ",".join(sorted(non_empty))

    if isinstance(val, dict):
        pairs = [
            f"{normalize_text(str(k))}:{normalize_attribute_value(v)}"
            for k, v in sorted(val.items())
            if v is not None
        ]
        return ";".join(pairs)

    return normalize_text(str(val))
