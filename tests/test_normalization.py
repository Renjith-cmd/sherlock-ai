import datetime
import pytest
from src.core.normalization import (
    extract_initials,
    name_fingerprint,
    normalize_attribute_value,
    normalize_text,
    soundex,
    strip_punctuation,
    tokenize_name,
)


def test_normalize_text_unicode_and_casefold():
    # German Eszett
    assert normalize_text("STRAßE") == "strasse"
    # Accents with NFKC
    assert normalize_text("Café") == "café"
    # Cyrillic
    assert normalize_text("Иван") == "иван"
    # Greek
    assert normalize_text("ΣΩΚΡΑΤΗΣ") == "σωκρατησ" or normalize_text("ΣΩΚΡΑΤΗΣ") == "σωκράτης".replace("ά", "α")


def test_normalize_text_whitespace_collapsing():
    assert normalize_text("  John   \t  Doe \n\n ") == "john doe"
    assert normalize_text("") == ""
    assert normalize_text(None) == ""
    assert normalize_text("   ") == ""


def test_strip_punctuation():
    # Replacing with space preserves token boundaries
    assert strip_punctuation("B. Wayne") == "b wayne"
    assert strip_punctuation("Smith, John") == "smith john"
    assert strip_punctuation("semi-conductor") == "semi conductor"
    # Without space replacement
    assert strip_punctuation("B. Wayne", replace_with_space=False) == "b wayne"
    assert strip_punctuation("U.S.A.", replace_with_space=False) == "usa"
    # Non-Latin characters preserved
    assert strip_punctuation("«Война и мир»") == "война и мир"
    assert strip_punctuation(None) == ""


def test_tokenize_name():
    assert tokenize_name("John Doe") == ["john", "doe"]
    assert tokenize_name("B. Wayne", min_length=2) == ["wayne"]
    assert tokenize_name("B. Wayne", min_length=1) == ["b", "wayne"]
    assert tokenize_name("  ", min_length=1) == []
    assert tokenize_name(None) == []


def test_extract_initials():
    # Multi-token organization
    assert extract_initials("International Business Machines") == "ibm"
    # Single acronym token
    assert extract_initials("IBM") == "ibm"
    assert extract_initials("NASA") == "nasa"
    # Person name
    assert extract_initials("Bruce Wayne") == "bw"
    # Empty
    assert extract_initials("") == ""
    assert extract_initials(None) == ""


def test_name_fingerprint():
    # Order-invariant representation
    assert name_fingerprint("Smith, John") == "john smith"
    assert name_fingerprint("John Smith") == "john smith"
    assert name_fingerprint("Alice Bob Charlie") == "alice bob charlie"
    assert name_fingerprint("Charlie Alice Bob") == "alice bob charlie"
    assert name_fingerprint("") == ""
    assert name_fingerprint(None) == ""


def test_soundex_latin_and_variations():
    # Spelling variation: Sarah vs Sara
    assert soundex("Sarah") == "S600"
    assert soundex("Sara") == "S600"
    # Connor
    assert soundex("Connor") == "C560"
    assert soundex("Conner") == "C560"
    # Accented Latin: José decomposes to Jose
    assert soundex("José") == "J200"
    assert soundex("Jose") == "J200"


def test_soundex_edge_cases():
    assert soundex("") == ""
    assert soundex(None) == ""
    assert soundex("1234") == ""
    # Non-Latin tokens return empty string safely
    assert soundex("Иван") == ""
    # Single letter
    assert soundex("A") == "A000"


def test_normalize_attribute_value():
    assert normalize_attribute_value(None) == ""
    assert normalize_attribute_value("  Detroit  ") == "detroit"
    assert normalize_attribute_value(1980) == "1980"
    assert normalize_attribute_value(True) == "true"
    
    # Dates
    d = datetime.date(1980, 1, 1)
    assert normalize_attribute_value(d) == "1980-01-01"
    
    # Collections
    assert normalize_attribute_value(["apple", "banana"]) == "apple,banana"
    assert normalize_attribute_value({"b": 2, "a": 1}) == "a:1;b:2"
