import pytest
from unittest.mock import Mock, patch
import json
from typing import List
from lib.index.terms.terms import (terms_from_txt)

# METHOD:
# def terms_from_txt(text: str, max_length: int = 100) -> List[str]:
def test_terms_from_txt():
    text = "This is a sample text"
    expected_terms = ["this", "is", "a", "sample", "text"]
    assert terms_from_txt(text) == expected_terms

def test_terms_from_txt_with_max_length():
    text = "This is a sample text"
    max_length = 3
    expected_terms = ["this", "is", "a"]
    assert terms_from_txt(text, max_length) == expected_terms

def test_terms_from_txt_with_characters_to_replace():
    text = "This is a sample text"
    characters_to_replace_before = ["a", "s"]
    characters_replace_by = ["b", "t"]
    expected_terms = ["thi", "is", "b", "sample", "text"]
    assert terms_from_txt(text, characters_to_replace_before, characters_replace_by) == expected_terms


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `terms_from_txt` method:
# ```python
# def test_terms_from_txt():
#     text = "This is a sample text"
#     expected_terms = ["this", "is", "a", "sample", "text"]
#     assert terms_from_txt(text) == expected_terms
# 
# def test_terms_from_txt_with_max_length():
#     text = "This is a sample text"
#     max_length = 3
#     expected_terms = ["this", "is", "a"]
#     assert terms_from_txt(text, max_length) == expected_terms
# 
# def test_terms_from_txt_with_characters_to_replace():
#     text = "This is a sample text"
#     characters_to_replace_before = ["a", "s"]
#     characters_replace_by = ["b", "t"]
#     expected_terms = ["thi", "is", "b", "sample", "text"]
#     assert terms_from_txt(text, characters_to_replace_before, characters_replace_by) == expected_terms
# ```

