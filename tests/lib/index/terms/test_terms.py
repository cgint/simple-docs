import pytest
from lib.terms import terms_from_txt

def test_basic_terms_extraction():
    text = "Hello, world!"
    assert terms_from_txt(text) == ["hello", "world"]

def test_with_special_characters():
    text = "Hello, <world>!"
    assert terms_from_txt(text) == ["hello", "world"]

def test_with_ignored_terms():
    text = "Hello world - #"
    assert terms_from_txt(text) == ["hello", "world"]

def test_max_length_constraint():
    text = "Hello " + "a" * 101
    assert terms_from_txt(text) == ["hello"]

def test_newlines_and_tabs():
    text = "Hello\nworld\t!"
    assert terms_from_txt(text) == ["hello", "world"]

def test_empty_string():
    assert terms_from_txt("") == []

def test_only_special_characters():
    text = "<>;:"
    assert terms_from_txt(text) == []
