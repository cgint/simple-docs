import pytest
from unittest.mock import Mock, patch
import json
from typing import List
from llama_index.core import Document
from llama_index.core.readers.json import JSONReader
from lib.json import (get_content_from_json_file)

# METHOD:
# def get_content_from_json_file(json_file_path: str, source_type: str, is_jsonl: bool = False) -> List[Document]:
def test_get_content_from_json_file_returns_list_of_documents():
    json_file_path = "test_data/test_json_file.json"
    source_type = "json"
    docs = get_content_from_json_file(json_file_path, source_type)
    assert isinstance(docs, list)
    for doc in docs:
        assert isinstance(doc, Document)

def test_get_content_from_json_file_returns_empty_list_when_file_does_not_exist():
    json_file_path = "test_data/non_existent_file.json"
    source_type = "json"
    docs = get_content_from_json_file(json_file_path, source_type)
    assert len(docs) == 0

def test_get_content_from_json_file_returns_empty_list_when_data_is_invalid():
    json_file_path = "test_data/invalid_json_file.json"
    source_type = "json"
    docs = get_content_from_json_file(json_file_path, source_type)
    assert len(docs) == 0

def test_get_content_from_json_file_sets_metadata_correctly():
    json_file_path = "test_data/test_json_file.json"
    source_type = "json"
    docs = get_content_from_json_file(json_file_path, source_type)
    for doc in docs:
        assert doc.metadata["source_id"] == json_file_path
        assert doc.metadata["source_type"] == source_type


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_content_from_json_file` method:
# 
# 1. Test that the method returns a list of documents when given a valid JSON file path and source type:
# ```python
# def test_get_content_from_json_file_returns_list_of_documents():
#     json_file_path = "test_data/test_json_file.json"
#     source_type = "json"
#     docs = get_content_from_json_file(json_file_path, source_type)
#     assert isinstance(docs, list)
#     for doc in docs:
#         assert isinstance(doc, Document)
# ```
# 2. Test that the method returns an empty list when given a JSON file path that does not exist:
# ```python
# def test_get_content_from_json_file_returns_empty_list_when_file_does_not_exist():
#     json_file_path = "test_data/non_existent_file.json"
#     source_type = "json"
#     docs = get_content_from_json_file(json_file_path, source_type)
#     assert len(docs) == 0
# ```
# 3. Test that the method returns an empty list when given a JSON file path that contains invalid JSON data:
# ```python
# def test_get_content_from_json_file_returns_empty_list_when_data_is_invalid():
#     json_file_path = "test_data/invalid_json_file.json"
#     source_type = "json"
#     docs = get_content_from_json_file(json_file_path, source_type)
#     assert len(docs) == 0
# ```
# 4. Test that the method sets the `source_id` and `source_type` metadata fields correctly for each document:
# ```python
# def test_get_content_from_json_file_sets_metadata_correctly():
#     json_file_path = "test_data/test_json_file.json"
#     source_type = "json"
#     docs = get_content_from_json_file(json_file_path, source_type)
#     for doc in docs:
#         assert doc.metadata["source_id"] == json_file_path
#         assert doc.metadata["source_type"] == source_type
# ```

