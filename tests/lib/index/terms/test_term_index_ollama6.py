import os
import pytest
from unittest.mock import Mock, patch
import json
import json
from llama_index.core import Document
import pandas as pd
from lib.index.helper import cur_simple_date_time_sec
from lib.index.terms.terms import terms_from_txt
from lib import constants
from lib.index.terms.term_index import (build_term_reference_index, write_term_references_to_file, count_terms_per_document, get_term_to_doc_items)

# METHOD:
# def build_term_reference_index(doc: Document):
def test_build_term_reference_index_single_doc():
    doc = Document("Test document", "This is a test document.")
    build_term_reference_index(doc)
    assert len(term_to_doc) == 3
    assert term_to_doc["test"] == {doc.doc_id: 1}
    assert term_to_doc["document"] == {doc.doc_id: 1}
    assert term_to_doc["this"] == {doc.doc_id: 1}
    assert len(doc_to_term) == 3
    assert doc_to_term[doc.doc_id] == {"test": 1, "document": 1, "this": 1}

def test_build_term_reference_index_multiple_docs():
    doc1 = Document("Test document 1", "This is a test document.")
    doc2 = Document("Test document 2", "This is another test document.")
    build_term_reference_index(doc1)
    build_term_reference_index(doc2)
    assert len(term_to_doc) == 6
    assert term_to_doc["test"] == {doc1.doc_id: 1, doc2.doc_id: 1}
    assert term_to_doc["document"] == {doc1.doc_id: 1, doc2.doc_id: 1}
    assert term_to_doc["this"] == {doc1.doc_id: 1, doc2.doc_id: 1}
    assert len(doc_to_term) == 6
    assert doc_to_term[doc1.doc_id] == {"test": 1, "document": 1, "this": 1}
    assert doc_to_term[doc2.doc_id] == {"test": 1, "document": 1, "this": 1}

def test_build_term_reference_index_duplicate_terms():
    doc = Document("Test document", "This is a test document. This is another test document.")
    build_term_reference_index(doc)
    assert len(term_to_doc) == 4
    assert term_to_doc["test"] == {doc.doc_id: 2}
    assert term_to_doc["document"] == {doc.doc_id: 1}
    assert term_to_doc["this"] == {doc.doc_id: 2}
    assert len(doc_to_term) == 4
    assert doc_to_term[doc.doc_id] == {"test": 2, "document": 1, "this": 2}

def test_build_term_reference_index_special_characters():
    doc = Document("Test document", "This is a test document. This is another test document.")
    build_term_reference_index(doc)
    assert len(term_to_doc) == 4
    assert term_to_doc["test"] == {doc.doc_id: 2}
    assert term_to_doc["document"] == {doc.doc_id: 1}
    assert term_to_doc["this"] == {doc.doc_id: 2}
    assert len(doc_to_term) == 4
    assert doc_to_term[doc.doc_id] == {"test": 2, "document": 1, "this": 2}


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `build_term_reference_index` method:
# 
# 1. Test that the method correctly builds the term reference index for a single document:
# ```python
# def test_build_term_reference_index_single_doc():
#     doc = Document("Test document", "This is a test document.")
#     build_term_reference_index(doc)
#     assert len(term_to_doc) == 3
#     assert term_to_doc["test"] == {doc.doc_id: 1}
#     assert term_to_doc["document"] == {doc.doc_id: 1}
#     assert term_to_doc["this"] == {doc.doc_id: 1}
#     assert len(doc_to_term) == 3
#     assert doc_to_term[doc.doc_id] == {"test": 1, "document": 1, "this": 1}
# ```
# 2. Test that the method correctly builds the term reference index for multiple documents:
# ```python
# def test_build_term_reference_index_multiple_docs():
#     doc1 = Document("Test document 1", "This is a test document.")
#     doc2 = Document("Test document 2", "This is another test document.")
#     build_term_reference_index(doc1)
#     build_term_reference_index(doc2)
#     assert len(term_to_doc) == 6
#     assert term_to_doc["test"] == {doc1.doc_id: 1, doc2.doc_id: 1}
#     assert term_to_doc["document"] == {doc1.doc_id: 1, doc2.doc_id: 1}
#     assert term_to_doc["this"] == {doc1.doc_id: 1, doc2.doc_id: 1}
#     assert len(doc_to_term) == 6
#     assert doc_to_term[doc1.doc_id] == {"test": 1, "document": 1, "this": 1}
#     assert doc_to_term[doc2.doc_id] == {"test": 1, "document": 1, "this": 1}
# ```
# 3. Test that the method correctly handles duplicate terms in a document:
# ```python
# def test_build_term_reference_index_duplicate_terms():
#     doc = Document("Test document", "This is a test document. This is another test document.")
#     build_term_reference_index(doc)
#     assert len(term_to_doc) == 4
#     assert term_to_doc["test"] == {doc.doc_id: 2}
#     assert term_to_doc["document"] == {doc.doc_id: 1}
#     assert term_to_doc["this"] == {doc.doc_id: 2}
#     assert len(doc_to_term) == 4
#     assert doc_to_term[doc.doc_id] == {"test": 2, "document": 1, "this": 2}
# ```
# 4. Test that the method correctly handles terms with special characters:
# ```python
# def test_build_term_reference_index_special_characters():
#     doc = Document("Test document", "This is a test document. This is another test document.")
#     build_term_reference_index(doc)
#     assert len(term_to_doc) == 4
#     assert term_to_doc["test"] == {doc.doc_id: 2}
#     assert term_to_doc["document"] == {doc.doc_id: 1}
#     assert term_to_doc["this"] == {doc.doc_id: 2}
#     assert len(doc_to_term) == 4
#     assert doc_to_term[doc.doc_id] == {"test": 2, "document": 1, "this": 2}
# ```
# 5. Test that the method correctly handles terms with punctuation:
# ```python
# def test_build_term_reference_index_punctuation():
#     doc = Document("Test document", "This is a test document. This is another test document.")
#     build_term_reference_index(doc)
#     assert len(term_to_doc) == 4
#     assert term_to_doc["test"] == {doc.doc_id: 2}
#     assert term_to_



# METHOD:
# def write_term_references_to_file():
def test_write_term_references_to_file(tmpdir):
    time = cur_simple_date_time_sec()
    with open(f"{constants.term_data_dir}/term_to_doc_{time}.json", "w") as f:
        json.dump({}, f, indent=2)
    with open(f"{constants.term_data_dir}/doc_to_term_{time}.json", "w") as f:
        json.dump({}, f, indent=2)
    
    write_term_references_to_file()
    
    assert os.path.exists(f"{constants.term_data_dir}/term_to_doc_{time}.json")
    assert os.path.exists(f"{constants.term_data_dir}/doc_to_term_{time}.json")

def test_write_term_references_to_file_with_data(tmpdir):
    time = cur_simple_date_time_sec()
    term_to_doc = {"term1": ["doc1", "doc2"], "term2": ["doc3"]}
    doc_to_term = {"doc1": ["term1", "term2"], "doc2": ["term1"], "doc3": ["term2"]}
    
    write_term_references_to_file(term_to_doc, doc_to_term)
    
    with open(f"{constants.term_data_dir}/term_to_doc_{time}.json", "r") as f:
        assert json.load(f) == term_to_doc
    with open(f"{constants.term_data_dir}/doc_to_term_{time}.json", "r") as f:
        assert json.load(f) == doc_to_term

def test_write_term_references_to_file_with_invalid_data(tmpdir):
    time = cur_simple_date_time_sec()
    
    with pytest.raises(ValueError):
        write_term_references_to_file({}, {})


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `write_term_references_to_file` method:
# 
# 1. Test that the method writes the term-to-document and document-to-term JSON files to the correct directory:
# ```python
# def test_write_term_references_to_file(tmpdir):
#     time = cur_simple_date_time_sec()
#     with open(f"{constants.term_data_dir}/term_to_doc_{time}.json", "w") as f:
#         json.dump({}, f, indent=2)
#     with open(f"{constants.term_data_dir}/doc_to_term_{time}.json", "w") as f:
#         json.dump({}, f, indent=2)
#     
#     write_term_references_to_file()
#     
#     assert os.path.exists(f"{constants.term_data_dir}/term_to_doc_{time}.json")
#     assert os.path.exists(f"{constants.term_data_dir}/doc_to_term_{time}.json")
# ```
# 2. Test that the method writes the correct data to the JSON files:
# ```python
# def test_write_term_references_to_file_with_data(tmpdir):
#     time = cur_simple_date_time_sec()
#     term_to_doc = {"term1": ["doc1", "doc2"], "term2": ["doc3"]}
#     doc_to_term = {"doc1": ["term1", "term2"], "doc2": ["term1"], "doc3": ["term2"]}
#     
#     write_term_references_to_file(term_to_doc, doc_to_term)
#     
#     with open(f"{constants.term_data_dir}/term_to_doc_{time}.json", "r") as f:
#         assert json.load(f) == term_to_doc
#     with open(f"{constants.term_data_dir}/doc_to_term_{time}.json", "r") as f:
#         assert json.load(f) == doc_to_term
# ```
# 3. Test that the method handles invalid input data correctly:
# ```python
# def test_write_term_references_to_file_with_invalid_data(tmpdir):
#     time = cur_simple_date_time_sec()
#     
#     with pytest.raises(ValueError):
#         write_term_references_to_file({}, {})
# ```



# METHOD:
# def count_terms_per_document():
def test_term_count_dataframe():
    term_count = count_terms_per_document()
    assert isinstance(term_count, pd.DataFrame)
    assert set(term_count.columns) == {"term", "num_docs", "num_all"}

def test_term_count_dataframe_values():
    term_count = count_terms_per_document()
    assert len(term_count) == 3
    assert term_count["term"][0] == "apple"
    assert term_count["num_docs"][0] == 2
    assert term_count["num_all"][0] == 10

def test_term_count_dataframe_sorting():
    term_count = count_terms_per_document()
    assert list(term_count["term"]) == ["banana", "orange", "apple"]

def test_term_count_dataframe_save():
    term_count = count_terms_per_document()
    assert os.path.exists(f"{constants.term_data_dir}/term_count_{cur_simple_date_time_sec()}.csv")


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `count_terms_per_document` method:
# 
# 1. Test that the method returns a DataFrame with the correct columns:
# ```python
# def test_term_count_dataframe():
#     term_count = count_terms_per_document()
#     assert isinstance(term_count, pd.DataFrame)
#     assert set(term_count.columns) == {"term", "num_docs", "num_all"}
# ```
# 2. Test that the method returns a DataFrame with the correct data:
# ```python
# def test_term_count_dataframe_values():
#     term_count = count_terms_per_document()
#     assert len(term_count) == 3
#     assert term_count["term"][0] == "apple"
#     assert term_count["num_docs"][0] == 2
#     assert term_count["num_all"][0] == 10
# ```
# 3. Test that the method sorts the DataFrame by `num_all` in descending order:
# ```python
# def test_term_count_dataframe_sorting():
#     term_count = count_terms_per_document()
#     assert list(term_count["term"]) == ["banana", "orange", "apple"]
# ```
# 4. Test that the method saves the DataFrame to a CSV file with the correct name:
# ```python
# def test_term_count_dataframe_save():
#     term_count = count_terms_per_document()
#     assert os.path.exists(f"{constants.term_data_dir}/term_count_{cur_simple_date_time_sec()}.csv")
# ```



# METHOD:
# def get_term_to_doc_items():
import pytest
from lib.index.html import get_term_to_doc_items

def test_get_term_to_doc_items():
    term_to_doc = {"term1": "doc1", "term2": "doc2"}
    assert get_term_to_doc_items(term_to_doc) == [("term1", "doc1"), ("term2", "doc2")]

def test_get_term_to_doc_items_empty():
    term_to_doc = {}
    assert get_term_to_doc_items(term_to_doc) == []

def test_get_term_to_doc_items_invalid_input():
    with pytest.raises(TypeError):
        get_term_to_doc_items("not a dict")


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_term_to_doc_items` method:
# ```python
# import pytest
# from lib.index.html import get_term_to_doc_items
# 
# def test_get_term_to_doc_items():
#     term_to_doc = {"term1": "doc1", "term2": "doc2"}
#     assert get_term_to_doc_items(term_to_doc) == [("term1", "doc1"), ("term2", "doc2")]
# 
# def test_get_term_to_doc_items_empty():
#     term_to_doc = {}
#     assert get_term_to_doc_items(term_to_doc) == []
# 
# def test_get_term_to_doc_items_invalid_input():
#     with pytest.raises(TypeError):
#         get_term_to_doc_items("not a dict")
# ```

