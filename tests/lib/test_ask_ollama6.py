import pytest
from unittest.mock import Mock, patch
import json
from lib.index.doc_sum_index import get_doc_sum_index_query_engine
from lib.vector_chroma import get_vector_query_engine, get_vector_ng_query_engine
from lib.index.terms.kg_num_term_neo4j import get_graph_query_engine
from lib.ask import (ask_for_vector_response, ask_for_vector_ng_response, ask_for_graph_response, ask_for_doc_sum_index_response)

# METHOD:
# def ask_for_vector_response(collection, question) -> str:
def test_ask_for_vector_response_valid():
    collection = "test_collection"
    question = "What is the capital of France?"
    expected_response = "Paris"
    assert ask_for_vector_response(collection, question) == expected_response

def test_ask_for_vector_response_invalid_collection():
    collection = "invalid_collection"
    question = "What is the capital of France?"
    with pytest.raises(ValueError):
        ask_for_vector_response(collection, question)

def test_ask_for_vector_response_invalid_question():
    collection = "test_collection"
    question = "Invalid question"
    with pytest.raises(ValueError):
        ask_for_vector_response(collection, question)

def test_ask_for_vector_response_empty_response():
    collection = "test_collection"
    question = "What is the capital of France?"
    expected_response = ""
    assert ask_for_vector_response(collection, question) == expected_response


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `ask_for_vector_response` method:
# 
# 1. Test that the method returns a string response when given a valid collection and question:
# ```python
# def test_ask_for_vector_response_valid():
#     collection = "test_collection"
#     question = "What is the capital of France?"
#     expected_response = "Paris"
#     assert ask_for_vector_response(collection, question) == expected_response
# ```
# 2. Test that the method raises a `ValueError` when given an invalid collection:
# ```python
# def test_ask_for_vector_response_invalid_collection():
#     collection = "invalid_collection"
#     question = "What is the capital of France?"
#     with pytest.raises(ValueError):
#         ask_for_vector_response(collection, question)
# ```
# 3. Test that the method raises a `ValueError` when given an invalid question:
# ```python
# def test_ask_for_vector_response_invalid_question():
#     collection = "test_collection"
#     question = "Invalid question"
#     with pytest.raises(ValueError):
#         ask_for_vector_response(collection, question)
# ```
# 4. Test that the method returns a response when given a valid collection and question, even if the response is empty:
# ```python
# def test_ask_for_vector_response_empty_response():
#     collection = "test_collection"
#     question = "What is the capital of France?"
#     expected_response = ""
#     assert ask_for_vector_response(collection, question) == expected_response
# ```



# METHOD:
# def ask_for_vector_ng_response(vector_storage_dir, question) -> str:
def test_ask_for_vector_ng_response_valid():
    vector_storage_dir = "path/to/vector/storage"
    question = "What is the capital of France?"
    expected_response = "Paris"
    assert ask_for_vector_ng_response(vector_storage_dir, question) == expected_response

def test_ask_for_vector_ng_response_invalid_directory():
    vector_storage_dir = "path/to/invalid/directory"
    question = "What is the capital of France?"
    with pytest.raises(ValueError):
        ask_for_vector_ng_response(vector_storage_dir, question)

def test_ask_for_vector_ng_response_invalid_question():
    vector_storage_dir = "path/to/vector/storage"
    question = 1234567890
    with pytest.raises(TypeError):
        ask_for_vector_ng_response(vector_storage_dir, question)

def test_ask_for_vector_ng_response_empty_question():
    vector_storage_dir = "path/to/vector/storage"
    question = ""
    expected_response = ""
    assert ask_for_vector_ng_response(vector_storage_dir, question) == expected_response


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `ask_for_vector_ng_response` method:
# 
# 1. Test that the method returns a string response when given a valid vector storage directory and question:
# ```python
# def test_ask_for_vector_ng_response_valid():
#     vector_storage_dir = "path/to/vector/storage"
#     question = "What is the capital of France?"
#     expected_response = "Paris"
#     assert ask_for_vector_ng_response(vector_storage_dir, question) == expected_response
# ```
# 2. Test that the method raises a `ValueError` when given an invalid vector storage directory:
# ```python
# def test_ask_for_vector_ng_response_invalid_directory():
#     vector_storage_dir = "path/to/invalid/directory"
#     question = "What is the capital of France?"
#     with pytest.raises(ValueError):
#         ask_for_vector_ng_response(vector_storage_dir, question)
# ```
# 3. Test that the method raises a `TypeError` when given an invalid question:
# ```python
# def test_ask_for_vector_ng_response_invalid_question():
#     vector_storage_dir = "path/to/vector/storage"
#     question = 1234567890
#     with pytest.raises(TypeError):
#         ask_for_vector_ng_response(vector_storage_dir, question)
# ```
# 4. Test that the method returns an empty string when given a valid vector storage directory and an empty question:
# ```python
# def test_ask_for_vector_ng_response_empty_question():
#     vector_storage_dir = "path/to/vector/storage"
#     question = ""
#     expected_response = ""
#     assert ask_for_vector_ng_response(vector_storage_dir, question) == expected_response
# ```



# METHOD:
# def ask_for_graph_response(vector_storage_dir, question) -> str:
def test_ask_for_graph_response_returns_string(vector_storage_dir, question):
    assert isinstance(ask_for_graph_response(vector_storage_dir, question), str)

def test_ask_for_graph_response_raises_value_error_with_invalid_vector_storage_dir(question):
    with pytest.raises(ValueError):
        ask_for_graph_response("invalid/path", question)

def test_ask_for_graph_response_raises_type_error_with_invalid_question(vector_storage_dir):
    with pytest.raises(TypeError):
        ask_for_graph_response(vector_storage_dir, 123)

def test_ask_for_graph_response_returns_response(vector_storage_dir, question):
    assert ask_for_graph_response(vector_storage_dir, question) == "some response"

def test_ask_for_graph_response_returns_response_with_special_characters(vector_storage_dir, question):
    assert ask_for_graph_response(vector_storage_dir, "some question with special characters") == "some response"


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `ask_for_graph_response` method:
# 
# 1. Test that the method returns a string response when given a valid vector storage directory and a question:
# ```python
# def test_ask_for_graph_response_returns_string(vector_storage_dir, question):
#     assert isinstance(ask_for_graph_response(vector_storage_dir, question), str)
# ```
# 2. Test that the method raises a `ValueError` when given an invalid vector storage directory:
# ```python
# def test_ask_for_graph_response_raises_value_error_with_invalid_vector_storage_dir(question):
#     with pytest.raises(ValueError):
#         ask_for_graph_response("invalid/path", question)
# ```
# 3. Test that the method raises a `TypeError` when given an invalid question:
# ```python
# def test_ask_for_graph_response_raises_type_error_with_invalid_question(vector_storage_dir):
#     with pytest.raises(TypeError):
#         ask_for_graph_response(vector_storage_dir, 123)
# ```
# 4. Test that the method returns a response when given a valid vector storage directory and a question:
# ```python
# def test_ask_for_graph_response_returns_response(vector_storage_dir, question):
#     assert ask_for_graph_response(vector_storage_dir, question) == "some response"
# ```
# 5. Test that the method returns a response when given a valid vector storage directory and a question with special characters:
# ```python
# def test_ask_for_graph_response_returns_response_with_special_characters(vector_storage_dir, question):
#     assert ask_for_graph_response(vector_storage_dir, "some question with special characters") == "some response"
# ```



# METHOD:
# def ask_for_doc_sum_index_response(storage_dir, question) -> str:
def test_ask_for_doc_sum_index_response_valid():
    storage_dir = "path/to/storage"
    question = "What is the meaning of life?"
    expected_response = "The meaning of life is to find purpose and fulfillment in life."
    assert ask_for_doc_sum_index_response(storage_dir, question) == expected_response

def test_ask_for_doc_sum_index_response_invalid_storage():
    storage_dir = "path/to/invalid/storage"
    question = "What is the meaning of life?"
    with pytest.raises(ValueError):
        ask_for_doc_sum_index_response(storage_dir, question)

def test_ask_for_doc_sum_index_response_invalid_question():
    storage_dir = "path/to/storage"
    question = ""
    with pytest.raises(ValueError):
        ask_for_doc_sum_index_response(storage_dir, question)

def test_ask_for_doc_sum_index_response_multiple_words():
    storage_dir = "path/to/storage"
    question = "What is the meaning of life? What is the purpose of life?"
    expected_response = "The meaning of life is to find purpose and fulfillment in life. The purpose of life is to seek happiness and fulfillment."
    assert ask_for_doc_sum_index_response(storage_dir, question) == expected_response

def test_ask_for_doc_sum_index_response_special_characters():
    storage_dir = "path/to/storage"
    question = "What is the meaning of life? What is the purpose of life?"
    expected_response = "The meaning of life is to find purpose and fulfillment in life. The purpose of life is to seek happiness and fulfillment."
    assert ask_for_doc_sum_index_response(storage_dir, question) == expected_response


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `ask_for_doc_sum_index_response` method:
# 
# 1. Test that the method returns a string response when given a valid storage directory and question:
# ```python
# def test_ask_for_doc_sum_index_response_valid():
#     storage_dir = "path/to/storage"
#     question = "What is the meaning of life?"
#     expected_response = "The meaning of life is to find purpose and fulfillment in life."
#     assert ask_for_doc_sum_index_response(storage_dir, question) == expected_response
# ```
# 2. Test that the method raises a `ValueError` when given an invalid storage directory:
# ```python
# def test_ask_for_doc_sum_index_response_invalid_storage():
#     storage_dir = "path/to/invalid/storage"
#     question = "What is the meaning of life?"
#     with pytest.raises(ValueError):
#         ask_for_doc_sum_index_response(storage_dir, question)
# ```
# 3. Test that the method raises a `ValueError` when given an invalid question:
# ```python
# def test_ask_for_doc_sum_index_response_invalid_question():
#     storage_dir = "path/to/storage"
#     question = ""
#     with pytest.raises(ValueError):
#         ask_for_doc_sum_index_response(storage_dir, question)
# ```
# 4. Test that the method returns a response when given a valid storage directory and a question with multiple words:
# ```python
# def test_ask_for_doc_sum_index_response_multiple_words():
#     storage_dir = "path/to/storage"
#     question = "What is the meaning of life? What is the purpose of life?"
#     expected_response = "The meaning of life is to find purpose and fulfillment in life. The purpose of life is to seek happiness and fulfillment."
#     assert ask_for_doc_sum_index_response(storage_dir, question) == expected_response
# ```
# 5. Test that the method returns a response when given a valid storage directory and a question with special characters:
# ```python
# def test_ask_for_doc_sum_index_response_special_characters():
#     storage_dir = "path/to/storage"
#     question = "What is the meaning of life? What is the purpose of life?"
#     expected_response = "The meaning of life is to find purpose and fulfillment in life. The purpose of life is to seek happiness and fulfillment."
#     assert ask_for_doc_sum_index_response(storage_dir, question) == expected_response
# ```

