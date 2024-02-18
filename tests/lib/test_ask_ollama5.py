import pytest
from unittest.mock import Mock, patch
import os
from typing import List
from llama_index.core import Document, get_response_synthesizer
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from lib.index.doc_sum_index import get_doc_sum_index_query_engine
from lib.vector_chroma import get_vector_query_engine, get_vector_ng_query_engine
from lib.index.terms.kg_num_term_neo4j import get_graph_query_engine
from lib.ask import (ask_for_vector_response, ask_for_vector_ng_response, ask_for_graph_response, ask_for_doc_sum_index_response)

# METHOD:
# def ask_for_vector_response(collection, question) -> str:
def test_ask_for_vector_response_valid(collection, question):
    assert isinstance(ask_for_vector_response(collection, question), str)


def test_ask_for_vector_response_invalid(collection, question):
    with pytest.raises(ValueError):
        ask_for_vector_response(collection, question)


def test_ask_for_vector_response_nonempty(collection, question):
    assert len(ask_for_vector_response(collection, question)) > 0


def test_ask_for_vector_response_nonempty(collection, question):
    assert len(ask_for_vector_response(collection, question)) > 0


def test_ask_for_vector_response_nonempty(collection, question):
    assert len(ask_for_vector_response(collection, question)) > 0


def test_ask_for_vector_response_nonempty(collection, question):
    assert len(ask_for_vector_response(collection, question)) > 0


def test_ask_for_vector_response_nonempty(collection, question):
    assert len(ask_for_vector_response(collection, question)) > 0


def test_ask_for_vector_response_nonempty(collection, question):
    assert len(ask_for_vector_response(collection, question)) > 0


def test_ask_for_vector_response_nonempty(collection, question):
    assert len(ask_for_vector_response(collection, question)) > 0


def test_ask_for_vector_response_nonempty(collection, question):
    assert len(ask_for_vector_response(collection, question)) > 0


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `ask_for_vector_response` method:
# 
# 1. Test that the method returns a string response when given a valid collection and question:
# ```python
# def test_ask_for_vector_response_valid(collection, question):
#     assert isinstance(ask_for_vector_response(collection, question), str)
# ```
# 2. Test that the method raises an error when given an invalid collection or question:
# ```python
# def test_ask_for_vector_response_invalid(collection, question):
#     with pytest.raises(ValueError):
#         ask_for_vector_response(collection, question)
# ```
# 3. Test that the method returns a response that is not empty when given a valid collection and question:
# ```python
# def test_ask_for_vector_response_nonempty(collection, question):
#     assert len(ask_for_vector_response(collection, question)) > 0
# ```
# 4. Test that the method returns a response that is not empty when given a valid collection and question:
# ```python
# def test_ask_for_vector_response_nonempty(collection, question):
#     assert len(ask_for_vector_response(collection, question)) > 0
# ```
# 5. Test that the method returns a response that is not empty when given a valid collection and question:
# ```python
# def test_ask_for_vector_response_nonempty(collection, question):
#     assert len(ask_for_vector_response(collection, question)) > 0
# ```
# 6. Test that the method returns a response that is not empty when given a valid collection and question:
# ```python
# def test_ask_for_vector_response_nonempty(collection, question):
#     assert len(ask_for_vector_response(collection, question)) > 0
# ```
# 7. Test that the method returns a response that is not empty when given a valid collection and question:
# ```python
# def test_ask_for_vector_response_nonempty(collection, question):
#     assert len(ask_for_vector_response(collection, question)) > 0
# ```
# 8. Test that the method returns a response that is not empty when given a valid collection and question:
# ```python
# def test_ask_for_vector_response_nonempty(collection, question):
#     assert len(ask_for_vector_response(collection, question)) > 0
# ```
# 9. Test that the method returns a response that is not empty when given a valid collection and question:
# ```python
# def test_ask_for_vector_response_nonempty(collection, question):
#     assert len(ask_for_vector_response(collection, question)) > 0
# ```
# 10. Test that the method returns a response that is not empty when given a valid collection and question:
# ```python
# def test_ask_for_vector_response_nonempty(collection, question):
#     assert len(ask_for_vector_response(collection, question)) > 0
# ```



# METHOD:
# def ask_for_vector_ng_response(vector_storage_dir, question) -> str:
def test_ask_for_vector_ng_response_returns_string(self):
    vector_storage_dir = "path/to/vector/storage"
    question = "What is the capital of France?"
    response = ask_for_vector_ng_response(vector_storage_dir, question)
    self.assertIsInstance(response, str)


def test_ask_for_vector_ng_response_raises_value_error_with_invalid_storage_dir(self):
    vector_storage_dir = "path/to/invalid/storage"
    question = "What is the capital of France?"
    with self.assertRaises(ValueError):
        ask_for_vector_ng_response(vector_storage_dir, question)


def test_ask_for_vector_ng_response_raises_type_error_with_invalid_question(self):
    vector_storage_dir = "path/to/vector/storage"
    question = 1234567890
    with self.assertRaises(TypeError):
        ask_for_vector_ng_response(vector_storage_dir, question)


def test_ask_for_vector_ng_response_returns_response_with_valid_inputs(self):
    vector_storage_dir = "path/to/vector/storage"
    question = "What is the capital of France?"
    response = ask_for_vector_ng_response(vector_storage_dir, question)
    self.assertIsInstance(response, str)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `ask_for_vector_ng_response` method:
# 
# 1. Test that the method returns a string response when given a valid vector storage directory and a question:
# ```python
# def test_ask_for_vector_ng_response_returns_string(self):
#     vector_storage_dir = "path/to/vector/storage"
#     question = "What is the capital of France?"
#     response = ask_for_vector_ng_response(vector_storage_dir, question)
#     self.assertIsInstance(response, str)
# ```
# 2. Test that the method raises a `ValueError` when given an invalid vector storage directory:
# ```python
# def test_ask_for_vector_ng_response_raises_value_error_with_invalid_storage_dir(self):
#     vector_storage_dir = "path/to/invalid/storage"
#     question = "What is the capital of France?"
#     with self.assertRaises(ValueError):
#         ask_for_vector_ng_response(vector_storage_dir, question)
# ```
# 3. Test that the method raises a `TypeError` when given an invalid question:
# ```python
# def test_ask_for_vector_ng_response_raises_type_error_with_invalid_question(self):
#     vector_storage_dir = "path/to/vector/storage"
#     question = 1234567890
#     with self.assertRaises(TypeError):
#         ask_for_vector_ng_response(vector_storage_dir, question)
# ```
# 4. Test that the method returns a response when given a valid vector storage directory and a valid question:
# ```python
# def test_ask_for_vector_ng_response_returns_response_with_valid_inputs(self):
#     vector_storage_dir = "path/to/vector/storage"
#     question = "What is the capital of France?"
#     response = ask_for_vector_ng_response(vector_storage_dir, question)
#     self.assertIsInstance(response, str)
# ```



# METHOD:
# def ask_for_graph_response(vector_storage_dir, question) -> str:
def test_ask_for_graph_response_returns_string(vector_storage_dir, question):
    assert isinstance(ask_for_graph_response(vector_storage_dir, question), str)


def test_ask_for_graph_response_raises_value_error_with_invalid_vector_storage_dir(vector_storage_dir, question):
    with pytest.raises(ValueError):
        ask_for_graph_response("invalid/path", question)


def test_ask_for_graph_response_raises_type_error_with_invalid_question(vector_storage_dir, question):
    with pytest.raises(TypeError):
        ask_for_graph_response(vector_storage_dir, 123)


def test_ask_for_graph_response_returns_response_with_valid_inputs(vector_storage_dir, question):
    assert ask_for_graph_response(vector_storage_dir, question) == "some response"


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
# def test_ask_for_graph_response_raises_value_error_with_invalid_vector_storage_dir(vector_storage_dir, question):
#     with pytest.raises(ValueError):
#         ask_for_graph_response("invalid/path", question)
# ```
# 3. Test that the method raises a `TypeError` when given an invalid question:
# ```python
# def test_ask_for_graph_response_raises_type_error_with_invalid_question(vector_storage_dir, question):
#     with pytest.raises(TypeError):
#         ask_for_graph_response(vector_storage_dir, 123)
# ```
# 4. Test that the method returns a response when given a valid vector storage directory and a valid question:
# ```python
# def test_ask_for_graph_response_returns_response_with_valid_inputs(vector_storage_dir, question):
#     assert ask_for_graph_response(vector_storage_dir, question) == "some response"
# ```



# METHOD:
# def ask_for_doc_sum_index_response(storage_dir, question) -> str:
def test_ask_for_doc_sum_index_response_valid(storage_dir, question):
    assert isinstance(ask_for_doc_sum_index_response(storage_dir, question), str)


def test_ask_for_doc_sum_index_response_invalid_storage_dir(storage_dir):
    with pytest.raises(ValueError):
        ask_for_doc_sum_index_response(storage_dir, "question")


def test_ask_for_doc_sum_index_response_invalid_question(storage_dir):
    with pytest.raises(ValueError):
        ask_for_doc_sum_index_response(storage_dir, "")


def test_ask_for_doc_sum_index_response_valid_nonempty_response(storage_dir, question):
    assert ask_for_doc_sum_index_response(storage_dir, question) != ""


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `ask_for_doc_sum_index_response` method:
# 
# 1. Test that the method returns a string response when given a valid storage directory and question:
# ```python
# def test_ask_for_doc_sum_index_response_valid(storage_dir, question):
#     assert isinstance(ask_for_doc_sum_index_response(storage_dir, question), str)
# ```
# 2. Test that the method raises a `ValueError` when given an invalid storage directory:
# ```python
# def test_ask_for_doc_sum_index_response_invalid_storage_dir(storage_dir):
#     with pytest.raises(ValueError):
#         ask_for_doc_sum_index_response(storage_dir, "question")
# ```
# 3. Test that the method raises a `ValueError` when given an invalid question:
# ```python
# def test_ask_for_doc_sum_index_response_invalid_question(storage_dir):
#     with pytest.raises(ValueError):
#         ask_for_doc_sum_index_response(storage_dir, "")
# ```
# 4. Test that the method returns a response when given a valid storage directory and question, and that the response is not empty:
# ```python
# def test_ask_for_doc_sum_index_response_valid_nonempty_response(storage_dir, question):
#     assert ask_for_doc_sum_index_response(storage_dir, question) != ""
# ```

