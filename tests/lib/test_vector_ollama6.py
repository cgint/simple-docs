import pytest
from unittest.mock import Mock, patch
import json
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
from lib.vector import (load_vector_index_storage_context, load_vector_index, persist_vector_index, get_vector_index, add_to_or_update_in_vector, operate_on_vector_index, assert_exists_vector_index, get_vector_chat_engine, get_vector_query_engine, get_vector_ng_query_engine)

# METHOD:
# def load_vector_index_storage_context(vector_storage_dir: str) -> StorageContext:
def test_load_vector_index_storage_context_with_valid_directory():
    storage_context = load_vector_index_storage_context("./test_data/vector_storage")
    assert isinstance(storage_context, StorageContext)
    assert storage_context.persist_dir == "./test_data/vector_storage"

def test_load_vector_index_storage_context_with_invalid_directory():
    with pytest.raises(ValueError):
        load_vector_index_storage_context("./test_data/non_existent_dir")


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `load_vector_index_storage_context` method:
# ```python
# def test_load_vector_index_storage_context_with_valid_directory():
#     storage_context = load_vector_index_storage_context("./test_data/vector_storage")
#     assert isinstance(storage_context, StorageContext)
#     assert storage_context.persist_dir == "./test_data/vector_storage"
# 
# def test_load_vector_index_storage_context_with_invalid_directory():
#     with pytest.raises(ValueError):
#         load_vector_index_storage_context("./test_data/non_existent_dir")
# ```
# These tests check that the `load_vector_index_storage_context` method returns a valid `StorageContext` object when given a valid directory, and raises a `ValueError` when given an invalid directory.



# METHOD:
# def load_vector_index(vector_storage_dir: str) -> VectorStoreIndex:
def test_load_vector_index_returns_vector_store_index(self):
    vector_storage_dir = "path/to/valid/directory"
    index = load_vector_index(vector_storage_dir)
    self.assertIsInstance(index, VectorStoreIndex)

def test_load_vector_index_raises_value_error_with_invalid_directory(self):
    vector_storage_dir = "path/to/invalid/directory"
    with self.assertRaises(ValueError):
        load_vector_index(vector_storage_dir)

def test_load_vector_index_returns_vector_store_index_with_valid_index_file(self):
    vector_storage_dir = "path/to/valid/directory"
    index_file = os.path.join(vector_storage_dir, "index.json")
    with open(index_file, "w") as f:
        json.dump({}, f)
    index = load_vector_index(vector_storage_dir)
    self.assertIsInstance(index, VectorStoreIndex)

def test_load_vector_index_raises_value_error_with_invalid_index_file(self):
    vector_storage_dir = "path/to/valid/directory"
    index_file = os.path.join(vector_storage_dir, "index.json")
    with open(index_file, "w") as f:
        json.dump({}, f)
    with self.assertRaises(ValueError):
        load_vector_index(vector_storage_dir)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `load_vector_index` method:
# 
# 1. Test that the method returns a `VectorStoreIndex` object when given a valid directory path:
# ```python
# def test_load_vector_index_returns_vector_store_index(self):
#     vector_storage_dir = "path/to/valid/directory"
#     index = load_vector_index(vector_storage_dir)
#     self.assertIsInstance(index, VectorStoreIndex)
# ```
# 2. Test that the method raises a `ValueError` when given an invalid directory path:
# ```python
# def test_load_vector_index_raises_value_error_with_invalid_directory(self):
#     vector_storage_dir = "path/to/invalid/directory"
#     with self.assertRaises(ValueError):
#         load_vector_index(vector_storage_dir)
# ```
# 3. Test that the method returns a `VectorStoreIndex` object when given a valid directory path and a valid index file:
# ```python
# def test_load_vector_index_returns_vector_store_index_with_valid_index_file(self):
#     vector_storage_dir = "path/to/valid/directory"
#     index_file = os.path.join(vector_storage_dir, "index.json")
#     with open(index_file, "w") as f:
#         json.dump({}, f)
#     index = load_vector_index(vector_storage_dir)
#     self.assertIsInstance(index, VectorStoreIndex)
# ```
# 4. Test that the method raises a `ValueError` when given an invalid index file:
# ```python
# def test_load_vector_index_raises_value_error_with_invalid_index_file(self):
#     vector_storage_dir = "path/to/valid/directory"
#     index_file = os.path.join(vector_storage_dir, "index.json")
#     with open(index_file, "w") as f:
#         json.dump({}, f)
#     with self.assertRaises(ValueError):
#         load_vector_index(vector_storage_dir)
# ```



# METHOD:
# def persist_vector_index(vector_index: VectorStoreIndex, vector_storage_dir: str):
def test_persist_vector_index(vector_index, vector_storage_dir):
    persist_vector_index(vector_index, vector_storage_dir)
    assert os.path.exists(os.path.join(vector_storage_dir, "vector-index"))

def test_persist_vector_index_with_invalid_directory(vector_index):
    with pytest.raises(ValueError) as excinfo:
        persist_vector_index(vector_index, "non-existent-dir")
    assert str(excinfo.value) == "Invalid directory"

def test_persist_vector_index_with_unwritable_directory(vector_index):
    with pytest.raises(ValueError) as excinfo:
        persist_vector_index(vector_index, "/root")
    assert str(excinfo.value) == "Directory is not writable"

def test_persist_vector_index_with_file_conflict(vector_index, vector_storage_dir):
    with open(os.path.join(vector_storage_dir, "vector-index"), "w") as f:
        f.write("test")
    with pytest.raises(ValueError) as excinfo:
        persist_vector_index(vector_index, vector_storage_dir)
    assert str(excinfo.value) == "File conflict"


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `persist_vector_index` method:
# 
# 1. Test that the method persists the vector index to the specified directory:
# ```python
# def test_persist_vector_index(vector_index, vector_storage_dir):
#     persist_vector_index(vector_index, vector_storage_dir)
#     assert os.path.exists(os.path.join(vector_storage_dir, "vector-index"))
# ```
# 2. Test that the method raises an error if the specified directory does not exist:
# ```python
# def test_persist_vector_index_with_invalid_directory(vector_index):
#     with pytest.raises(ValueError) as excinfo:
#         persist_vector_index(vector_index, "non-existent-dir")
#     assert str(excinfo.value) == "Invalid directory"
# ```
# 3. Test that the method raises an error if the specified directory is not writable:
# ```python
# def test_persist_vector_index_with_unwritable_directory(vector_index):
#     with pytest.raises(ValueError) as excinfo:
#         persist_vector_index(vector_index, "/root")
#     assert str(excinfo.value) == "Directory is not writable"
# ```
# 4. Test that the method raises an error if the specified directory contains a file with the same name as the vector index:
# ```python
# def test_persist_vector_index_with_file_conflict(vector_index, vector_storage_dir):
#     with open(os.path.join(vector_storage_dir, "vector-index"), "w") as f:
#         f.write("test")
#     with pytest.raises(ValueError) as excinfo:
#         persist_vector_index(vector_index, vector_storage_dir)
#     assert str(excinfo.value) == "File conflict"
# ```



# METHOD:
# def get_vector_index(vector_storage_dir: str) -> VectorStoreIndex:
def test_returns_vector_store_index(tmpdir):
    vector_storage_dir = tmpdir / "vectors"
    os.makedirs(vector_storage_dir)
    with open(os.path.join(vector_storage_dir, "document1.json"), "w") as f:
        json.dump({"id": 1}, f)
    vector_index = get_vector_index(vector_storage_dir)
    assert isinstance(vector_index, VectorStoreIndex)

def test_returns_empty_vector_store_index(tmpdir):
    vector_storage_dir = tmpdir / "vectors"
    os.makedirs(vector_storage_dir)
    vector_index = get_vector_index(vector_storage_dir)
    assert isinstance(vector_index, VectorStoreIndex)

def test_raises_file_not_found_error(tmpdir):
    vector_storage_dir = tmpdir / "vectors"
    with pytest.raises(FileNotFoundError):
        get_vector_index(vector_storage_dir)

def test_raises_value_error(tmpdir):
    vector_storage_dir = tmpdir / "vectors"
    os.makedirs(vector_storage_dir)
    with open(os.path.join(vector_storage_dir, "document1.json"), "w") as f:
        json.dump({"id": 1}, f)
    vector_index = get_vector_index(vector_storage_dir)
    assert isinstance(vector_index, VectorStoreIndex)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_vector_index` method:
# 
# 1. Test that the method returns a `VectorStoreIndex` object when the directory exists and is not empty:
# ```python
# def test_returns_vector_store_index(tmpdir):
#     vector_storage_dir = tmpdir / "vectors"
#     os.makedirs(vector_storage_dir)
#     with open(os.path.join(vector_storage_dir, "document1.json"), "w") as f:
#         json.dump({"id": 1}, f)
#     vector_index = get_vector_index(vector_storage_dir)
#     assert isinstance(vector_index, VectorStoreIndex)
# ```
# 2. Test that the method returns a `VectorStoreIndex` object when the directory exists and is empty:
# ```python
# def test_returns_empty_vector_store_index(tmpdir):
#     vector_storage_dir = tmpdir / "vectors"
#     os.makedirs(vector_storage_dir)
#     vector_index = get_vector_index(vector_storage_dir)
#     assert isinstance(vector_index, VectorStoreIndex)
# ```
# 3. Test that the method raises a `FileNotFoundError` when the directory does not exist:
# ```python
# def test_raises_file_not_found_error(tmpdir):
#     vector_storage_dir = tmpdir / "vectors"
#     with pytest.raises(FileNotFoundError):
#         get_vector_index(vector_storage_dir)
# ```
# 4. Test that the method raises a `ValueError` when the directory is not a valid vector storage directory:
# ```python
# def test_raises_value_error(tmpdir):
#     vector_storage_dir = tmpdir / "vectors"
#     os.makedirs(vector_storage_dir)
#     with open(os.path.join(vector_storage_dir, "document1.json"), "w") as f:
#         json.dump({"id": 1}, f)
#     vector_index = get_vector_index(vector_storage_dir)
#     assert isinstance(vector_index, VectorStoreIndex)
# ```



# METHOD:
# def add_to_or_update_in_vector(vector_storage_dir: str, documents: List[Document]) -> VectorStoreIndex:
def test_returns_vector_store_index(self):
    vector_storage_dir = "test/data"
    documents = [Document("id", "title", "content")]
    result = add_to_or_update_in_vector(vector_storage_dir, documents)
    self.assertIsInstance(result, VectorStoreIndex)

def test_raises_value_error_with_invalid_vector_storage_dir(self):
    vector_storage_dir = "test/data"
    documents = [Document("id", "title", "content")]
    with self.assertRaises(ValueError):
        add_to_or_update_in_vector(vector_storage_dir, documents)

def test_raises_type_error_with_invalid_documents(self):
    vector_storage_dir = "test/data"
    documents = [1, 2, 3]
    with self.assertRaises(TypeError):
        add_to_or_update_in_vector(vector_storage_dir, documents)

def test_updates_vector_store_index(self):
    vector_storage_dir = "test/data"
    documents = [Document("id", "title", "content")]
    result = add_to_or_update_in_vector(vector_storage_dir, documents)
    self.assertEqual(result.documents, documents)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `add_to_or_update_in_vector` method:
# 
# 1. Test that the method returns a `VectorStoreIndex` object when called with valid arguments:
# ```python
# def test_returns_vector_store_index(self):
#     vector_storage_dir = "test/data"
#     documents = [Document("id", "title", "content")]
#     result = add_to_or_update_in_vector(vector_storage_dir, documents)
#     self.assertIsInstance(result, VectorStoreIndex)
# ```
# 2. Test that the method raises a `ValueError` when called with an invalid `vector_storage_dir`:
# ```python
# def test_raises_value_error_with_invalid_vector_storage_dir(self):
#     vector_storage_dir = "test/data"
#     documents = [Document("id", "title", "content")]
#     with self.assertRaises(ValueError):
#         add_to_or_update_in_vector(vector_storage_dir, documents)
# ```
# 3. Test that the method raises a `TypeError` when called with an invalid `documents` argument:
# ```python
# def test_raises_type_error_with_invalid_documents(self):
#     vector_storage_dir = "test/data"
#     documents = [1, 2, 3]
#     with self.assertRaises(TypeError):
#         add_to_or_update_in_vector(vector_storage_dir, documents)
# ```
# 4. Test that the method updates the `VectorStoreIndex` object when called with valid arguments:
# ```python
# def test_updates_vector_store_index(self):
#     vector_storage_dir = "test/data"
#     documents = [Document("id", "title", "content")]
#     result = add_to_or_update_in_vector(vector_storage_dir, documents)
#     self.assertEqual(result.documents, documents)
# ```



# METHOD:
# def operate_on_vector_index(vector_storage_dir: str, operation=lambda: None) -> VectorStoreIndex:
def test_returns_vector_store_index(self):
    vector_storage_dir = "test_data/vectors"
    operation = lambda x: None
    result = operate_on_vector_index(vector_storage_dir, operation)
    self.assertIsInstance(result, VectorStoreIndex)

def test_performs_operation(self):
    vector_storage_dir = "test_data/vectors"
    operation = lambda x: x + 1
    result = operate_on_vector_index(vector_storage_dir, operation)
    self.assertEqual(result[0], 2)

def test_persists_vector_store_index(self):
    vector_storage_dir = "test_data/vectors"
    operation = lambda x: None
    result = operate_on_vector_index(vector_storage_dir, operation)
    self.assertTrue(os.path.exists(vector_storage_dir))

def test_raises_error_if_operation_not_callable(self):
    vector_storage_dir = "test_data/vectors"
    operation = 123
    with self.assertRaises(TypeError):
        operate_on_vector_index(vector_storage_dir, operation)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `operate_on_vector_index` method:
# 
# 1. Test that the method returns a `VectorStoreIndex` object after performing an operation on it:
# ```python
# def test_returns_vector_store_index(self):
#     vector_storage_dir = "test_data/vectors"
#     operation = lambda x: None
#     result = operate_on_vector_index(vector_storage_dir, operation)
#     self.assertIsInstance(result, VectorStoreIndex)
# ```
# 2. Test that the method performs the given operation on the `VectorStoreIndex` object:
# ```python
# def test_performs_operation(self):
#     vector_storage_dir = "test_data/vectors"
#     operation = lambda x: x + 1
#     result = operate_on_vector_index(vector_storage_dir, operation)
#     self.assertEqual(result[0], 2)
# ```
# 3. Test that the method persists the `VectorStoreIndex` object to disk after performing an operation on it:
# ```python
# def test_persists_vector_store_index(self):
#     vector_storage_dir = "test_data/vectors"
#     operation = lambda x: None
#     result = operate_on_vector_index(vector_storage_dir, operation)
#     self.assertTrue(os.path.exists(vector_storage_dir))
# ```
# 4. Test that the method raises an error if the `operation` parameter is not callable:
# ```python
# def test_raises_error_if_operation_not_callable(self):
#     vector_storage_dir = "test_data/vectors"
#     operation = 123
#     with self.assertRaises(TypeError):
#         operate_on_vector_index(vector_storage_dir, operation)
# ```



# METHOD:
# def assert_exists_vector_index(vector_storage_dir: str):
import pytest
from pathlib import Path

def test_exists_vector_index():
    vector_storage_dir = "path/to/vector/index"
    assert os.path.exists(vector_storage_dir)
    assert_exists_vector_index(vector_storage_dir)

def test_not_exists_vector_index():
    vector_storage_dir = "path/to/non-existent/vector/index"
    with pytest.raises(ValueError):
        assert_exists_vector_index(vector_storage_dir)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `assert_exists_vector_index` method:
# ```python
# import pytest
# from pathlib import Path
# 
# def test_exists_vector_index():
#     vector_storage_dir = "path/to/vector/index"
#     assert os.path.exists(vector_storage_dir)
#     assert_exists_vector_index(vector_storage_dir)
# 
# def test_not_exists_vector_index():
#     vector_storage_dir = "path/to/non-existent/vector/index"
#     with pytest.raises(ValueError):
#         assert_exists_vector_index(vector_storage_dir)
# ```
# These test cases cover the following scenarios:
# 
# * `test_exists_vector_index`: This test case checks that the method raises no error when the vector index directory exists. It also checks that the method returns no value, as it is a void method.
# * `test_not_exists_vector_index`: This test case checks that the method raises a `ValueError` when the vector index directory does not exist. It also checks that the method returns no value, as it is a void method.



# METHOD:
# def get_vector_chat_engine(vector_storage_dir: str) -> BaseChatEngine:
def test_get_vector_chat_engine_returns_base_chat_engine():
    vector_storage_dir = "path/to/vector/index"
    assert isinstance(get_vector_chat_engine(vector_storage_dir), BaseChatEngine)

def test_get_vector_chat_engine_raises_value_error():
    vector_storage_dir = "path/to/non-existent/vector/index"
    with pytest.raises(ValueError):
        get_vector_chat_engine(vector_storage_dir)

def test_get_vector_chat_engine_returns_base_chat_engine_with_loaded_index():
    vector_storage_dir = "path/to/vector/index"
    assert get_vector_chat_engine(vector_storage_dir).as_chat_engine().is_loaded()

def test_get_vector_chat_engine_returns_base_chat_engine_with_loaded_index_and_can_perform_chat():
    vector_storage_dir = "path/to/vector/index"
    engine = get_vector_chat_engine(vector_storage_dir)
    assert engine.as_chat_engine().is_loaded()
    response = engine.respond("Hello, how are you?")
    assert response == "I'm doing well, thank you for asking!"


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_vector_chat_engine` method:
# 
# 1. Test that the method returns a `BaseChatEngine` object when the vector index exists:
# ```python
# def test_get_vector_chat_engine_returns_base_chat_engine():
#     vector_storage_dir = "path/to/vector/index"
#     assert isinstance(get_vector_chat_engine(vector_storage_dir), BaseChatEngine)
# ```
# 2. Test that the method raises a `ValueError` when the vector index does not exist:
# ```python
# def test_get_vector_chat_engine_raises_value_error():
#     vector_storage_dir = "path/to/non-existent/vector/index"
#     with pytest.raises(ValueError):
#         get_vector_chat_engine(vector_storage_dir)
# ```
# 3. Test that the method returns a `BaseChatEngine` object when the vector index exists and is loaded correctly:
# ```python
# def test_get_vector_chat_engine_returns_base_chat_engine_with_loaded_index():
#     vector_storage_dir = "path/to/vector/index"
#     assert get_vector_chat_engine(vector_storage_dir).as_chat_engine().is_loaded()
# ```
# 4. Test that the method returns a `BaseChatEngine` object when the vector index exists and is loaded correctly, and that the engine can be used to perform chat operations:
# ```python
# def test_get_vector_chat_engine_returns_base_chat_engine_with_loaded_index_and_can_perform_chat():
#     vector_storage_dir = "path/to/vector/index"
#     engine = get_vector_chat_engine(vector_storage_dir)
#     assert engine.as_chat_engine().is_loaded()
#     response = engine.respond("Hello, how are you?")
#     assert response == "I'm doing well, thank you for asking!"
# ```



# METHOD:
# def get_vector_query_engine(vector_storage_dir: str) -> BaseQueryEngine:
def test_get_vector_query_engine_returns_query_engine():
    with patch("lib.index.html.assert_exists_vector_index") as mock_assert_exists_vector_index:
        mock_assert_exists_vector_index.return_value = True
        vector_storage_dir = "test/data"
        query_engine = get_vector_query_engine(vector_storage_dir)
        assert isinstance(query_engine, BaseQueryEngine)

def test_get_vector_query_engine_raises_exception():
    with patch("lib.index.html.assert_exists_vector_index") as mock_assert_exists_vector_index:
        mock_assert_exists_vector_index.return_value = False
        vector_storage_dir = "test/data"
        with pytest.raises(Exception):
            get_vector_query_engine(vector_storage_dir)

def test_get_vector_query_engine_returns_query_engine_when_index_exists():
    with patch("lib.index.html.assert_exists_vector_index") as mock_assert_exists_vector_index:
        mock_assert_exists_vector_index.return_value = True
        vector_storage_dir = "test/data"
        query_engine = get_vector_query_engine(vector_storage_dir)
        assert isinstance(query_engine, BaseQueryEngine)

def test_get_vector_query_engine_raises_exception_when_index_does_not_exist():
    with patch("lib.index.html.assert_exists_vector_index") as mock_assert_exists_vector_index:
        mock_assert_exists_vector_index.return_value = False
        vector_storage_dir = "test/data"
        with pytest.raises(Exception):
            get_vector_query_engine(vector_storage_dir)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_vector_query_engine` method:
# 
# 1. Test that the method returns a query engine instance when the vector index exists:
# ```python
# def test_get_vector_query_engine_returns_query_engine():
#     with patch("lib.index.html.assert_exists_vector_index") as mock_assert_exists_vector_index:
#         mock_assert_exists_vector_index.return_value = True
#         vector_storage_dir = "test/data"
#         query_engine = get_vector_query_engine(vector_storage_dir)
#         assert isinstance(query_engine, BaseQueryEngine)
# ```
# 2. Test that the method raises an exception when the vector index does not exist:
# ```python
# def test_get_vector_query_engine_raises_exception():
#     with patch("lib.index.html.assert_exists_vector_index") as mock_assert_exists_vector_index:
#         mock_assert_exists_vector_index.return_value = False
#         vector_storage_dir = "test/data"
#         with pytest.raises(Exception):
#             get_vector_query_engine(vector_storage_dir)
# ```
# 3. Test that the method returns a query engine instance when the vector index exists and the `load_vector_index` method is called:
# ```python
# def test_get_vector_query_engine_returns_query_engine_when_index_exists():
#     with patch("lib.index.html.assert_exists_vector_index") as mock_assert_exists_vector_index:
#         mock_assert_exists_vector_index.return_value = True
#         vector_storage_dir = "test/data"
#         query_engine = get_vector_query_engine(vector_storage_dir)
#         assert isinstance(query_engine, BaseQueryEngine)
# ```
# 4. Test that the method raises an exception when the `load_vector_index` method is called and the vector index does not exist:
# ```python
# def test_get_vector_query_engine_raises_exception_when_index_does_not_exist():
#     with patch("lib.index.html.assert_exists_vector_index") as mock_assert_exists_vector_index:
#         mock_assert_exists_vector_index.return_value = False
#         vector_storage_dir = "test/data"
#         with pytest.raises(Exception):
#             get_vector_query_engine(vector_storage_dir)
# ```



# METHOD:
# def get_vector_ng_query_engine(vector_storage_dir: str) -> BaseQueryEngine:
def test_returns_base_query_engine():
    engine = get_vector_ng_query_engine("test_dir")
    assert isinstance(engine, BaseQueryEngine)

def test_raises_error_if_vector_storage_dir_does_not_exist():
    with pytest.raises(AssertionError):
        get_vector_ng_query_engine("non_existent_dir")

def test_returns_retriever_query_engine():
    engine = get_vector_ng_query_engine("test_dir")
    assert isinstance(engine, RetrieverQueryEngine)

def test_returns_vector_index_retriever():
    engine = get_vector_ng_query_engine("test_dir")
    assert isinstance(engine.retriever, VectorIndexRetriever)

def test_returns_response_synthesizer():
    engine = get_vector_ng_query_engine("test_dir")
    assert isinstance(engine.response_synthesizer, ResponseSynthesizer)

def test_returns_response_synthesizer_with_correct_response_mode():
    engine = get_vector_ng_query_engine("test_dir")
    assert engine.response_synthesizer.response_mode == "tree_summarize"


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_vector_ng_query_engine` method:
# 
# 1. Test that the method returns a `BaseQueryEngine` object:
# ```python
# def test_returns_base_query_engine():
#     engine = get_vector_ng_query_engine("test_dir")
#     assert isinstance(engine, BaseQueryEngine)
# ```
# 2. Test that the method raises an error if the `vector_storage_dir` does not exist:
# ```python
# def test_raises_error_if_vector_storage_dir_does_not_exist():
#     with pytest.raises(AssertionError):
#         get_vector_ng_query_engine("non_existent_dir")
# ```
# 3. Test that the method returns a `RetrieverQueryEngine` object:
# ```python
# def test_returns_retriever_query_engine():
#     engine = get_vector_ng_query_engine("test_dir")
#     assert isinstance(engine, RetrieverQueryEngine)
# ```
# 4. Test that the method returns a `VectorIndexRetriever` object:
# ```python
# def test_returns_vector_index_retriever():
#     engine = get_vector_ng_query_engine("test_dir")
#     assert isinstance(engine.retriever, VectorIndexRetriever)
# ```
# 5. Test that the method returns a `ResponseSynthesizer` object:
# ```python
# def test_returns_response_synthesizer():
#     engine = get_vector_ng_query_engine("test_dir")
#     assert isinstance(engine.response_synthesizer, ResponseSynthesizer)
# ```
# 6. Test that the method returns a `ResponseSynthesizer` object with the correct response mode:
# ```python
# def test_returns_response_synthesizer_with_correct_response_mode():
#     engine = get_vector_ng_query_engine("test_dir")
#     assert engine.response_synthesizer.response_mode == "tree_summarize"
# ```

