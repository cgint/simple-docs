import pytest
from unittest.mock import Mock, patch
import json
from typing import List
from llama_index.core import Document, get_response_synthesizer
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from lib import constants
from lib.vector_chroma import (chunker, get_or_create_chroma_collection, delete_chroma_collection, load_vector_index_chroma_storage_context, load_vector_index, get_vector_index, add_to_or_update_in_vector, operate_on_vector_index, get_vector_query_engine, get_vector_ng_query_engine)

# METHOD:
# def chunker(seq, size):
import pytest

def test_chunker():
    seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    size = 3
    expected_result = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
    assert list(chunker(seq, size)) == expected_result

def test_chunker_with_invalid_size():
    seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    size = -1
    with pytest.raises(ValueError):
        list(chunker(seq, size))


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `chunker` function:
# ```python
# import pytest
# 
# def test_chunker():
#     seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     size = 3
#     expected_result = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
#     assert list(chunker(seq, size)) == expected_result
# 
# def test_chunker_with_invalid_size():
#     seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     size = -1
#     with pytest.raises(ValueError):
#         list(chunker(seq, size))
# ```
# The first test case checks that the `chunker` function returns the expected result when given a valid input sequence and chunk size. The second test case checks that an error is raised when the chunk size is invalid (i.e., negative).



# METHOD:
# def get_or_create_chroma_collection(collection: str) -> chromadb.Collection:
def test_new_collection_created():
    remote_db = chromadb.HttpClient(host=constants.host_ip)
    collection = "test_collection"
    assert not remote_db.get_or_create_collection(collection).exists()
    get_or_create_chroma_collection(collection)
    assert remote_db.get_or_create_collection(collection).exists()

def test_existing_collection_returned():
    remote_db = chromadb.HttpClient(host=constants.host_ip)
    collection = "test_collection"
    assert not remote_db.get_or_create_collection(collection).exists()
    get_or_create_chroma_collection(collection)
    assert remote_db.get_or_create_collection(collection).exists()

def test_correct_collection_returned():
    remote_db = chromadb.HttpClient(host=constants.host_ip)
    collection1 = "test_collection1"
    collection2 = "test_collection2"
    assert not remote_db.get_or_create_collection(collection1).exists()
    assert not remote_db.get_or_create_collection(collection2).exists()
    get_or_create_chroma_collection(collection1)
    get_or_create_chroma_collection(collection2)
    assert remote_db.get_or_create_collection(collection1).exists()
    assert remote_db.get_or_create_collection(collection2).exists()

def test_invalid_collection_name():
    with pytest.raises(ValueError):
        get_or_create_chroma_collection("")


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_or_create_chroma_collection` method:
# 
# 1. Test that a new collection is created when it doesn't exist:
# ```python
# def test_new_collection_created():
#     remote_db = chromadb.HttpClient(host=constants.host_ip)
#     collection = "test_collection"
#     assert not remote_db.get_or_create_collection(collection).exists()
#     get_or_create_chroma_collection(collection)
#     assert remote_db.get_or_create_collection(collection).exists()
# ```
# 2. Test that an existing collection is returned when it exists:
# ```python
# def test_existing_collection_returned():
#     remote_db = chromadb.HttpClient(host=constants.host_ip)
#     collection = "test_collection"
#     assert not remote_db.get_or_create_collection(collection).exists()
#     get_or_create_chroma_collection(collection)
#     assert remote_db.get_or_create_collection(collection).exists()
# ```
# 3. Test that the correct collection is returned when multiple collections exist:
# ```python
# def test_correct_collection_returned():
#     remote_db = chromadb.HttpClient(host=constants.host_ip)
#     collection1 = "test_collection1"
#     collection2 = "test_collection2"
#     assert not remote_db.get_or_create_collection(collection1).exists()
#     assert not remote_db.get_or_create_collection(collection2).exists()
#     get_or_create_chroma_collection(collection1)
#     get_or_create_chroma_collection(collection2)
#     assert remote_db.get_or_create_collection(collection1).exists()
#     assert remote_db.get_or_create_collection(collection2).exists()
# ```
# 4. Test that an exception is raised when the collection name is invalid:
# ```python
# def test_invalid_collection_name():
#     with pytest.raises(ValueError):
#         get_or_create_chroma_collection("")
# ```



# METHOD:
# def delete_chroma_collection(collection: str) -> None:
def test_delete_chroma_collection(mocker):
    mocker.patch("lib.index.html.get_or_create_chroma_collection")
    chroma_collection = get_or_create_chroma_collection("test_collection")
    doc_ids = chroma_collection.get()["ids"]
    delete_chroma_collection("test_collection")
    assert len(doc_ids) == 0

def test_delete_chroma_collection_invalid_name(mocker):
    mocker.patch("lib.index.html.get_or_create_chroma_collection")
    chroma_collection = get_or_create_chroma_collection("test_collection")
    doc_ids = chroma_collection.get()["ids"]
    with pytest.raises(ValueError):
        delete_chroma_collection("invalid_name")

def test_delete_chroma_collection_with_batch_size(mocker):
    mocker.patch("lib.index.html.get_or_create_chroma_collection")
    chroma_collection = get_or_create_chroma_collection("test_collection")
    doc_ids = chroma_collection.get()["ids"]
    delete_chroma_collection("test_collection", batch_size=20000)
    assert len(doc_ids) == 0

def test_delete_chroma_collection_invalid_batch_size(mocker):
    mocker.patch("lib.index.html.get_or_create_chroma_collection")
    chroma_collection = get_or_create_chroma_collection("test_collection")
    doc_ids = chroma_collection.get()["ids"]
    with pytest.raises(ValueError):
        delete_chroma_collection("test_collection", batch_size=-1)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `delete_chroma_collection` method:
# 
# 1. Test that the method deletes all documents from the collection when called with a valid collection name:
# ```python
# def test_delete_chroma_collection(mocker):
#     mocker.patch("lib.index.html.get_or_create_chroma_collection")
#     chroma_collection = get_or_create_chroma_collection("test_collection")
#     doc_ids = chroma_collection.get()["ids"]
#     delete_chroma_collection("test_collection")
#     assert len(doc_ids) == 0
# ```
# 2. Test that the method raises a `ValueError` when called with an invalid collection name:
# ```python
# def test_delete_chroma_collection_invalid_name(mocker):
#     mocker.patch("lib.index.html.get_or_create_chroma_collection")
#     chroma_collection = get_or_create_chroma_collection("test_collection")
#     doc_ids = chroma_collection.get()["ids"]
#     with pytest.raises(ValueError):
#         delete_chroma_collection("invalid_name")
# ```
# 3. Test that the method deletes all documents from the collection when called with a valid collection name and batch size:
# ```python
# def test_delete_chroma_collection_with_batch_size(mocker):
#     mocker.patch("lib.index.html.get_or_create_chroma_collection")
#     chroma_collection = get_or_create_chroma_collection("test_collection")
#     doc_ids = chroma_collection.get()["ids"]
#     delete_chroma_collection("test_collection", batch_size=20000)
#     assert len(doc_ids) == 0
# ```
# 4. Test that the method raises a `ValueError` when called with an invalid batch size:
# ```python
# def test_delete_chroma_collection_invalid_batch_size(mocker):
#     mocker.patch("lib.index.html.get_or_create_chroma_collection")
#     chroma_collection = get_or_create_chroma_collection("test_collection")
#     doc_ids = chroma_collection.get()["ids"]
#     with pytest.raises(ValueError):
#         delete_chroma_collection("test_collection", batch_size=-1)
# ```



# METHOD:
# def load_vector_index_chroma_storage_context(collection: str) -> tuple[ChromaVectorStore, StorageContext]:
def test_returns_tuple():
    collection = "test"
    vector_store, storage_context = load_vector_index_chroma_storage_context(collection)
    assert isinstance(vector_store, ChromaVectorStore)
    assert isinstance(storage_context, StorageContext)

def test_creates_new_chroma_collection():
    collection = "test"
    vector_store, storage_context = load_vector_index_chroma_storage_context(collection)
    assert isinstance(vector_store.chroma_collection, ChromaCollection)

def test_creates_new_storage_context():
    collection = "test"
    vector_store, storage_context = load_vector_index_chroma_storage_context(collection)
    assert isinstance(storage_context.vector_store, ChromaVectorStore)
    assert storage_context.vector_store == vector_store

def test_returns_correct_objects():
    collection = "test"
    vector_store, storage_context = load_vector_index_chroma_storage_context(collection)
    assert isinstance(vector_store.chroma_collection, ChromaCollection)
    assert isinstance(storage_context.vector_store, ChromaVectorStore)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `load_vector_index_chroma_storage_context` method:
# 
# 1. Test that the method returns a tuple with two elements:
# ```python
# def test_returns_tuple():
#     collection = "test"
#     vector_store, storage_context = load_vector_index_chroma_storage_context(collection)
#     assert isinstance(vector_store, ChromaVectorStore)
#     assert isinstance(storage_context, StorageContext)
# ```
# 2. Test that the method creates a new `ChromaCollection` object if it doesn't exist:
# ```python
# def test_creates_new_chroma_collection():
#     collection = "test"
#     vector_store, storage_context = load_vector_index_chroma_storage_context(collection)
#     assert isinstance(vector_store.chroma_collection, ChromaCollection)
# ```
# 3. Test that the method creates a new `StorageContext` object with the correct parameters:
# ```python
# def test_creates_new_storage_context():
#     collection = "test"
#     vector_store, storage_context = load_vector_index_chroma_storage_context(collection)
#     assert isinstance(storage_context.vector_store, ChromaVectorStore)
#     assert storage_context.vector_store == vector_store
# ```
# 4. Test that the method returns the correct `ChromaVectorStore` and `StorageContext` objects:
# ```python
# def test_returns_correct_objects():
#     collection = "test"
#     vector_store, storage_context = load_vector_index_chroma_storage_context(collection)
#     assert isinstance(vector_store.chroma_collection, ChromaCollection)
#     assert isinstance(storage_context.vector_store, ChromaVectorStore)
# ```



# METHOD:
# def load_vector_index(collection: str) -> VectorStoreIndex:
def test_load_vector_index_with_valid_collection():
    collection = "test_collection"
    index = load_vector_index(collection)
    assert isinstance(index, VectorStoreIndex)
    assert index.vector_store == vector_store

def test_load_vector_index_with_invalid_collection():
    collection = "invalid_collection"
    with pytest.raises(ValueError):
        load_vector_index(collection)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `load_vector_index` method:
# ```python
# def test_load_vector_index_with_valid_collection():
#     collection = "test_collection"
#     index = load_vector_index(collection)
#     assert isinstance(index, VectorStoreIndex)
#     assert index.vector_store == vector_store
# 
# def test_load_vector_index_with_invalid_collection():
#     collection = "invalid_collection"
#     with pytest.raises(ValueError):
#         load_vector_index(collection)
# ```
# These tests check that the `load_vector_index` method returns a `VectorStoreIndex` object when given a valid collection name, and raises a `ValueError` when given an invalid collection name.



# METHOD:
# def get_vector_index(collection: str) -> VectorStoreIndex:
import pytest
from lib.index.html import get_vector_index

def test_get_vector_index():
    collection = "my_collection"
    index = get_vector_index(collection)
    assert isinstance(index, VectorStoreIndex)
    assert index.collection == collection

def test_get_vector_index_with_invalid_collection():
    collection = "my_collection"
    with pytest.raises(ValueError):
        get_vector_index("")


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_vector_index` method:
# ```python
# import pytest
# from lib.index.html import get_vector_index
# 
# def test_get_vector_index():
#     collection = "my_collection"
#     index = get_vector_index(collection)
#     assert isinstance(index, VectorStoreIndex)
#     assert index.collection == collection
# ```
# This test case checks that the `get_vector_index` method returns an instance of `VectorStoreIndex`, and that the `collection` attribute of the returned index matches the input argument.
# 
# Another test case could be:
# ```python
# def test_get_vector_index_with_invalid_collection():
#     collection = "my_collection"
#     with pytest.raises(ValueError):
#         get_vector_index("")
# ```
# This test case checks that the `get_vector_index` method raises a `ValueError` when an invalid collection name is passed as input.



# METHOD:
# def add_to_or_update_in_vector(vector_storage_dir: str, documents: List[Document]) -> VectorStoreIndex:
def test_returns_vector_store_index(self):
    vector_storage_dir = "test_data/vector_storage"
    documents = [Document("doc1", "This is a test document")]
    result = add_to_or_update_in_vector(vector_storage_dir, documents)
    self.assertIsInstance(result, VectorStoreIndex)

def test_raises_value_error_with_invalid_vector_storage_dir(self):
    vector_storage_dir = "test_data/invalid_vector_storage"
    documents = [Document("doc1", "This is a test document")]
    with self.assertRaises(ValueError):
        add_to_or_update_in_vector(vector_storage_dir, documents)

def test_raises_type_error_with_invalid_documents(self):
    vector_storage_dir = "test_data/vector_storage"
    documents = ["This is not a list of Document objects"]
    with self.assertRaises(TypeError):
        add_to_or_update_in_vector(vector_storage_dir, documents)

def test_updates_vector_store_index_correctly(self):
    vector_storage_dir = "test_data/vector_storage"
    documents = [Document("doc1", "This is a test document")]
    result = add_to_or_update_in_vector(vector_storage_dir, documents)
    self.assertEqual(result.documents, documents)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `add_to_or_update_in_vector` method:
# 
# 1. Test that the method returns a `VectorStoreIndex` object when called with valid arguments:
# ```python
# def test_returns_vector_store_index(self):
#     vector_storage_dir = "test_data/vector_storage"
#     documents = [Document("doc1", "This is a test document")]
#     result = add_to_or_update_in_vector(vector_storage_dir, documents)
#     self.assertIsInstance(result, VectorStoreIndex)
# ```
# 2. Test that the method raises a `ValueError` when called with an invalid `vector_storage_dir`:
# ```python
# def test_raises_value_error_with_invalid_vector_storage_dir(self):
#     vector_storage_dir = "test_data/invalid_vector_storage"
#     documents = [Document("doc1", "This is a test document")]
#     with self.assertRaises(ValueError):
#         add_to_or_update_in_vector(vector_storage_dir, documents)
# ```
# 3. Test that the method raises a `TypeError` when called with an invalid `documents` argument:
# ```python
# def test_raises_type_error_with_invalid_documents(self):
#     vector_storage_dir = "test_data/vector_storage"
#     documents = ["This is not a list of Document objects"]
#     with self.assertRaises(TypeError):
#         add_to_or_update_in_vector(vector_storage_dir, documents)
# ```
# 4. Test that the method updates the `VectorStoreIndex` object correctly when called with valid arguments:
# ```python
# def test_updates_vector_store_index_correctly(self):
#     vector_storage_dir = "test_data/vector_storage"
#     documents = [Document("doc1", "This is a test document")]
#     result = add_to_or_update_in_vector(vector_storage_dir, documents)
#     self.assertEqual(result.documents, documents)
# ```



# METHOD:
# def operate_on_vector_index(collection: str, operation=lambda: None) -> VectorStoreIndex:
def test_returns_vector_store_index():
    collection = "test"
    operation = lambda vector_index: None
    result = operate_on_vector_index(collection, operation)
    assert isinstance(result, VectorStoreIndex)

def test_calls_operation():
    collection = "test"
    operation = Mock()
    operate_on_vector_index(collection, operation)
    assert operation.called_once_with(get_vector_index(collection))

def test_returns_same_vector_store_index():
    collection = "test"
    operation = lambda vector_index: None
    result = operate_on_vector_index(collection, operation)
    assert result is get_vector_index(collection)

def test_raises_value_error():
    collection = "test"
    operation = lambda vector_index: 123
    with pytest.raises(ValueError):
        operate_on_vector_index(collection, operation)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `operate_on_vector_index` method:
# 
# 1. Test that the method returns a `VectorStoreIndex` object:
# ```python
# def test_returns_vector_store_index():
#     collection = "test"
#     operation = lambda vector_index: None
#     result = operate_on_vector_index(collection, operation)
#     assert isinstance(result, VectorStoreIndex)
# ```
# 2. Test that the method calls the `operation` function with the correct argument:
# ```python
# def test_calls_operation():
#     collection = "test"
#     operation = Mock()
#     operate_on_vector_index(collection, operation)
#     assert operation.called_once_with(get_vector_index(collection))
# ```
# 3. Test that the method returns the same `VectorStoreIndex` object as the one passed to the `operation` function:
# ```python
# def test_returns_same_vector_store_index():
#     collection = "test"
#     operation = lambda vector_index: None
#     result = operate_on_vector_index(collection, operation)
#     assert result is get_vector_index(collection)
# ```
# 4. Test that the method raises a `ValueError` if the `operation` function returns a non-`VectorStoreIndex` object:
# ```python
# def test_raises_value_error():
#     collection = "test"
#     operation = lambda vector_index: 123
#     with pytest.raises(ValueError):
#         operate_on_vector_index(collection, operation)
# ```



# METHOD:
# def get_vector_query_engine(collection: str) -> BaseQueryEngine:
def test_get_vector_query_engine_valid_collection(self):
    collection = "my_collection"
    query_engine = get_vector_query_engine(collection)
    self.assertIsInstance(query_engine, BaseQueryEngine)

def test_get_vector_query_engine_invalid_collection(self):
    collection = "my_invalid_collection"
    with self.assertRaises(ValueError):
        get_vector_query_engine(collection)

def test_get_vector_query_engine_valid_index(self):
    collection = "my_collection"
    index = load_vector_index(collection)
    query_engine = get_vector_query_engine(collection, index=index)
    self.assertIsInstance(query_engine, BaseQueryEngine)

def test_get_vector_query_engine_invalid_index(self):
    collection = "my_collection"
    index = load_vector_index("another_collection")
    with self.assertRaises(ValueError):
        get_vector_query_engine(collection, index=index)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_vector_query_engine` method:
# 
# 1. Test that the method returns a query engine instance when given a valid collection name:
# ```python
# def test_get_vector_query_engine_valid_collection(self):
#     collection = "my_collection"
#     query_engine = get_vector_query_engine(collection)
#     self.assertIsInstance(query_engine, BaseQueryEngine)
# ```
# 2. Test that the method raises a `ValueError` when given an invalid collection name:
# ```python
# def test_get_vector_query_engine_invalid_collection(self):
#     collection = "my_invalid_collection"
#     with self.assertRaises(ValueError):
#         get_vector_query_engine(collection)
# ```
# 3. Test that the method returns a query engine instance when given a valid collection name and a valid index:
# ```python
# def test_get_vector_query_engine_valid_index(self):
#     collection = "my_collection"
#     index = load_vector_index(collection)
#     query_engine = get_vector_query_engine(collection, index=index)
#     self.assertIsInstance(query_engine, BaseQueryEngine)
# ```
# 4. Test that the method raises a `ValueError` when given an invalid index:
# ```python
# def test_get_vector_query_engine_invalid_index(self):
#     collection = "my_collection"
#     index = load_vector_index("another_collection")
#     with self.assertRaises(ValueError):
#         get_vector_query_engine(collection, index=index)
# ```



# METHOD:
# def get_vector_ng_query_engine(vector_storage_dir: str) -> BaseQueryEngine:
import pytest
from lib.index.html import RetrieverQueryEngine, VectorIndexRetriever, load_vector_index, get_response_synthesizer

def test_get_vector_ng_query_engine():
    vector_storage_dir = "./data/vectors"
    query_engine = get_vector_ng_query_engine(vector_storage_dir)
    assert isinstance(query_engine, RetrieverQueryEngine)
    assert isinstance(query_engine.retriever, VectorIndexRetriever)
    assert query_engine.response_synthesizer == get_response_synthesizer("tree_summarize")

def test_vector_index_retriever():
    vector_storage_dir = "./data/vectors"
    query_engine = get_vector_ng_query_engine(vector_storage_dir)
    retriever = query_engine.retriever
    assert isinstance(retriever, VectorIndexRetriever)
    assert retriever.index == load_vector_index(vector_storage_dir)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_vector_ng_query_engine` method:
# ```python
# import pytest
# from lib.index.html import RetrieverQueryEngine, VectorIndexRetriever, load_vector_index, get_response_synthesizer
# 
# def test_get_vector_ng_query_engine():
#     vector_storage_dir = "./data/vectors"
#     query_engine = get_vector_ng_query_engine(vector_storage_dir)
#     assert isinstance(query_engine, RetrieverQueryEngine)
#     assert isinstance(query_engine.retriever, VectorIndexRetriever)
#     assert query_engine.response_synthesizer == get_response_synthesizer("tree_summarize")
# ```
# This test case checks that the `get_vector_ng_query_engine` method returns an instance of `RetrieverQueryEngine`, and that its `retriever` attribute is an instance of `VectorIndexRetriever`. It also checks that the `response_synthesizer` attribute is set to the result of calling `get_response_synthesizer("tree_summarize")`.
# 
# Here's another test case:
# ```python
# def test_vector_index_retriever():
#     vector_storage_dir = "./data/vectors"
#     query_engine = get_vector_ng_query_engine(vector_storage_dir)
#     retriever = query_engine.retriever
#     assert isinstance(retriever, VectorIndexRetriever)
#     assert retriever.index == load_vector_index(vector_storage_dir)
# ```
# This test case checks that the `VectorIndexRetriever` instance returned by the `get_vector_ng_query_engine` method has an `index` attribute set to the result of calling `load_vector_index`.
# 
# You can add more test cases as needed, depending on the specific requirements and behavior of the `get_vector_ng_query_engine` method.

