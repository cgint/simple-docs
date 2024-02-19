import pytest
from unittest.mock import Mock, patch
import json
from typing import List
from llama_index.core import Document, KnowledgeGraphIndex, StorageContext
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core import load_index_from_storage
import os
import atexit
import shutil
from lib.index.kg_classic import (load_kg_graph_index_storage_context, persist_kg_graph_index, delete_kg_graph_index, load_kg_graph_index, get_kg_graph_index, operate_on_kg_graph_index, add_to_or_update_in_kg_graph, get_kg_graph_query_engine)

# METHOD:
# def load_kg_graph_index_storage_context(kg_graph_storage_dir: str) -> StorageContext:
def test_load_kg_graph_index_storage_context_valid_dir():
    storage_context = load_kg_graph_index_storage_context("./test_data/kg_graph")
    assert isinstance(storage_context, StorageContext)

def test_load_kg_graph_index_storage_context_invalid_dir():
    with pytest.raises(ValueError):
        load_kg_graph_index_storage_context("./test_data/kg_graph/non_existent")

def test_load_kg_graph_index_storage_context_valid_dir_properties():
    storage_context = load_kg_graph_index_storage_context("./test_data/kg_graph")
    assert storage_context.persist_dir == "./test_data/kg_graph"


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `load_kg_graph_index_storage_context` method:
# 
# 1. Test that the method returns a `StorageContext` object when given a valid directory path:
# ```python
# def test_load_kg_graph_index_storage_context_valid_dir():
#     storage_context = load_kg_graph_index_storage_context("./test_data/kg_graph")
#     assert isinstance(storage_context, StorageContext)
# ```
# 2. Test that the method raises a `ValueError` when given an invalid directory path:
# ```python
# def test_load_kg_graph_index_storage_context_invalid_dir():
#     with pytest.raises(ValueError):
#         load_kg_graph_index_storage_context("./test_data/kg_graph/non_existent")
# ```
# 3. Test that the method returns a `StorageContext` object with the correct properties when given a valid directory path:
# ```python
# def test_load_kg_graph_index_storage_context_valid_dir_properties():
#     storage_context = load_kg_graph_index_storage_context("./test_data/kg_graph")
#     assert storage_context.persist_dir == "./test_data/kg_graph"
# ```



# METHOD:
# def persist_kg_graph_index(idx: KnowledgeGraphIndex, kg_graph_storage_dir: str):
def test_persist_kg_graph_index(idx, kg_graph_storage_dir):
    doc_count = len(idx.docstore.docs)
    persist_kg_graph_index(idx, kg_graph_storage_dir)
    assert len(os.listdir(kg_graph_storage_dir)) == doc_count

def test_persist_kg_graph_index_prints_message(idx, kg_graph_storage_dir):
    expected_message = f"Persisting {len(idx.docstore.docs)} docs for kg_graph to {kg_graph_storage_dir} ..."
    with patch("builtins.print") as mock_print:
        persist_kg_graph_index(idx, kg_graph_storage_dir)
        mock_print.assert_called_once_with(expected_message)

def test_persist_kg_graph_index_raises_error_if_storage_dir_does_not_exist(idx, kg_graph_storage_dir):
    with pytest.raises(ValueError) as excinfo:
        persist_kg_graph_index(idx, "non-existent-directory")
    assert str(excinfo.value) == "Storage directory does not exist"


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `persist_kg_graph_index` method:
# 
# 1. Test that the method persists the correct number of documents to the storage directory:
# ```python
# def test_persist_kg_graph_index(idx, kg_graph_storage_dir):
#     doc_count = len(idx.docstore.docs)
#     persist_kg_graph_index(idx, kg_graph_storage_dir)
#     assert len(os.listdir(kg_graph_storage_dir)) == doc_count
# ```
# 2. Test that the method prints the correct message to the console:
# ```python
# def test_persist_kg_graph_index_prints_message(idx, kg_graph_storage_dir):
#     expected_message = f"Persisting {len(idx.docstore.docs)} docs for kg_graph to {kg_graph_storage_dir} ..."
#     with patch("builtins.print") as mock_print:
#         persist_kg_graph_index(idx, kg_graph_storage_dir)
#         mock_print.assert_called_once_with(expected_message)
# ```
# 3. Test that the method raises an error if the storage directory does not exist:
# ```python
# def test_persist_kg_graph_index_raises_error_if_storage_dir_does_not_exist(idx, kg_graph_storage_dir):
#     with pytest.raises(ValueError) as excinfo:
#         persist_kg_graph_index(idx, "non-existent-directory")
#     assert str(excinfo.value) == "Storage directory does not exist"
# ```



# METHOD:
# def delete_kg_graph_index(kg_graph_storage_dir: str):
def test_deletes_directory():
    kg_graph_storage_dir = "test/path"
    delete_kg_graph_index(kg_graph_storage_dir)
    assert not os.path.exists(kg_graph_storage_dir)

def test_does_not_delete_invalid_directory():
    kg_graph_storage_dir = "test/path"
    with patch("os.path.exists", return_value=False):
        delete_kg_graph_index(kg_graph_storage_dir)
        assert os.path.exists(kg_graph_storage_dir)

def test_prints_message():
    kg_graph_storage_dir = "test/path"
    with patch("builtins.print") as mock_print:
        delete_kg_graph_index(kg_graph_storage_dir)
        mock_print.assert_called_once_with(f"Deleting kg_graph at {kg_graph_storage_dir} ...")


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `delete_kg_graph_index` method:
# 
# 1. Test that the method deletes the directory at the specified path:
# ```python
# def test_deletes_directory():
#     kg_graph_storage_dir = "test/path"
#     delete_kg_graph_index(kg_graph_storage_dir)
#     assert not os.path.exists(kg_graph_storage_dir)
# ```
# 2. Test that the method does not delete any directories if the specified path is invalid:
# ```python
# def test_does_not_delete_invalid_directory():
#     kg_graph_storage_dir = "test/path"
#     with patch("os.path.exists", return_value=False):
#         delete_kg_graph_index(kg_graph_storage_dir)
#         assert os.path.exists(kg_graph_storage_dir)
# ```
# 3. Test that the method prints a message to the console:
# ```python
# def test_prints_message():
#     kg_graph_storage_dir = "test/path"
#     with patch("builtins.print") as mock_print:
#         delete_kg_graph_index(kg_graph_storage_dir)
#         mock_print.assert_called_once_with(f"Deleting kg_graph at {kg_graph_storage_dir} ...")
# ```



# METHOD:
# def load_kg_graph_index(kg_graph_storage_dir: str) -> KnowledgeGraphIndex:
def test_load_kg_graph_index_with_valid_data(tmpdir):
    kg_graph_storage_dir = tmpdir / "kg-graph"
    os.makedirs(kg_graph_storage_dir)
    with open(kg_graph_storage_dir / "kg-graph.json", "w") as f:
        json.dump({"nodes": [], "edges": []}, f)
    
    kg_graph = load_kg_graph_index(kg_graph_storage_dir)
    
    assert isinstance(kg_graph, KnowledgeGraphIndex)

def test_load_kg_graph_index_with_empty_data(tmpdir):
    kg_graph_storage_dir = tmpdir / "kg-graph"
    
    kg_graph = load_kg_graph_index(kg_graph_storage_dir)
    
    assert isinstance(kg_graph, KnowledgeGraphIndex)

def test_load_kg_graph_index_with_invalid_data(tmpdir):
    kg_graph_storage_dir = tmpdir / "kg-graph"
    os.makedirs(kg_graph_storage_dir)
    with open(kg_graph_storage_dir / "kg-graph.json", "w") as f:
        json.dump({"nodes": [], "edges": []}, f)
    
    with pytest.raises(ValueError):
        load_kg_graph_index(kg_graph_storage_dir)

def test_load_kg_graph_index_with_initialization_failure(tmpdir):
    kg_graph_storage_dir = tmpdir / "kg-graph"
    
    with pytest.raises(FileNotFoundError):
        load_kg_graph_index(kg_graph_storage_dir)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `load_kg_graph_index` method:
# 
# 1. Test that the method returns a `KnowledgeGraphIndex` object when the storage directory exists and contains valid data:
# ```python
# def test_load_kg_graph_index_with_valid_data(tmpdir):
#     kg_graph_storage_dir = tmpdir / "kg-graph"
#     os.makedirs(kg_graph_storage_dir)
#     with open(kg_graph_storage_dir / "kg-graph.json", "w") as f:
#         json.dump({"nodes": [], "edges": []}, f)
#     
#     kg_graph = load_kg_graph_index(kg_graph_storage_dir)
#     
#     assert isinstance(kg_graph, KnowledgeGraphIndex)
# ```
# 2. Test that the method returns a `KnowledgeGraphIndex` object when the storage directory does not exist and needs to be initialized:
# ```python
# def test_load_kg_graph_index_with_empty_data(tmpdir):
#     kg_graph_storage_dir = tmpdir / "kg-graph"
#     
#     kg_graph = load_kg_graph_index(kg_graph_storage_dir)
#     
#     assert isinstance(kg_graph, KnowledgeGraphIndex)
# ```
# 3. Test that the method raises a `ValueError` when the storage directory exists but contains invalid data:
# ```python
# def test_load_kg_graph_index_with_invalid_data(tmpdir):
#     kg_graph_storage_dir = tmpdir / "kg-graph"
#     os.makedirs(kg_graph_storage_dir)
#     with open(kg_graph_storage_dir / "kg-graph.json", "w") as f:
#         json.dump({"nodes": [], "edges": []}, f)
#     
#     with pytest.raises(ValueError):
#         load_kg_graph_index(kg_graph_storage_dir)
# ```
# 4. Test that the method raises a `FileNotFoundError` when the storage directory does not exist and needs to be initialized, but the initialization fails:
# ```python
# def test_load_kg_graph_index_with_initialization_failure(tmpdir):
#     kg_graph_storage_dir = tmpdir / "kg-graph"
#     
#     with pytest.raises(FileNotFoundError):
#         load_kg_graph_index(kg_graph_storage_dir)
# ```



# METHOD:
# def get_kg_graph_index(graph_storage_dir: str) -> KnowledgeGraphIndex:
import pytest
from kg_graph import KnowledgeGraphIndex

def test_get_kg_graph_index_with_valid_directory(tmpdir):
    graph_storage_dir = tmpdir.mkdir("graph_storage")
    index = get_kg_graph_index(str(graph_storage_dir))
    assert isinstance(index, KnowledgeGraphIndex)
    assert index.graph_storage_dir == str(graph_storage_dir)

def test_get_kg_graph_index_with_invalid_directory():
    with pytest.raises(ValueError):
        get_kg_graph_index("invalid/path")


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_kg_graph_index` method:
# ```python
# import pytest
# from kg_graph import KnowledgeGraphIndex
# 
# def test_get_kg_graph_index_with_valid_directory(tmpdir):
#     graph_storage_dir = tmpdir.mkdir("graph_storage")
#     index = get_kg_graph_index(str(graph_storage_dir))
#     assert isinstance(index, KnowledgeGraphIndex)
#     assert index.graph_storage_dir == str(graph_storage_dir)
# 
# def test_get_kg_graph_index_with_invalid_directory():
#     with pytest.raises(ValueError):
#         get_kg_graph_index("invalid/path")
# ```
# These tests check that the `get_kg_graph_index` method returns a valid `KnowledgeGraphIndex` object when given a valid directory, and raises a `ValueError` when given an invalid directory.



# METHOD:
# def operate_on_kg_graph_index(kg_graph_index_dir: str, operation=lambda: None) -> KnowledgeGraphIndex:
def test_successful_operation():
    kg_graph_index_dir = "test/data"
    operation = lambda idx: None
    expected_result = get_kg_graph_index(kg_graph_index_dir)
    actual_result = operate_on_kg_graph_index(kg_graph_index_dir, operation)
    assert actual_result == expected_result

def test_failed_operation():
    kg_graph_index_dir = "test/data"
    operation = lambda idx: raise Exception("Test exception")
    with pytest.raises(Exception):
        operate_on_kg_graph_index(kg_graph_index_dir, operation)

def test_persist_to_disk():
    kg_graph_index_dir = "test/data"
    operation = lambda idx: None
    with patch("lib.kg_graph.persist_kg_graph_index") as mock_persist:
        operate_on_kg_graph_index(kg_graph_index_dir, operation)
        mock_persist.assert_called_once()

def test_no_persist_to_disk():
    kg_graph_index_dir = "test/data"
    operation = lambda idx: raise Exception("Test exception")
    with patch("lib.kg_graph.persist_kg_graph_index") as mock_persist:
        with pytest.raises(Exception):
            operate_on_kg_graph_index(kg_graph_index_dir, operation)
        mock_persist.assert_not_called()


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `operate_on_kg_graph_index` method:
# 
# 1. Test that the function returns the correct value when the operation is successful:
# ```python
# def test_successful_operation():
#     kg_graph_index_dir = "test/data"
#     operation = lambda idx: None
#     expected_result = get_kg_graph_index(kg_graph_index_dir)
#     actual_result = operate_on_kg_graph_index(kg_graph_index_dir, operation)
#     assert actual_result == expected_result
# ```
# 2. Test that the function raises an error when the operation fails:
# ```python
# def test_failed_operation():
#     kg_graph_index_dir = "test/data"
#     operation = lambda idx: raise Exception("Test exception")
#     with pytest.raises(Exception):
#         operate_on_kg_graph_index(kg_graph_index_dir, operation)
# ```
# 3. Test that the function persists the index to disk when the operation is successful:
# ```python
# def test_persist_to_disk():
#     kg_graph_index_dir = "test/data"
#     operation = lambda idx: None
#     with patch("lib.kg_graph.persist_kg_graph_index") as mock_persist:
#         operate_on_kg_graph_index(kg_graph_index_dir, operation)
#         mock_persist.assert_called_once()
# ```
# 4. Test that the function does not persist the index to disk when the operation fails:
# ```python
# def test_no_persist_to_disk():
#     kg_graph_index_dir = "test/data"
#     operation = lambda idx: raise Exception("Test exception")
#     with patch("lib.kg_graph.persist_kg_graph_index") as mock_persist:
#         with pytest.raises(Exception):
#             operate_on_kg_graph_index(kg_graph_index_dir, operation)
#         mock_persist.assert_not_called()
# ```



# METHOD:
# def add_to_or_update_in_kg_graph(graph_storage_dir: str, documents: List[Document]):
def test_update_kg_graph_index(tmpdir):
    graph_storage_dir = tmpdir / "graph"
    documents = [Document("test1"), Document("test2")]
    add_to_or_update_in_kg_graph(graph_storage_dir, documents)
    assert len(graph_storage_dir.listdir()) == 1

def test_no_update_empty_docs(tmpdir):
    graph_storage_dir = tmpdir / "graph"
    documents = []
    add_to_or_update_in_kg_graph(graph_storage_dir, documents)
    assert len(graph_storage_dir.listdir()) == 0

def test_no_update_missing_dir(tmpdir):
    graph_storage_dir = tmpdir / "missing"
    documents = [Document("test1"), Document("test2")]
    with pytest.raises(FileNotFoundError):
        add_to_or_update_in_kg_graph(graph_storage_dir, documents)

def test_update_missing_index(tmpdir):
    graph_storage_dir = tmpdir / "graph"
    documents = [Document("test1"), Document("test2")]
    add_to_or_update_in_kg_graph(graph_storage_dir, documents)
    assert len(graph_storage_dir.listdir()) == 1

def test_update_corrupted_index(tmpdir):
    graph_storage_dir = tmpdir / "graph"
    documents = [Document("test1"), Document("test2")]
    add_to_or_update_in_kg_graph(graph_storage_dir, documents)
    assert len(graph_storage_dir.listdir()) == 1


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `add_to_or_update_in_kg_graph` method:
# 
# 1. Test that the method updates the KG graph index when the documents list is not empty and the graph storage directory exists:
# ```python
# def test_update_kg_graph_index(tmpdir):
#     graph_storage_dir = tmpdir / "graph"
#     documents = [Document("test1"), Document("test2")]
#     add_to_or_update_in_kg_graph(graph_storage_dir, documents)
#     assert len(graph_storage_dir.listdir()) == 1
# ```
# 2. Test that the method does not update the KG graph index when the documents list is empty:
# ```python
# def test_no_update_empty_docs(tmpdir):
#     graph_storage_dir = tmpdir / "graph"
#     documents = []
#     add_to_or_update_in_kg_graph(graph_storage_dir, documents)
#     assert len(graph_storage_dir.listdir()) == 0
# ```
# 3. Test that the method raises an error when the graph storage directory does not exist:
# ```python
# def test_no_update_missing_dir(tmpdir):
#     graph_storage_dir = tmpdir / "missing"
#     documents = [Document("test1"), Document("test2")]
#     with pytest.raises(FileNotFoundError):
#         add_to_or_update_in_kg_graph(graph_storage_dir, documents)
# ```
# 4. Test that the method updates the KG graph index when the documents list is not empty and the graph storage directory exists, but the index file does not exist:
# ```python
# def test_update_missing_index(tmpdir):
#     graph_storage_dir = tmpdir / "graph"
#     documents = [Document("test1"), Document("test2")]
#     add_to_or_update_in_kg_graph(graph_storage_dir, documents)
#     assert len(graph_storage_dir.listdir()) == 1
# ```
# 5. Test that the method updates the KG graph index when the documents list is not empty and the graph storage directory exists, but the index file is corrupted:
# ```python
# def test_update_corrupted_index(tmpdir):
#     graph_storage_dir = tmpdir / "graph"
#     documents = [Document("test1"), Document("test2")]
#     add_to_or_update_in_kg_graph(graph_storage_dir, documents)
#     assert len(graph_storage_dir.listdir()) == 1
# ```



# METHOD:
# def get_kg_graph_query_engine(graph_storage_dir: str) -> BaseQueryEngine:
def test_get_kg_graph_query_engine_valid_dir(tmpdir):
    # Create a temporary directory for the graph storage
    tmpdir = Path(tmpdir)
    os.mkdir(tmpdir / "graph")

    # Call the method with the temporary directory as input
    query_engine = get_kg_graph_query_engine(str(tmpdir))

    # Assert that a query engine instance was returned
    assert isinstance(query_engine, BaseQueryEngine)

def test_get_kg_graph_query_engine_invalid_dir():
    # Call the method with a non-existent directory as input
    with pytest.raises(ValueError):
        get_kg_graph_query_engine("non-existent/directory")

def test_get_kg_graph_query_engine_invalid_storage(tmpdir):
    # Create a temporary directory for the graph storage
    tmpdir = Path(tmpdir)
    os.mkdir(tmpdir / "graph")

    # Write some random data to the directory
    with open(tmpdir / "random.txt", "w") as f:
        f.write("This is not a graph storage!")

    # Call the method with the temporary directory as input
    with pytest.raises(ValueError):
        get_kg_graph_query_engine(str(tmpdir))


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_kg_graph_query_engine` method:
# 
# 1. Test that the method returns a query engine instance when given a valid graph storage directory:
# ```python
# def test_get_kg_graph_query_engine_valid_dir(tmpdir):
#     # Create a temporary directory for the graph storage
#     tmpdir = Path(tmpdir)
#     os.mkdir(tmpdir / "graph")
# 
#     # Call the method with the temporary directory as input
#     query_engine = get_kg_graph_query_engine(str(tmpdir))
# 
#     # Assert that a query engine instance was returned
#     assert isinstance(query_engine, BaseQueryEngine)
# ```
# 2. Test that the method raises an error when given an invalid graph storage directory:
# ```python
# def test_get_kg_graph_query_engine_invalid_dir():
#     # Call the method with a non-existent directory as input
#     with pytest.raises(ValueError):
#         get_kg_graph_query_engine("non-existent/directory")
# ```
# 3. Test that the method raises an error when given a directory that is not a graph storage:
# ```python
# def test_get_kg_graph_query_engine_invalid_storage(tmpdir):
#     # Create a temporary directory for the graph storage
#     tmpdir = Path(tmpdir)
#     os.mkdir(tmpdir / "graph")
# 
#     # Write some random data to the directory
#     with open(tmpdir / "random.txt", "w") as f:
#         f.write("This is not a graph storage!")
# 
#     # Call the method with the temporary directory as input
#     with pytest.raises(ValueError):
#         get_kg_graph_query_engine(str(tmpdir))
# ```

