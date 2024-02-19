import pytest
from unittest.mock import Mock, patch
import json
import os
from llama_index.core import DocumentSummaryIndex, StorageContext, get_response_synthesizer, load_index_from_storage
from llama_index.core.query_engine import BaseQueryEngine
import atexit
import shutil
from lib.index.doc_sum_index import (create_doc_summary_index, delete_doc_summary_index, persist_index, operate_on_doc_sum_index, get_doc_sum_index, load_doc_sum_index, get_doc_sum_index_query_engine)

# METHOD:
# def create_doc_summary_index(documents, storage_dir):
def test_create_doc_summary_index(documents, storage_dir):
    doc_sum_index = create_doc_summary_index(documents, storage_dir)
    assert isinstance(doc_sum_index, DocumentSummaryIndex)
    assert len(doc_sum_index.ref_docs) == len(documents)

def test_create_doc_summary_index_empty_list(storage_dir):
    with pytest.raises(ValueError):
        create_doc_summary_index([], storage_dir)

def test_create_doc_summary_index_invalid_documents(storage_dir):
    with pytest.raises(TypeError):
        create_doc_summary_index([1, 2, 3], storage_dir)

def test_create_doc_summary_index_invalid_storage_dir(documents):
    with pytest.raises(ValueError):
        create_doc_summary_index(documents, "/path/to/invalid/directory")


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `create_doc_summary_index` method:
# 
# 1. Test that the method returns a valid document summary index when given a list of documents and a storage directory:
# ```python
# def test_create_doc_summary_index(documents, storage_dir):
#     doc_sum_index = create_doc_summary_index(documents, storage_dir)
#     assert isinstance(doc_sum_index, DocumentSummaryIndex)
#     assert len(doc_sum_index.ref_docs) == len(documents)
# ```
# 2. Test that the method raises a `ValueError` when given an empty list of documents:
# ```python
# def test_create_doc_summary_index_empty_list(storage_dir):
#     with pytest.raises(ValueError):
#         create_doc_summary_index([], storage_dir)
# ```
# 3. Test that the method raises a `TypeError` when given a list of documents that are not instances of `Document`:
# ```python
# def test_create_doc_summary_index_invalid_documents(storage_dir):
#     with pytest.raises(TypeError):
#         create_doc_summary_index([1, 2, 3], storage_dir)
# ```
# 4. Test that the method raises a `ValueError` when given an invalid storage directory:
# ```python
# def test_create_doc_summary_index_invalid_storage_dir(documents):
#     with pytest.raises(ValueError):
#         create_doc_summary_index(documents, "/path/to/invalid/directory")
# ```



# METHOD:
# def delete_doc_summary_index(doc_sum_index_dir: str):
def test_deletes_doc_summary_index(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda x: True)
    with patch("builtins.print") as mock_print:
        delete_doc_summary_index("/some/dir")
        mock_print.assert_called_once_with(f"Deleting doc_sum_index at /some/dir ...")

def test_deletes_doc_summary_index_if_exists(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda x: True)
    with patch("shutil.rmtree") as mock_rmtree:
        delete_doc_summary_index("/some/dir")
        mock_rmtree.assert_called_once_with("/some/dir")

def test_does_not_delete_doc_summary_index_if_not_exists(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda x: False)
    with patch("shutil.rmtree") as mock_rmtree:
        delete_doc_summary_index("/some/dir")
        mock_rmtree.assert_not_called()


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `delete_doc_summary_index` method:
# 
# 1. Test that the method prints a message indicating that it is deleting the index:
# ```python
# def test_deletes_doc_summary_index(monkeypatch):
#     monkeypatch.setattr("os.path.exists", lambda x: True)
#     with patch("builtins.print") as mock_print:
#         delete_doc_summary_index("/some/dir")
#         mock_print.assert_called_once_with(f"Deleting doc_sum_index at /some/dir ...")
# ```
# 2. Test that the method deletes the index directory if it exists:
# ```python
# def test_deletes_doc_summary_index_if_exists(monkeypatch):
#     monkeypatch.setattr("os.path.exists", lambda x: True)
#     with patch("shutil.rmtree") as mock_rmtree:
#         delete_doc_summary_index("/some/dir")
#         mock_rmtree.assert_called_once_with("/some/dir")
# ```
# 3. Test that the method does not delete the index directory if it does not exist:
# ```python
# def test_does_not_delete_doc_summary_index_if_not_exists(monkeypatch):
#     monkeypatch.setattr("os.path.exists", lambda x: False)
#     with patch("shutil.rmtree") as mock_rmtree:
#         delete_doc_summary_index("/some/dir")
#         mock_rmtree.assert_not_called()
# ```



# METHOD:
# def persist_index(idx: DocumentSummaryIndex, doc_sum_index_dir: str):
def test_persist_index_with_empty_idx():
    idx = DocumentSummaryIndex()
    doc_sum_index_dir = "test_dir"
    persist_index(idx, doc_sum_index_dir)
    assert len(os.listdir(doc_sum_index_dir)) == 0

def test_persist_index_with_non_empty_idx():
    idx = DocumentSummaryIndex()
    idx.docstore.docs = [Document("test1"), Document("test2")]
    doc_sum_index_dir = "test_dir"
    persist_index(idx, doc_sum_index_dir)
    assert len(os.listdir(doc_sum_index_dir)) == 2


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `persist_index` method:
# ```python
# def test_persist_index_with_empty_idx():
#     idx = DocumentSummaryIndex()
#     doc_sum_index_dir = "test_dir"
#     persist_index(idx, doc_sum_index_dir)
#     assert len(os.listdir(doc_sum_index_dir)) == 0
# 
# def test_persist_index_with_non_empty_idx():
#     idx = DocumentSummaryIndex()
#     idx.docstore.docs = [Document("test1"), Document("test2")]
#     doc_sum_index_dir = "test_dir"
#     persist_index(idx, doc_sum_index_dir)
#     assert len(os.listdir(doc_sum_index_dir)) == 2
# ```
# These tests check that the `persist_index` method correctly persists an empty and a non-empty index to the specified directory.



# METHOD:
# def operate_on_doc_sum_index(doc_sum_index_dir: str, operation=lambda: None) -> DocumentSummaryIndex:
def test_operation_called_with_correct_index(mocker):
    mock_operation = mocker.Mock()
    operate_on_doc_sum_index("test_dir", mock_operation)
    assert mock_operation.call_count == 1
    assert mock_operation.call_args[0][0] == get_doc_sum_index("test_dir")

def test_index_persisted_after_operation(mocker):
    mock_operation = mocker.Mock()
    operate_on_doc_sum_index("test_dir", mock_operation)
    assert persist_index.call_count == 1
    assert persist_index.call_args[0][0] == get_doc_sum_index("test_dir")

def test_atexit_registration(mocker):
    mock_operation = mocker.Mock()
    operate_on_doc_sum_index("test_dir", mock_operation)
    assert atexit.register.call_count == 1
    assert atexit.register.call_args[0][0] == persist_index

def test_atexit_unregistration(mocker):
    mock_operation = mocker.Mock()
    operate_on_doc_sum_index("test_dir", mock_operation)
    assert atexit.unregister.call_count == 1
    assert atexit.unregister.call_args[0][0] == persist_index

def test_function_returns_correct_index(mocker):
    mock_operation = mocker.Mock()
    result = operate_on_doc_sum_index("test_dir", mock_operation)
    assert result == get_doc_sum_index("test_dir")


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `operate_on_doc_sum_index` method:
# 
# 1. Test that the operation is called with the correct index object:
# ```python
# def test_operation_called_with_correct_index(mocker):
#     mock_operation = mocker.Mock()
#     operate_on_doc_sum_index("test_dir", mock_operation)
#     assert mock_operation.call_count == 1
#     assert mock_operation.call_args[0][0] == get_doc_sum_index("test_dir")
# ```
# 2. Test that the index is persisted after the operation:
# ```python
# def test_index_persisted_after_operation(mocker):
#     mock_operation = mocker.Mock()
#     operate_on_doc_sum_index("test_dir", mock_operation)
#     assert persist_index.call_count == 1
#     assert persist_index.call_args[0][0] == get_doc_sum_index("test_dir")
# ```
# 3. Test that the atexit registration is called with the correct function:
# ```python
# def test_atexit_registration(mocker):
#     mock_operation = mocker.Mock()
#     operate_on_doc_sum_index("test_dir", mock_operation)
#     assert atexit.register.call_count == 1
#     assert atexit.register.call_args[0][0] == persist_index
# ```
# 4. Test that the atexit unregistration is called with the correct function:
# ```python
# def test_atexit_unregistration(mocker):
#     mock_operation = mocker.Mock()
#     operate_on_doc_sum_index("test_dir", mock_operation)
#     assert atexit.unregister.call_count == 1
#     assert atexit.unregister.call_args[0][0] == persist_index
# ```
# 5. Test that the function returns the correct index object:
# ```python
# def test_function_returns_correct_index(mocker):
#     mock_operation = mocker.Mock()
#     result = operate_on_doc_sum_index("test_dir", mock_operation)
#     assert result == get_doc_sum_index("test_dir")
# ```



# METHOD:
# def get_doc_sum_index(doc_sum_index_dir) -> DocumentSummaryIndex:
def test_get_doc_sum_index_returns_document_summary_index(self):
    doc_sum_index_dir = "path/to/doc_sum_index"
    expected_result = DocumentSummaryIndex()
    actual_result = get_doc_sum_index(doc_sum_index_dir)
    self.assertEqual(actual_result, expected_result)

def test_get_doc_sum_index_raises_value_error_with_invalid_directory_path(self):
    doc_sum_index_dir = "path/to/invalid/directory"
    with self.assertRaises(ValueError):
        get_doc_sum_index(doc_sum_index_dir)

def test_get_doc_sum_index_returns_document_summary_index_with_valid_directory_path(self):
    doc_sum_index_dir = "path/to/doc_sum_index"
    expected_result = DocumentSummaryIndex()
    with patch("lib.index.html.load_doc_sum_index") as mock_load_doc_sum_index:
        mock_load_doc_sum_index.return_value = expected_result
        actual_result = get_doc_sum_index(doc_sum_index_dir)
    self.assertEqual(actual_result, expected_result)

def test_get_doc_sum_index_raises_value_error_with_invalid_directory_path(self):
    doc_sum_index_dir = "path/to/doc_sum_index"
    with patch("lib.index.html.load_doc_sum_index") as mock_load_doc_sum_index:
        mock_load_doc_sum_index.return_value = None
        with self.assertRaises(ValueError):
            get_doc_sum_index(doc_sum_index_dir)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_doc_sum_index` method:
# 
# 1. Test that the method returns a `DocumentSummaryIndex` object when given a valid directory path:
# ```python
# def test_get_doc_sum_index_returns_document_summary_index(self):
#     doc_sum_index_dir = "path/to/doc_sum_index"
#     expected_result = DocumentSummaryIndex()
#     actual_result = get_doc_sum_index(doc_sum_index_dir)
#     self.assertEqual(actual_result, expected_result)
# ```
# 2. Test that the method raises a `ValueError` when given an invalid directory path:
# ```python
# def test_get_doc_sum_index_raises_value_error_with_invalid_directory_path(self):
#     doc_sum_index_dir = "path/to/invalid/directory"
#     with self.assertRaises(ValueError):
#         get_doc_sum_index(doc_sum_index_dir)
# ```
# 3. Test that the method returns a `DocumentSummaryIndex` object when given a valid directory path and the `load_doc_sum_index` function returns a valid result:
# ```python
# def test_get_doc_sum_index_returns_document_summary_index_with_valid_directory_path(self):
#     doc_sum_index_dir = "path/to/doc_sum_index"
#     expected_result = DocumentSummaryIndex()
#     with patch("lib.index.html.load_doc_sum_index") as mock_load_doc_sum_index:
#         mock_load_doc_sum_index.return_value = expected_result
#         actual_result = get_doc_sum_index(doc_sum_index_dir)
#     self.assertEqual(actual_result, expected_result)
# ```
# 4. Test that the method raises a `ValueError` when given a valid directory path but the `load_doc_sum_index` function returns an invalid result:
# ```python
# def test_get_doc_sum_index_raises_value_error_with_invalid_directory_path(self):
#     doc_sum_index_dir = "path/to/doc_sum_index"
#     with patch("lib.index.html.load_doc_sum_index") as mock_load_doc_sum_index:
#         mock_load_doc_sum_index.return_value = None
#         with self.assertRaises(ValueError):
#             get_doc_sum_index(doc_sum_index_dir)
# ```



# METHOD:
# def load_doc_sum_index(doc_sum_index_dir) -> DocumentSummaryIndex:
def test_create_new_index(tmpdir):
    doc_sum_index_dir = tmpdir / "doc_sum_index"
    assert not os.path.exists(doc_sum_index_dir)
    load_doc_sum_index(doc_sum_index_dir)
    assert os.path.exists(doc_sum_index_dir)

def test_load_existing_index(tmpdir):
    doc_sum_index_dir = tmpdir / "doc_sum_index"
    os.makedirs(doc_sum_index_dir)
    idx = DocumentSummaryIndex.from_documents([], show_progress=True)
    persist_index(idx, doc_sum_index_dir)
    assert os.path.exists(doc_sum_index_dir / "index.json")
    load_doc_sum_index(doc_sum_index_dir)

def test_invalid_path(tmpdir):
    doc_sum_index_dir = tmpdir / "doc_sum_index"
    with pytest.raises(ValueError):
        load_doc_sum_index("")

def test_not_a_directory(tmpdir):
    doc_sum_index_dir = tmpdir / "doc_sum_index"
    with pytest.raises(ValueError):
        load_doc_sum_index(doc_sum_index_dir)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `load_doc_sum_index` method:
# 
# 1. Test that the method creates a new index if it doesn't exist:
# ```python
# def test_create_new_index(tmpdir):
#     doc_sum_index_dir = tmpdir / "doc_sum_index"
#     assert not os.path.exists(doc_sum_index_dir)
#     load_doc_sum_index(doc_sum_index_dir)
#     assert os.path.exists(doc_sum_index_dir)
# ```
# 2. Test that the method loads an existing index if it exists:
# ```python
# def test_load_existing_index(tmpdir):
#     doc_sum_index_dir = tmpdir / "doc_sum_index"
#     os.makedirs(doc_sum_index_dir)
#     idx = DocumentSummaryIndex.from_documents([], show_progress=True)
#     persist_index(idx, doc_sum_index_dir)
#     assert os.path.exists(doc_sum_index_dir / "index.json")
#     load_doc_sum_index(doc_sum_index_dir)
# ```
# 3. Test that the method raises an error if the index directory is not a valid path:
# ```python
# def test_invalid_path(tmpdir):
#     doc_sum_index_dir = tmpdir / "doc_sum_index"
#     with pytest.raises(ValueError):
#         load_doc_sum_index("")
# ```
# 4. Test that the method raises an error if the index directory is not a directory:
# ```python
# def test_not_a_directory(tmpdir):
#     doc_sum_index_dir = tmpdir / "doc_sum_index"
#     with pytest.raises(ValueError):
#         load_doc_sum_index(doc_sum_index_dir)
# ```



# METHOD:
# def get_doc_sum_index_query_engine(doc_sum_index_dir: str) -> BaseQueryEngine:
def test_returns_query_engine():
    doc_sum_index_dir = "path/to/doc_sum_index"
    query_engine = get_doc_sum_index_query_engine(doc_sum_index_dir)
    assert isinstance(query_engine, BaseQueryEngine)

def test_loads_document_summary_index():
    doc_sum_index_dir = "path/to/doc_sum_index"
    query_engine = get_doc_sum_index_query_engine(doc_sum_index_dir)
    assert query_engine.doc_sum_index == load_doc_sum_index(doc_sum_index_dir)

def test_sets_response_synthesizer():
    doc_sum_index_dir = "path/to/doc_sum_index"
    query_engine = get_doc_sum_index_query_engine(doc_sum_index_dir)
    assert isinstance(query_engine.response_synthesizer, TreeSummarizeResponseSynthesizer)

def test_sets_use_async():
    doc_sum_index_dir = "path/to/doc_sum_index"
    query_engine = get_doc_sum_index_query_engine(doc_sum_index_dir)
    assert query_engine.response_synthesizer.use_async == True


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_doc_sum_index_query_engine` method:
# 
# 1. Test that the method returns a query engine instance:
# ```python
# def test_returns_query_engine():
#     doc_sum_index_dir = "path/to/doc_sum_index"
#     query_engine = get_doc_sum_index_query_engine(doc_sum_index_dir)
#     assert isinstance(query_engine, BaseQueryEngine)
# ```
# 2. Test that the method loads the document summary index from the specified directory:
# ```python
# def test_loads_document_summary_index():
#     doc_sum_index_dir = "path/to/doc_sum_index"
#     query_engine = get_doc_sum_index_query_engine(doc_sum_index_dir)
#     assert query_engine.doc_sum_index == load_doc_sum_index(doc_sum_index_dir)
# ```
# 3. Test that the method sets the response synthesizer to a `TreeSummarizeResponseSynthesizer` instance:
# ```python
# def test_sets_response_synthesizer():
#     doc_sum_index_dir = "path/to/doc_sum_index"
#     query_engine = get_doc_sum_index_query_engine(doc_sum_index_dir)
#     assert isinstance(query_engine.response_synthesizer, TreeSummarizeResponseSynthesizer)
# ```
# 4. Test that the method sets the `use_async` parameter to `True`:
# ```python
# def test_sets_use_async():
#     doc_sum_index_dir = "path/to/doc_sum_index"
#     query_engine = get_doc_sum_index_query_engine(doc_sum_index_dir)
#     assert query_engine.response_synthesizer.use_async == True
# ```

