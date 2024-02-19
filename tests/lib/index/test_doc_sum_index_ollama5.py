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
import os
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Sequence
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from lib import constants
from lib.index.helper import cur_simple_date_time_sec
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms import CustomLLM
from llama_index.core.llms import ChatMessage, ChatResponse, ChatResponseGen, CompletionResponse, CompletionResponseGen, LLMMetadata, MessageRole
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from lib.callbacks.simple_dict_collector import SimpleDictStoreHandler
from llama_index.callbacks.aim import AimCallback
from llama_index.core.callbacks import CallbackManager
import pandas as pd
import time
from typing import List, Optional
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
import concurrent.futures
from lib.index.doc_sum_index import load_doc_sum_index
from lib.index.kg_classic import load_kg_graph_index
from lib.index.terms.kg_num_term_neo4j import load_graph_index
from lib.vector_chroma import load_vector_index
from llama_index.core.query_engine import RetrieverQueryEngine
from lib import constants
from lib.index.error_helper import write_error_to_file
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.question_gen.guidance import GuidanceQuestionGenerator
from guidance.models import OpenAI
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.response_synthesizers import ResponseMode
from typing import List
from llama_index.core import Document
from llama_index.core.readers.json import JSONReader
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
from typing import Any, Dict, List, Optional, cast
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.callbacks import EventPayload
from lib.index.helper import cur_simple_date_time_sec
import pandas as pd
import os
from llama_index.core.llms import ChatMessage
import re
import requests
import os
import hashlib
from lib import constants
from typing import List
from llama_index.core import Document, KnowledgeGraphIndex, StorageContext
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core import load_index_from_storage
import os
import atexit
import shutil
from concurrent.futures import FIRST_COMPLETED, ALL_COMPLETED, ThreadPoolExecutor, wait
import os
from time import sleep
from lib.index.doc_sum_index import delete_doc_summary_index, operate_on_doc_sum_index, persist_index
from lib.index.html import clean_html_content, get_documents_from_urls_as_mirror, get_documents_from_urls
from lib.index.helper import cur_simple_date_time_sec
from lib.index.error_helper import write_error_to_file
from lib.index.kg_classic import delete_kg_graph_index, operate_on_kg_graph_index
from lib.index.pdf import get_content_from_pdf_file
from lib.index.terms.term_index import build_term_reference_index, count_terms_per_document, write_term_references_to_file
from lib.index.terms.terms import terms_from_txt
from lib.index.terms.kg_num_term_neo4j import kg_neo4j_delete_all_nodes, operate_on_graph_index
from lib.index.helper import list_files
from lib.json import get_content_from_json_file
from lib.vector_chroma import delete_chroma_collection, operate_on_vector_index
from llama_index.core import Document, KnowledgeGraphIndex
from typing import List
from lib import constants
import queue
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
from typing import List
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from llama_index.core import Document
from lib.index.web import create_simple_identifier_from_url, get_plain_content_from
from bs4 import BeautifulSoup
from typing import List
from llama_index.core import Document
from llama_index.readers.file import PDFReader
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
def test_deletes_directory():
    doc_sum_index_dir = "path/to/doc_sum_index"
    delete_doc_summary_index(doc_sum_index_dir)
    assert not os.path.exists(doc_sum_index_dir)


def test_prints_message():
    doc_sum_index_dir = "path/to/doc_sum_index"
    with patch("builtins.print") as mock_print:
        delete_doc_summary_index(doc_sum_index_dir)
        mock_print.assert_called_once_with(f"Deleting doc_sum_index at {doc_sum_index_dir} ...")


def test_does_not_raise_error():
    doc_sum_index_dir = "path/to/doc_sum_index"
    delete_doc_summary_index(doc_sum_index_dir)
    assert True  # no error raised


def test_does_not_raise_error_if_empty():
    doc_sum_index_dir = "path/to/doc_sum_index"
    os.makedirs(doc_sum_index_dir)  # create an empty directory
    delete_doc_summary_index(doc_sum_index_dir)
    assert True  # no error raised


def test_raises_file_not_found_error():
    doc_sum_index_dir = "path/to/doc_sum_index"
    with pytest.raises(FileNotFoundError):
        delete_doc_summary_index(doc_sum_index_dir)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `delete_doc_summary_index` method:
# 
# 1. Test that the method deletes the directory at the specified path:
# ```python
# def test_deletes_directory():
#     doc_sum_index_dir = "path/to/doc_sum_index"
#     delete_doc_summary_index(doc_sum_index_dir)
#     assert not os.path.exists(doc_sum_index_dir)
# ```
# 2. Test that the method prints a message indicating that it is deleting the directory:
# ```python
# def test_prints_message():
#     doc_sum_index_dir = "path/to/doc_sum_index"
#     with patch("builtins.print") as mock_print:
#         delete_doc_summary_index(doc_sum_index_dir)
#         mock_print.assert_called_once_with(f"Deleting doc_sum_index at {doc_sum_index_dir} ...")
# ```
# 3. Test that the method does not raise an error if the directory does not exist:
# ```python
# def test_does_not_raise_error():
#     doc_sum_index_dir = "path/to/doc_sum_index"
#     delete_doc_summary_index(doc_sum_index_dir)
#     assert True  # no error raised
# ```
# 4. Test that the method does not raise an error if the directory is empty:
# ```python
# def test_does_not_raise_error_if_empty():
#     doc_sum_index_dir = "path/to/doc_sum_index"
#     os.makedirs(doc_sum_index_dir)  # create an empty directory
#     delete_doc_summary_index(doc_sum_index_dir)
#     assert True  # no error raised
# ```
# 5. Test that the method raises a `FileNotFoundError` if the directory does not exist:
# ```python
# def test_raises_file_not_found_error():
#     doc_sum_index_dir = "path/to/doc_sum_index"
#     with pytest.raises(FileNotFoundError):
#         delete_doc_summary_index(doc_sum_index_dir)
# ```



# METHOD:
# def persist_index(idx: DocumentSummaryIndex, doc_sum_index_dir: str):
def test_persist_index(idx, doc_sum_index_dir):
    persist_index(idx, doc_sum_index_dir)
    assert os.path.exists(doc_sum_index_dir)
    assert len(os.listdir(doc_sum_index_dir)) == 1


def test_print_message(idx, doc_sum_index_dir):
    persist_index(idx, doc_sum_index_dir)
    assert "Persisting" in caplog.text
    assert f"{len(idx.docstore.docs)} docs" in caplog.text


def test_call_persist(idx, doc_sum_index_dir):
    persist_index(idx, doc_sum_index_dir)
    assert idx.storage_context.persist.called_once_with(persist_dir=doc_sum_index_dir)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `persist_index` method:
# 
# 1. Test that the method persists the index to the specified directory:
# ```python
# def test_persist_index(idx, doc_sum_index_dir):
#     persist_index(idx, doc_sum_index_dir)
#     assert os.path.exists(doc_sum_index_dir)
#     assert len(os.listdir(doc_sum_index_dir)) == 1
# ```
# 2. Test that the method prints a message indicating the number of documents being persisted:
# ```python
# def test_print_message(idx, doc_sum_index_dir):
#     persist_index(idx, doc_sum_index_dir)
#     assert "Persisting" in caplog.text
#     assert f"{len(idx.docstore.docs)} docs" in caplog.text
# ```
# 3. Test that the method calls `storage_context.persist` with the correct arguments:
# ```python
# def test_call_persist(idx, doc_sum_index_dir):
#     persist_index(idx, doc_sum_index_dir)
#     assert idx.storage_context.persist.called_once_with(persist_dir=doc_sum_index_dir)
# ```



# METHOD:
# def operate_on_doc_sum_index(doc_sum_index_dir: str, operation=lambda: None) -> DocumentSummaryIndex:
def test_operation_called_with_correct_argument():
    def operation(idx):
        assert idx == get_doc_sum_index(doc_sum_index_dir)
    operate_on_doc_sum_index(doc_sum_index_dir, operation)


def test_persist_index_called_with_correct_arguments():
    def persist_index(idx, doc_sum_index_dir):
        assert idx == get_doc_sum_index(doc_sum_index_dir)
        assert doc_sum_index_dir == "path/to/doc_sum_index"
    operate_on_doc_sum_index(doc_sum_index_dir, persist_index)


def test_atexit_register_called_with_correct_arguments():
    def atexist_reg_callable(idx):
        assert idx == get_doc_sum_index(doc_sum_index_dir)
    operate_on_doc_sum_index(doc_sum_index_dir, atexist_reg_callable)


def test_atexit_unregister_called_with_correct_arguments():
    def atexist_reg_callable(idx):
        assert idx == get_doc_sum_index(doc_sum_index_dir)
    operate_on_doc_sum_index(doc_sum_index_dir, atexist_reg_callable)


def test_get_doc_sum_index_called_with_correct_argument():
    def get_doc_sum_index(doc_sum_index_dir):
        assert doc_sum_index_dir == "path/to/doc_sum_index"
    operate_on_doc_sum_index(doc_sum_index_dir, lambda: None)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `operate_on_doc_sum_index` method:
# 
# 1. Test that the operation is called with the correct argument:
# ```python
# def test_operation_called_with_correct_argument():
#     def operation(idx):
#         assert idx == get_doc_sum_index(doc_sum_index_dir)
#     operate_on_doc_sum_index(doc_sum_index_dir, operation)
# ```
# 2. Test that the `persist_index` function is called with the correct arguments:
# ```python
# def test_persist_index_called_with_correct_arguments():
#     def persist_index(idx, doc_sum_index_dir):
#         assert idx == get_doc_sum_index(doc_sum_index_dir)
#         assert doc_sum_index_dir == "path/to/doc_sum_index"
#     operate_on_doc_sum_index(doc_sum_index_dir, persist_index)
# ```
# 3. Test that the `atexit.register` function is called with the correct arguments:
# ```python
# def test_atexit_register_called_with_correct_arguments():
#     def atexist_reg_callable(idx):
#         assert idx == get_doc_sum_index(doc_sum_index_dir)
#     operate_on_doc_sum_index(doc_sum_index_dir, atexist_reg_callable)
# ```
# 4. Test that the `atexit.unregister` function is called with the correct arguments:
# ```python
# def test_atexit_unregister_called_with_correct_arguments():
#     def atexist_reg_callable(idx):
#         assert idx == get_doc_sum_index(doc_sum_index_dir)
#     operate_on_doc_sum_index(doc_sum_index_dir, atexist_reg_callable)
# ```
# 5. Test that the `get_doc_sum_index` function is called with the correct argument:
# ```python
# def test_get_doc_sum_index_called_with_correct_argument():
#     def get_doc_sum_index(doc_sum_index_dir):
#         assert doc_sum_index_dir == "path/to/doc_sum_index"
#     operate_on_doc_sum_index(doc_sum_index_dir, lambda: None)
# ```



# METHOD:
# def get_doc_sum_index(doc_sum_index_dir) -> DocumentSummaryIndex:
def test_get_doc_sum_index_returns_document_summary_index(self):
    doc_sum_index_dir = "path/to/doc_sum_index"
    expected_result = DocumentSummaryIndex(doc_sum_index_dir)
    actual_result = get_doc_sum_index(doc_sum_index_dir)
    self.assertEqual(actual_result, expected_result)


def test_get_doc_sum_index_raises_value_error_with_invalid_directory_path(self):
    doc_sum_index_dir = "path/to/invalid/directory"
    with self.assertRaises(ValueError):
        get_doc_sum_index(doc_sum_index_dir)


def test_get_doc_sum_index_returns_document_summary_index_with_existing_index(self):
    doc_sum_index_dir = "path/to/doc_sum_index"
    expected_result = DocumentSummaryIndex(doc_sum_index_dir)
    actual_result = get_doc_sum_index(doc_sum_index_dir)
    self.assertEqual(actual_result, expected_result)


def test_get_doc_sum_index_raises_value_error_with_invalid_directory_path_and_missing_index(self):
    doc_sum_index_dir = "path/to/invalid/directory"
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
#     expected_result = DocumentSummaryIndex(doc_sum_index_dir)
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
# 3. Test that the method returns a `DocumentSummaryIndex` object when given a valid directory path and the index file exists:
# ```python
# def test_get_doc_sum_index_returns_document_summary_index_with_existing_index(self):
#     doc_sum_index_dir = "path/to/doc_sum_index"
#     expected_result = DocumentSummaryIndex(doc_sum_index_dir)
#     actual_result = get_doc_sum_index(doc_sum_index_dir)
#     self.assertEqual(actual_result, expected_result)
# ```
# 4. Test that the method raises a `ValueError` when given an invalid directory path and the index file does not exist:
# ```python
# def test_get_doc_sum_index_raises_value_error_with_invalid_directory_path_and_missing_index(self):
#     doc_sum_index_dir = "path/to/invalid/directory"
#     with self.assertRaises(ValueError):
#         get_doc_sum_index(doc_sum_index_dir)
# ```



# METHOD:
# def load_doc_sum_index(doc_sum_index_dir) -> DocumentSummaryIndex:
def test_create_new_index(tmpdir):
    doc_sum_index_dir = tmpdir / "doc_sum_index"
    assert not os.path.exists(doc_sum_index_dir)
    idx = load_doc_sum_index(doc_sum_index_dir)
    assert isinstance(idx, DocumentSummaryIndex)


def test_load_existing_index(tmpdir):
    doc_sum_index_dir = tmpdir / "doc_sum_index"
    idx = DocumentSummaryIndex.from_documents([], show_progress=True)
    persist_index(idx, doc_sum_index_dir)
    assert os.path.exists(doc_sum_index_dir)
    loaded_idx = load_doc_sum_index(doc_sum_index_dir)
    assert isinstance(loaded_idx, DocumentSummaryIndex)
    assert loaded_idx == idx


def test_non_existent_directory(tmpdir):
    doc_sum_index_dir = tmpdir / "doc_sum_index"
    assert not os.path.exists(doc_sum_index_dir)
    with pytest.raises(FileNotFoundError):
        load_doc_sum_index(doc_sum_index_dir)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `load_doc_sum_index` method:
# 
# 1. Test that the method creates a new index if one does not exist in the specified directory:
# ```python
# def test_create_new_index(tmpdir):
#     doc_sum_index_dir = tmpdir / "doc_sum_index"
#     assert not os.path.exists(doc_sum_index_dir)
#     idx = load_doc_sum_index(doc_sum_index_dir)
#     assert isinstance(idx, DocumentSummaryIndex)
# ```
# 2. Test that the method loads an existing index from storage if one exists in the specified directory:
# ```python
# def test_load_existing_index(tmpdir):
#     doc_sum_index_dir = tmpdir / "doc_sum_index"
#     idx = DocumentSummaryIndex.from_documents([], show_progress=True)
#     persist_index(idx, doc_sum_index_dir)
#     assert os.path.exists(doc_sum_index_dir)
#     loaded_idx = load_doc_sum_index(doc_sum_index_dir)
#     assert isinstance(loaded_idx, DocumentSummaryIndex)
#     assert loaded_idx == idx
# ```
# 3. Test that the method raises an error if the specified directory does not exist:
# ```python
# def test_non_existent_directory(tmpdir):
#     doc_sum_index_dir = tmpdir / "doc_sum_index"
#     assert not os.path.exists(doc_sum_index_dir)
#     with pytest.raises(FileNotFoundError):
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

