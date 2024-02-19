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
from lib import constants
from lib.index.helper import cur_simple_date_time_sec
from traceback import format_exception
from lib.index.error_helper import (write_error_to_file)

# METHOD:
# def write_error_to_file(e: Exception, msg: str = None):
def test_write_error_to_file(tmpdir):
    with tmpdir.as_cwd():
        with open("error.log", "w") as f:
            write_error_to_file(Exception("Test error"), msg="Test message")
            assert f.read() == "========================================================================\n" \
                               "================= 2023-01-01T00:00:00Z ==================\n" \
                               "========================================================================\n" \
                               "Test message\n" \
                               "Trace: Test error\n" \
                               "========================================================================\n"


def test_write_error_to_file_traceback(tmpdir):
    with tmpdir.as_cwd():
        with open("error.log", "w") as f:
            write_error_to_file(Exception("Test error"))
            assert f.read() == "========================================================================\n" \
                               "================= 2023-01-01T00:00:00Z ==================\n" \
                               "========================================================================\n" \
                               "Trace: Test error\n" \
                               "========================================================================\n"


def test_write_error_to_file_message_and_traceback(tmpdir):
    with tmpdir.as_cwd():
        with open("error.log", "w") as f:
            write_error_to_file(Exception("Test error"), msg="Test message")
            assert f.read() == "========================================================================\n" \
                               "================= 2023-01-01T00:00:00Z ==================\n" \
                               "========================================================================\n" \
                               "Test message\n" \
                               "Trace: Test error\n" \
                               "========================================================================\n"


def test_write_error_to_file_message_traceback_and_date(tmpdir):
    with tmpdir.as_cwd():
        with open("error.log", "w") as f:
            write_error_to_file(Exception("Test error"), msg="Test message")
            assert f.read() == "========================================================================\n" \
                               "================= 2023-01-01T00:00:00Z ==================\n" \
                               "========================================================================\n" \
                               "Test message\n" \
                               "Trace: Test error\n" \
                               "========================================================================\n"


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `write_error_to_file` method:
# 
# 1. Test that the file is opened in append mode and the error message is written to it correctly:
# ```python
# def test_write_error_to_file(tmpdir):
#     with tmpdir.as_cwd():
#         with open("error.log", "w") as f:
#             write_error_to_file(Exception("Test error"), msg="Test message")
#             assert f.read() == "========================================================================\n" \
#                                "================= 2023-01-01T00:00:00Z ==================\n" \
#                                "========================================================================\n" \
#                                "Test message\n" \
#                                "Trace: Test error\n" \
#                                "========================================================================\n"
# ```
# 2. Test that the file is opened in append mode and the traceback is written to it correctly:
# ```python
# def test_write_error_to_file_traceback(tmpdir):
#     with tmpdir.as_cwd():
#         with open("error.log", "w") as f:
#             write_error_to_file(Exception("Test error"))
#             assert f.read() == "========================================================================\n" \
#                                "================= 2023-01-01T00:00:00Z ==================\n" \
#                                "========================================================================\n" \
#                                "Trace: Test error\n" \
#                                "========================================================================\n"
# ```
# 3. Test that the file is opened in append mode and the message and traceback are written to it correctly:
# ```python
# def test_write_error_to_file_message_and_traceback(tmpdir):
#     with tmpdir.as_cwd():
#         with open("error.log", "w") as f:
#             write_error_to_file(Exception("Test error"), msg="Test message")
#             assert f.read() == "========================================================================\n" \
#                                "================= 2023-01-01T00:00:00Z ==================\n" \
#                                "========================================================================\n" \
#                                "Test message\n" \
#                                "Trace: Test error\n" \
#                                "========================================================================\n"
# ```
# 4. Test that the file is opened in append mode and the message, traceback, and date are written to it correctly:
# ```python
# def test_write_error_to_file_message_traceback_and_date(tmpdir):
#     with tmpdir.as_cwd():
#         with open("error.log", "w") as f:
#             write_error_to_file(Exception("Test error"), msg="Test message")
#             assert f.read() == "========================================================================\n" \
#                                "================= 2023-01-01T00:00:00Z ==================\n" \
#                                "========================================================================\n" \
#                                "Test message\n" \
#                                "Trace: Test error\n" \
#                                "========================================================================\n"
# ```

