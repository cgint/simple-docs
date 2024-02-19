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
from lib.callbacks.simple_dict_collector import (__init__, _write_llm_event, _append_to_store, start_trace, end_trace, on_event_start, on_event_end)

# METHOD:
#     def start_trace(self, trace_id: Optional[str] = None) -> None:
def test_start_trace_returns_none(self):
    self.assertIsNone(start_trace())


def test_start_trace_raises_type_error_with_invalid_trace_id(self):
    with self.assertRaises(TypeError):
        start_trace(trace_id=123)


def test_start_trace_raises_value_error_with_empty_string_trace_id(self):
    with self.assertRaises(ValueError):
        start_trace(trace_id="")


def test_start_trace_raises_value_error_with_whitespace_string_trace_id(self):
    with self.assertRaises(ValueError):
        start_trace(trace_id="   ")


def test_start_trace_raises_value_error_with_non_ascii_string_trace_id(self):
    with self.assertRaises(ValueError):
        start_trace(trace_id="😀")


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `start_trace` method:
# 
# 1. Test that the method returns `None`:
# ```python
# def test_start_trace_returns_none(self):
#     self.assertIsNone(start_trace())
# ```
# 2. Test that the method raises a `TypeError` if an invalid type is passed as the `trace_id` argument:
# ```python
# def test_start_trace_raises_type_error_with_invalid_trace_id(self):
#     with self.assertRaises(TypeError):
#         start_trace(trace_id=123)
# ```
# 3. Test that the method raises a `ValueError` if an empty string is passed as the `trace_id` argument:
# ```python
# def test_start_trace_raises_value_error_with_empty_string_trace_id(self):
#     with self.assertRaises(ValueError):
#         start_trace(trace_id="")
# ```
# 4. Test that the method raises a `ValueError` if a string containing only whitespace is passed as the `trace_id` argument:
# ```python
# def test_start_trace_raises_value_error_with_whitespace_string_trace_id(self):
#     with self.assertRaises(ValueError):
#         start_trace(trace_id="   ")
# ```
# 5. Test that the method raises a `ValueError` if a string containing non-ASCII characters is passed as the `trace_id` argument:
# ```python
# def test_start_trace_raises_value_error_with_non_ascii_string_trace_id(self):
#     with self.assertRaises(ValueError):
#         start_trace(trace_id="😀")
# ```



# METHOD:
#     def end_trace(
def test_end_trace_returns_none(self):
    self.assertIsNone(end_trace())


def test_end_trace_raises_type_error_if_trace_id_is_not_a_string(self):
    with self.assertRaises(TypeError):
        end_trace(trace_id=123)


def test_end_trace_raises_value_error_if_trace_map_is_not_a_dictionary(self):
    with self.assertRaises(ValueError):
        end_trace(trace_map=["test", "test2"])


def test_end_trace_raises_key_error_if_trace_id_is_not_in_trace_map(self):
    with self.assertRaises(KeyError):
        end_trace(trace_id="test", trace_map={"test2": ["test3"]})


def test_end_trace_returns_none_if_trace_id_is_in_trace_map(self):
    self.assertIsNone(end_trace(trace_id="test", trace_map={"test": ["test2"]}))


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `end_trace` method:
# 
# 1. Test that the method returns `None`:
# ```python
# def test_end_trace_returns_none(self):
#     self.assertIsNone(end_trace())
# ```
# 2. Test that the method raises a `TypeError` if `trace_id` is not a string:
# ```python
# def test_end_trace_raises_type_error_if_trace_id_is_not_a_string(self):
#     with self.assertRaises(TypeError):
#         end_trace(trace_id=123)
# ```
# 3. Test that the method raises a `ValueError` if `trace_map` is not a dictionary:
# ```python
# def test_end_trace_raises_value_error_if_trace_map_is_not_a_dictionary(self):
#     with self.assertRaises(ValueError):
#         end_trace(trace_map=["test", "test2"])
# ```
# 4. Test that the method raises a `KeyError` if `trace_id` is not in `trace_map`:
# ```python
# def test_end_trace_raises_key_error_if_trace_id_is_not_in_trace_map(self):
#     with self.assertRaises(KeyError):
#         end_trace(trace_id="test", trace_map={"test2": ["test3"]})
# ```
# 5. Test that the method returns `None` if `trace_id` is in `trace_map`:
# ```python
# def test_end_trace_returns_none_if_trace_id_is_in_trace_map(self):
#     self.assertIsNone(end_trace(trace_id="test", trace_map={"test": ["test2"]}))
# ```



# METHOD:
#     def on_event_start(
def test_on_event_start_returns_string(self):
    event_type = CBEventType.START
    payload = {"key": "value"}
    event_id = "test-event"
    parent_id = "test-parent"
    kwargs = {}
    result = on_event_start(event_type, payload, event_id, parent_id, **kwargs)
    self.assertIsInstance(result, str)


def test_on_event_start_returns_empty_string(self):
    result = on_event_start()
    self.assertEqual(result, "")


def test_on_event_start_raises_type_error(self):
    event_type = CBEventType.START
    payload = {"key": "value"}
    event_id = 12345
    parent_id = "test-parent"
    kwargs = {}
    with self.assertRaises(TypeError):
        on_event_start(event_type, payload, event_id, parent_id, **kwargs)


def test_on_event_start_raises_value_error(self):
    event_type = CBEventType.START
    payload = {"key": "value"}
    event_id = ""
    parent_id = "test-parent"
    kwargs = {}
    with self.assertRaises(ValueError):
        on_event_start(event_type, payload, event_id, parent_id, **kwargs)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `on_event_start` method:
# 
# 1. Test that the method returns a string when called with valid arguments:
# ```python
# def test_on_event_start_returns_string(self):
#     event_type = CBEventType.START
#     payload = {"key": "value"}
#     event_id = "test-event"
#     parent_id = "test-parent"
#     kwargs = {}
#     result = on_event_start(event_type, payload, event_id, parent_id, **kwargs)
#     self.assertIsInstance(result, str)
# ```
# 2. Test that the method returns an empty string when called with no arguments:
# ```python
# def test_on_event_start_returns_empty_string(self):
#     result = on_event_start()
#     self.assertEqual(result, "")
# ```
# 3. Test that the method raises a `TypeError` when called with invalid arguments:
# ```python
# def test_on_event_start_raises_type_error(self):
#     event_type = CBEventType.START
#     payload = {"key": "value"}
#     event_id = 12345
#     parent_id = "test-parent"
#     kwargs = {}
#     with self.assertRaises(TypeError):
#         on_event_start(event_type, payload, event_id, parent_id, **kwargs)
# ```
# 4. Test that the method raises a `ValueError` when called with invalid arguments:
# ```python
# def test_on_event_start_raises_value_error(self):
#     event_type = CBEventType.START
#     payload = {"key": "value"}
#     event_id = ""
#     parent_id = "test-parent"
#     kwargs = {}
#     with self.assertRaises(ValueError):
#         on_event_start(event_type, payload, event_id, parent_id, **kwargs)
# ```



# METHOD:
#     def on_event_end(
def test_on_event_end_noop(self):
    self._write_llm_event = Mock()
    on_event_end(CBEventType.OTHER, None)
    self._write_llm_event.assert_not_called()


def test_on_event_end_with_payload(self):
    self._write_llm_event = Mock()
    on_event_end(CBEventType.LLM, {"key": "value"})
    self._write_llm_event.assert_called_once_with({"key": "value"})


def test_on_event_end_no_payload(self):
    self._write_llm_event = Mock()
    on_event_end(CBEventType.LLM, None)
    self._write_llm_event.assert_not_called()


def test_on_event_end_noop_with_payload(self):
    self._write_llm_event = Mock()
    on_event_end(CBEventType.OTHER, {"key": "value"})
    self._write_llm_event.assert_not_called()


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `on_event_end` method:
# 
# 1. Test that the method does nothing if the event type is not LLM:
# ```python
# def test_on_event_end_noop(self):
#     self._write_llm_event = Mock()
#     on_event_end(CBEventType.OTHER, None)
#     self._write_llm_event.assert_not_called()
# ```
# 2. Test that the method calls `_write_llm_event` with the correct payload if the event type is LLM and a payload is provided:
# ```python
# def test_on_event_end_with_payload(self):
#     self._write_llm_event = Mock()
#     on_event_end(CBEventType.LLM, {"key": "value"})
#     self._write_llm_event.assert_called_once_with({"key": "value"})
# ```
# 3. Test that the method does nothing if the event type is LLM but no payload is provided:
# ```python
# def test_on_event_end_no_payload(self):
#     self._write_llm_event = Mock()
#     on_event_end(CBEventType.LLM, None)
#     self._write_llm_event.assert_not_called()
# ```
# 4. Test that the method does nothing if the event type is not LLM and a payload is provided:
# ```python
# def test_on_event_end_noop_with_payload(self):
#     self._write_llm_event = Mock()
#     on_event_end(CBEventType.OTHER, {"key": "value"})
#     self._write_llm_event.assert_not_called()
# ```

