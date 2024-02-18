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
from lib.hybrid_query import (character_sum_cutoff, get_hybrid_query_engine, wrap_in_sub_question_engine, __init__, _retrieve, __init__, _postprocess_nodes, __init__, _postprocess_nodes, __init__, _postprocess_nodes, __init__, _postprocess_nodes, retrieve_parallel, retrieve_in_parallel)

# METHOD:
# def character_sum_cutoff(nodes: List[NodeWithScore], top_n: int) -> List[NodeWithScore]:
def test_character_sum_cutoff_empty_list():
    nodes = []
    top_n = 10
    assert character_sum_cutoff(nodes, top_n) == []

def test_character_sum_cutoff_single_node():
    nodes = [NodeWithScore("test", 1)]
    top_n = 4
    assert character_sum_cutoff(nodes, top_n) == [NodeWithScore("test", 1)]

def test_character_sum_cutoff_multiple_nodes():
    nodes = [NodeWithScore("test1", 1), NodeWithScore("test2", 2), NodeWithScore("test3", 3)]
    top_n = 6
    assert character_sum_cutoff(nodes, top_n) == [NodeWithScore("test1", 1), NodeWithScore("test2", 2), NodeWithScore("test3", 3)]

def test_character_sum_cutoff_with_cutoff():
    nodes = [NodeWithScore("test1", 1), NodeWithScore("test2", 2), NodeWithScore("test3", 3)]
    top_n = 5
    assert character_sum_cutoff(nodes, top_n) == [NodeWithScore("test1", 1), NodeWithScore("test2", 2)]


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `character_sum_cutoff` method:
# ```python
# def test_character_sum_cutoff_empty_list():
#     nodes = []
#     top_n = 10
#     assert character_sum_cutoff(nodes, top_n) == []
# 
# def test_character_sum_cutoff_single_node():
#     nodes = [NodeWithScore("test", 1)]
#     top_n = 4
#     assert character_sum_cutoff(nodes, top_n) == [NodeWithScore("test", 1)]
# 
# def test_character_sum_cutoff_multiple_nodes():
#     nodes = [NodeWithScore("test1", 1), NodeWithScore("test2", 2), NodeWithScore("test3", 3)]
#     top_n = 6
#     assert character_sum_cutoff(nodes, top_n) == [NodeWithScore("test1", 1), NodeWithScore("test2", 2), NodeWithScore("test3", 3)]
# 
# def test_character_sum_cutoff_with_cutoff():
#     nodes = [NodeWithScore("test1", 1), NodeWithScore("test2", 2), NodeWithScore("test3", 3)]
#     top_n = 5
#     assert character_sum_cutoff(nodes, top_n) == [NodeWithScore("test1", 1), NodeWithScore("test2", 2)]
# ```



# METHOD:
# def get_hybrid_query_engine(query_engine_options) -> RetrieverQueryEngine:
def test_get_hybrid_query_engine():
    query_engine_options = {
        "variant": "bm25",
        "reranker_model": "none"
    }
    engine = get_hybrid_query_engine(query_engine_options)
    assert isinstance(engine, RetrieverQueryEngine)


def test_get_cached_hybrid_query_engine():
    query_engine_options = {
        "variant": "bm25",
        "reranker_model": "none"
    }
    engine1 = get_hybrid_query_engine(query_engine_options)
    engine2 = get_hybrid_query_engine(query_engine_options)
    assert engine1 is engine2


def test_get_new_hybrid_query_engine():
    query_engine_options = {
        "variant": "bm25",
        "reranker_model": "none"
    }
    engine1 = get_hybrid_query_engine(query_engine_options)
    query_engine_options["variant"] = "kggraph"
    engine2 = get_hybrid_query_engine(query_engine_options)
    assert engine1 is not engine2


def test_get_new_hybrid_query_engine_with_different_retrievers():
    query_engine_options = {
        "variant": "bm25",
        "reranker_model": "none"
    }
    engine1 = get_hybrid_query_engine(query_engine_options)
    query_engine_options["retriever"] = "vector"
    engine2 = get_hybrid_query_engine(query_engine_options)
    assert engine1 is not engine2


def test_get_new_hybrid_query_engine_with_different_post_processors():
    query_engine_options = {
        "variant": "bm25",
        "reranker_model": "none"
    }
    engine1 = get_hybrid_query_engine(query_engine_options)
    query_engine_options["post_processors"] = [PrintingPostProcessor("no reranking")]
    engine2 = get_hybrid_query_engine(query_engine_options)
    assert engine1 is not engine2


# FULL-RESPONSE:
# # assistant: 
# This is a complex method that creates a hybrid query engine based on the provided options. The method first checks if the engine has already been created and stored in the `cached_hybrid_retriever_engine` dictionary. If it has, it returns the cached engine. Otherwise, it creates a new engine by combining different retrievers and post-processors based on the provided options.
# 
# Here are some test cases that you can use to ensure that this method works correctly:
# 
# 1. Test that the method returns a `RetrieverQueryEngine` object when given valid input:
# ```python
# def test_get_hybrid_query_engine():
#     query_engine_options = {
#         "variant": "bm25",
#         "reranker_model": "none"
#     }
#     engine = get_hybrid_query_engine(query_engine_options)
#     assert isinstance(engine, RetrieverQueryEngine)
# ```
# 2. Test that the method returns a cached engine when given the same options as before:
# ```python
# def test_get_cached_hybrid_query_engine():
#     query_engine_options = {
#         "variant": "bm25",
#         "reranker_model": "none"
#     }
#     engine1 = get_hybrid_query_engine(query_engine_options)
#     engine2 = get_hybrid_query_engine(query_engine_options)
#     assert engine1 is engine2
# ```
# 3. Test that the method creates a new engine when given different options:
# ```python
# def test_get_new_hybrid_query_engine():
#     query_engine_options = {
#         "variant": "bm25",
#         "reranker_model": "none"
#     }
#     engine1 = get_hybrid_query_engine(query_engine_options)
#     query_engine_options["variant"] = "kggraph"
#     engine2 = get_hybrid_query_engine(query_engine_options)
#     assert engine1 is not engine2
# ```
# 4. Test that the method creates a new engine when given different options for retrievers and post-processors:
# ```python
# def test_get_new_hybrid_query_engine_with_different_retrievers():
#     query_engine_options = {
#         "variant": "bm25",
#         "reranker_model": "none"
#     }
#     engine1 = get_hybrid_query_engine(query_engine_options)
#     query_engine_options["retriever"] = "vector"
#     engine2 = get_hybrid_query_engine(query_engine_options)
#     assert engine1 is not engine2
# ```
# 5. Test that the method creates a new engine when given different options for post-processors:
# ```python
# def test_get_new_hybrid_query_engine_with_different_post_processors():
#     query_engine_options = {
#         "variant": "bm25",
#         "reranker_model": "none"
#     }
#     engine1 = get_hybrid_query_engine(query_engine_options)
#     query_engine_options["post_processors"] = [PrintingPostProcessor("no reranking")]
#     engine2 = get_hybrid_query_engine(query_engine_options)
#     assert engine1 is not engine2
# ```



# METHOD:
# def wrap_in_sub_question_engine(query_engine: RetrieverQueryEngine) -> RetrieverQueryEngine:
def test_returned_query_engine_is_instance_of_sub_question_engine():
    query_engine = RetrieverQueryEngine()
    wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
    assert isinstance(wrapped_query_engine, SubQuestionQueryEngine)


def test_use_async_is_set_to_true():
    query_engine = RetrieverQueryEngine()
    wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
    assert wrapped_query_engine.use_async is True


def test_query_engine_tools_is_set_correctly():
    query_engine = RetrieverQueryEngine()
    wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
    assert len(wrapped_query_engine.query_engine_tools) == 1
    assert isinstance(wrapped_query_engine.query_engine_tools[0], QueryEngineTool)
    assert wrapped_query_engine.query_engine_tools[0].metadata.name == "hybrid_query_engine"


def test_question_gen_is_set_correctly():
    query_engine = RetrieverQueryEngine()
    wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
    assert isinstance(wrapped_query_engine.question_gen, GuidanceQuestionGenerator)
    assert wrapped_query_engine.question_gen.guidance_llm == OpenAI(model=constants.guidance_gpt_version)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `wrap_in_sub_question_engine` method:
# 
# 1. Test that the returned query engine is an instance of `SubQuestionQueryEngine`:
# ```python
# def test_returned_query_engine_is_instance_of_sub_question_engine():
#     query_engine = RetrieverQueryEngine()
#     wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
#     assert isinstance(wrapped_query_engine, SubQuestionQueryEngine)
# ```
# 2. Test that the `use_async` parameter is set to `True`:
# ```python
# def test_use_async_is_set_to_true():
#     query_engine = RetrieverQueryEngine()
#     wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
#     assert wrapped_query_engine.use_async is True
# ```
# 3. Test that the `query_engine_tools` parameter is set correctly:
# ```python
# def test_query_engine_tools_is_set_correctly():
#     query_engine = RetrieverQueryEngine()
#     wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
#     assert len(wrapped_query_engine.query_engine_tools) == 1
#     assert isinstance(wrapped_query_engine.query_engine_tools[0], QueryEngineTool)
#     assert wrapped_query_engine.query_engine_tools[0].metadata.name == "hybrid_query_engine"
# ```
# 4. Test that the `question_gen` parameter is set correctly:
# ```python
# def test_question_gen_is_set_correctly():
#     query_engine = RetrieverQueryEngine()
#     wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
#     assert isinstance(wrapped_query_engine.question_gen, GuidanceQuestionGenerator)
#     assert wrapped_query_engine.question_gen.guidance_llm == OpenAI(model=constants.guidance_gpt_version)
# ```

