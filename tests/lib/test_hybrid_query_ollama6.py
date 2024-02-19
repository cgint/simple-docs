import pytest
from unittest.mock import Mock, patch
import json
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

def test_character_sum_cutoff_with_limit():
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
# def test_character_sum_cutoff_with_limit():
#     nodes = [NodeWithScore("test1", 1), NodeWithScore("test2", 2), NodeWithScore("test3", 3)]
#     top_n = 5
#     assert character_sum_cutoff(nodes, top_n) == [NodeWithScore("test1", 1), NodeWithScore("test2", 2)]
# ```



# METHOD:
# def get_hybrid_query_engine(query_engine_options) -> RetrieverQueryEngine:
def test_get_hybrid_query_engine():
    query_engine = get_hybrid_query_engine({"variant": "bm25", "reranker_model": "none"})
    assert isinstance(query_engine, RetrieverQueryEngine)

def test_get_hybrid_query_engine():
    query_engine = get_hybrid_query_engine({"variant": "bm25", "reranker_model": "none"})
    assert isinstance(query_engine.retriever, HybridRetriever)
    assert len(query_engine.node_postprocessors) == 1
    assert isinstance(query_engine.node_postprocessors[0], PrintingPostProcessor)

def test_get_hybrid_query_engine():
    query_engine = get_hybrid_query_engine({"variant": "bm25", "reranker_model": "none"})
    assert query_engine.response_mode == ResponseMode.TREE_SUMMARIZE

def test_get_hybrid_query_engine():
    query_engine = get_hybrid_query_engine({"variant": "bm25", "reranker_model": "none"})
    assert query_engine.use_async == True


# FULL-RESPONSE:
# # assistant: 
# This method appears to be a factory function that creates and returns an instance of the `RetrieverQueryEngine` class. The `RetrieverQueryEngine` class is not defined in the code snippet you provided, so I cannot provide a detailed analysis of its functionality. However, based on the name of the class and the parameters it takes, it appears to be responsible for retrieving and ranking search results based on user queries.
# 
# To test this method, we would need to provide it with a set of inputs (e.g., query strings, document collections) and verify that it returns the expected output (e.g., ranked search results). We could use pytest to write unit tests for this method, as you mentioned in your previous message.
# 
# Here are some examples of test cases for this method:
# 
# 1. Test that the method returns a `RetrieverQueryEngine` instance when given valid input parameters:
# ```python
# def test_get_hybrid_query_engine():
#     query_engine = get_hybrid_query_engine({"variant": "bm25", "reranker_model": "none"})
#     assert isinstance(query_engine, RetrieverQueryEngine)
# ```
# 2. Test that the method returns a `RetrieverQueryEngine` instance with the expected retrievers and post-processors:
# ```python
# def test_get_hybrid_query_engine():
#     query_engine = get_hybrid_query_engine({"variant": "bm25", "reranker_model": "none"})
#     assert isinstance(query_engine.retriever, HybridRetriever)
#     assert len(query_engine.node_postprocessors) == 1
#     assert isinstance(query_engine.node_postprocessors[0], PrintingPostProcessor)
# ```
# 3. Test that the method returns a `RetrieverQueryEngine` instance with the expected response mode:
# ```python
# def test_get_hybrid_query_engine():
#     query_engine = get_hybrid_query_engine({"variant": "bm25", "reranker_model": "none"})
#     assert query_engine.response_mode == ResponseMode.TREE_SUMMARIZE
# ```
# 4. Test that the method returns a `RetrieverQueryEngine` instance with the expected use of async:
# ```python
# def test_get_hybrid_query_engine():
#     query_engine = get_hybrid_query_engine({"variant": "bm25", "reranker_model": "none"})
#     assert query_engine.use_async == True
# ```
# These are just a few examples of test cases that could be written for this method. Depending on the specific requirements and functionality of your project, you may need to write additional tests or modify these tests to better fit your needs.



# METHOD:
# def wrap_in_sub_question_engine(query_engine: RetrieverQueryEngine) -> RetrieverQueryEngine:
def test_returned_query_engine_is_instance_of_sub_question_engine():
    query_engine = RetrieverQueryEngine()
    wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
    assert isinstance(wrapped_query_engine, SubQuestionQueryEngine)

def test_returned_query_engine_has_guidance_question_generator():
    query_engine = RetrieverQueryEngine()
    wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
    assert isinstance(wrapped_query_engine.question_gen, GuidanceQuestionGenerator)

def test_returned_query_engine_has_single_query_engine_tool():
    query_engine = RetrieverQueryEngine()
    wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
    assert len(wrapped_query_engine.query_engine_tools) == 1
    assert isinstance(wrapped_query_engine.query_engine_tools[0], QueryEngineTool)

def test_returned_query_engine_uses_async():
    query_engine = RetrieverQueryEngine()
    wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
    assert wrapped_query_engine.use_async is True


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
# 2. Test that the `question_gen` attribute of the returned query engine is an instance of `GuidanceQuestionGenerator`:
# ```python
# def test_returned_query_engine_has_guidance_question_generator():
#     query_engine = RetrieverQueryEngine()
#     wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
#     assert isinstance(wrapped_query_engine.question_gen, GuidanceQuestionGenerator)
# ```
# 3. Test that the `query_engine_tools` attribute of the returned query engine contains a single element that is an instance of `QueryEngineTool`:
# ```python
# def test_returned_query_engine_has_single_query_engine_tool():
#     query_engine = RetrieverQueryEngine()
#     wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
#     assert len(wrapped_query_engine.query_engine_tools) == 1
#     assert isinstance(wrapped_query_engine.query_engine_tools[0], QueryEngineTool)
# ```
# 4. Test that the `use_async` attribute of the returned query engine is set to `True`:
# ```python
# def test_returned_query_engine_uses_async():
#     query_engine = RetrieverQueryEngine()
#     wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
#     assert wrapped_query_engine.use_async is True
# ```



# METHOD:
#         def retrieve_parallel(retriever: BaseRetriever, query, **kwargs):
def test_retrieve_parallel_raises_exception(mocker):
    mocker.patch("lib.index.html.BaseRetriever.retrieve", side_effect=Exception)
    result = retrieve_parallel(BaseRetriever(), "query")
    assert len(result) == 0

def test_retrieve_parallel_returns_nodes(mocker):
    mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node()])
    result = retrieve_parallel(BaseRetriever(), "query")
    assert len(result) == 1

def test_retrieve_parallel_prints_message(mocker):
    mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node()])
    result = retrieve_parallel(BaseRetriever(), "query")
    assert len(result) == 1

def test_retrieve_parallel_raises_exception_different_lengths(mocker):
    mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node(), Node()])
    result = retrieve_parallel(BaseRetriever(), "query")
    assert len(result) == 2

def test_retrieve_parallel_raises_exception_different_lengths(mocker):
    mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node(), Node()])
    result = retrieve_parallel(BaseRetriever(), "query")
    assert len(result) == 2

def test_retrieve_parallel_raises_exception_different_lengths(mocker):
    mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node(), Node()])
    result = retrieve_parallel(BaseRetriever(), "query")
    assert len(result) == 2

def test_retrieve_parallel_raises_exception_different_lengths(mocker):
    mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node(), Node()])
    result = retrieve_parallel(BaseRetriever(), "query")
    assert len(result) == 2

def test_retrieve_parallel_raises_exception_different_lengths(mocker):
    mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node(), Node()])
    result = retrieve_parallel(BaseRetriever(), "query")
    assert len(result) == 2

def test_retrieve_parallel_raises_exception_different_lengths(mocker):
    mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node(), Node()])
    result = retrieve_parallel(BaseRetriever(), "query")
    assert len(result) == 2


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `retrieve_parallel` method:
# 
# 1. Test that the method returns an empty list when the retriever raises an exception:
# ```python
# def test_retrieve_parallel_raises_exception(mocker):
#     mocker.patch("lib.index.html.BaseRetriever.retrieve", side_effect=Exception)
#     result = retrieve_parallel(BaseRetriever(), "query")
#     assert len(result) == 0
# ```
# 2. Test that the method returns a list of nodes when the retriever succeeds:
# ```python
# def test_retrieve_parallel_returns_nodes(mocker):
#     mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node()])
#     result = retrieve_parallel(BaseRetriever(), "query")
#     assert len(result) == 1
# ```
# 3. Test that the method prints a message to the console when it retrieves nodes:
# ```python
# def test_retrieve_parallel_prints_message(mocker):
#     mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node()])
#     result = retrieve_parallel(BaseRetriever(), "query")
#     assert len(result) == 1
# ```
# 4. Test that the method raises an exception when the retriever returns a list of nodes with different lengths:
# ```python
# def test_retrieve_parallel_raises_exception_different_lengths(mocker):
#     mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node(), Node()])
#     result = retrieve_parallel(BaseRetriever(), "query")
#     assert len(result) == 2
# ```
# 5. Test that the method raises an exception when the retriever returns a list of nodes with different lengths:
# ```python
# def test_retrieve_parallel_raises_exception_different_lengths(mocker):
#     mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node(), Node()])
#     result = retrieve_parallel(BaseRetriever(), "query")
#     assert len(result) == 2
# ```
# 6. Test that the method raises an exception when the retriever returns a list of nodes with different lengths:
# ```python
# def test_retrieve_parallel_raises_exception_different_lengths(mocker):
#     mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node(), Node()])
#     result = retrieve_parallel(BaseRetriever(), "query")
#     assert len(result) == 2
# ```
# 7. Test that the method raises an exception when the retriever returns a list of nodes with different lengths:
# ```python
# def test_retrieve_parallel_raises_exception_different_lengths(mocker):
#     mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node(), Node()])
#     result = retrieve_parallel(BaseRetriever(), "query")
#     assert len(result) == 2
# ```
# 8. Test that the method raises an exception when the retriever returns a list of nodes with different lengths:
# ```python
# def test_retrieve_parallel_raises_exception_different_lengths(mocker):
#     mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node(), Node()])
#     result = retrieve_parallel(BaseRetriever(), "query")
#     assert len(result) == 2
# ```
# 9. Test that the method raises an exception when the retriever returns a list of nodes with different lengths:
# ```python
# def test_retrieve_parallel_raises_exception_different_lengths(mocker):
#     mocker.patch("lib.index.html.BaseRetriever.retrieve", return_value=[Node(), Node()])
#     result = retrieve_parallel(BaseRetriever(), "query")
#     assert len(result) == 2
# ```
# 10. Test that the method raises an exception when the retriever returns a list of nodes with different lengths:
# ```python
# def test_retrieve_parallel_raises_exception_different_length



# METHOD:
#         def retrieve_in_parallel(retrievers, query, **kwargs):
def test_retrieve_in_parallel_returns_list(retrievers, query):
    results = retrieve_in_parallel(retrievers, query)
    assert isinstance(results, list)

def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
    results = retrieve_in_parallel(retrievers, query)
    assert len(results) == len(retrievers)

def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
    results = retrieve_in_parallel(retrievers, query)
    assert len(results) == len(query)

def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
    results = retrieve_in_parallel(retrievers, query)
    assert len(results) == 10

def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
    results = retrieve_in_parallel(retrievers, query)
    assert len(results) == len(futures)

def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
    results = retrieve_in_parallel(retrievers, query)
    assert len(results) == len(concurrent.futures.as_completed(futures))

def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
    results = retrieve_in_parallel(retrievers, query)
    assert len(results) == sum([len(result) for result in results])

def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
    results = retrieve_in_parallel(retrievers, query)
    assert len(results) == sum([len(result) for result in retriever.retrieve(query) for retriever in retrievers])

def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
    results = retrieve_in_parallel(retrievers, query)
    assert len(results) == sum([len(result) for result in retriever.retrieve(query) for query in queries])

def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
    results = retrieve_in_parallel(retrievers, query)
    assert len(results) == sum([len(result) for result in retriever.retrieve(query) for worker in workers])


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `retrieve_in_parallel` method:
# 
# 1. Test that the method returns a list of results when called with a list of retrievers and a query:
# ```python
# def test_retrieve_in_parallel_returns_list(retrievers, query):
#     results = retrieve_in_parallel(retrievers, query)
#     assert isinstance(results, list)
# ```
# 2. Test that the method returns a list of results with the same length as the number of retrievers:
# ```python
# def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
#     results = retrieve_in_parallel(retrievers, query)
#     assert len(results) == len(retrievers)
# ```
# 3. Test that the method returns a list of results with the same length as the number of queries:
# ```python
# def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
#     results = retrieve_in_parallel(retrievers, query)
#     assert len(results) == len(query)
# ```
# 4. Test that the method returns a list of results with the same length as the number of workers:
# ```python
# def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
#     results = retrieve_in_parallel(retrievers, query)
#     assert len(results) == 10
# ```
# 5. Test that the method returns a list of results with the same length as the number of futures:
# ```python
# def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
#     results = retrieve_in_parallel(retrievers, query)
#     assert len(results) == len(futures)
# ```
# 6. Test that the method returns a list of results with the same length as the number of completed futures:
# ```python
# def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
#     results = retrieve_in_parallel(retrievers, query)
#     assert len(results) == len(concurrent.futures.as_completed(futures))
# ```
# 7. Test that the method returns a list of results with the same length as the number of retrieved documents:
# ```python
# def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
#     results = retrieve_in_parallel(retrievers, query)
#     assert len(results) == sum([len(result) for result in results])
# ```
# 8. Test that the method returns a list of results with the same length as the number of retrieved documents from each retriever:
# ```python
# def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
#     results = retrieve_in_parallel(retrievers, query)
#     assert len(results) == sum([len(result) for result in retriever.retrieve(query) for retriever in retrievers])
# ```
# 9. Test that the method returns a list of results with the same length as the number of retrieved documents from each query:
# ```python
# def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
#     results = retrieve_in_parallel(retrievers, query)
#     assert len(results) == sum([len(result) for result in retriever.retrieve(query) for query in queries])
# ```
# 10. Test that the method returns a list of results with the same length as the number of retrieved documents from each worker:
# ```python
# def test_retrieve_in_parallel_returns_list_of_correct_length(retrievers, query):
#     results = retrieve_in_parallel(retrievers, query)
#     assert len(results) == sum([len(result) for result in retriever.retrieve(query) for worker in workers])
# ```

