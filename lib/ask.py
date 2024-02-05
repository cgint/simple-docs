from lib.hybrid_query import get_hybrid_query_engine
from lib.index.doc_sum_index import get_doc_sum_index_query_engine
from lib.vector_chroma import get_vector_query_engine, get_vector_ng_query_engine
from lib.index.terms.kg_num_term_neo4j import get_graph_query_engine
from llama_index.core.response.schema import Response

def ask_for_vector_response(service_context, vector_storage_dir, question) -> str:
    return get_vector_query_engine(service_context, vector_storage_dir).query(question).response

def ask_for_vector_ng_response(service_context, vector_storage_dir, question) -> str:
    return get_vector_ng_query_engine(service_context, vector_storage_dir).query(question).response

def ask_for_graph_response(service_context, vector_storage_dir, question) -> str:
    return get_graph_query_engine(service_context, vector_storage_dir).query(question).response

def ask_for_doc_sum_index_response(service_context, storage_dir, question) -> str:
    return get_doc_sum_index_query_engine(service_context, storage_dir).query(question).response

def ask_hybrid_query_engine(service_context, storage_dir, question) -> Response:
    return get_hybrid_query_engine(service_context, storage_dir).query(question)