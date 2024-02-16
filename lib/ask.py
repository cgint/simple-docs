from lib.index.doc_sum_index import get_doc_sum_index_query_engine
from lib.vector_chroma import get_vector_query_engine, get_vector_ng_query_engine
from lib.index.terms.kg_num_term_neo4j import get_graph_query_engine

def ask_for_vector_response(collection, question) -> str:
    return get_vector_query_engine(collection).query(question).response

def ask_for_vector_ng_response(vector_storage_dir, question) -> str:
    return get_vector_ng_query_engine(vector_storage_dir).query(question).response

def ask_for_graph_response(vector_storage_dir, question) -> str:
    return get_graph_query_engine(vector_storage_dir).query(question).response

def ask_for_doc_sum_index_response(storage_dir, question) -> str:
    return get_doc_sum_index_query_engine(storage_dir).query(question).response