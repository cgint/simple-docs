from typing import List
from llama_index import Document, KnowledgeGraphIndex, StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from neo4j import GraphDatabase
from llama_index.core.base_query_engine import BaseQueryEngine
import os

host_ip = os.getenv("HOST_IP", "host.docker.internal")
user = "neo4j"
pwd = ""
uri = f"neo4j://{host_ip}:8687"
g_db = "neo4j"

def kg_neo4j_delete_all_nodes():
    with GraphDatabase.driver(uri=uri, auth=(user, pwd)).session(database=g_db) as session:
        session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))

def load_graph_index_neo4j_storage_context(collection: str) -> (Neo4jGraphStore, StorageContext):
    graph_store = Neo4jGraphStore(user, pwd, uri, collection)
    return graph_store, StorageContext.from_defaults(graph_store=graph_store)

def load_graph_index(service_context, graph_storage_dir: str) -> KnowledgeGraphIndex:
    collection = g_db # graph_storage_dir.replace("/", "_").replace("_", "")
    _, storage_context = load_graph_index_neo4j_storage_context(collection)
    return KnowledgeGraphIndex.from_documents([],
        service_context=service_context,
        storage_context=storage_context,
        index_id=collection,
        kg_triplet_extract_fn=kg_triplet_extract_fn_term_noop
    )

def kg_triplet_extract_fn_term_noop(_: str):
    """Do not extract any triplets at this stage. Will be done after the document is indexed."""
    return []

def get_graph_index(service_context, graph_storage_dir: str) -> KnowledgeGraphIndex:
    return load_graph_index(service_context, graph_storage_dir)

def operate_on_graph_index(service_context, graph_storage_dir: str, operation=lambda: None):
    operation(get_graph_index(service_context, graph_storage_dir))


def add_to_or_update_in_graph(service_context, graph_storage_dir: str, documents: List[Document]):
    operate_on_graph_index(
        service_context,
        graph_storage_dir,
        lambda graph_index: graph_index.refresh_ref_docs(documents)
    )

def get_graph_query_engine(service_context, graph_storage_dir: str) -> BaseQueryEngine:
    return load_graph_index(service_context, graph_storage_dir).as_query_engine()
