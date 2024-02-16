from typing import List
from llama_index.core import Document, KnowledgeGraphIndex, StorageContext
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core import load_index_from_storage
import os

def load_kg_graph_index_storage_context(kg_graph_storage_dir: str) -> StorageContext:
    return StorageContext.from_defaults(persist_dir=kg_graph_storage_dir)

def persist_kg_graph_index(idx: KnowledgeGraphIndex, kg_graph_storage_dir: str):
    doc_count = len(idx.docstore.docs)
    print(f"Persisting {doc_count} docs for kg_graph to {kg_graph_storage_dir} ...")
    idx.storage_context.persist(persist_dir=kg_graph_storage_dir)

def delete_kg_graph_index(kg_graph_storage_dir: str):
    print(f"Deleting kg_graph at {kg_graph_storage_dir} ...")
    if os.path.exists(kg_graph_storage_dir):
        import shutil
        shutil.rmtree(kg_graph_storage_dir)

def load_kg_graph_index(kg_graph_storage_dir: str) -> KnowledgeGraphIndex:
    if not os.path.exists(kg_graph_storage_dir):
        print(f"About to initialize an empty kg-graph ...")
        kg_graph = KnowledgeGraphIndex.from_documents(
            []
        )
        persist_kg_graph_index(kg_graph, kg_graph_storage_dir)
    return load_index_from_storage(
        storage_context=load_kg_graph_index_storage_context(kg_graph_storage_dir)
    )

def get_kg_graph_index(graph_storage_dir: str) -> KnowledgeGraphIndex:
    return load_kg_graph_index(graph_storage_dir)

def operate_on_kg_graph_index(kg_graph_index_dir: str, operation=lambda: None) -> KnowledgeGraphIndex:
    import atexit
    idx = get_kg_graph_index(kg_graph_index_dir)
    atexist_reg_callable = atexit.register(persist_kg_graph_index, idx, kg_graph_index_dir)
    try:
        operation(idx)
    finally:
        persist_kg_graph_index(idx, kg_graph_index_dir)
        atexit.unregister(atexist_reg_callable)
    return idx


def add_to_or_update_in_kg_graph(graph_storage_dir: str, documents: List[Document]):
    operate_on_kg_graph_index(
        graph_storage_dir,
        lambda graph_index: graph_index.refresh_ref_docs(documents)
    )

def get_kg_graph_query_engine(graph_storage_dir: str) -> BaseQueryEngine:
    return load_kg_graph_index(graph_storage_dir).as_query_engine()
