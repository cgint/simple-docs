import os
from typing import List
from llama_index import Document, get_response_synthesizer
from llama_index import VectorStoreIndex
from llama_index import StorageContext
from llama_index import load_index_from_storage
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

def load_vector_index_storage_context(vector_storage_dir: str) -> StorageContext:
    return StorageContext.from_defaults(persist_dir=vector_storage_dir)

def load_vector_index(service_context, vector_storage_dir: str) -> VectorStoreIndex:
    return load_index_from_storage(
        service_context=service_context,
        storage_context=load_vector_index_storage_context(vector_storage_dir)
    )

def persist_vector_index(vector_index: VectorStoreIndex, vector_storage_dir: str):
    print(f"Storing vector-index to {vector_storage_dir} ...")
    vector_index.storage_context.persist(persist_dir=vector_storage_dir)

def get_vector_index(service_context, vector_storage_dir: str) -> VectorStoreIndex:
    if not os.path.exists(vector_storage_dir):
        print(f"About to initialize an empty vector-index ...")
        vector_index = VectorStoreIndex.from_documents(
            [],
            service_context=service_context
        )
        persist_vector_index(vector_index, vector_storage_dir)
    return load_vector_index(service_context, vector_storage_dir)

def add_to_or_update_in_vector(service_context, vector_storage_dir: str, documents: List[Document]) -> VectorStoreIndex:
    return operate_on_vector_index(
        service_context,
        vector_storage_dir,
        lambda vector_index: vector_index.refresh_ref_docs(documents)
    )
    
def operate_on_vector_index(service_context, vector_storage_dir: str, operation=lambda: None) -> VectorStoreIndex:
    vector_index = get_vector_index(service_context, vector_storage_dir)
    operation(vector_index)
    persist_vector_index(vector_index, vector_storage_dir)
    return vector_index

def assert_exists_vector_index(vector_storage_dir: str):
    if not os.path.exists(vector_storage_dir):
        raise ValueError(f"Vector index not found at {vector_storage_dir}")

def get_vector_chat_engine(service_context, vector_storage_dir: str) -> BaseChatEngine:
    assert_exists_vector_index(vector_storage_dir)
    return load_vector_index(service_context, vector_storage_dir).as_chat_engine()

def get_vector_query_engine(service_context, vector_storage_dir: str) -> BaseQueryEngine:
    assert_exists_vector_index(vector_storage_dir)
    return load_vector_index(service_context, vector_storage_dir).as_query_engine()

def get_vector_ng_query_engine(service_context, vector_storage_dir: str) -> BaseQueryEngine:
    assert_exists_vector_index(vector_storage_dir)
    # configure retriever
    return RetrieverQueryEngine(
            retriever=VectorIndexRetriever(
            index=load_vector_index(service_context, vector_storage_dir),
            similarity_top_k=3,
        ),
            response_synthesizer=get_response_synthesizer(
            response_mode="tree_summarize",
        )
    )