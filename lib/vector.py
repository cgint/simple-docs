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

def load_vector_index_storage_context(vector_storage_dir: str) -> StorageContext:
    return StorageContext.from_defaults(persist_dir=vector_storage_dir)

def load_vector_index(vector_storage_dir: str) -> VectorStoreIndex:
    return load_index_from_storage(
        storage_context=load_vector_index_storage_context(vector_storage_dir)
    )

def delete_vector_index(vector_storage_dir: str):
    print(f"Deleting vector at {vector_storage_dir} ...")
    if os.path.exists(vector_storage_dir):
        import shutil
        shutil.rmtree(vector_storage_dir)

def persist_vector_index(vector_index: VectorStoreIndex, vector_storage_dir: str):
    print(f"Storing vector-index to {vector_storage_dir} ...")
    vector_index.storage_context.persist(persist_dir=vector_storage_dir)

def get_vector_index(vector_storage_dir: str) -> VectorStoreIndex:
    if not os.path.exists(vector_storage_dir):
        print("About to initialize an empty vector-index ...")
        vector_index = VectorStoreIndex.from_documents(
            []
        )
        persist_vector_index(vector_index, vector_storage_dir)
    return load_vector_index(vector_storage_dir)

def add_to_or_update_in_vector(vector_storage_dir: str, documents: List[Document]) -> VectorStoreIndex:
    return operate_on_vector_index(
        vector_storage_dir,
        lambda vector_index: vector_index.refresh_ref_docs(documents)
    )
    
def operate_on_vector_index(vector_storage_dir: str, operation=lambda: None) -> VectorStoreIndex:
    vector_index = get_vector_index(vector_storage_dir)
    operation(vector_index)
    persist_vector_index(vector_index, vector_storage_dir)
    return vector_index

def assert_exists_vector_index(vector_storage_dir: str):
    if not os.path.exists(vector_storage_dir):
        raise ValueError(f"Vector index not found at {vector_storage_dir}")

def get_vector_chat_engine(vector_storage_dir: str) -> BaseChatEngine:
    assert_exists_vector_index(vector_storage_dir)
    return load_vector_index(vector_storage_dir).as_chat_engine()

def get_vector_query_engine(vector_storage_dir: str) -> BaseQueryEngine:
    assert_exists_vector_index(vector_storage_dir)
    return load_vector_index(vector_storage_dir).as_query_engine()

def get_vector_ng_query_engine(vector_storage_dir: str) -> BaseQueryEngine:
    assert_exists_vector_index(vector_storage_dir)
    # configure retriever
    return RetrieverQueryEngine(
            retriever=VectorIndexRetriever(
            index=load_vector_index(vector_storage_dir),
            similarity_top_k=3,
        ),
            response_synthesizer=get_response_synthesizer(
            response_mode="tree_summarize",
        )
    )