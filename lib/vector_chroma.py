from typing import List
from llama_index import Document, get_response_synthesizer
from llama_index import VectorStoreIndex
from llama_index import StorageContext
from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

import os

host_ip = os.getenv("HOST_IP", "host.docker.internal")

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_or_create_chroma_collection(collection: str) -> chromadb.Collection:
    remote_db = chromadb.HttpClient(host=host_ip)
    return remote_db.get_or_create_collection(collection)

def delete_chroma_collection(collection: str) -> None:
    chroma_collection = get_or_create_chroma_collection(collection)
    doc_ids = chroma_collection.get()["ids"]
    print(f"Deleting {len(doc_ids)} documents from chroma_collection ...")
    batch_size = 20000
    for i, batch in enumerate(chunker(doc_ids, batch_size)):
        print(f"Deleting batch {i+1}/{len(batch)} ...")
        chroma_collection.delete(ids=batch)
    print(f"Collection.count()={chroma_collection.count()} ...")


def load_vector_index_chroma_storage_context(collection: str) -> tuple[ChromaVectorStore, StorageContext]:
    chroma_collection = get_or_create_chroma_collection(collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return vector_store, storage_context

def load_vector_index(service_context, collection: str) -> VectorStoreIndex:
    vector_store, _ = load_vector_index_chroma_storage_context(collection)
    return VectorStoreIndex.from_vector_store(
        service_context=service_context,
        vector_store=vector_store
    )

def get_vector_index(service_context, collection: str) -> VectorStoreIndex:
    return load_vector_index(service_context, collection)

def add_to_or_update_in_vector(service_context, vector_storage_dir: str, documents: List[Document]) -> VectorStoreIndex:
    return operate_on_vector_index(
        service_context,
        vector_storage_dir,
        lambda vector_index: vector_index.refresh_ref_docs(documents)
    )
    
def operate_on_vector_index(service_context, collection: str, operation=lambda: None) -> VectorStoreIndex:
    vector_index = get_vector_index(service_context, collection)
    operation(vector_index)
    return vector_index

def get_vector_query_engine(service_context, collection: str) -> BaseQueryEngine:
    return load_vector_index(service_context, collection).as_query_engine()

def get_vector_ng_query_engine(service_context, vector_storage_dir: str) -> BaseQueryEngine:
    return RetrieverQueryEngine(
        retriever=VectorIndexRetriever(
            service_context=service_context,
            index=load_vector_index(service_context, vector_storage_dir),
            similarity_top_k=3,
        ),
        response_synthesizer=get_response_synthesizer(
            response_mode="tree_summarize",
        )
    )