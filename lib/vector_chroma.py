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

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_or_create_chroma_collection(collection: str) -> chromadb.Collection:
    remote_db = chromadb.HttpClient(host=constants.host_ip)
    return remote_db.get_or_create_collection(collection)

def delete_chroma_collection(collection: str) -> None:
    chroma_collection = get_or_create_chroma_collection(collection)
    doc_ids = chroma_collection.get()["ids"]
    print(f"Deleting {len(doc_ids)} documents from chroma_collection ...")
    batch_size = 20000
    for i, batch in enumerate(chunker(doc_ids, batch_size)):
        print(f"Deleting batch {i+1} with batch-size of {len(batch)} items ...")
        chroma_collection.delete(ids=batch)
    print(f"Collection.count()={chroma_collection.count()} ...")


def load_vector_index_chroma_storage_context(collection: str) -> tuple[ChromaVectorStore, StorageContext]:
    chroma_collection = get_or_create_chroma_collection(collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return vector_store, storage_context

def load_vector_index(collection: str) -> VectorStoreIndex:
    vector_store, _ = load_vector_index_chroma_storage_context(collection)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )

def get_vector_index(collection: str) -> VectorStoreIndex:
    return load_vector_index(collection)

def add_to_or_update_in_vector(vector_storage_dir: str, documents: List[Document]) -> VectorStoreIndex:
    return operate_on_vector_index(
        vector_storage_dir,
        lambda vector_index: vector_index.refresh_ref_docs(documents)
    )
    
def operate_on_vector_index(collection: str, operation=lambda: None) -> VectorStoreIndex:
    vector_index = get_vector_index(collection)
    operation(vector_index)
    return vector_index

def get_vector_query_engine(collection: str) -> BaseQueryEngine:
    return load_vector_index(collection).as_query_engine()

def get_vector_ng_query_engine(vector_storage_dir: str) -> BaseQueryEngine:
    return RetrieverQueryEngine(
        retriever=VectorIndexRetriever(
            index=load_vector_index(vector_storage_dir),
            similarity_top_k=3,
        ),
        response_synthesizer=get_response_synthesizer(
            response_mode="tree_summarize",
        )
    )