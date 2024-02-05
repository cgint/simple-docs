from typing import List
from llama_index import Document, get_response_synthesizer
from llama_index import VectorStoreIndex
from llama_index import StorageContext
from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers import BaseRetriever

from lib.llm import get_callback_manager

def load_vector_index_storage_context(vector_storage_dir: str) -> StorageContext:
    return StorageContext.from_defaults(persist_dir=vector_storage_dir)

def load_vector_index_chroma_storage_context(collection: str) -> (ChromaVectorStore, StorageContext):
    remote_db = chromadb.HttpClient(host="host.docker.internal")
    chroma_collection = remote_db.get_or_create_collection(collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return vector_store, storage_context

def load_vector_index(service_context, vector_storage_dir: str) -> VectorStoreIndex:
    collection = vector_storage_dir.replace("/", "_").replace("_", "")
    vector_store, _ = load_vector_index_chroma_storage_context(collection)
    return VectorStoreIndex.from_vector_store(
        service_context=service_context,
        #storage_context=load_vector_index_storage_context(vector_storage_dir)
        vector_store=vector_store
    )

def get_vector_index(service_context, vector_storage_dir: str) -> VectorStoreIndex:
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
    return vector_index

def get_vector_query_engine(service_context, vector_storage_dir: str) -> BaseQueryEngine:
    return load_vector_index(service_context, vector_storage_dir).as_query_engine()

def get_vector_ng_query_engine(service_context, vector_storage_dir: str) -> BaseQueryEngine:
    return RetrieverQueryEngine(
        retriever=VectorIndexRetriever(
            #callback_manager=get_callback_manager(),
            service_context=service_context,
            index=load_vector_index(service_context, vector_storage_dir),
            similarity_top_k=3,
        ),
        response_synthesizer=get_response_synthesizer(
            response_mode="tree_summarize",
        )
        #,callback_manager=get_callback_manager()
    )