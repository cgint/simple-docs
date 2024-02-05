import os
from lib.index.doc_sum_index import create_doc_summary_index

from lib.index.html import clean_html_content, get_documents_from_urls_as_mirror, get_documents_from_urls
from lib.index.helper import cur_simple_date_time_sec
from lib.index.pdf import get_content_from_pdf_file
from lib.index.terms.term_index import build_term_reference_index, count_terms_per_document, get_term_to_doc_items, write_term_references_to_file
from lib.index.terms.terms import terms_from_txt
from lib.index.web import create_simple_identifier_from_url
from lib.index.terms.kg_num_term_neo4j import kg_neo4j_delete_all_nodes, operate_on_graph_index

from lib.vector_chroma import add_to_or_update_in_vector, operate_on_vector_index
from llama_index import Document, KnowledgeGraphIndex
from typing import List
import queue
import threading

documents_remembered = []

def init_consumer_threads(service_context, storage_dir):
    q_main = queue.Queue(maxsize=100)

    q_vector = queue.Queue(maxsize=100)
    q_graph = queue.Queue(maxsize=100)
    q_term_index = queue.Queue(maxsize=100)
    q_remember_documents = queue.Queue(maxsize=100)

    # Start consuming in parallel
    consumer_threads = []
    consumer_threads.append(threading.Thread(target=queue_distributor, args=(q_main, [q_vector, q_graph, q_term_index, q_remember_documents])))
    consumer_threads.append(threading.Thread(target=index_consume_documents_on_vector, args=(service_context, storage_dir, "VectorIndex", q_vector)))
    consumer_threads.append(threading.Thread(target=index_consume_documents_on_graph, args=(service_context, storage_dir, "GraphIndex", q_graph)))
    consumer_threads.append(threading.Thread(target=index_consume_documents_on_term_index, args=("TermIndex", q_term_index)))
    consumer_threads.append(threading.Thread(target=index_consume_remember_documents, args=("RememberDocs", q_remember_documents)))
    for t in consumer_threads:
        t.start()
    
    return q_main, consumer_threads

def register_main_queue_push(q, doc: Document):
    doc_id = doc.doc_id if doc else "None"
    print(f"Pushing document '{doc_id}' to main_queue ...")
    q.put(doc)

def async_index(service_context, storage_dir, index_dir, index_dir_done):
    print("Deleting neo4j nodes ...")
    kg_neo4j_delete_all_nodes()
    q_main, consumer_threads = init_consumer_threads(service_context, storage_dir)
    
    # Start producing
    index_produce_documents(index_dir, index_dir_done, lambda doc: register_main_queue_push(q_main, doc))

    # Tell consumer to stop and wait for it to finish
    q_main.put(None)

    # count docs per term
    write_term_references_to_file()
    count_terms_per_document()

    create_doc_summary_index(documents_remembered, service_context, storage_dir)
    
    for t in consumer_threads:
        t.join()

    # Apply term to doc on graph index
    #operate_on_graph_index(service_context, vector_storage_dir, apply_term_to_doc_on_graph_index)

def apply_term_to_doc_on_graph_index(idx: KnowledgeGraphIndex):
    print(f"Applying term to doc on graph index ...")
    for term, doc_count in get_term_to_doc_items():
        for doc_id in doc_count.keys():
            idx.upsert_triplet((term, "in", doc_id))

def queue_distributor(q, consumers):
    while True:
        doc = q.get()
        for c in consumers:
            c.put(doc)
        q.task_done()
        if doc is None:
            break  # Exit if None is received

def index_consume_remember_documents(log_name, q):
    global documents_remembered
    index_consume_documents(log_name, q, documents_remembered.append)

def index_consume_documents_on_vector(service_context, vector_storage_dir, log_name, q):
    processor = lambda idx: index_consume_documents(log_name, q, lambda doc: idx.refresh_ref_docs([doc]))
    operate_on_vector_index(service_context, vector_storage_dir, processor)

def index_consume_documents_on_graph(service_context, graph_storage_dir, log_name, q):
    processor = lambda idx: index_consume_documents(log_name, q, lambda doc: idx_update_doc_on_graph(idx, doc))
    operate_on_graph_index(service_context, graph_storage_dir, processor)

def idx_update_doc_on_graph(idx: KnowledgeGraphIndex, doc: Document):
    terms = terms_from_txt(doc.text)
    for term in terms:
        idx.upsert_triplet_and_node((term, "in", doc.metadata["source_id"]), doc)

def index_consume_documents_on_term_index(log_name, q):
    index_consume_documents(log_name, q, build_term_reference_index)

def index_consume_documents(log_name, q, process=lambda: None):
    counter = 0
    while True:
        doc = q.get()
        counter += 1
        if doc is None:
            q.task_done()
            break  # Exit if None is received
        print(f"{log_name} - Indexing document #{counter} with id {doc.doc_id} ...")
        process(doc)
        q.task_done()

def index(service_context, vector_storage_dir, index_dir, index_dir_done):
    async_index(service_context, vector_storage_dir, index_dir, index_dir_done)

def index_sync(service_context, vector_storage_dir, index_dir, index_dir_done):
    documents = []
    index_produce_documents(index_dir, index_dir_done, lambda doc: documents.append(doc))
    print(f"Indexing {len(documents)} documents ...")
    add_to_or_update_in_vector(service_context, vector_storage_dir, documents)

def index_produce_documents(index_dir, index_dir_done, producer_sink=lambda: Document):
    print(f"Indexing documents from {index_dir} ...")
    files_to_index = [f for f in os.listdir(index_dir) if not f.startswith(".")]
    files_to_index = [f for f in files_to_index if os.path.isfile(os.path.join(index_dir, f))]
    if len(files_to_index) == 0:
        print(f"No documents to index in {index_dir}.")
        return
    
    run_index_time = cur_simple_date_time_sec()
    print(f"Creating documents: {len(files_to_index)} for index {run_index_time}")
    for file in files_to_index:
        print(f"Indexing {file} ...")
        file_full_path = os.path.join(index_dir, file)
        if file == "fetch_urls.txt":
            get_documents_from_urls(read_relevant_lines(file_full_path), producer_sink)
        elif file == "mirror.txt":
            for url in read_relevant_lines(file_full_path):
                simple_url = create_simple_identifier_from_url(url)
                mirror_dir = f"/data/index_inbox/mirror/{simple_url}/{run_index_time}/"
                os.makedirs(mirror_dir, exist_ok=True)
                get_documents_from_urls_as_mirror(mirror_dir, url, producer_sink)
        elif file.endswith(".txt"):
            with open(file_full_path, "r") as f:
                producer_sink(Document(text=f.read(), metadata={"source_id": file, "source_type": "txt"}))
        elif file.endswith(".json"):
            Document.from_json(file_full_path)
            with open(file_full_path, "r") as f:
                doc = Document.from_json(f.read(), {"source_id": file, "source_type": "json"})
                producer_sink(doc)
        elif file.endswith(".md"):
            with open(file_full_path, "r") as f:
                producer_sink(Document(text=f.read(), metadata={"source_id": file, "source_type": "txt"}))
        elif file.endswith(".html"):
            with open(file_full_path, "r") as f:
                producer_sink(Document(text=clean_html_content(f.read()), metadata={"source_id": file, "source_type": "txt"}))
        elif file.endswith(".pdf"):
            for doc in get_content_from_pdf_file(file_full_path):
                producer_sink(doc)
    for file in files_to_index:
        file_full_path = os.path.join(index_dir, file)
        target_file_name = file
        if file == "fetch_urls.txt":
            target_file_name = f"fetch_urls_{run_index_time}.txt"
        if file == "mirror.txt":
            target_file_name = f"mirror_{run_index_time}.txt"
        to_path = os.path.join(index_dir_done, target_file_name)
        print(f"Moving {file_full_path} to {to_path} ...")
        os.rename(file_full_path, to_path)


def read_relevant_lines(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f.read().splitlines() if line.strip() != "" and not line.strip().startswith("#")]
