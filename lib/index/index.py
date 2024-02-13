from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import os
from time import sleep
from lib.index.doc_sum_index import operate_on_doc_sum_index

from lib.index.html import clean_html_content, get_documents_from_urls_as_mirror, get_documents_from_urls
from lib.index.helper import cur_simple_date_time_sec
from lib.index.pdf import get_content_from_pdf_file
from lib.index.terms.term_index import build_term_reference_index, count_terms_per_document, get_term_to_doc_items, write_term_references_to_file
from lib.index.terms.terms import terms_from_txt
from lib.index.terms.kg_num_term_neo4j import kg_neo4j_delete_all_nodes, operate_on_graph_index
from lib.index.helper import list_files
from lib.vector_chroma import delete_chroma_collection, operate_on_vector_index
from llama_index import Document, KnowledgeGraphIndex
from typing import List
from lib import constants
import queue
import threading

main_queue_size = 1
worker_queue_sizes = 5
RETRY_ATTEMPTS_MAX = 3

def init_consumer_threads(service_context, doc_sum_index_dir, collection, g_db):
    q_main = queue.Queue(maxsize=main_queue_size)

    q_vector = queue.Queue(maxsize=worker_queue_sizes)
    q_graph = queue.Queue(maxsize=worker_queue_sizes)
    q_term_index = queue.Queue(maxsize=worker_queue_sizes)
    q_doc_sum = queue.Queue(maxsize=worker_queue_sizes)

    # Start consuming in parallel
    consumer_threads = []
    consumer_threads.append(threading.Thread(target=queue_fan_out, args=(q_main, [q_vector, q_graph, q_term_index, q_doc_sum])))
    consumer_threads.append(threading.Thread(target=index_consume_documents_on_vector, args=(service_context, collection, "VectorIndex", q_vector)))
    consumer_threads.append(threading.Thread(target=index_consume_documents_on_graph, args=(service_context, g_db, "GraphIndex", q_graph)))
    consumer_threads.append(threading.Thread(target=index_consume_documents_on_term_index, args=("TermIndex", q_term_index)))
    consumer_threads.append(threading.Thread(target=index_consume_documents_on_doc_sum, args=(service_context, doc_sum_index_dir, "DocSumIndex", q_doc_sum)))
    for t in consumer_threads:
        t.start()
    
    return q_main, consumer_threads

def register_main_queue_push(q_main: queue.Queue, doc: Document):
    # doc_id = doc.doc_id if doc else "None"
    # q_size = q_main.qsize()
    # print(f"Pushing document '{doc_id}' to main_queue (size={q_size}) ...")
    q_main.put(doc)

def async_index(service_context, doc_sum_index_dir, collection, g_db, index_dir, index_dir_done):
    # print("Deleting neo4j nodes ...")
    # kg_neo4j_delete_all_nodes()
    # print("Deleting chroma entries ...")
    # delete_chroma_collection(collection)
    
    
    # Start producing
    try:
        q_main, consumer_threads = init_consumer_threads(service_context, doc_sum_index_dir, collection, g_db)
        index_produce_documents(constants.max_files_to_index_per_run, index_dir, index_dir_done, lambda doc: register_main_queue_push(q_main, doc))
    finally:
        # Tell consumer to stop and wait for it to finish
        print("Telling consumers to finish once their queues are empty ...")
        q_main.put(None)
        for t in consumer_threads:
            t.join()

    # count docs per term
    write_term_references_to_file()
    count_terms_per_document()

def queue_fan_out(q, consumers):
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

def index_consume_documents_on_vector(service_context, collection, log_name, q):
    processor = lambda idx: index_consume_documents(log_name, q, lambda doc: try_refresh_ref_docs_retry_max(RETRY_ATTEMPTS_MAX, idx, doc))
    operate_on_vector_index(service_context, collection, processor)

def index_consume_documents_on_doc_sum(service_context, storage_dir, log_name, q):
    processor = lambda idx: index_consume_documents_threading_workers(log_name, q, lambda doc: try_refresh_ref_docs_retry_max(RETRY_ATTEMPTS_MAX, idx, doc), 3)
    operate_on_doc_sum_index(service_context, storage_dir, processor)

def try_refresh_ref_docs_retry_max(attempts , idx, doc):
    for i in range(attempts):
        try:
            idx.refresh_ref_docs([doc])
            break
        except Exception as e:
            if i < 2:
                print(f" ERROR-RETRYING: Error refreshing ref_docs after a bit. The error: {e}")
                sleep(0.25)
            else:
                raise e

from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading

def index_consume_documents_threading_workers(log_name, q, process=lambda x: None, max_threads=1, max_queued_tasks=5):
    counter = 0
    futures = set()

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        while True:
            counter += 1
            doc = q.get()
            if doc is None:
                q.task_done()
                break  # Exit if None is received

            # Check if we need to wait for some tasks to complete
            unfinished_count = len({future for future in futures if not future.done()})
            if unfinished_count >= max_queued_tasks:
                print(f" {log_name} - q(size={q.qsize()}) - Waiting for one of {unfinished_count} active tasks to complete before submitting next (backpressure) ...")
                # Wait for at least one task to complete
                _, futures = wait(futures, return_when=FIRST_COMPLETED)

            # Submit new task
            futures.add(executor.submit(process, doc))
            print(f" {log_name} - q(size={q.qsize()}) - Indexing document #{counter} with id {doc.doc_id} ...")
            q.task_done()

        # Wait for all submitted tasks to complete
        for future in as_completed(futures):
            future.result()  # This also helps in raising exceptions if any occurred

def index_consume_documents_on_graph(service_context, collection, log_name, q):
    processor = lambda idx: index_consume_documents(log_name, q, lambda doc: idx_update_doc_on_graph(idx, doc))
    operate_on_graph_index(service_context, collection, processor)

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
        print(f" {log_name} - q(size={q.qsize()}) - Indexing document #{counter} with id {doc.doc_id} ...")
        process(doc)
        q.task_done()

def index(service_context, doc_sum_index_dir, collection, g_db, index_dir, index_dir_done):
    async_index(service_context, doc_sum_index_dir, collection, g_db, index_dir, index_dir_done)


extensions_to_treat_as_plain_txt = ["tf", "yml", "yaml", "txt", "md", "java", "scala", "js", "ts", "xml", "kts", "gradle", "groovy", "py"]
known_extensions = ["json", "html", "pdf"] + extensions_to_treat_as_plain_txt

def index_produce_documents(max_files_to_process: int, index_dir: str, index_dir_done: str, producer_sink=lambda: Document):
    print(f"Looking for documents under {index_dir} to be indexed recursively ...")
    files_to_index = list_files(index_dir)
    if len(files_to_index) == 0:
        print(f"No documents to index in {index_dir}.")
        return
    
    run_index_time = cur_simple_date_time_sec()
    full_file_count = len(files_to_index)
    file_counter = 0
    print(f"Creating documents: {full_file_count} for index {run_index_time}")
    for file in files_to_index:
        file_full_path = file
        file_extension = file_full_path.split('.')[-1]
        if file_extension not in known_extensions:
            print(f"Skipping unknown file extension: {file_extension} for file {file_full_path}")
            continue
        file_counter += 1
        if max_files_to_process is not None and file_counter > max_files_to_process:
            print(f"Max files to process ({max_files_to_process}) reached. Will stop producing tasks ...")
            break
        print(f"Indexing {file_counter}/{full_file_count} {file} ...")
        single_file_processed = False
        if file == "fetch_urls.txt":
            get_documents_from_urls(read_relevant_lines(file_full_path), producer_sink)
        elif file == "mirror.txt":
            for url in read_relevant_lines(file_full_path):
                # simple_url = create_simple_identifier_from_url(url)
                # mirror_dir = f"{index_dir}/mirror/{simple_url}/{run_index_time}/"
                # os.makedirs(mirror_dir, exist_ok=True)
                get_documents_from_urls_as_mirror(None, url, producer_sink)
        elif file_extension == "json":
            single_file_processed = True
            with open(file_full_path, "r") as f:
                doc = Document.from_json(f.read(), {"source_id": file, "source_type": "json"})
                producer_sink(doc)
        elif file_extension == "html":
            single_file_processed = True
            with open(file_full_path, "r") as f:
                producer_sink(Document(text=clean_html_content(f.read()), metadata={"source_id": file, "source_type": "html"}))
        elif file_extension == "pdf":
            single_file_processed = True
            for doc in get_content_from_pdf_file(file_full_path):
                producer_sink(doc)
        elif file_extension in extensions_to_treat_as_plain_txt:
            single_file_processed = True
            with open(file_full_path, "r") as f:
                producer_sink(Document(text=f.read(), metadata={"source_id": file, "source_type": file_extension}))
        if single_file_processed:
            target_file_name = file.replace(index_dir, index_dir_done)
            # print(f" !!! NOT !!! - Moving {file_full_path} to {to_path} ... to preserve for the next run.")
            print(f"Moving {file_full_path} to {target_file_name} ...")
            os.makedirs(os.path.dirname(target_file_name), exist_ok=True)
            os.rename(file_full_path, target_file_name)
    

def read_relevant_lines(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f.read().splitlines() if line.strip() != "" and not line.strip().startswith("#")]
