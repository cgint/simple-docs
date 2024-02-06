from dataclasses import Field
import time
from typing import List, Optional
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore, QueryBundle
from llama_index.retrievers import BaseRetriever
import concurrent.futures

from lib.index.doc_sum_index import get_doc_sum_index_query_engine, load_doc_sum_index
from lib.index.terms.kg_num_term_neo4j import get_graph_query_engine, load_graph_index
from lib.vector_chroma import get_vector_query_engine, load_vector_index
from llama_index.query_engine import RetrieverQueryEngine

class HybridRetriever(BaseRetriever):
    def __init__(self, retrievers: List[BaseRetriever]):
        self.retrievers = retrievers
        super().__init__()

    def _retrieve(self, query, **kwargs):

        def retrieve_parallel(retriever: BaseRetriever, query, **kwargs):
            start_time = time.time()
            result = retriever.retrieve(query, **kwargs)
            end_time = time.time()
            duration_rounded = round(end_time - start_time, 2)
            size_of_nodes_contents = sum([len(n.node.text) for n in result])
            print(f"Retrieved {len(result)} nodes (size={size_of_nodes_contents}) from {retriever.__class__} in {duration_rounded} seconds.")
            return result

        def retrieve_in_parallel(retrievers, query, **kwargs):
            print(f"Retrieving from {len(retrievers)} retrievers in parallel ...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(retrieve_parallel, retriever, query, **kwargs) for retriever in retrievers]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            return results

        node_lists = retrieve_in_parallel(self.retrievers, query, **kwargs)
        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for single_list in node_lists:
            for n in single_list:
                if n.node.node_id not in node_ids:
                    all_nodes.append(n)
                    node_ids.add(n.node.node_id)
        return all_nodes
    


class PrintingPostProcessor(BaseNodePostprocessor):
    print_identifier: str
    print_rank_scores: bool
    def __init__(self, print_identifier: str, print_rank_scores: bool = False):
        super().__init__(print_identifier=print_identifier, print_rank_scores=print_rank_scores)

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        size_of_nodes_contents = sum([len(n.node.text) for n in nodes])
        cur_time_h_m_s = time.strftime("%H:%M:%S", time.localtime())
        print(f"{cur_time_h_m_s} - {len(nodes)} nodes / {size_of_nodes_contents} characters {self.print_identifier}.")
        if self.print_rank_scores:
            for n in nodes:
                first_text_chars = n.node.text[:30]
                print(f"Source Node: {n.score} - {n.node.node_id} - {first_text_chars} - {n.node.metadata}")
        return nodes


def get_hybrid_query_engine(service_context, storage_dir) -> RetrieverQueryEngine:
    from llama_index.postprocessor import SentenceTransformerRerank
    from llama_index.retrievers import BM25Retriever
    retriever_k = 5
    reranker_k = 2
    doc_sum_index = load_doc_sum_index(service_context, storage_dir)
    print(f"Pushing {len(doc_sum_index.docstore.docs)} from doc_sum_index to BM25Retriever.")
    bm25_retriever = BM25Retriever.from_defaults(docstore=doc_sum_index.docstore, similarity_top_k=retriever_k)
    print(f"Done.")
    retrievers = []
    retrievers.append(bm25_retriever)
    retrievers.append(load_vector_index(service_context, storage_dir).as_retriever(similarity_top_k=retriever_k))
    retrievers.append(doc_sum_index.as_retriever(similarity_top_k=retriever_k))
    retrievers.append(load_graph_index(service_context, storage_dir).as_retriever(similarity_top_k=retriever_k))
    info_k = f"re-ranker [retriever_k={retriever_k}, reranker_k={reranker_k}]"
    return RetrieverQueryEngine.from_args(
        retriever=HybridRetriever(retrievers),
        node_postprocessors=[
            PrintingPostProcessor(f"before {info_k}", True), 
            SentenceTransformerRerank(top_n=reranker_k, model="BAAI/bge-reranker-base"), 
            PrintingPostProcessor(f"after  {info_k}", True)
        ],
        service_context=service_context,
    )
        