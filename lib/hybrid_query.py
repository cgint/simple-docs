import time
from typing import List, Optional
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore, QueryBundle
from llama_index.retrievers import BaseRetriever
import concurrent.futures

from lib.index.doc_sum_index import load_doc_sum_index
from lib.index.terms.kg_num_term_neo4j import load_graph_index
from lib.vector_chroma import load_vector_index
from llama_index.query_engine import RetrieverQueryEngine

class HybridRetriever(BaseRetriever):
    """Retrieves nodes from multiple retrievers and combines the results."""
    def __init__(self, retrievers: List[BaseRetriever]):
        self.retrievers = retrievers
        super().__init__()

    def _retrieve(self, query, **kwargs):

        def retrieve_parallel(retriever: BaseRetriever, query, **kwargs):
            try:
                start_time = time.time()
                print(f"Retrieving {retriever.__class__} ...")
                result = retriever.retrieve(query, **kwargs)
                end_time = time.time()
                duration_rounded = round(end_time - start_time, 2)
                size_of_nodes_contents = sum([len(n.node.text) for n in result])
                print(f"Retrieved {len(result)} nodes (size={size_of_nodes_contents}) from {retriever.__class__} in {duration_rounded} seconds.")
            except Exception as e:
                print(f"Error retrieving from {retriever.__class__}: {e}")
                result = []
            return result

        def retrieve_in_parallel(retrievers, query, **kwargs):
            # print(f"Retrieving from {len(retrievers)} retrievers in parallel ...")
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
                if n.node.node_id not in node_ids and not n.node.text == "No relationships found.":
                    all_nodes.append(n)
                    node_ids.add(n.node.node_id)
        return all_nodes
    
class PrintingPostProcessor(BaseNodePostprocessor):
    """Prints the number of nodes and the total character count of the nodes' text."""
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
                first_text_chars_one_line = n.node.text[:30].replace("\n", " ")
                source_info = " - "+n.node.metadata['source_id'] if 'source_id' in n.node.metadata else ""
                print(f"    {n.node.node_id} - {n.score} - {first_text_chars_one_line}{source_info}")
        return nodes
    
class SimpleNodeListCutoff(BaseNodePostprocessor):
    """Adds nodes to the resultlist until the total number of nodes reaches the cutoff."""
    top_n: int
    def __init__(self, top_n: int):
        super().__init__(top_n=top_n)
        self.top_n = top_n

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        return nodes[:self.top_n]

def character_sum_cutoff(nodes: List[NodeWithScore], top_n: int) -> List[NodeWithScore]:
        result = []
        total_chars = 0
        for n in nodes:
            len_with_cur_node = total_chars + len(n.node.text)
            if len_with_cur_node > top_n:
                break
            result.append(n)
            total_chars = len_with_cur_node
        return result

class SimpleCharacterSumCutoff(BaseNodePostprocessor):
    """Adds nodes to the resultlist until the total character count of the nodes' text reaches the cutoff."""
    top_n: int
    def __init__(self, top_n: int):
        super().__init__(top_n=top_n)
        self.top_n = top_n

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        return character_sum_cutoff(nodes, self.top_n)

DEFAULT_SKIP_WORDS = ["!", "?", ",", "the", "and", "or", "of", "in", "to", "for", "on", "with", "by", "from", "at", "as", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "but", "not", "no", "nor", "so", "if", "when", "where", "which", "what", "who", "whom", "whose", "why", "how", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs", "myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves", "this", "that", "these", "those", "here", "there", "this", "that", "these", "those", "here", "there", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "but", "not", "no", "nor", "so", "if", "when", "where", "which", "what", "who", "whom", "whose", "why", "how", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs", "myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves", "this", "that", "these", "those", "here", "there", "that", "these", "those"]
class QueryWordmatchReranker(BaseNodePostprocessor):
    """Reranks nodes based on the number of query words that are found in the node's text."""
    skip_words: List[str]
    def __init__(self, skip_words: List[str] = DEFAULT_SKIP_WORDS):
        super().__init__(skip_words=skip_words)
        self.skip_words = skip_words

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        Sets the score of the node to the number of matching words excluding skip_words, sorts and returns top_n
        The words in the node text are iterated so the score is higher when a word is found more often in the node text.
        """
        if query_bundle is not None and query_bundle.query_str is not None and query_bundle.query_str != "":
            query_str_lower_words = [word.lower() for word in query_bundle.query_str.split() if word.lower() not in self.skip_words]
            print(f"QueryWordmatchReranker: Checking for words in TextNodes ... {query_str_lower_words}")
            for n in nodes:
                node_text_lower = n.node.text.lower()
                word_intersect_sum = len([word for word in node_text_lower.split() if word in query_str_lower_words])
                word_overlap       = len([word for word in query_str_lower_words if word in node_text_lower])
                n.score = word_overlap * word_intersect_sum
            nodes.sort(key=lambda x: x.score, reverse=True)
        return nodes


cached_hybrid_retriever_engine = {}
def get_hybrid_query_engine(service_context, query_engine_options, doc_sum_index_dir, collection, g_db) -> RetrieverQueryEngine:
    global cached_hybrid_retriever_engine
    variant = query_engine_options["variant"]
    reranker_model = query_engine_options["reranker_model"]
    engine_cache_key = f"{variant}-{reranker_model}"
    if not engine_cache_key in cached_hybrid_retriever_engine:
        print("Creating new hybrid query engine ...")
        from llama_index.postprocessor import SentenceTransformerRerank
        from llama_index.retrievers import BM25Retriever
        from llama_index.response_synthesizers import ResponseMode
        retriever_k = int(query_engine_options["retriever_k"])
        if "bm25" in variant or "docsum" in variant:
            doc_sum_index = load_doc_sum_index(service_context, doc_sum_index_dir)
        retrievers = []
        if "bm25" in variant:
            print(f"Pushing {len(doc_sum_index.docstore.docs)} from doc_sum_index to BM25Retriever.")
            retriever = BM25Retriever.from_defaults(docstore=doc_sum_index.docstore, similarity_top_k=retriever_k)
            print(f"Done.")
            retrievers.append(retriever)
        if "vector" in variant:
            retrievers.append(load_vector_index(service_context, collection).as_retriever(similarity_top_k=retriever_k))
        if "docsum" in variant:
            retrievers.append(doc_sum_index.as_retriever(similarity_top_k=retriever_k))
        if "graph" in variant:
            retrievers.append(load_graph_index(service_context, g_db).as_retriever(similarity_top_k=retriever_k))
        
        reranker_k = int(query_engine_options["reranker_k"])
        if "none" == reranker_model:
            post_processors = [PrintingPostProcessor("no reranking")]
        elif "cutoff-nodecount" == query_engine_options["reranker_model"]:
            post_processors = [
                    PrintingPostProcessor(f"before cutoff at {reranker_k} nodes"), 
                    SimpleNodeListCutoff(top_n=reranker_k), 
                    PrintingPostProcessor(f"after cutoff at {reranker_k} nodes", True)
                ]
        elif "cutoff-charsum" == query_engine_options["reranker_model"]:
            post_processors = [
                    PrintingPostProcessor(f"before cutoff at {reranker_k} characters"), 
                    SimpleCharacterSumCutoff(top_n=reranker_k), 
                    PrintingPostProcessor(f"after cutoff at {reranker_k} characters", True)
                ]
        elif "query-wordmatch" == query_engine_options["reranker_model"]:
            post_processors = [
                    PrintingPostProcessor(f"before QueryWordmatchReranker"), 
                    QueryWordmatchReranker(), 
                    PrintingPostProcessor(f"after QueryWordmatchReranker", True)
                ]
        elif "query-wordmatch-nodecount" == query_engine_options["reranker_model"]:
            post_processors = [
                    PrintingPostProcessor(f"before QueryWordmatchReranker and cutoff at {reranker_k} nodes"), 
                    QueryWordmatchReranker(), 
                    SimpleNodeListCutoff(top_n=reranker_k), 
                    PrintingPostProcessor(f"after QueryWordmatchReranker and cutoff at {reranker_k} nodes", True)
                ]
        elif "query-wordmatch-charsum" == query_engine_options["reranker_model"]:
            post_processors = [
                    PrintingPostProcessor(f"before QueryWordmatchReranker cutoff at {reranker_k} characters"), 
                    QueryWordmatchReranker(), 
                    SimpleCharacterSumCutoff(top_n=reranker_k), 
                    PrintingPostProcessor(f"after QueryWordmatchReranker cutoff at {reranker_k} characters", True)
                ]
        else:
            info_k = f"re-ranker [retriever_k={retriever_k}, reranker_k={reranker_k}]"
            post_processors = [
                    PrintingPostProcessor(f"before {info_k}"), 
                    SentenceTransformerRerank(top_n=reranker_k, model=query_engine_options["reranker_model"]), 
                    PrintingPostProcessor(f"after {info_k}", True)
                ]
        cached_hybrid_retriever_engine[engine_cache_key] = wrap_in_sub_question_engine(RetrieverQueryEngine.from_args(
            retriever=HybridRetriever(retrievers),
            node_postprocessors=post_processors,
            service_context=service_context
        ), service_context)
    return cached_hybrid_retriever_engine[engine_cache_key]

def wrap_in_sub_question_engine(query_engine: RetrieverQueryEngine, service_context) -> RetrieverQueryEngine:
    """Wraps the given query engine in a sub question engine."""
    from llama_index.query_engine import SubQuestionQueryEngine
    from llama_index.tools import QueryEngineTool, ToolMetadata
    from llama_index.question_gen.guidance_generator import GuidanceQuestionGenerator
    from guidance.models import OpenAI
    return SubQuestionQueryEngine.from_defaults(
        service_context=service_context,
        question_gen= GuidanceQuestionGenerator.from_defaults(guidance_llm=OpenAI(model="gpt-3.5-turbo")),
        query_engine_tools=[
            QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name="hybrid_query_engine",
                    description=(
                        "Provides information about everything the user might ask."
                    )
                )
            )
        ]
    )