import os
from lib.llm import get_aim_callback, get_callback_manager, get_csv_callback_handler, get_llm
from llama_index.embeddings import FastEmbedEmbedding
from llama_index import set_global_service_context, set_global_handler
from llama_index import ServiceContext
from lib.ask import ask_for_vector_response, ask_for_vector_ng_response, ask_for_graph_response, ask_for_doc_sum_index_response, ask_hybrid_query_engine
import llama_index

index_dir = "/data/index_inbox"
index_dir_done = "/data/index_inbox/done"

vector_storage_dir = "/data/vector_index"
doc_sum_index_dir = "/data/doc_sum_index"

def get_service_context(llm_options):
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    #llm = get_llm("together", "mixtral-together", "mistralai/Mixtral-8x7B-Instruct-v0.1")
    # llm = get_llm("ollama", "neural-chat")
    llm = get_llm(llm_options["engine"], llm_options["model"], llm_options["openai_model"])
    print(f"About to instanciate Embed Model {embed_model_name} using FastEmbedEmbedding ...")
    embed_cache_dir = "/data/fastembed_cache/"
    embed_model = FastEmbedEmbedding(model_name=embed_model_name, cache_dir=embed_cache_dir)
    return ServiceContext.from_defaults(
        llm=llm, 
        chunk_size=512, 
        embed_model=embed_model # embed_model="local"
        #,callback_manager=get_callback_manager()
    )

def get_params_from_env():
    command = os.environ.get("PARAM_COMMAND", "")
    if command != "index" and command != "ask":
        print("Usage:")
        print("  PARAM_COMMAND: 'index', 'ask'")
        print(f"  Data from {index_dir} will be indexed and moved to {index_dir_done} when done.")
        exit(1)
    llm_options = {
        "engine": os.environ.get("LLM_ENGINE", "ollama"),
        "model": os.environ.get("LLM_MODEL", "neural-chat"),
        "openai_model": os.environ.get("OPENAI_MODEL")
    }
    return command, llm_options

def ask_questions(service_context, index_dir, question):
    print(f"Looking for an answer for question: {question}")
    answer_full = ask_hybrid_query_engine(service_context, index_dir, question)
    answer = answer_full.response
    if answer == "" or answer == None:
        answer = "No answer found."
    print("Answer: " + answer)
    print("\n")
    #print(f"  Answer MetaData: {str(answer_full.metadata)}")
    print(f"  Source Nodes ({len(answer_full.source_nodes)}) :")
    for n in answer_full.source_nodes:
        first_text_chars = n.node.text[:30]
        print(f"    Source Node: {n.score} - {n.node_id} - {first_text_chars} - {answer_full.metadata[n.node_id]}")
    print("\n")


if __name__ == "__main__":
    command, llm_options = get_params_from_env()

    experiment_name = f"{llm_options['engine']}_{llm_options['model']}" + (f"_{llm_options['openai_model']}" if llm_options['openai_model'] is not None else "")
    aim_run_params = { 
        'query_variant': 'hybrid', 
        'command': command,
        'llm': llm_options 
        }
    llama_index.global_handler = get_aim_callback(experiment_name, "/data/aim", aim_run_params)

    service_context = get_service_context(llm_options)
    set_global_service_context(service_context)

    fixed_questions = os.environ.get("ASK_SENTENCES")
    if fixed_questions is not None:
        fixed_questions = fixed_questions.split("###")
        for question in fixed_questions:
            ask_questions(service_context, doc_sum_index_dir, question)
    
    elif command == "index":
        from lib.index.index import index
        index(service_context, vector_storage_dir, index_dir, index_dir_done)
    else:
        # loop with input until "end" is given
        while True:
            question = input("Ask (type 'e' or 'end' to finish): ").strip()
            if question == "":
                continue
            if question == "end" or question == "e":
                break
            ask_questions(service_context, doc_sum_index_dir, question)
        print("\nBye.")