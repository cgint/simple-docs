import os
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Sequence
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.ollama import Ollama
from lib import constants
from lib.index.helper import cur_simple_date_time_sec
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms import CustomLLM
from llama_index.core.llms import ChatMessage, ChatResponse, ChatResponseGen, CompletionResponse, CompletionResponseGen, LLMMetadata, MessageRole

class MultiOllamaRoundRobin(CustomLLM):
    ollama_workers: List[Ollama]
    ollama_main: Ollama
    ollama_busy: List[bool]

    def __init__(self, ollama_workers: List[Ollama]):
        super().__init__(ollama_workers=ollama_workers, ollama_main=ollama_workers[0], ollama_busy = [False for _ in ollama_workers])

    def execute_on_free_worker(self, task_func: Callable):
        chosen_index = None
        attempts = 0
        while chosen_index is None:
            attempts += 1
            for i, busy in enumerate(self.ollama_busy):
                if not busy:
                    chosen_index = i
                    self.ollama_busy[i] = True
                    break
            if chosen_index is None:
                sleep(0.1)
        
        print(f" -- MultiOllamaRoundRobin --- --- --- --- --- --- --- --- --- --- -- Using Ollama-Instance {chosen_index} --- --- --- -- ATTEMPTS {attempts} --- --- ---")

        # Execute the task on the chosen worker
        try:
            response = task_func(self.ollama_workers[chosen_index])
        finally:
            self.ollama_busy[chosen_index] = False

        return response

    @llm_chat_callback()
    def chat(self, messages, **kwargs):
        answer = self.execute_on_free_worker(lambda worker: worker.chat(messages, **kwargs))
        self.write_to_csv(messages, answer)
        return answer

    def write_to_csv(self, messages, answer):
        clz = self.class_name()
        filename = f"{constants.data_base_dir}/{constants.run_start_time_id}_{clz}_chat_log.csv"
        import pandas as pd
        ts = cur_simple_date_time_sec()
        df = pd.DataFrame({
            "time_id": [ts for _ in messages],
            "role": [m.role for m in messages],
            "message": [m.content for m in messages],
            "answer": [answer for _ in messages],
        })
        df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
    
    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        return self.execute_on_free_worker(lambda worker: worker.stream_chat(messages, **kwargs))
    
    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self.execute_on_free_worker(lambda worker: worker.complete(prompt, formatted, **kwargs))
    
    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        return self.execute_on_free_worker(lambda worker: worker.stream_complete(prompt, formatted, **kwargs))
    
    @classmethod
    def class_name(cls) -> str:
        return "MultiOllamaRoundRobin"
    
    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return self.ollama_main.metadata
