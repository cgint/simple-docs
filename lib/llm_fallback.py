import os
from typing import Any, Callable, List, Sequence
from lib import constants
from lib.index.helper import cur_simple_date_time_sec
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.llms import CustomLLM
from llama_index.core.llms import ChatMessage, ChatResponseGen, CompletionResponse, CompletionResponseGen, LLMMetadata

class MultiLlmFallback(CustomLLM):
    llm_prio_list: List[BaseLLM]

    def __init__(self, llm_prio_list: List[BaseLLM]):
        super().__init__(llm_prio_list=llm_prio_list)

    def execute_and_fallback_on_error(self, task_func: Callable):
        exceptions = []
        for chosen_index, llm in enumerate(self.llm_prio_list):
            try:
                answer = task_func(llm)
                message = f" -- MultiLlmFallback --- --- --- --- --- --- --- --- --- --- -- Successful answer from LLM on list-index {chosen_index} --- --- --- -- --- --- ---"
                message += f" ### Exceptions: {exceptions}" if exceptions else ""
                print(message)
                return answer
            except Exception as e:
                print(f" XXX --- MultiLlmFallback --- Exception from LLM on list-index {chosen_index} --- XXX - {e} - XXX ---")
                exceptions.append(e)
        
        
        message = f" -- MultiLlmFallback --- --- --- --- --- --- --- --- --- --- -- Unsuccessful even with fallback --- --- --- -- --- --- ---"
        message += f" ### Exceptions: {exceptions}" if exceptions else ""
        raise Exception(message)

    @llm_chat_callback()
    def chat(self, messages, **kwargs):
        answer = self.execute_and_fallback_on_error(lambda worker: worker.chat(messages, **kwargs))
        self.write_to_csv(messages, answer)
        return answer

    def write_to_csv(self, messages, answer):
        clz = self.class_name()
        filename = f"{constants.data_base_dir}/{constants.run_start_time_id}_{clz}_chat_log_fallback.csv"
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
        return self.execute_and_fallback_on_error(lambda worker: worker.stream_chat(messages, **kwargs))
    
    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self.execute_and_fallback_on_error(lambda worker: worker.complete(prompt, formatted, **kwargs))
    
    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        return self.execute_and_fallback_on_error(lambda worker: worker.stream_complete(prompt, formatted, **kwargs))
    
    @classmethod
    def class_name(cls) -> str:
        return "MultiLlmFallback"
    
    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return self.llm_prio_list[0].metadata
