# Import modules
from typing import Any, Mapping
from dotenv import load_dotenv
from langchain.llms.base import LLM
from llama_index import (
    PromptHelper, 
    SimpleDirectoryReader,
    GPTListIndex,
    LLMPredictor,
    ServiceContext 
)
import torch
from transformers import pipeline
import os

# Load env handles
load_dotenv()

max_token = 256

prompt_helper = PromptHelper(
    # maximum_input size
    max_input_size=1024,
    # num of output tokens
    num_output=max_token,
    # max chunk overlap
    max_chunk_overlap=20
)

# create an oop class
class LocalCustomLLM(LLM):
    # 60 gb model from facebook / opensource
    potential_model_one = "models/opt-iml-1.3b" # 2.36 gb model
    model_name = potential_model_one

    # Define pipeline
    pipeline = pipeline("text-generation", 
                        model=model_name,
                        # device="cuda:0"
                        model_kwargs={
                            "torch_dtype": torch.bfloat16
                        }
                    )
    # constructor
    def _call(self, prompt:str, stop=None) -> str:
        # Get the first item and specifically the generated text key
        # response includes the prompt
        response = self.pipeline(prompt, max_new_tokens=max_token)[0]["generated_text"]
        # Take the length of the prompt, and start from there: Only returns the newly generated tokens
        return response[len(prompt):]

    @property
    def _identifying_params(self):
        return {"name_of_model: ": self.model_name}
    
    @property
    def _llm_type(self):
        return "custom"
    

def create_index():
    """
    Responsible for creating an index
    """
    print("Creating index")
    # wrapper around the LLMChain from LangChain
    llm = LLMPredictor(llm=LocalCustomLLM())
    service_context = ServiceContext(
        llm_predictor=llm,
        prompt_helper=prompt_helper
    )
    # Simple directory read and load the data in
    docs = SimpleDirectoryReader("documents").load_data()

    index = GPTListIndex.from_documents(docs, service_context=service_context)
    print("Done creating index")
    return index

def execute_query():
    response = index.query(
        "How does a timestamp server work?",
        # We can add words to exclude from the search for a more custom search
        exclude_keywords=["Network"],
        # Required keywords (optional)
        # required_keywords=["timestamp"],
        response_mode="no_text"
    )
    
    return response 

if __name__ == "__main__":

    # embeddings file
    filename = "embed_app_embeddings.json"
    if not os.path.exists():
        print("No local cache of the embeddings, downloading from hugging face...")
        index = create_index()
        index.save_to_disk(filename)
    else:
        print("Loading local cache of the embeddings...") 
        index = GPTListIndex.load_from_disk(filename)       
    
    response = execute_query()
    print(response)