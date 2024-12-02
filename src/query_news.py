import os
import torch
from dotenv import load_dotenv
from llama_index import StorageContext, load_index_from_storage
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

# Load environment variables
load_dotenv()

lc_embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
print("Embedding model downloaded successfully.")

# Create Langchain embedding model
print("Creating Langchain embedding...")
embed_model = LangchainEmbedding(lc_embed_model)




llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)


storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# new version of llama index uses query_engine.query()
query_engine = index.as_query_engine()

# response = query_engine.query("What are some near-term risks to Nvidia?")
# print(response)


response = query_engine.query("Tell me about Google's new supercomputer.")
print(response)




