import os
import torch
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
import streamlit as st

# Load environment variables
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACE_API_KEY

lc_embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

embed_model = LangchainEmbedding(lc_embed_model)

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    device_map="auto",
    model_kwargs={
        "torch_dtype": torch.float16,
        "load_in_8bit": False,
        "use_auth_token": HUGGINGFACE_API_KEY
    }
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900


storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()


st.title('Financial Stock Analysis using LlamaIndex')

st.header("Reports:")

report_type = st.selectbox(
    'What type of report do you want?',
    ('Single Stock Outlook', 'Competitor Analysis'))


if report_type == 'Single Stock Outlook':
    symbol = st.text_input("Stock Symbol")

    if symbol:
        with st.spinner(f'Generating report for {symbol}...'):
            response = query_engine.query(f"Write a report on the outlook for {symbol} stock from the years 2023-2027. Be sure to include potential risks and headwinds.")
            print(type(response))

            st.write(str(response))
            

if report_type == 'Competitor Analysis':
    symbol1 = st.text_input("Stock Symbol 1")
    symbol2 = st.text_input("Stock Symbol 2")

    if symbol1 and symbol2:
        with st.spinner(f'Generating report for {symbol1} vs. {symbol2}...'):
            response = query_engine.query(f"Write a report on the competition between {symbol1} stock and {symbol2} stock.")

            st.write(str(response))




