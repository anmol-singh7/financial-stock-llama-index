import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
print("Loading environment variables...")
load_dotenv()

# Set up HuggingFace API key
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
os.environ['HUGGINGFACE_API_KEY'] = HUGGINGFACE_API_KEY
print("HuggingFace API Key set from .env file.")

# Get the root folder and articles folder paths
root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
articles_folder = os.path.join(root_folder, 'articles')

# Ensure the articles folder exists
if not os.path.exists(articles_folder):
    print(f"Error: The articles folder '{articles_folder}' does not exist.")
else:
    print(f"Articles folder found at: {articles_folder}")

# Load documents from the articles folder
print("Loading documents from the articles folder...")
documents = SimpleDirectoryReader(articles_folder).load_data()
print(f"Loaded {len(documents)} documents.")

# Define embedding model
print("Downloading the embedding model...")
lc_embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
print("Embedding model downloaded successfully.")

# Create Langchain embedding model
print("Creating Langchain embedding...")
embed_model = LangchainEmbedding(lc_embed_model)
print("Langchain embedding created.")

# Create index with embedding model
print("Creating the vector store index...")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
print("Index created successfully.")

# Ensure the storage directory exists or create it
storage_dir = os.path.join(root_folder, 'storage')
if not os.path.exists(storage_dir):
    print(f"Storage directory not found. Creating storage directory at: {storage_dir}")
    os.makedirs(storage_dir)
else:
    print(f"Storage directory found at: {storage_dir}")

# Persist index
print(f"Persisting the index in the storage directory: {storage_dir}")
index.storage_context.persist(persist_dir=storage_dir)

print("Index created and persisted successfully in the 'storage' directory.")
