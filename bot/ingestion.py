import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
import datasets
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from typing import List, Optional  # Import List and Optional for type annotations
from langchain.docstore.document import Document as LangchainDocument


EMBEDDING_MODEL_NAME = "thenlper/gte-small"

pd.set_option("display.max_colwidth", None)

def read_token_from_file(file_path="token.txt"):
    with open(file_path, "r") as file:
        return file.read().strip()

# We use a hierarchical list of separators specifically tailored for splitting Markdown documents
# This list is taken from LangChain's MarkdownTextSplitter class
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] ,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def process_documents():
    # Load dataset from Hugging Face
    ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")

    # Convert raw knowledge base into LangchainDocument format
    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in ds
    ]

    # Process documents (split into chunks)
    docs_processed = split_documents(
        512,  # Chunk size adapted to the model's capabilities
        RAW_KNOWLEDGE_BASE,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )
    return docs_processed

def add_embeddings_to_qdrant(docs_processed):
    """Create embeddings for documents and add them to Qdrant."""
    
    URL=read_token_from_file("keys/qdrant_URL.txt") 
    API_KEY=read_token_from_file("keys/qdrant.txt")

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,  # Enable multiprocessing
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    # # Initialize Qdrant client
    # qdrant_client = QdrantClient(
    #     url=URL, 
    #     api_key=API_KEY,
    # )

    # Initialize Qdrant vector store
    vector_store = QdrantVectorStore.from_documents(
        docs_processed,
        embedding_model,
        url = URL,
        prefer_grpc=True,
        api_key=API_KEY,
        collection_name="test_1",
        force_recreate = True
    )

if __name__ == '__main__':
    # Only the main process should execute this
    print(f"Model's maximum sequence length: {SentenceTransformer(EMBEDDING_MODEL_NAME).max_seq_length}")

    docs_processed = process_documents()

    print("Created embedding model")
    add_embeddings_to_qdrant(docs_processed)

    print("Documents and their embeddings have been successfully added to Qdrant.")
