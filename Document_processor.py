import os
import pickle
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# File name used to cache the generated vector store locally
VECTOR_STORE_FILE = "vector_store.pkl"

def process_and_store_pdf(pdf_path):
    """Process the PDF, split it into chunks, generate embeddings using Hugging Face, and store in Qdrant."""

    # Load vector store from disk if already cached
    if os.path.exists(VECTOR_STORE_FILE):
        print("✅ Loading cached embeddings from disk...")
        with open(VECTOR_STORE_FILE, "rb") as f:
            return pickle.load(f)

    print(f"🚀 Processing PDF: {pdf_path}")

    # Load PDF content and split into overlapping text chunks for better semantic indexing
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    print(f"Total document chunks: {len(docs)}")

    # Initialize a Hugging Face embedding model for generating numerical vectors from text
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Convert split documents into LangChain-compatible Document objects for structured indexing
    langchain_docs = [Document(page_content=doc.page_content) for doc in docs]

    # Initialize an in-memory Qdrant vector store with the processed documents and their embeddings
    client = QdrantClient(":memory:")
    vector_store = Qdrant.from_documents(
        documents=langchain_docs,
        embedding=embeddings_model,
        location=":memory:",
        collection_name="pdf_documents"
    )

    # Persist the vector store to disk for reuse across sessions
    with open(VECTOR_STORE_FILE, "wb") as f:
        pickle.dump(vector_store, f)

    print("✅ Local Embeddings Cached.")
    return vector_store
