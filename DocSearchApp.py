from fastapi import FastAPI
from Document_processor import process_and_store_pdf
import os
import pickle

# Define the file name used to persist the vector store locally
VECTOR_STORE_FILE = "vector_store.pkl"

app = FastAPI()

# Attempt to load the vector store from a pickle file if it exists; otherwise, process a PDF and create embeddings
if os.path.exists(VECTOR_STORE_FILE):
    print("✅ Loading cached vector store from disk...")
    with open(VECTOR_STORE_FILE, "rb") as f:
        vector_store = pickle.load(f)
else:
    print("🚀 Processing PDF and creating embeddings...")
    vector_store = process_and_store_pdf("/Users/ishitaagrawal/Documents/PDFs/sample.pdf")

# Health check endpoint to confirm the API is running
@app.get("/")
def read_root():
    return {"message": "AI Search Engine Running!"}

# Endpoint for querying similar content from the embedded PDF data using a search query
@app.get("/search")
def search(query: str):
    results = vector_store.similarity_search(query)
    return {
        "results": [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in results
        ]
    }

# Entry point to run the FastAPI application using uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
