import os
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Directory to store downloaded models and embeddings
MODELS_DIR = os.path.expanduser("~/models_cache")
os.makedirs(MODELS_DIR, exist_ok=True)

# File to persist FAISS vector store
VECTOR_STORE_FILE = "vector_store.pkl"

class SimpleRAGApp:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None

        print("Setting up RAG Application...")

        # Load sentence-transformer embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=os.path.join(MODELS_DIR, "embeddings")
        )

        # Load Phi-2 LLM model for text generation
        self._setup_llm()

    def _setup_llm(self):
        model_id = "microsoft/phi-2"

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=os.path.join(MODELS_DIR, "llm"),
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=os.path.join(MODELS_DIR, "llm"),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            top_p=0.95
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

    def load_pdf(self, pdf_path):
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file '{pdf_path}' not found!")
            return False

        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        print("Splitting PDF into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")

        if os.path.exists(VECTOR_STORE_FILE):
            print("✅ Loading cached vector store...")
            with open(VECTOR_STORE_FILE, "rb") as f:
                self.vector_store = pickle.load(f)
            self.vector_store.add_documents(chunks)
        else:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)

        with open(VECTOR_STORE_FILE, "wb") as f:
            pickle.dump(self.vector_store, f)
        print("✅ Vector store cached successfully!")

        custom_prompt_template = """
You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer the question directly and concisely based on the context provided:
"""
        PROMPT = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        print("PDF processed and indexed successfully!")
        return True

    def ask_question(self, question):
        if self.qa_chain is None:
            print("Please load a PDF first using the 'load_pdf' function")
            return None

        print(f"Question: {question}")
        print("Searching for relevant information...")

        try:
            result = self.qa_chain.invoke({"query": question})
        except Exception as e:
            print(f"Error: {e}")
            result = self.qa_chain({"query": question})

        return {
            "answer": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        }

def main():
    app = SimpleRAGApp()

    while True:
        print("\n" + "=" * 50)
        print("Simple RAG App - PDF Question Answering")
        print("=" * 50)
        print("1. Load PDF")
        print("2. Ask Question")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ")

        if choice == "1":
            pdf_path = input("Enter PDF file path: ")
            app.load_pdf(pdf_path)

        elif choice == "2":
            if app.qa_chain is None:
                print("Please load a PDF first!")
                continue

            question = input("Enter your question: ")
            result = app.ask_question(question)

            print("\nAnswer:")
            print(result["answer"])

            print("\nSources:")
            for i, source in enumerate(result["sources"], 1):
                print(f"\nSource {i}:")
                print(source[:300] + "..." if len(source) > 300 else source)

        elif choice == "3":
            print("Thank you for using Simple RAG App!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
