Overview

This project is a collection of independent Python scripts that demonstrate various AI and NLP capabilities, including:

Retrieval-Augmented Generation (RAG) with PDFs
Embedding-based document search using FAISS
Custom assistant agents using Hugging Face transformers
Fine-tuning a sentiment analysis model using BERT
Each script is standalone and can be run individually based on your use case.

File Descriptions

📄 Autogen.py
Description:
Demonstrates a multi-agent conversation system using AutoGen, with agents that respond using a Hugging Face GPT-2 model. Includes role-based agents like a Project Manager and Software Engineer in a simulated chat loop.

To run:

python Autogen.py

📄 DocSearchApp.py
Description:
FastAPI application that allows users to search a preprocessed PDF using embedding-based similarity. Loads from a precomputed vector store or processes the PDF if not available.

To run:

uvicorn DocSearchApp:app --reload

📄 Document_processor.py
Description:
(Internal Use) Used by other scripts to load a PDF, split it into chunks, embed the text using HuggingFace models, and store it in a FAISS vector store. Not meant to be executed directly.

📄 FineTuneSentimentTrainer.py
Description:
Fine-tunes a distilbert-base-uncased model for binary sentiment classification. Includes tokenization, training, early stopping, saving, and inference testing on example text.

To run:

python FineTuneSentimentTrainer.py

📄 SimpleRAGApp.py
Description:
Implements a console-based Retrieval-Augmented Generation (RAG) app. Loads a PDF, embeds the content, and allows users to ask questions and get answers from the context using Microsoft's Phi-2 language model.

To run:

python SimpleRAGApp.py

🧾 Dependencies

Install them using:

pip install -r requirements.txt
Some major packages used:

transformers
sentence-transformers
torch
langchain
faiss-cpu
uvicorn
fastapi
For best performance with LLMs, ensure your system supports CUDA if using GPU.
