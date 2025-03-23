Overview

This project is a collection of independent Python scripts that demonstrate various AI and NLP capabilities, including:

Retrieval-Augmented Generation (RAG) with PDFs
Embedding-based document search using FAISS
Custom assistant agents using Hugging Face transformers
Fine-tuning a sentiment analysis model using BERT
Each script is standalone and can be run individually based on your use case.
#-------------------------------------------------------------------------
🧾 First Install Dependencies

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

#------------------------------------------------------------
File Descriptions

📄 Autogen.py
Description:
Demonstrates a multi-agent conversation system using AutoGen, with agents that respond using a Hugging Face GPT-2 model. Includes role-based agents like a Project Manager and Software Engineer in a simulated chat loop.

To run:

python Autogen.py

Note - You will get response like below

Device set to use cpu

--- Round 1 ---
UserAgent:


--- Round 2 ---
ProjectManagerAgent:
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
For the first four months of 2015, the FBI collected nearly $25 billion in wiretaps. The FBI, for instance, collected over $1 billion from individuals and businesses associated with the companies they target.

We're also seeing evidence of widespread

--- Round 3 ---
AssistantAgent:
For the first four months of 2015, the FBI collected nearly $25 billion in wiretaps. The FBI, for instance, collected over $1 billion from individuals and businesses associated with the companies they target.

We're also seeing evidence of widespread domestic spying, including the NSA gathering data from hundreds of thousands of phone carriers that also include Verizon and AT&T. In addition, Verizon reportedly used the NSA data collection to build its own spyware and to eavesdrop on millions of American Internet users

--- Round 4 ---
SoftwareEngineerAgent:
For the first four months of 2015, the FBI collected nearly $25 billion in wiretaps. The FBI, for instance, collected over $1 billion from individuals and businesses associated with the companies they target.

We're also seeing evidence of widespread domestic spying, including the NSA gathering data from hundreds of thousands of phone carriers that also include Verizon and AT&T. In addition, Verizon reportedly used the NSA data collection to build its own spyware and to eavesdrop on millions of American Internet users, in violation of the Espionage Act.

As of June, the NSA collected over 14 billion phone calls, almost all of them from Americans. A government watchdog report has the NSA collecting over 17 billion telephone calls for each year.




#-----------------------------------------------------------------------------
📄 DocSearchApp.py
Description:
FastAPI application that allows users to search a preprocessed PDF using embedding-based similarity. Loads from a precomputed vector store or processes the PDF if not available.

To run:

uvicorn DocSearchApp:app --reload or python DocSearchApp

Then access with search query : http://localhost:8000/search?query=newton%27s%20first%20law

Try clearing cache if you are seeing old data: rm vector_store.pkl 

📄 Document_processor.py
Description:
(Internal Use) Used by other scripts to load a PDF, split it into chunks, embed the text using HuggingFace models, and store it in a FAISS vector store. Not meant to be executed directly.
#-------------------------------------------------------
📄 FineTuneSentimentTrainer.py
Description:
Fine-tunes a distilbert-base-uncased model for binary sentiment classification. Includes tokenization, training, early stopping, saving, and inference testing on example text.

To run:

python FineTuneSentimentTrainer.py

Note- You will get below response.

Text: I'm really impressed with the quality!
Sentiment: Positive

Text: This product is absolutely useless.
Sentiment: Negative
#----------------------------------------------------------------------------
📄 SimpleRAGApp.py
Description:
Implements a console-based Retrieval-Augmented Generation (RAG) app. Loads a PDF, embeds the content, and allows users to ask questions and get answers from the context using Microsoft's Phi-2 language model.

To run:

python SimpleRAGApp.py

You will be asked the below questions:
1. Load PDF
2. Ask Question
3. Exit

What would be your response:

Press 1.
Write --> sample.pdf

Press 2.
Ask --> Give me one theory in physics? # or any question related to the writing in sample.pdf

Note - You will get following response with contect and soruce info:

Question: Give me one theory of physics

Answer the question directly and concisely based on the context provided:

According to the context, one theory of physics is Quantum Mechanics.


#-----------------------The End-----------------------------------------------
