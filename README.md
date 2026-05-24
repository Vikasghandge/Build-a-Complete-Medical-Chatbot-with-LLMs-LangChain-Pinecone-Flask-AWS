# Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS
This is GenAI project to create Medical ChatBot
# 🏥 Medical RAG Chatbot using LangChain, Pinecone & OpenAI

## 📌 Project Overview

This project is a **Medical RAG (Retrieval-Augmented Generation) Chatbot** built using:

* LangChain
* Pinecone Vector Database
* OpenAI GPT Models
* HuggingFace Embeddings
* Python

The main goal of this project is to create an AI chatbot that can:

* Read medical PDF documents
* Convert medical text into embeddings
* Store embeddings inside Pinecone
* Retrieve relevant medical information
* Generate accurate answers using GPT models

Instead of giving general AI answers, this chatbot answers only from the uploaded medical documents.

---

# ⚙️ Requirements Installation

## 📌 Why We Need Requirements

Before starting the project, we need to install all required Python libraries.

These libraries help us with:

* LLM integration
* Vector database connection
* PDF reading
* Embedding generation
* Prompt handling
* RAG pipeline creation

---

## 📦 Install Dependencies

```bash
pip install langchain==0.3.26
pip install flask==3.1.1
pip install sentence-transformers==4.1.0
pip install pypdf==5.6.1
pip install python-dotenv==1.1.0
pip install langchain-pinecone==0.2.8
pip install langchain-openai==0.3.24
pip install langchain-community==0.3.26
```

---

# 🔐 Environment Variables Setup

## 📌 Why We Use .env File

We never hardcode API keys directly inside the code.

Instead, we store secrets inside a `.env` file.

This helps in:

* Better security
* Cleaner code
* Easy environment management

---

## 📄 Create `.env` File

```env
PINECONE_API_KEY="your_pinecone_api_key"
OPENAI_API_KEY="your_openai_api_key"
```

---

# 📥 Load Environment Variables

## 📌 Why We Load Environment Variables

Python cannot automatically read `.env` values.

So we use `load_dotenv()` to load API keys into the application.

---

## ✅ Code

```python
from dotenv import load_dotenv

load_dotenv()
```

---

# 📚 Load Medical PDF

## 📌 Why We Load PDF

The chatbot needs medical knowledge.

So we load the medical PDF document which contains all medical information.

This PDF becomes the knowledge source for our chatbot.

---

## ✅ Code

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../data/Medical_book.pdf")

extracted_data = loader.load()
```

---

# ✂️ Split Text into Chunks

## 📌 Why Text Splitting is Required

LLMs cannot process huge PDF content directly.

So we split the PDF into smaller chunks.

Benefits:

* Better retrieval
* Better embeddings
* Faster searching
* Reduced token usage

---

## ✅ Code

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20
)

texts_chunks = text_splitter.split_documents(extracted_data)
```

---

# 🧠 Generate Embeddings

## 📌 What Are Embeddings?

Embeddings are vector representations of text.

They convert text into numerical form so that AI systems can understand semantic meaning.

Example:

* Similar medical concepts get similar vectors
* Helps in semantic search

---

## 📌 Why We Use HuggingFace Embeddings

We use HuggingFace embedding models because:

* Free and open source
* Good semantic understanding
* Fast performance
* Works well with RAG systems

---

## ✅ Code

```python
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name
)
```

---

# 🗂️ Setup Pinecone Vector Database

## 📌 Why We Use Pinecone

Pinecone is a vector database.

It stores embeddings and helps in:

* Fast similarity search
* Semantic retrieval
* Large-scale vector storage

Instead of searching plain text, Pinecone searches vector similarity.

---

## ✅ Create Pinecone Client

```python
from pinecone import Pinecone
import os

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)
```

---

# 📌 Create Vector Index

## 📌 Why We Create an Index

An index is like a database collection.

It stores all embedding vectors.

---

## ✅ Code

```python
index_name = "medical-chatbot"
```

---

# 📤 Store Embeddings into Pinecone

## 📌 What This Step Does

This step:

* Takes text chunks
* Converts them into embeddings
* Stores vectors inside Pinecone

This is one of the most important steps in the RAG pipeline.

---

## ✅ Code

```python
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunks,
    embedding=embeddings,
    index_name=index_name
)
```

---

# 🔎 Create Retriever

## 📌 What is a Retriever?

Retriever searches relevant information from Pinecone.

Whenever the user asks a question:

* Retriever searches similar vectors
* Finds relevant medical content
* Sends context to the LLM

---

## ✅ Code

```python
retriever = docsearch.as_retriever()
```

---

# 🤖 Setup OpenAI Chat Model

## 📌 Why We Use GPT Model

Retriever only finds relevant context.

GPT model generates human-like answers.

It combines:

* User question
* Retrieved medical context
* AI reasoning

Then generates the final answer.

---

## ✅ Code

```python
from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(
    model="gpt-4o"
)
```

---

# 📝 Create Prompt Template

## 📌 Why Prompt Engineering is Important

Prompt controls chatbot behavior.

This prompt tells the chatbot:

* Answer only from retrieved context
* Keep answers short
* Do not hallucinate
* Say "I don't know" if answer is unavailable

---

## ✅ Code

```python
from langchain.prompts import ChatPromptTemplate

system_prompt = (
    "You are an Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise. "
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
```

---

# 🔗 Create RAG Chain

## 📌 What is a RAG Chain?

RAG means:

**Retrieval-Augmented Generation**

Flow:

User Question → Retriever → Relevant Context → GPT → Final Answer

This combines:

* Retriever
* Prompt
* LLM model

into one pipeline.

---

## ✅ Import Required Chains

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
```

---

## ✅ Create QA Chain

```python
question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)
```

---

## ✅ Create Retrieval Chain

```python
rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)
```

---

# 🧪 Test the Chatbot

## 📌 What This Step Does

This step sends a medical question to the chatbot.

Then:

1. Retriever searches relevant medical content
2. GPT generates the answer
3. Final response gets printed

---

## ✅ Code

```python
response = rag_chain.invoke(
    {"input": "what is Acromegaly and gigantism?"}
)

print(response["answer"])
```

---

# 🏗️ Complete Architecture Flow

```text
Medical PDF
     ↓
PDF Loader
     ↓
Text Splitting
     ↓
Embeddings Generation
     ↓
Store Embeddings in Pinecone
     ↓
Create Retriever
     ↓
User Question
     ↓
Retriever Finds Similar Context
     ↓
Prompt + Context Sent to GPT
     ↓
GPT Generates Final Answer
```

---

# ❌ Errors Faced During Development

## 1. NameError

### Reason

Variables disappeared after notebook restart.

### Solution

Run all notebook cells again.

---

## 2. Pinecone API Error

### Reason

API key not loaded properly.

### Solution

Use `.env` file and `load_dotenv()`.

---

## 3. OpenAI RateLimit Error

### Reason

Quota exceeded.

### Solution

Add billing or generate new API key.

---

## 4. ModuleNotFoundError

### Reason

Required packages were missing.

### Solution

Install packages using pip.

---

# 💡 Key Learnings from This Project

Through this project, the following concepts were learned:

* LangChain framework
* Vector databases
* Pinecone indexing
* Embedding generation
* Prompt engineering
* OpenAI integration
* RAG architecture
* Semantic search
* LLM pipelines
* AI chatbot development

---

# 📄 Resume Description

Built a Medical RAG Chatbot using LangChain, Pinecone, OpenAI, and HuggingFace embeddings. Implemented PDF document loading, text chunking, vector embeddings generation, Pinecone vector storage, semantic retrieval, prompt engineering, and GPT-based response generation using Retrieval-Augmented Generation architecture.

---

# 🎤 Interview Explanation

“I worked on a Medical RAG Chatbot project where medical PDF documents were converted into embeddings using HuggingFace models and stored inside Pinecone vector database. When users ask questions, the retriever fetches relevant medical content and GPT generates the final answer. I worked on LangChain pipelines, vector databases, embeddings, prompt engineering, and end-to-end RAG implementation.”

---

# 🚀 Final Output

The final system successfully:

✅ Reads medical PDFs

✅ Stores vector embeddings

✅ Retrieves relevant medical information

✅ Generates AI-based medical answers

✅ Uses complete RAG architecture
