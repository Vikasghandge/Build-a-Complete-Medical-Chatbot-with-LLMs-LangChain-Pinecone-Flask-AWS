# Groq + LangChain Setup Guide

## Step 1: Create a Groq API Key

1. Open the Groq Console:

https://console.groq.com/keys

2. Sign in using your account.

3. Click on:

```text
Create API Key
```

4. Copy the generated API key.

Example:

```text
gsk_xxxxxxxxxxxxxxxxxxxxxxxxx
```

---

# Step 2: Create Project Folder

Create a new folder for your project.

Example:

```text
grok-project
```

Open this folder in VS Code.

---

# Step 3: Create Virtual Environment (Recommended)

Open terminal inside VS Code and run:

```bash
python -m venv venv
```

Activate virtual environment.

## Windows

```bash
venv\Scripts\activate
```

## Linux / Mac

```bash
source venv/bin/activate
```

---

# Step 4: Install Required Packages

Install LangChain, Groq integration, and dotenv.

```bash
pip install langchain langchain-groq python-dotenv
```

---

# Step 5: Create `.env` File

Inside your project folder create a file named:

```text
.env
```

Add your Groq API key inside it.

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxx
```

Important:

* Do not use quotes
* Do not add spaces
* Keep `.env` secret

---

# Step 6: Create Python File

Create a file named:

```text
app.py
```

---

# Step 7: Write Python Code

Add the following code inside `app.py`.

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

# Load environment variables
load_dotenv(".env")

# Create LLM model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Ask question
response = llm.invoke("Hello")

# Print response
print(response.content)
```

---

# Step 8: Run the Project

Run the Python file.

```bash
python app.py
```

---

# Expected Output

Example output:

```text
Hello! How can I assist you today?
```

---

# Project Structure

```text
grok-project/
│
├── .env
├── app.py
└── venv/
```

---

# Explanation of Code

## Load Environment Variables

```python
load_dotenv(".env")
```

Loads API key from `.env` file.

---

## Create LLM

```python
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)
```

Initializes Groq model using LangChain.

---

## Ask Question

```python
response = llm.invoke("Hello")
```

Sends prompt to the AI model.

---

## Print Response

```python
print(response.content)
```

Displays AI response.

---

# Available Groq Models

You can use:

```python
model="llama-3.1-8b-instant"
```

```python
model="llama-3.3-70b-versatile"
```

```python
model="gemma2-9b-it"
```

---

# Common Errors

## 1. Invalid API Key

Error:

```text
401 Unauthorized
```

Fix:

* Check API key
* Generate new key
* Verify `.env` file

---

## 2. API Key Not Found

Error:

```text
GroqError: api_key must be set
```

Fix:

```python
load_dotenv(".env")
```

Make sure `.env` exists.

---

## 3. Model Decommissioned

Error:

```text
model_decommissioned
```

Fix:
Use active model:

```python
model="llama-3.1-8b-instant"
```

---

# Interactive Chatbot Example

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv(".env")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    response = llm.invoke(question)

    print("AI:", response.content)
```

---

# Run Chatbot

```bash
python app.py
```

Example:

```text
You: What is Kubernetes?
AI: Kubernetes is a container orchestration platform...
```

---

# Summary

You learned:

* How to generate Groq API key
* How to store key in `.env`
* How to load environment variables
* How to initialize LangChain Groq model
* How to ask questions to LLM
* How to build a simple chatbot
