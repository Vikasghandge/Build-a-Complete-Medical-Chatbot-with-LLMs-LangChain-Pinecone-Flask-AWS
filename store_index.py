from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# Step 1 - Load PDF
extracted_data = load_pdf("data/Medical_book.pdf")


# Step 2 - Split Text
text_chunks = text_split(extracted_data)


# Step 3 - Create Embeddings
embeddings = download_hugging_face_embeddings()


# Step 4 - Connect Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"


# Step 5 - Create Index if Not Exists
existing_indexes = [index.name for index in pc.list_indexes()]

if index_name not in existing_indexes:

    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


# Step 6 - Store Vectors
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("Pinecone Vector Store Created Successfully")
