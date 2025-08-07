from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()


def load_and_index_documents(pdf_dir: Path, qdrant_url: str, collection_name: str):
    # STEP-1 Initialize OpenAI Embedding Model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

    # STEP-2 Connect to Qdrant instance
    client = QdrantClient(url=qdrant_url)

    # STEP-3 Ensure collection exists (create if not)
    # Note: We use a dummy embedding to get the embedding size
    dummy_vector_size = len(embedding_model.embed_query("test"))
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dummy_vector_size, distance=Distance.COSINE),
    )

    # STEP-4 Prepare LangChain Vector Store with Qdrant client
    vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=embedding_model
    )

    # STEP-5 Configure text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)

    # STEP-6 Get all PDF files in directory
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠️ No PDF files found.")
        return

    # STEP-7 Loop through each PDF and process
    for pdf_file in pdf_files:
        print(f"📄 Processing: {pdf_file.name}")

        # Load and parse PDF using LangChain loader
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        # Add unique metadata to each document
        for doc in docs:
            doc.metadata["source"] = pdf_file.name

        # Split documents into smaller chunks
        split_docs = text_splitter.split_documents(docs)
        print(f"🔗 {len(split_docs)} chunks created from {pdf_file.name}")

        # Store chunks in vector DB
        vector_store.add_documents(split_docs)

    print(
        f"✅ Successfully indexed {len(pdf_files)} PDFs into collection '{collection_name}'"
    )


if __name__ == "__main__":
    pdf_directory = Path(__file__).parent.parent / "CUAD_subset"
    qdrant_url = "http://localhost:6333"
    collection = "cuad_contracts"

    load_and_index_documents(pdf_directory, qdrant_url, collection)
