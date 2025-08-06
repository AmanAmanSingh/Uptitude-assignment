from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

import os

load_dotenv()

def load_and_index_documents(pdf_dir: Path, qdrant_url: str, collection_name: str):
    all_docs = []

    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"📄 Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        # Add metadata to each document
        for doc in docs:
            doc.metadata["source"] = pdf_file.name  # Unique & meaningful

        all_docs.extend(docs)

    if not all_docs:
        print("⚠️ No documents found. Exiting.")
        return

    print(f"✅ Loaded {len(all_docs)} pages from {len(list(pdf_dir.glob('*.pdf')))} PDFs")

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400
    )
    split_docs = text_splitter.split_documents(all_docs)

    print(f"🔗 Created {len(split_docs)} chunks")

    # Embeddings
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

    # Store into ONE collection with metadata
    vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        url=qdrant_url,
        collection_name=collection_name,
        embedding=embedding_model
    )

    print("✅ All documents indexed into collection:", collection_name)

if __name__ == "__main__":
    pdf_directory = Path(__file__).parent / "CUAD_subset"
    qdrant_url = "http://localhost:6333"
    collection = "cuad_contracts"  

    load_and_index_documents(pdf_directory, qdrant_url, collection)
