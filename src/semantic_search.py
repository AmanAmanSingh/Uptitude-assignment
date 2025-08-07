import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from openai import OpenAI

load_dotenv()

# Load OpenAI API Key
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def semantic_clause_search(query: str, top_k: int = 3):
    # Initialize embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Connect to Qdrant
    client = QdrantClient(url="http://localhost:6333")

    # Load vector store (same collection used in indexing)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="cuad_contracts",
        embedding=embeddings,
    )

    # Perform semantic search
    results = vector_store.similarity_search(query, k=top_k)

    print(f'\n🔍 Top {top_k} results for: "{query}"\n')
    for i, doc in enumerate(results, 1):
        print(f"{i}. 📄 Source: {doc.metadata.get('source')}")
        print(doc.page_content)
        print("-" * 80)

    #  Send to LLM for refinement/summary
    llm_refine_results_with_summary(query, results)


def llm_refine_results_with_summary(query: str, documents: list):
    print("\n🧠 Asking LLM to refine/answer based on top results...\n")

    combined_context = "\n\n".join([doc.page_content for doc in documents])[
        :12000
    ]  # limit for token safety

    prompt = f"""
You are a legal assistant. Given the following contract excerpts and the user query, provide a concise, human-readable answer.

Query:
"{query}"

Contract Excerpts:
\"\"\"
{combined_context}
\"\"\"

Answer:"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        print(answer)

    except Exception as e:
        print(f"❌ Error from LLM: {e}")


if __name__ == "__main__":
    user_query = input("🔎 Enter your query: ")
    semantic_clause_search(user_query)
