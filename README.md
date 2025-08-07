

# 📄 LLM-Powered Legal Contract Processing

This project demonstrates a complete document ingestion and information extraction pipeline for legal contracts using **LLMs (Large Language Models)** and **vector databases**. It processes PDF contracts, extracts key legal clauses, summarizes agreements, and enables semantic search using vector embeddings.

---


##  Steps usage instructions




### 1. Clone the repo
```bash
https://github.com/AmanAmanSingh/Uptitude-assignment.git
```

### 2. Go to correct file path
    cd Uptitude-assignment

## 3. Environment Variables

To run this project, you will need to add the environment variables to your .env file
```bash
OPENAI_API_KEY=abc****xyz
```

## 4. Install dependencies
```bash
python3 -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

## 5. Run Qdrant vector DB
Make sure Docker Desktop is running on your machine. Then, launch Qdrant using:
```bash
docker-compose up -d
```



## 🚀 How to Run Or Test project working 

🔹 Step 1: Index the documents into Qdrant:

```bash
python3 src/indexing.py
```

🔹 Step 2: Extract legal clauses using LLM:
```bash
python3 src/extract_clauses.py

```


🔹 Step 3: Generate contract summaries using LLM:
```bash
python3 src/summarize_contracts.py
```

🔹 Step 4: Perform semantic search over contracts :
```bash
python3 src/semantic_search.py
```

##  Output

| File                      | Description                                  |
| ------------------------- | -------------------------------------------- |
| `extracted_clauses.json`  | Extracted legal clauses for each contract    |
| `contract_summaries.json` | Concise LLM-generated summaries per contract |


## Each entry in the JSON contains:

```bash
{
  "contract_id": "Sample.pdf",
  "summary": "...",
  "termination_clause": "...",
  "confidentiality_clause": "...",
  "liability_clause": "..."
}
```
## 🛠 Tech Stack
- LangChain – LLM abstraction & document processing

- OpenAI API – GPT-3.5 for clause extraction and summarization

- Qdrant – Vector search engine

- PyPDFLoader – PDF parsing

- Docker – Containerized Qdrant service
## Tech Stack
**Server:** Python

