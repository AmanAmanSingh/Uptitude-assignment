import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
from utils import extract_text_from_pdf


load_dotenv()

# Initialize OpenAI client using your key
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_clauses_with_llm(contract_text: str, contract_id: str) -> dict:
    prompt = f"""
You are a legal AI assistant. Extract the following clauses from the contract text provided below:
1. Termination clause
2. Confidentiality clause
3. Liability clause

Please return the output in this JSON format:
{{
  "contract_id": "{contract_id}",
  "termination_clause": "...",
  "confidentiality_clause": "...",
  "liability_clause": "..."
}}

Contract Text:
\"\"\"
{contract_text[:12000]}  # Truncate if very long
\"\"\"
"""

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    # Try parsing the JSON from LLM response
    try:
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"❌ Error parsing LLM response for {contract_id}: {e}")
        return {
            "contract_id": contract_id,
            "termination_clause": "",
            "confidentiality_clause": "",
            "liability_clause": "",
        }


def extract_clauses_from_all_pdfs(pdf_dir: Path, output_path: str):
    results = []
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("⚠️ No PDF files found.")
        return

    for pdf_file in pdf_files:
        print(f"📄 Processing: {pdf_file.name}")
        contract_text = extract_text_from_pdf(str(pdf_file))
        clauses = extract_clauses_with_llm(contract_text, pdf_file.name)
        results.append(clauses)

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"✅ Extracted clauses from {len(results)} contracts and saved to {output_path}"
    )


if __name__ == "__main__":
    pdf_directory = Path(__file__).parent.parent / "CUAD_subset"
    output_file = Path(__file__).parent.parent / "extracted_clauses.json"
    extract_clauses_from_all_pdfs(pdf_directory, output_file)
