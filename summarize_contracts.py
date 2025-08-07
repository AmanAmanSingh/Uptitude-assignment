import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
from utils import extract_text_from_pdf


# Load environment variables (for OpenAI key)
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_contract_summary(contract_text: str, contract_id: str) -> dict:
    """Uses OpenAI LLM to summarize the contract based on key points."""

    prompt = f"""
You are a legal contract summarization assistant.

Please read the contract text below and generate a concise summary (100–150 words) covering:
- Purpose of the agreement
- Key obligations of each party
- Notable risks or penalties

Return in this format:
{{
  "contract_id": "{contract_id}",
  "summary": "Your summary here..."
}}

Contract Text:
\"\"\"
{contract_text[:12000]}  # truncated to avoid token limits
\"\"\"
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"❌ Error summarizing {contract_id}: {e}")
        return {
            "contract_id": contract_id,
            "summary": "Error generating summary"
        }

def summarize_all_contracts(pdf_dir: Path, output_path: str):
    """Process each PDF and save summaries to JSON."""
    summaries = []
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("⚠️ No PDF files found.")
        return

    for pdf_file in pdf_files:
        print(f"📄 Summarizing: {pdf_file.name}")
        contract_text = extract_text_from_pdf(str(pdf_file))
        summary = generate_contract_summary(contract_text, pdf_file.name)
        summaries.append(summary)

    with open(output_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"✅ Saved summaries of {len(summaries)} contracts to {output_path}")

if __name__ == "__main__":
    pdf_directory = Path(__file__).parent / "CUAD_subset"
    output_file = "contract_summaries.json"
    summarize_all_contracts(pdf_directory, output_file)
