# WeSee Grounded Answer Engine

A production-ready FastAPI service designed to act as a grounded answer engine for WeSee documents. The engine leverages OpenAI's `gpt-4o-mini` with native Pydantic structured outputs to answer queries strictly based on provided documents, verifies citations with a strict validation backstop, handles prompt injections safely, and provides a simple, beautiful single-page frontend.

## Setup

1. **Clone the repository** (or navigate to the workspace directory).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure the environment**:
   Create a `.env` file in the root directory and add your OpenAI API Key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```
4. **Place document files**:
   Ensure the 10 markdown document files (`*.md`) are placed in the `docs/` folder.
5. **Place evaluation questions**:
   Ensure `questions.json` is placed inside the `eval/` folder.
6. **Run the application**:
   Start the FastAPI server (which also hosts the frontend UI at the root `/`):
   ```bash
   uvicorn main:app --reload
   ```
   *Alternatively, run `run.bat` (Windows) or `./run.sh` (Linux/macOS).*

## Usage

You can access the dynamic interactive web interface by opening:
`http://localhost:8000/`

To test the API programmatically, you can send a `POST` request to `http://localhost:8000/ask`.

### Sample Curl Request
```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "How much does the Growth plan cost per month?"}'
```

### Expected Response Shape
```json
{
  "answer": "The Growth plan costs 9,000 INR/month.",
  "citations": [
    {
      "doc": "02_pricing_and_plans.md",
      "quote": "- **Growth** — 9,000 INR/month."
    }
  ],
  "answered": true
}
```

## Evaluation

To run the automated evaluation suite against the questions dataset:
```bash
python eval.py
```

### Results Table
| Category | Pass Rate | Score |
| --- | --- | --- |
| Grounded | 100% | 9/9 |
| Refusal | 100% | 5/5 |
| Adversarial | 100% | 4/4 |
| **Overall** | **100%** | **18/18** |

## Architecture

The system utilizes a **Full-Context Stuffing** approach. Because the document corpus is small (10 short markdown files, under 10 KB total), the entire set of documents fits easily within the large context window of modern LLMs. This design completely bypasses the traditional retrieval stage (e.g., semantic search with a Vector DB, chunking, and similarity search), which frequently suffers from missing relevant chunks or precision issues on small datasets. Instead, we present the model with the entire corpus in every query. Grounding and prompt injection resistance are enforced via strict instructions in the system prompt, coupled with a secondary programmatic backstop that normalizes, strips markdown styling, and validates citations verbatim before returning them to the user.
