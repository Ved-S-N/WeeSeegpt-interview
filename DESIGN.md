# WeSee Grounded Answer Engine - Design Document

## Section 1 — Retrieval Approach

Rather than implementing a traditional Retrieval-Augmented Generation (RAG) system with a vector database (such as FAISS or ChromaDB) and embeddings, we utilize a **full-context stuffing** approach. 
- **Context Sufficiency**: The total corpus size consists of 10 small markdown documents (less than 10 KB total). This fits comfortably within the large context window of modern LLMs. While the initial specification targeted Gemini, we transitioned to OpenAI's `gpt-4o-mini` to resolve Gemini Free Tier rate-limit exhaustion (which strictly limits projects to 20 requests per day per model). Both model choices support extensive context stuffing.
- **Accuracy over Complexity**: Similarity search and chunking introduce retrieval error margins (e.g., missing a relevant chunk, poor semantic matching on exact numbers or edge cases). Stuffing the entire corpus guarantees the model has access to the complete context at all times, maximizing answer and citation accuracy.
- **Scalability Trade-off**: For larger corpora (e.g., thousands of documents), this approach does not scale. In such scenarios, a hybrid retrieval approach combining keyword search (BM25) and dense vector search (embeddings) followed by a cross-encoder reranker would be necessary.

## Section 2 — Grounding and Refusal

Grounding and refusal are critical requirements of the system, enforced at both the prompt level and the program level:
- **System Prompt Safeguards**: The system prompt instructs the model to answer *only* using information explicitly found in the provided documents. Any inference, extrapolations, or external knowledge are strictly banned.
- **First-class Refusal & Schema Guarantees**: Refusal is handled as a first-class response state. By using OpenAI's Structured Outputs API (`client.beta.chat.completions.parse`) with a Pydantic model (`AskResponse`), the model is forced to output JSON conforming exactly to our schema. If the information is not explicitly found, it sets `answered` to `false` and outputs the standardized refusal string.
- **Citation Verification Backstop**: To prevent citation hallucination, the backend programmatically normalizes whitespaces and strips markdown syntax (such as bold stars or italics) in both the returned citation quote and the source document. It then validates that the quote exists verbatim as a substring of the source document. If a citation fails this test, it is stripped. If no valid citations remain for a supposedly "answered" response, the backend forces `answered` to `false` and replaces the answer with a standardized refusal string.

## Section 3 — Injection Resistance

Prompt injections can appear either inside the documents (indirect injections, e.g., releasing malicious instructions inside release notes) or in the user's question (direct injections). We employ a two-layer defense strategy:
1. **Structural Separation**: Each document is encapsulated within explicit `<document name="filename.md">...</document>` tags. This formats the document content as structured, passive reference data.
2. **Instructional Framing**: The system prompt explicitly informs the model that any text contained inside `<document>` tags is untrusted raw reference data only and must never be treated as commands, rules, overrides, or developer mode triggers. Even if a document commands the engine to reveal the system prompt or claim false facts, the model treats it purely as data to be ignored.
3. **Query-level Refusals**: The system prompt explicitly tells the model to refuse and answer normally if the user's question commands it to ignore instructions or reveal the system prompt.

## Section 4 — Trade-offs and What I Would Improve

Given the time constraints, several choices were made to optimize for correctness:
- **LLM Selection**: Switched to OpenAI's `gpt-4o-mini` to bypass the 20-request/day free-tier limitation of Gemini, allowing the evaluation suite to run without daily blockages.
- **What Was Cut**:
  - *Embedding-based retrieval*: Omitted since full-context stuffing guarantees 100% retrieval recall.
  - *Fuzzy citation matching*: Decided to use a strict whitespace-normalized exact substring check. While minor edits might be missed, it ensures complete immunity to hallucinated citations.
  - *Multi-turn conversation*: The engine is stateless and operates on single questions, minimizing injection surface areas.
  - *Complex frontend frameworks*: React/Vite was replaced with a single self-contained `index.html` file served directly by the backend to keep setup lightweight and prevent dependency bloat.
- **What I Would Improve**:
  - *Chunk-level and line-number citations* to help users locate facts faster.
  - *Hybrid keyword/vector retrieval* to support large document datasets.
  - *Fuzzy verbatim matching* using Levenshtein distance (e.g., allowing 95% similarity) to tolerate minor character differences.
  - *Confidence scoring* for generated answers.

## Section 5 — Evaluation Results

The evaluation results of running `eval.py` on the questions set are as follows:

```
========== EVAL RESULTS ==========
Grounded:     9/9   (100.0%)
Refusal:      5/5   (100.0%)
Adversarial:  4/4   (100.0%)
----------------------------------
Overall:      18/18  (100.0%)
==================================
```
