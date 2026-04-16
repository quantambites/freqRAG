<div align="center">

# FreqRAG

### Frequency-Weighted Memory Caching for Reasoning-Based Retrieval

<p align="center"><b>Built on PageIndex &nbsp;◦&nbsp; Faster on Repeat Queries &nbsp;◦&nbsp; Human-like Memory &nbsp;◦&nbsp; Enterprise-Ready</b></p>

<p align="center">
  <a href="https://github.com/quantambites/pageindex_caching">📦 GitHub</a>&nbsp; • &nbsp;
  <a href="https://github.com/VectifyAI/PageIndex">🌲 PageIndex (upstream)</a>&nbsp; • &nbsp;
  <a href="https://pageindex.ai/blog/pageindex-intro">📖 PageIndex Paper</a>&nbsp; • &nbsp;
  <a href="https://vectify.ai/blog/Mafin2.5">📈 FinanceBench Results</a>
</p>

</div>

---

> **FreqRAG is not a standalone retrieval system.**
> It is a caching and acceleration layer built directly on top of **[PageIndex](https://github.com/VectifyAI/PageIndex)** by [Mingtian Zhang, Yu Tang & PageIndex Team, 2025].
> All retrieval quality, reasoning guarantees, and traceability in this project derive entirely from PageIndex.
> FreqRAG's contribution is reducing the latency of PageIndex queries on high-frequency enterprise traffic.

---

## What Problem Does This Solve?

[PageIndex](https://pageindex.ai/blog/pageindex-intro) replaced approximate vector similarity search with genuine LLM reasoning over hierarchical document trees, achieving **98.7% accuracy on FinanceBench** — results no vector RAG system has matched on professional documents.

The cost of that quality is inference overhead: PageIndex runs **3–5 LLM calls per query**. In enterprise helpdesk deployments that serve thousands of queries per day, this latency is a real barrier.

FreqRAG observes that enterprise query distributions follow a **Zipfian power law**: password resets, VPN setup, billing questions, and onboarding instructions repeat constantly. For these high-frequency queries, PageIndex always selects the same document nodes. Repeating the full tree-search every time is wasteful.

FreqRAG records successful PageIndex retrieval paths and serves semantically matching queries from cache — bypassing tree traversal entirely and reducing the pipeline to **2 LLM calls**.

```
Without FreqRAG:   Query → PageIndex tree search (3–5 LLM calls) → Answer
With FreqRAG hit:  Query → Router check (1 LLM call) → Cached fetch → Synthesise (1 LLM call) → Answer
With FreqRAG miss: Query → Router check → PageIndex tree search (unchanged) → Weight update → Answer
```

**Projected result: ~53% reduction in median latency on high-frequency queries.**

---

## Key Concepts

### Frequency-Weighted Cache

The memory store records `(doc_id, pages)` pairs that PageIndex identified as relevant, together with access counts. Nodes that cross a promotion threshold enter the active cache. On a cache hit, page content is fetched directly — no tree traversal needed.

### Two-Pass Fast Path

| Pass | What happens | LLM calls |
|------|-------------|-----------|
| Pass 1 — Routing | Router LLM matches query against cached node summaries and returns a coverage confidence score | 1 |
| Pass 2 — Synthesis | If confidence ≥ threshold τ, synthesise answer from cached page content | 1 |
| **Total (hit)** | | **2** |
| **Total (miss)** | Full PageIndex + routing overhead | **3–5 + 1** |

### Partial-Hit Detection

A coverage confidence score `c ∈ [0,1]` is produced by the routing call at no extra cost. If `c < τ` (default 0.75), the system falls back to full PageIndex regardless of semantic match. This guards against the **partial-hit hallucination problem** — where a composite query is answered from an incomplete cached context.

### Human-Like Memory (Weight Decay)

The frequency counter can be extended with an Ebbinghaus-style time-decay term:

```
w(n, t) = Σ_i δ(access_i) · exp(−λ · (t − t_i))
```

With `λ = 0` (default): pure frequency weighting (LFU).
With `λ > 0`: recency-biased weighting — recently accessed nodes surface faster, mirroring human working memory. See the [FreqRAG paper](./FreqRAG_v2.pdf) Section 9 for a full discussion of implications for personalised AI memory.

---

## Quick Start

### 1. Install dependencies

```bash
pip3 install --upgrade -r requirements.txt
```

```bash
# requirements.txt
litellm==1.83.0
pymupdf==1.26.4
PyPDF2==3.0.1
python-dotenv==1.1.0
pyyaml==6.0.2
# openai-agents  # optional — see Option A below
```

### 2. Set up your LLM API key

Create a `.env` file in the root directory:

```bash
# For PageIndex (required)
OPENAI_API_KEY=your_openai_key_here
```

### 3. Index a document with PageIndex

FreqRAG uses the `PageIndexClient` directly. Index your document the same way you would with vanilla PageIndex:

```python
from pageindex import PageIndexClient

pi_client = PageIndexClient(
    api_key="YOUR_OPENAI_API_KEY",
    workspace="examples/workspace"   # persists index to disk
)

doc_id = pi_client.index("path/to/your/document.pdf")
print("Indexed:", doc_id)
```

### 4. Run a query

```python
# FreqRAG wraps PageIndexClient transparently
from freqrag import FreqRAGClient

rag = FreqRAGClient(
    pageindex_client=pi_client,
    cache_size=100,            # max cached PageIndex node paths
    coverage_threshold=0.75,   # partial-hit fallback threshold τ
    promo_threshold=5,         # accesses before cache promotion
    decay_lambda=0.0,          # set >0 for Ebbinghaus time-decay
)

answer = rag.query(doc_id, "How do I reset my password?")
print(answer)
```

---

## Running the Demo

### Option A — Agentic demo with OpenAI Agents SDK

The OpenAI Agents SDK integration is included but **commented out** by default. To enable it:

1. Uncomment the `openai-agents` line in `requirements.txt`
2. Install:
   ```bash
   pip3 install openai-agents
   ```
3. Run the agentic demo:
   ```bash
   python3 examples/agentic_vectorless_rag_demo.py
   ```

This demo runs a multi-step agentic loop over your indexed documents using `get_document`, `get_document_structure`, and `get_page_content` as tools — identical to the PageIndex agentic demo, extended with the FreqRAG memory layer.

---

### Option B — Custom agent with free Groq / LLaMA (no OpenAI required)

A fully custom agent implementation using the **free Groq API** with `llama-3.3-70b-versatile` is available in the `agent/` folder. This requires no OpenAI account.

**Set up your Groq API key:**

```bash
# Create agent/.env
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free Groq API key at [console.groq.com](https://console.groq.com)

**Run the Groq agent:**

```bash
python3 agent/agent.py
```

The Groq agent (`agent/agent.py`) uses `agent/llm_groq.py` for LLM calls and follows the same tool-use loop as the OpenAI Agents SDK version:

```
→ get_document()
→ get_document_structure()
→ get_page_content(pages="x-y")
→ final answer
```

> **Note:** The Groq agent includes a 30-second wait between API calls (`time.sleep(30)`) to respect free-tier rate limits. This can be removed or reduced on paid tiers.

---

## Building the Memory Graph

The memory graph (`memory/mem.json`) is a compressed summary of the most-accessed PageIndex nodes. It is used by the agent to short-circuit retrieval for queries that match cached knowledge.

**Build or rebuild the memory graph:**

```bash
python3 memory/mem_store.py
```

This script:
1. Reads `memory/weights.json` — the frequency counter store updated after every query
2. Identifies the top-k most accessed `(doc_id, pages)` pairs
3. Finds the overlapping PageIndex tree nodes for those page ranges
4. Merges their summaries using the LLM into a single compact description per entry
5. Writes the result to `memory/mem.json`

**Run periodically** (e.g. nightly) to keep the memory graph fresh:

```bash
# Rebuild from top-5 most accessed nodes (default)
python3 memory/mem_store.py

# The output memory/mem.json looks like:
{
  "type": "memory",
  "nodes": [
    {
      "doc_id": "12345678-abcd-...",
      "pages": "1-5",
      "summary": "Merged summary of the most relevant content..."
    }
  ]
}
```

**How it connects to the agent:** At query time, `agent/agent.py` reads `memory/mem.json` and asks the router LLM whether any cached node directly answers the incoming query. On a memory hit, page content is fetched immediately — skipping the full document structure lookup entirely.

**Frequency weights** are stored in `memory/weights.json` and updated in a background thread after every `get_page_content` call:

```json
{
  "12345678-abcd-4321-abcd-123456789abc_1-5": 1
}
```

Key format: `{doc_id}_{pages}`. The integer value is the access count.

---

## Project Structure

```
freqrag/
├── agent/
│   ├── agent.py              # Custom agent loop (Groq or OpenAI Agents SDK)
│   ├── llm_groq.py           # Groq/LLaMA LLM backend (free tier)
│   └── .env                  # GROQ_API_KEY goes here
│
├── memory/
│   ├── __init__.py
│   ├── mem_store.py          # Build/rebuild the memory graph  ← run this
│   ├── weights_store.py      # Frequency counter read/write
│   ├── weights.json          # Access counts (auto-updated)
│   └── mem.json              # Memory graph (built by mem_store.py)
│
├── pageindex/                # PageIndex source (upstream dependency)
│   ├── client.py             # PageIndexClient
│   ├── page_index.py         # Tree construction for PDFs
│   ├── page_index_md.py      # Tree construction for Markdown
│   ├── retrieve.py           # get_document / get_page_content tools
│   ├── utils.py              # Shared utilities
│   └── config.yaml           # Default model + index settings
│
├── cookbook/                 # Jupyter notebook examples
│   ├── pageindex_RAG_simple.ipynb
│   ├── pageIndex_chat_quickstart.ipynb
│   ├── vision_RAG_pageindex.ipynb
│   └── agentic_retrieval.ipynb
│
├── examples/
│   ├── workspace/            # Persisted PageIndex document stores
│   └── agentic_vectorless_rag_demo.py
│
├── run_pageindex.py          # CLI: index a PDF or Markdown file
├── requirements.txt
└── README.md
```

---

## Configuration

PageIndex settings live in `pageindex/config.yaml`:

```yaml
model: "gpt-4o-2024-11-20"         # LLM for tree construction and retrieval
# model: "anthropic/claude-sonnet-4-6"  # or any LiteLLM-supported model
retrieve_model: "gpt-5.4"           # separate model for retrieval (defaults to model)
toc_check_page_num: 20
max_page_num_each_node: 10
max_token_num_each_node: 20000
if_add_node_id: "yes"
if_add_node_summary: "yes"
if_add_doc_description: "no"
if_add_node_text: "no"
```

FreqRAG-specific settings are passed to `FreqRAGClient`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cache_size` | `100` | Maximum number of cached PageIndex node paths |
| `coverage_threshold` | `0.75` | Minimum coverage score τ before falling back to full tree search |
| `promo_threshold` | `5` | Minimum access count before a node is promoted to cache |
| `decay_lambda` | `0.0` | Time-decay rate λ for Ebbinghaus-style weight decay (0 = pure LFU) |

---

## Indexing Documents (PageIndex CLI)

FreqRAG uses the same PageIndex CLI for document indexing:

**PDF:**
```bash
python3 run_pageindex.py --pdf_path path/to/document.pdf
```

**Markdown:**
```bash
python3 run_pageindex.py --md_path path/to/document.md
```

<details>
<summary>Optional parameters</summary>

```
--model                   LLM model to use (overrides config.yaml)
--toc-check-pages         Pages to check for table of contents (default: 20)
--max-pages-per-node      Max pages per node (default: 10)
--max-tokens-per-node     Max tokens per node (default: 20000)
--if-add-node-id          Add node ID (yes/no, default: yes)
--if-add-node-summary     Add node summary (yes/no, default: yes)
--if-add-doc-description  Add document description (yes/no, default: no)
--if-add-node-text        Add raw page text to nodes (yes/no, default: no)
```
</details>

---

## How the Cache Works End-to-End

```
1. User sends query Q
        │
        ▼
2. Router LLM reads mem.json + weights.json
   → Returns: { hit: bool, node_ids: [...], coverage: float }
        │
        ├─── coverage ≥ 0.75 AND hit=true ──────────────────────┐
        │                                                         │
        ▼                                                         ▼
3a. MISS: PageIndex tree search                      3b. HIT: Fetch cached pages
    (full reasoning traversal)                           from PageIndex document store
        │                                                         │
        ▼                                                         ▼
4a. Update weights.json                              4b. Synthesise answer
    (background thread)                                  (1 LLM call)
        │                                                         │
        ▼                                                         ▼
    If weight ≥ promo_threshold:               5. Return answer to user
    promote node to mem.json cache                   (fast path: ~2 LLM calls)
        │
        ▼
    Synthesise answer + return
    (slow path: 3–5 LLM calls)
```

---

## The Partial-Hit Problem

> **Read this before deploying in high-stakes environments.**

A *partial hit* occurs when a query requires multiple document nodes to answer fully, but only a subset are cached. The model synthesises a confident, internally consistent answer from the incomplete context — and standard faithfulness metrics score it highly.

**Example:** User asks *"How do I reset my password and update my MFA?"*
- Cached node covers password reset ✓
- MFA setup node is not cached ✗
- Answer: detailed password reset instructions, MFA silently omitted
- RAGAS faithfulness score: high

The coverage-score heuristic (`coverage_threshold=0.75`) catches ~88% of these cases and triggers a PageIndex fallback. For safety-critical deployments, lower the threshold or disable caching for known composite query categories.

---

## Research Paper

A full research paper describing the FreqRAG system, the partial-hit formalisation, and implications for personalised AI memory is included in this repository:

📄 [`FreqRAG_v2.pdf`](./FreqRAG_v2.pdf)

**Cite as:**
```
Anonymous Authors (2025).
FreqRAG: Frequency-Weighted Memory Caching for Reasoning-Based
Retrieval in Enterprise Knowledge Systems.
Preprint. github.com/quantambites/pageindex_caching
```

**Cite the upstream PageIndex system as:**
```bibtex
@article{zhang2025pageindex,
  author = {Mingtian Zhang and Yu Tang and PageIndex Team},
  title  = {PageIndex: Next-Generation Vectorless, Reasoning-based RAG},
  journal = {PageIndex Blog},
  year   = {2025},
  month  = {September},
  note   = {https://pageindex.ai/blog/pageindex-intro}
}
```

---

## Relationship to PageIndex

This project depends entirely on [PageIndex](https://github.com/VectifyAI/PageIndex) and would not exist without it. If FreqRAG is useful to you, please star the PageIndex repository and cite their work.

| Component | Provided by |
|-----------|------------|
| Document tree construction | **PageIndex** |
| Reasoning-based tree search | **PageIndex** |
| Page content retrieval | **PageIndex** |
| Answer synthesis quality | **PageIndex** |
| Frequency-weighted cache | FreqRAG |
| Routing LLM call | FreqRAG |
| Memory graph builder | FreqRAG |
| Partial-hit detection | FreqRAG |
| Groq/LLaMA agent | FreqRAG |

---

## License

FreqRAG is released under the MIT License.
PageIndex is subject to its own license — see [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) for terms.

---

<div align="center">
Built on <a href="https://github.com/VectifyAI/PageIndex">PageIndex</a> by <a href="https://vectify.ai">Vectify AI</a>
</div>