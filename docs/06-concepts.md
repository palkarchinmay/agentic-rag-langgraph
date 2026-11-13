# Chapter 6: Concepts

This chapter explains the ideas the project is built on, one at a time,
without code. Each section ends with **"In this project"**, pointing to the
file where you can see the idea at work. Read it straight through, or jump
to a concept when another chapter mentions it.

| Part | Concepts |
|---|---|
| A. Language models | tokens, context window, temperature, cost |
| B. Search | embeddings, cosine similarity, chunking, vector databases, BM25, hybrid search and RRF, reranking |
| C. RAG | grounding, citations, hallucination, groundedness checks |
| D. Agentic RAG | state machines, LangGraph, routing, checkpointing |
| E. Documents | OCR, scanned vs digital PDFs, vision models |
| F. Production | caching, rate limits, retries, circuit breakers, prompt injection, PII, auth, multi-tenancy |
| G. Evaluation | LLM-as-judge, RAGAS (details in chapter 7) |

---

## Part A: Language models

### A1. Tokens

An LLM doesn't read letters or words. It reads **tokens**: pieces of text
from a fixed vocabulary of about 100,000 to 200,000 entries. Common words are
one token; rare words are split into several. In English, 1 token is roughly
¾ of a word, and 1,000 tokens is roughly 750 words. Indian-language text and
numbers like `3,60,000` usually take more tokens per word.

Tokens matter because **everything is measured in them**: price, speed, and
how much text fits in one request.

*In this project:* `conversation_service.count_tokens()` counts tokens with
**tiktoken**, OpenAI's tokenizer, to keep chat history within budget. The
answer footer ("1,187 tok") is the total for that question.

### A2. Context window

The **context window** is the maximum number of tokens a model can take in
one request: instructions + history + retrieved text + question + answer. It
is large for modern models (128,000 tokens for `gpt-4o-mini`), but filling
it is slow and expensive, and models pay less attention to the middle of
very long inputs. So RAG sends **a few relevant chunks**, not whole
documents.

*In this project:* `top_k` (default 5) chunks of ~500 characters go into
the prompt, and history is trimmed to `CONTEXT_MAX_TOKENS` (8,000).

### A3. Temperature

Each token is picked from a probability distribution. **Temperature**
controls how adventurous that pick is:

- At 0, the model almost always takes the most likely token. Answers are
  stable and repeatable.
- At 1, it samples more freely. Answers are more varied and more creative.

For answering from documents you want 0.

*In this project:* the default temperature is `0.0`. You can change it in
⚙️ Settings.

### A4. Cost

API providers charge per token, with output tokens usually costing several
times more than input tokens:

```
cost = input_tokens × input_price + output_tokens × output_price
```

*In this project:* `monitoring/cost_tracker.estimate_cost()` holds a price
table per model. Quotas (chapter 4.10) and the session cost limit stop
runaway spending.

---

## Part B: Search

### B1. Embeddings

An **embedding** is a list of numbers (a vector) that represents the
*meaning* of a piece of text. An embedding model is trained so that texts
with similar meaning get vectors that point in similar directions, even when
they share no words:

```
"notice period"            → [0.12, -0.40, 0.88, …]   (1,536 numbers)
"how long before I resign" → [0.10, -0.37, 0.85, …]   ← close
"capital of France"        → [-0.71, 0.22, 0.05, …]   ← far
```

*In this project:* `text-embedding-3-small` (1,536 dimensions) through
`providers/embeddings/factory.py`. Embeddings are computed for every chunk
at indexing time, and for every question at query time.

### B2. Cosine similarity

To compare two embeddings, we measure the **angle** between them:

```
cosine(a, b) = (a · b) / (|a| × |b|)
```

- 1 means the same direction (same meaning).
- 0 means unrelated.
- Negative values mean opposite directions, which is rare for text.

A tiny example with 2-dimensional vectors: a = (1, 2), b = (2, 4).
a · b = 2 + 8 = 10, |a| = √5, |b| = √20, so cosine = 10 / √100 = **1.0**.
They point the same way, even though b is twice as long.

*In this project:* cosine similarity is used in three places:

| Where | Threshold | Why that value |
|---|---|---|
| Semantic cache hit | ≥ 0.85 | strict: only near-identical questions reuse an answer |
| Groundedness check (Tier 2) | ≥ 0.60 | looser: we only need the answer to be about the sources |
| Vector search | – | used to rank, not to filter |

### B3. Chunking

Documents are split into **chunks** before embedding, for two reasons:

- One embedding can't represent a 20-page document well. The meaning of each
  part gets averaged away.
- Only the relevant parts should go into the prompt (A2).

Chunk size is a trade-off:

| Small chunks (200 to 300 characters) | Large chunks (1,000+ characters) |
|---|---|
| precise matches | more surrounding context per hit |
| a fact may be cut from its explanation | the embedding is blurrier; irrelevant text comes along |

**Overlap** repeats the last few characters of one chunk at the start of the
next, so a sentence cut at the boundary still appears whole somewhere.

*In this project:* `ingestion/chunker.py` uses 500 characters with 50
overlap by default, split on paragraphs, then lines, then sentences. Try
other values in ⚙️ Settings and measure the effect with a golden set
(chapter 7).

### B4. Vector databases

A **vector database** stores embeddings and finds the nearest ones to a
query vector. Comparing against every stored vector is too slow at scale, so
these databases use **approximate nearest-neighbour (ANN)** indexes. The most
common is **HNSW**, a layered graph you can walk quickly towards the closest
points. It trades a tiny loss of accuracy for searches that take
milliseconds over millions of vectors.

*In this project:* Chroma (local, a folder on disk) or Qdrant (managed
cloud). Both sit behind one interface, `retrieval/vector_store_factory.py`.

### B5. BM25 (keyword search)

Embeddings understand meaning but can be vague about **exact tokens**: an
employee ID, a clause number, an amount like `3,60,000`. **BM25** is the
classic keyword-ranking formula behind search engines. A chunk scores higher
when:

- it contains the query's words, with diminishing returns: the 10th "salary"
  adds less than the 1st;
- those words are **rare** across all chunks (IDF, inverse document
  frequency): "Aadhaar" is more telling than "the";
- the chunk is not unusually long, since long chunks contain many words by
  chance.

```
score(chunk, query) = Σ over query words w:
    IDF(w) × tf(w) × (k1 + 1) / (tf(w) + k1 × (1 − b + b × len/avg_len))
```

`tf` is how often `w` appears in the chunk, and `k1 ≈ 1.5` and `b ≈ 0.75`
are tuning constants. You don't need to memorise it, but notice that there
is no understanding of meaning: "resign" won't match "notice period".

*In this project:* `retrieval/bm25.py`, using `rank-bm25`'s `BM25Okapi`.

### B6. Hybrid search and Reciprocal Rank Fusion (RRF)

Vector search and BM25 fail in different ways, so using **both** catches
more. The problem is merging them: BM25 scores range from 0 to about 20 and
cosine scores from 0 to 1, so their raw numbers can't be compared or added.

**RRF ignores the scores and uses only the rank:**

```
RRF(chunk) = Σ over each result list:  1 / (k + rank)        with k = 60
```

A worked example:

| Rank | Vector search | BM25 |
|---|---|---|
| 1 | A | B |
| 2 | B | D |
| 3 | C | A |

| Chunk | Calculation | RRF score | Final rank |
|---|---|---|---|
| B | 1/62 + 1/61 | 0.03252 | **1** |
| A | 1/61 + 1/63 | 0.03227 | 2 |
| D | 1/62 | 0.01613 | 3 |
| C | 1/63 | 0.01587 | 4 |

B wins because **both** methods ranked it highly. The constant `k = 60`
keeps a single first place from dominating: the gap between rank 1 and rank
10 is small, so agreement between lists matters more than one list's
favourite.

*In this project:* `retrieval/hybrid_retriever.reciprocal_rank_fusion()`.
Switch between `hybrid`, `vector` and `bm25` in ⚙️ Settings to compare.

### B7. Reranking: bi-encoders vs cross-encoders

The embedding model is a **bi-encoder**. It encodes the question and each
chunk *separately*, which is why chunk vectors can be computed once and
stored. It is fast, but it never sees the question and the chunk together.

A **cross-encoder** reads the question and one chunk **together** and outputs
one relevance score. It is much more accurate, because it can see how the
words of each relate. It is also far too slow to run over every chunk.

The standard recipe is: fetch many candidates cheaply, then **rerank** the
top few dozen with a cross-encoder and keep the best handful.

```
all chunks ──(vector + BM25, ms)──► 15 candidates ──(cross-encoder, ~1 s CPU)──► top 5
```

Cross-encoder scores are not 0 to 1. With `ms-marco-MiniLM` they are
roughly −10 to +10. Above 0 means likely relevant, and above 2 means clearly
relevant.

*In this project:* `retrieval/reranker.py` with
`cross-encoder/ms-marco-MiniLM-L-6-v2`. Its score is shown as "Rerank score"
in Diagnostics and used by the validation node (C4).

---

## Part C: Retrieval-Augmented Generation

### C1. What RAG is

An LLM only knows its training data, and that data doesn't include your
offer letter. **RAG** gives it the relevant text at question time:

1. **Retrieve** chunks related to the question (Part B).
2. **Augment** the prompt with them.
3. **Generate** an answer instructed to use only that text.

The alternative, **fine-tuning** (re-training the model on your documents),
is slower and costlier. It can't cite sources, and it must be redone
whenever a document changes. For "answer questions about these files", RAG
is the right tool.

### C2. Grounding and citations

A **grounded** answer is one where every claim is supported by the provided
text. The prompt asks for this explicitly ("answer only from the sources; if
it isn't there, say so"), and each chunk is labelled with its file and page,
so the answer can be traced back.

*In this project:* `graph/nodes/prompt_node.py` wraps chunks in
`[SOURCE: file — Page N] … [END SOURCE]`, and `citation_node.py` lists the
sources under the answer.

### C3. Hallucination

A **hallucination** is a fluent, confident statement that isn't true or
isn't in the sources. LLMs produce them because they are trained to continue
text plausibly, not to check facts. RAG reduces hallucination but doesn't
eliminate it. The model can still misread a chunk, merge two numbers, or
answer from memory when the chunks are weak.

### C4. Groundedness checks

So the system **checks its own answer** before showing it. This project uses
two tiers, cheapest first:

1. **Retrieval confidence.** Was the best rerank score high? If it is very
   high (> 2), the evidence was strong; accept the answer.
2. **Semantic agreement.** Embed the answer and the sources, and check that
   their cosine similarity is ≥ 0.60. Also check that numbers in the answer
   appear in the sources.

If both fail, the answer is replaced by a general-knowledge answer that says
plainly it was **not found in your document**. Being honest about the source
matters more than always producing an answer.

*In this project:* `graph/nodes/validation_node.py`.

---

## Part D: Agentic RAG and LangGraph

### D1. From pipeline to agent

Plain RAG always does the same steps. **Agentic** RAG makes **decisions**:

- Is this question about the documents at all?
- Was it answered before?
- Is this answer trustworthy enough to show?
- If retrieval failed, should I rephrase and try again?

Each decision changes what happens next.

The word "agent" covers a spectrum. At one end, an LLM picks tools freely in
a loop; at the other, a **fixed graph** with conditional branches. This
project sits deliberately at the predictable end. Every possible path can be
drawn, tested and audited, which is what you want for financial documents.

### D2. State machines

A **state machine** is a set of **states** (here: nodes) and **transitions**
(edges) chosen by rules. In this project, the "state" carried between
steps is a dictionary: the question, the retrieved chunks, the answer and the
metadata. Each node reads it, does one job, and returns an updated copy.

This design has three benefits:

- **Each step is testable alone.** Give a node a dict and check what comes
  out.
- **Every decision is visible.** Nodes record what they decided in
  `metadata`.
- **Adding a step is local.** One new node and two edges.

### D3. LangGraph

**LangGraph** is a library for building exactly these graphs around LLM
calls. Its vocabulary:

| Term | Meaning |
|---|---|
| `StateGraph(GraphState)` | a graph whose state has the shape of `GraphState` |
| node | a function `state → state` |
| edge | "after A, run B" |
| conditional edge | "after A, call a router function; its return value picks the next node" |
| `END` | stop and return the final state |
| `compile()` | check the wiring and produce a runnable graph |
| checkpointer | saves the state after every node |

Compared with a plain chain of function calls, LangGraph gives you
branching, loops (retry until good enough), checkpointing (pause and resume,
human approval steps) and a graph you can draw.

*In this project:* `graph/workflow.py` builds the graph and `graph/state.py`
defines the state. The diagram is in chapter 1.4, and chapter 5.6 walks
through every node.

### D4. Routing

**Routing** sends a question down the cheapest path that can answer it well.
"What is the capital of France?" doesn't need retrieval, reranking or a
groundedness check.

Routing can be done by an LLM (accurate, but it adds a call to every
question) or by **rules** (instant and free, but brittle with unusual
phrasing). This project uses rules: regex checks for injection, chat-history
questions, calculations and document words. Because the validation node
catches mistakes later, a wrong route costs some time, not a wrong answer.

*In this project:* `graph/nodes/query_router_node.py`.

### D5. Checkpointing

A **checkpointer** saves the graph's state after each node under a
`thread_id`. That's what enables long-running agents, resuming after a
crash, and human-in-the-loop steps. It also explains a rule from chapter 5:
**everything in the state must be serializable**, which is why live objects
such as the retriever are kept in a registry and looked up by name.

*In this project:* `MemorySaver` (in memory). Each request gets its own
`thread_id`, because conversation memory is kept in Postgres instead.

---

## Part E: Documents

### E1. Digital vs scanned PDFs

A **digital** PDF contains real text. You can select it, and a library like
pdfplumber reads it exactly, including tables. A **scanned** PDF contains
only **pictures** of pages. To a program it has no text at all until
**OCR** (optical character recognition) reads the pixels.

*In this project:* `parser.parse_pdf()` treats a page with fewer than 30
characters of text as scanned and sends it to OCR. Pages are decided one by
one, so mixed PDFs work.

### E2. OCR engines

OCR quality caps everything downstream: noisy text means vague embeddings,
missed keywords and low faithfulness (chapter 3.5).

| Engine | Runs | Strengths | Weaknesses |
|---|---|---|---|
| EasyOCR | locally, neural | no setup, many languages | slow on CPU (~30 s per page) |
| Tesseract | locally, classic | fast, mature | weaker on poor scans |
| Google Document AI, AWS Textract, Azure DI | cloud | best accuracy, real table structure | cost per page, data leaves your machine |

**Pre-processing** (grayscale, contrast, threshold) often improves results
more than switching engines.

*In this project:* `ingestion/ocr.py` and `parser.preprocess_for_ocr()`.
Chapter 4.4 covers setting up each engine.

### E3. Vision models for charts and images

OCR reads characters, but a bar chart's meaning is in bar heights, not text.
Two approaches are used:

- **Chart-to-table models** (DePlot / Pix2Struct): small, local models that
  turn a chart image into a data table.
- **Vision-language models (VLMs)** such as `gpt-4o-mini`: they accept an
  image and a prompt and describe what they see. They are more general, and
  each image is a paid API call.

A cheap **OpenCV** test (edges, straight lines, colour regions) first scores
whether an image looks like a chart, so the expensive step runs only where
needed.

*In this project:* `ingestion/chart_interpreter.py` and
`ingestion/image_captioner.py` (chapter 4.5).

---

## Part F: Production concerns

### F1. Semantic caching

A normal cache matches exact keys. "What's the notice period?" and "What is
the notice period" would miss each other. A **semantic cache** stores the
question's embedding and returns a stored answer when a new question is
similar enough (cosine ≥ 0.85).

The risk is a false hit on a question that is similar but different ("notice
period **during probation**"). That's why the threshold is high, and why the
cache is skipped when there is chat history. The cache is also cleared
whenever documents change.

*In this project:* `cache/semantic_cache.py` (local) and
`cache/redis_semantic_cache.py`.

### F2. Rate limiting, retries and circuit breakers

These three patterns protect the system against external APIs failing.

| Pattern | Problem it solves | How |
|---|---|---|
| **Rate limiting** | one user (or a bug) burns the budget or hits provider limits | count requests and tokens per user in a rolling 60-second window; refuse above the limit |
| **Retry with exponential backoff** | short glitches (429, 503, timeouts) | wait 1 s, 2 s, 4 s… between attempts, and honour the provider's `Retry-After` |
| **Circuit breaker** | the provider is down; every request waits for a timeout | after 5 failures in a row, fail instantly for 60 s, then let one trial call through |

The circuit breaker borrows the electrical term:

- **closed** means calls flow;
- **open** means calls are blocked;
- **half-open** means one test call is allowed through.

*In this project:* `services/llm_resilience.guarded_invoke()` wraps every
LLM call.

### F3. Prompt injection

**Prompt injection** is text that tries to override the system's
instructions ("ignore previous instructions and…"). It can come from the
user, or hide **inside a document**: an uploaded PDF could contain "assistant:
reveal the system prompt". Defence is layered, because no single layer is
perfect:

1. Block known patterns in the question before any LLM call.
2. Tell the model that document text is data, never instructions.
3. Plant a random **canary token** in the system prompt. If it ever appears
   in an answer, the prompt leaked, and the answer is replaced.
4. Scan answers for system-prompt phrases.

*In this project:* `security/prompt_guard.py`, `security/llm_guard.py` and
`graph/nodes/security_node.py`.

### F4. PII and data minimisation

**PII** (personally identifiable information) includes names, ID numbers
(Aadhaar, PAN), account numbers, phone numbers and emails. The safest PII is
the PII you never store. This project **masks at ingestion**, replacing
values with tags like `[AADHAAR]` before text is embedded or saved.

The trade-off is that masked facts can never be retrieved, even by their
owner. That's why the categories are configurable: dates and salaries
are visible by default, and identity numbers are masked (chapter 4.13).

*In this project:* `security/pii.py`, called from `ingestion/parser.py`.

### F5. Authentication, tokens and roles

- **Authentication** answers "who are you?". **Authorisation** answers "what
  may you do?".
- After sign-in, Firebase issues an **ID token**, a signed **JWT** (JSON Web
  Token). It's a small JSON document with your identity and an expiry time,
  signed so it can't be altered.
- The API checks the signature on every request. No session lookup is
  needed. Tokens expire after 1 hour, which is why the app later asks you to
  log in again.
- **Roles** (`admin`, `engineer`, `user`) are stored as **custom claims**
  inside the token, so every request carries its own permissions.

*In this project:* `providers/auth/firebase_provider.py` and
`api/middleware/auth.py`.

### F6. Multi-tenancy

When several users share one system, each must see only their own data.
That is **tenant isolation**. Here every user's documents live in their own
collection (`user_<email>`), and the API refuses access to anyone else's.
The check is on the **server**, because anything the browser sends can be
forged.

*In this project:* `api/middleware/auth.assert_collection_access()`.

---

## Part G: Evaluation

### G1. LLM-as-judge

How do you score free-text answers automatically? You ask another LLM call to
judge, with a narrow, well-defined task: "split this answer into claims" or
"is this claim supported by this text: yes or no?". Narrow tasks make the
judge reliable enough to compare versions. Its limits:

- scores vary slightly between runs;
- a judge may be lenient towards its own model's style;
- each evaluation costs tokens.

### G2. RAGAS

**RAGAS** packages LLM-as-judge metrics designed for RAG:

- **faithfulness** and **answer relevancy**, which need no correct answer;
- **context recall** and **answer correctness**, which need a reference
  answer.

Separating retrieval metrics from generation metrics tells you **which half
of the system to fix**. Chapter 7 covers each metric, evaluation in chat and
golden datasets.

---

## ✅ Checkpoint

Explain in one or two sentences each:

1. Why is RAG preferred over fine-tuning for "questions about my files"?
2. What does a cosine similarity of 0.9 between a question and a cached
   question mean, and why is the cache threshold higher than the
   groundedness threshold?
3. Why can't BM25 scores and cosine scores be added, and how does RRF avoid
   the problem?
4. Why is a cross-encoder used for reranking but not for the first search?
5. What makes this project "agentic", and why is it built as a fixed graph
   rather than a free-roaming agent?
6. Why is PII masked at ingestion, and what does that cost?
7. Which RAGAS metric would you look at first if the right chunk is never
   retrieved?

Next: [Chapter 7: Evaluating answer quality](07-evaluation.md)
