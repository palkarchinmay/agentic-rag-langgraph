# Chapter 4: Feature guide

This chapter is a tour of everything the app can do: what each feature is
for, how to use it, how to set it up, and where its code lives.

| Section | Feature |
|---|---|
| 4.1 | Roles and chat profiles (tabs) |
| 4.2 | 💬 Chat: upload, ask, cite |
| 4.3 | ⚙️ Settings panel |
| 4.4 | OCR engines: EasyOCR, Tesseract, Google Document AI, AWS Textract, Azure DI |
| 4.5 | 🖼️ Vision mode (charts and images) |
| 4.6 | 📊 Diagnostics tab and RAGAS evaluation |
| 4.7 | 🔌 Connections tab |
| 4.8 | 👥 User management |
| 4.9 | ⚙️ Admin panel |
| 4.10 | 💰 Usage & FinOps (cost tracking and quotas) |
| 4.11 | Conversation memory |
| 4.12 | Semantic cache |
| 4.13 | Security: prompt-injection guard, PII masking, groundedness check |
| 4.14 | LLM resilience: retries, rate limits, circuit breaker |
| 4.15 | Multi-provider LLMs and embeddings |
| 4.16 | Chat command reference |
| 4.17 | Which features need which services |

---

## 4.1 Roles and chat profiles

Every user has one of three **roles**, stored as a Firebase custom claim
(`role`) on their account:

| Role | Intended for | Sees these tabs |
|---|---|---|
| `user` | Business users | 💬 Chat |
| `engineer` | People tuning the RAG pipeline | 💬 Chat, 📊 Diagnostics, 🔌 Connections |
| `admin` | Operators | All of the above, plus 👥 User Management, 💰 Usage & FinOps, ⚙️ Admin Panel |

The tabs are Chainlit **chat profiles**: the dropdown next to "Chat" at the
top of the screen. Switching profile starts a new session showing that
panel.

**How it works.** At login the API returns the role from the Firebase token
(`api/routers/auth.py`). `ui/auth.py → set_chat_profiles()` builds the
list of tabs from it, and `ui/chat_handlers.py → on_chat_start()` renders
the matching panel. The **API enforces roles independently** with
`require_role("admin")` dependencies (`api/middleware/auth.py`), so hiding
a tab is a convenience, not the security boundary. Also, the API ignores
any `user_role` the client sends and uses the one in the verified token,
so a user can't promote themselves.

---

## 4.2 💬 Chat

The main tab, available to everyone.

- **Upload** with the 📎 button: PDF, DOCX, PPTX, XLSX, TXT, MD, and CSV
  (CSV is treated as a golden dataset, see 4.6). Images inside PDFs, Word
  and PowerPoint files are handled by OCR and vision mode. Each user's documents
  go into their own collection `user_<email>`; nobody else can query it.
- **Ask** anything. The router decides whether to search your documents.
- **Answer footer**: route (`RAG` / `General`), latency, tokens, estimated
  cost.
- **Sources**: file name and page for each chunk used.
- **📊 Evaluate (RAGAS)** button under each answer (engineers and admins).
- **Used 🔍 Diagnostics** expander (engineers and admins): the retrieved
  chunks with their scores, so you can see *why* the model answered as it did.
- **Duplicate guard**: uploading a file that's already indexed in this
  session asks "Re-index anyway?" instead of silently adding duplicates.

Code: `ui/chat_handlers.py`, `ui/upload_handler.py`, `ui/query_handler.py`.

---

## 4.3 ⚙️ Settings panel

Open with the ⚙️ icon next to the message box. Settings are per user and
survive profile switches.

| Setting | Options | Effect |
|---|---|---|
| 🤖 LLM model | `openai:gpt-4o-mini`, `openai:gpt-4o`, `anthropic:…`, `google:gemini-…`, `azure:…`, `bedrock:…` | Model that writes answers |
| Temperature | 0 to 1 | 0 = deterministic, best for factual Q&A |
| 🔍 Embedding model | OpenAI, Azure, Google, Bedrock, HuggingFace | Must match the one used at indexing time |
| 📚 Retrieval mode | `hybrid` / `vector` / `bm25` | Which indexes to search (chapter 6 explains why hybrid usually wins) |
| Top-K | 1 to 20 | Chunks passed to the LLM |
| Cross-encoder reranking | on / off | Re-scores candidates for precision |
| ✂️ Chunking method, size, overlap | `recursive` / `sentence` / `fixed`, 100–2000, 0–200 | How documents are split at indexing |
| 👁️ OCR method | `easyocr`, `tesseract`, `google_docai`, `none` | See 4.4 |
| 🖼️ Vision mode + caption model | on/off, any vision-capable model | See 4.5 |
| 🌍 Display timezone | | For timestamps in logs and usage |
| 🔌 Connection URLs | API, Qdrant, Redis, Supabase | Shown for reference; edit in the Connections tab |

The session panel on the right lists indexed documents, conversation ID and
evaluations. `/reset` restores all defaults.

> ⚠️ **Changing the embedding model after indexing** makes new queries
> incompatible with stored vectors. `/clear` and re-index after switching.

Code: `ui/settings_actions.py`.

---

## 4.4 OCR engines

OCR (optical character recognition) turns **scanned pages** into text.
Digital PDFs don't need it: `parser.py` extracts their text directly and
only sends a page to OCR when it finds no real text on it ("scanned").

All engines sit behind one function, `app/ingestion/ocr.py → extract_text()`.
Each engine's SDK is imported lazily, so an engine you don't configure costs
nothing and can't crash startup.

| Engine | Setting value | Runs | Cost | Best for | Speed (CPU) |
|---|---|---|---|---|---|
| **EasyOCR** | `easyocr` (default) | Locally, deep learning | Free | Noisy or rotated scans; works out of the box | ~30 s/page |
| **Tesseract** | `tesseract` | Locally, needs a system binary | Free | Clean, high-contrast scans | ~1–3 s/page |
| **Google Document AI** | `google_docai` | Google Cloud | Paid per page | Mixed scripts, stamps, handwriting; best overall on Indian financial documents | ~1–2 s/page |
| **AWS Textract** | `textract` | AWS | Paid per page | Structured financial documents: tables and forms | ~1–2 s/page |
| **Azure Document Intelligence** | `azure_di` | Azure | Paid per page | Invoices, forms | ~1–2 s/page |

If an engine fails or isn't configured, the page falls back to EasyOCR (or
returns no text for cloud engines), and the API log says so.

> ℹ️ The settings dropdown offers EasyOCR, Tesseract, Google Document AI and
> `none`. Textract and Azure DI are selected with `DEFAULT_OCR_METHOD` in
> `.env`, or the `ocr_method` field of `POST /api/v1/index`.

### EasyOCR (default)
Nothing to set up. The first use downloads its models (~100 MB) to
`~/.EasyOCR/`.

### Tesseract
1. Install the binary:
   - **Windows:** installer from https://github.com/UB-Mannheim/tesseract/wiki
   - **macOS:** `brew install tesseract`
   - **Linux / Docker:** `apt install tesseract-ocr tesseract-ocr-eng`
2. Windows only: the installer doesn't add it to `PATH`, so tell the app where it is:
   ```ini
   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```
3. Choose **tesseract** in Settings → OCR method, `/clear`, and re-upload.

If the binary isn't found, the log says
`Tesseract binary not found ... falling back to easyocr`.

### Google Document AI
1. In Google Cloud: enable the **Document AI API**, then **Document AI →
   Processors → Create processor → Document OCR**. Note the processor ID
   and region (`us` or `eu`).
2. Create a service account with the **Document AI API User** role and
   download its JSON key.
3. `.env`:
   ```ini
   GCP_PROJECT_ID=your-project-id
   GCP_LOCATION=us
   GCP_PROCESSOR_ID=abc123def456
   GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\docai-service-account.json
   ```
4. Choose **google_docai** in Settings.

### AWS Textract
1. Create an IAM user (or role) with the `AmazonTextractFullAccess`
   policy, or a narrower one allowing `textract:DetectDocumentText` and
   `textract:AnalyzeDocument`.
2. `.env`:
   ```ini
   AWS_ACCESS_KEY_ID=AKIA...
   AWS_SECRET_ACCESS_KEY=...
   AWS_DEFAULT_REGION=ap-south-1
   DEFAULT_OCR_METHOD=textract
   ```
Textract also has a table extractor (`extract_tables_textract`) that returns
tables as Markdown, which keeps rows and columns intact for the LLM.

### Azure Document Intelligence
```ini
AZURE_DI_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DI_KEY=...
AZURE_DI_MODEL=prebuilt-read        # or prebuilt-layout for tables
DEFAULT_OCR_METHOD=azure_di
```

### `none`
Skips OCR entirely: scanned pages contribute no text. Useful for indexing a
large digital-only corpus quickly.

> 💡 **Try it:** index the same scanned PDF with EasyOCR and then with
> Document AI (`/clear` in between), ask the same question, and compare the
> RAGAS faithfulness scores. Better OCR → cleaner chunks → higher faithfulness.

---

## 4.5 🖼️ Vision mode (charts and images)

Financial documents are full of charts and images that OCR alone reads
badly ("Revenue 2023 2024 120 140" with no idea what goes with what).

**How it works** (`app/ingestion/chart_interpreter.py`,
`app/ingestion/image_captioner.py`):

1. **Detection always runs.** Every image region on a page gets a chart
   score from OpenCV heuristics (lines, axes, colour blocks). You saw these
   in the log: `Image detected (score=0.22) but Vision Mode is off -- OCR fallback`.
2. **Vision mode off (default):** images are only OCR'd. Free.
3. **Vision mode on:** each image is sent to a **vision-language model**
   (VLM) with one of two prompts:
   - **Chart prompt:** chart type, axis labels and units, every readable
     data point *as a table*, and the trend.
   - **General image prompt:** a factual description of the image,
     transcribing any text in it, under 200 words.

   The description is stored as a chunk tagged `[CHARTS]`, so questions like
   "What was revenue growth in FY24?" can be answered from a bar chart.
4. **Caption model:** choose it in Settings → *Vision/Caption model*, for
   example `openai:gpt-4o-mini` or `google:gemini-2.5-flash` (fast and cheap).

**Cost:** one VLM call per image, typically 1 to 3 seconds and a fraction
of a cent each. That's why it's opt-in. A 50-page report with 40 charts
might cost a few cents to index.

**Offline fallback:** if no caption model is configured, local
**DePlot / Pix2Struct** models can interpret charts for free, but they take
30 to 90+ seconds per chart on CPU and produce weaker descriptions. The
code keeps them as an explicit zero-cost option.

---

## 4.6 📊 Diagnostics tab and RAGAS evaluation

For **engineers and admins**. It answers the question "is my RAG pipeline
any good, and is it getting better?"

### Per-answer evaluation
Click **📊 Evaluate (RAGAS)** under any answer. In 15 to 30 seconds you get:

| Metric | Question it answers |
|---|---|
| **Faithfulness** | Are the answer's claims supported by the retrieved chunks? (hallucination check) |
| **Answer relevancy** | Does the answer actually address the question? |

Scores range from 0 to 1, shown with 🟢 / 🟡 / 🔴 indicators. The evaluation runs
via `POST /api/v1/evaluate`, and results are cached for 2 hours.

### The Diagnostics tab
Switch the profile to **📊 Diagnostics** to see every evaluation from your
session in one table, with actions to:
- **📊 Export All** / **Export This**: download as Excel
  (question, answer, sources, scores, model, latency), or use `/export`
- **🎯 Golden Dataset Test**: batch evaluation (below)
- **🗑️ Clear All Evaluations**

### Golden dataset (batch evaluation)
A **golden dataset** is a fixed list of questions with known answers that
you re-run after every change (new chunk size, new OCR engine, new model) to
measure the effect.

1. Create a CSV:
   ```csv
   question,reference_answer
   What is the notice period after probation?,60 days
   What is the joining date?,1 August 2026
   ```
2. In 💬 Chat, upload the CSV with 📎.
3. Type `/golden`. Every question is run through the pipeline and scored,
   with a progress bar and a summary table at the end.

> ℹ️ Evaluations survive profile switches only when Redis is configured
> (`CACHE_BACKEND=redis`); in local mode they last for the session.

Full guide, including what each metric means and how to build a good golden set: [chapter 7](07-evaluation.md).

Code: `ui/evaluation_actions.py`, `app/evaluation/ragas_evaluator.py`,
`app/graph/nodes/ragas_node.py`, `api/routers/evaluate.py`.

---

## 4.7 🔌 Connections tab

For **engineers and admins**. It shows every external service the app
depends on, with a **🔍 Test** button for each:

| Service | What the test does | Success shows |
|---|---|---|
| FastAPI backend | `GET /health` | status, HTTP code |
| Qdrant vector DB | `GET /collections` with the API key | list of collections |
| Redis / Upstash | `PING` + memory info | memory used |
| Supabase / Postgres | connect + simple query | server version |

URLs are **masked** in the output (`mask_url`), so passwords embedded in
connection strings never appear on screen. The Postgres test also lists the
tables it finds, so you can confirm `audit_logs`, `conversations` and
`user_quotas` were created.

Use it first whenever something's wrong: it tells you in seconds which
dependency is down.

Code: `ui/settings_actions.py → render_connections_panel(), _test_single_connection()`.

---

## 4.8 👥 User management

For **admins**. All operations go through the API
(`/api/v1/admin/users*`), which calls the configured auth provider. Today
that's Firebase. Cognito and Azure AD stubs exist in `app/providers/auth/`,
so the provider can be swapped without changing the UI.

| Action | How |
|---|---|
| List users (paginated) | **👤 List Users** |
| Create a user | `/create_user email=jane@example.com password=Secret123 role=user name="Jane Doe"` |
| Change role | Pick a user → **user / engineer / admin** |
| Disable / enable | **🚫 Disable** / **✅ Enable** (a disabled user can't log in, but their data is kept) |
| Delete | **🗑️ Delete User**, then confirm |
| First admin | `python scripts/create_first_admin.py` (chapter 2) |
| Promote from the CLI | `python scripts/promote_to_admin.py` |

**How roles are stored:** `FirebaseAuthProvider.set_user_role()` writes
`{"role": "admin"}` into the user's Firebase **custom claims**. Claims are
embedded in every ID token Firebase issues, so the API can read the role
from the token without a database lookup. A role change takes effect the
next time the user logs in.

**Google sign-in:** when someone signs in with Google, Chainlit completes
the OAuth handshake, then calls `POST /api/v1/auth/login/sso` with a shared
`INTERNAL_SERVICE_SECRET` header. The API finds or creates the Firebase user
(default role `user`) and returns a token. End users can't call this
endpoint themselves, because they don't have the secret.

Code: `ui/user_management_actions.py`, `api/routers/admin.py`,
`app/providers/auth/`.

---

## 4.9 ⚙️ Admin panel

For **admins**: day-to-day operations in one place.

| Area | Actions |
|---|---|
| **Stats** | Requests, error rate, average latency, tokens, cost and cache-hit rate for the last 24 h (**🔄 Refresh**); also `/stats` |
| **Logs** | Recent audit-log entries, filterable by user (**👤 per-user** buttons, **📋 Load more**); also `/logs` |
| **Semantic cache** | **📋 View cache keys**, **🗑️ Clear all** (after changing prompts or models) |
| **Collections** | **📊 View collections** with chunk counts, **🗑️ Clear my collection** |
| **Conversations** | **💬 View**, **🗑️ Cleanup old (> 2 days)**, **Clear mine**, **Clear ALL** (admin-guarded) |

Logs and stats come from the `audit_logs` table (4.17).

Code: `ui/admin_actions.py`, `api/routers/admin.py`.

---

## 4.10 💰 Usage & FinOps

For **admins**: who is spending what, and stopping runaway costs.

- **Usage per user** for the current month: requests, tokens, estimated
  cost, with relative bars.
- **📈 Trend**: daily usage this month.
- **🧮 By model**: which models cost the most.
- **Quotas**: per user, monthly limits on tokens, cost (USD) and/or
  requests. Set them from the panel or with
  `/set_quota uid=... tokens=500000 cost=5 requests=1000`.
- `/usage from=2026-11-01 to=2026-11-30 user=...` gives the same as a command.

**Enforcement:** `api/routers/query.py` calls `check_quota()` *before* any
retrieval or LLM work, so an over-quota user is refused instantly and costs
nothing. Quotas reset on the 1st of each month (UTC).

**Cost estimation:** every request's tokens are priced from the
`MODEL_PRICING` table in `app/monitoring/cost_tracker.py` (for example
gpt-4o-mini at $0.15 / $0.60 per million input / output tokens). These are
**estimates**, not invoice data, and are labelled as such everywhere.

Code: `ui/usage_actions.py`, `app/services/usage_service.py`.

---

## 4.11 Conversation memory

Follow-up questions ("and after probation?") need the previous turns.

- Each chat gets a **conversation ID**, and messages are stored in Postgres
  (`conversations` and `messages` tables).
- Token counts are computed once, at write time, with tiktoken.
- For each new question, `build_context()` walks backwards through the
  history and keeps as many recent messages as fit in `CONTEXT_MAX_TOKENS`
  (a **sliding window**).
- Once a conversation passes `SUMMARIZE_THRESHOLD_TURNS`, older messages are
  **summarized by the LLM** into one short message, so long chats keep their
  gist without blowing the context budget.
- Retention: `CONVERSATION_RETENTION_DAYS` (cleanup from the Admin panel).

Without a database, the UI still sends recent chat history with each
request; only server-side persistence and summarization are disabled.

Code: `app/services/conversation_service.py`, `api/routers/conversations.py`.

---

## 4.12 Semantic cache

If someone asks "What's the notice period?" and later "How long is the
notice period?", the second question shouldn't cost another LLM call.

- The question is **embedded**, then compared by cosine similarity with
  cached questions **in the same collection**.
- Similarity ≥ `CACHE_SIMILARITY_THRESHOLD` (default **0.85**) → return the
  cached answer instantly (`cache=hit` in the log, cost $0).
- Entries expire after `CACHE_TTL_SECONDS` (default 1 h). Indexing new
  documents **invalidates** that collection's cache, so stale answers can't
  survive a re-index.
- Backends: local JSON files (`CACHE_BACKEND=local`) or Redis/Upstash
  (`CACHE_BACKEND=redis`, required when running several API instances).

Code: `app/cache/`, `app/graph/nodes/cache_node.py`.

---

## 4.13 Security features

| Layer | Where | What it does |
|---|---|---|
| **Authentication** | Firebase + `api/middleware/auth.py` | Every API call except `/health` and login needs a valid Firebase ID token |
| **Authorization** | `require_role()` | Admin endpoints check the role in the token |
| **Data isolation** | `api/middleware/auth.py` (`assert_collection_access`) | Each user's documents live in `user_<email>`. Index, query and delete all check that a non-admin uses only their own collection (403 otherwise); admins can reach any |
| **Prompt-injection guard** | `app/security/prompt_guard.py`, `security_node` | Regex rule groups (instruction override, persona injection, system-prompt extraction, jailbreaks, hidden instructions in code blocks) block the query before any LLM call |
| **PII masking** | `app/security/pii.py`, called from `parser.py` | Three passes replace sensitive values with tags **before** chunks are embedded or stored: (1) structural regexes (Aadhaar, PAN, IFSC, GST, IBAN, card numbers, account numbers, phone, email, dates, IPs…); (2) label-based (`Name:`, `DOB:`, `Address:`, `SWIFT:`/`BIC:`, `Salary:` → the value after the label); (3) OCR-tolerant variants for noisy scans. Which categories are masked is configurable (see below) |
| **Groundedness check** | `validation_node.py` | Tier 1: is the top reranker score high enough? Tier 2: is the answer semantically close (≥ 0.60) to the sources? If both fail, the document answer is discarded and replaced by a general-knowledge answer that starts with "This information was not found in your uploaded document", so the user is never shown an unsupported answer as if it came from their file |
| **Secrets management** | `app/config/secrets.py` | `SECRETS_BACKEND=local` reads `.env`; `gcp` loads secrets from Google Secret Manager at startup |
| **Audit log** | `api/middleware/logging.py` | Every query is logged with user, route, tokens, cost and latency |

> ⚖️ **The privacy / usefulness trade-off.** Masking happens *before*
> indexing, so a masked value can never be retrieved, not even by its
> owner. If dates and salaries were masked, "What is my joining date?" or
> "What is the CTC?" could never be answered from an offer letter. So the
> masking is configurable in `.env`:
>
> | Variable | Default | Effect |
> |---|---|---|
> | `PII_MASKING_ENABLED` | `true` | `false` turns masking off completely |
> | `PII_SKIP_CATEGORIES` | `DATE_ISO,DATE_DMY,DATE,SALARY` | Tags that stay **visible**. Set it to empty (`PII_SKIP_CATEGORIES=`) to mask everything, or add tags such as `NAME` or `PHONE` |
>
> Identity and account data (Aadhaar, PAN, IBAN, account numbers, card
> numbers) stay masked by default. SWIFT/BIC codes are masked only after a
> `SWIFT:` or `BIC:` label, because a bare "8 or 11 capital letters" pattern
> also matched ordinary upper-case words such as `COMPANY`.
>
> The setting applies **at indexing time**. After changing it, restart the
> API, run `/clear`, and upload the documents again. `detect_pii` (used by
> the security node on questions) is not affected by the skip list.
---

## 4.14 LLM resilience

LLM APIs fail: 429 rate limits, 503 overloads, timeouts. Every LLM call
goes through `guarded_invoke()` (`app/services/llm_resilience.py`):

| Layer | Default | Setting |
|---|---|---|
| **Retry with exponential backoff** on 429/500/503/timeouts | 3 retries, 1 s → 30 s | `LLM_MAX_RETRIES`, `LLM_RETRY_BASE_DELAY`, `LLM_RETRY_MAX_DELAY` |
| **No retry on permanent errors** (401, 400) | | |
| **Request timeout** | 60 s | `LLM_REQUEST_TIMEOUT` |
| **Per-user rate limit** (rolling 60 s) | 20 requests, 50k tokens | `LLM_RPM_LIMIT`, `LLM_TPM_LIMIT` |
| **Circuit breaker**: after N consecutive failures, fail fast for a cool-down instead of hammering a broken provider | 5 failures, 60 s | `LLM_CB_FAILURES`, `LLM_CB_TIMEOUT` |
| **Session cost limit** | $1.00 | `LLM_SESSION_COST_LIMIT` |

---

## 4.15 Multi-provider LLMs and embeddings

`app/providers/llm/factory.py` and `app/providers/embeddings/factory.py`
return a LangChain model for a `provider:model` string. Supported:

| Provider | LLM | Embeddings | Keys in `.env` |
|---|---|---|---|
| OpenAI | ✅ | ✅ | `OPENAI_API_KEY` |
| Azure OpenAI | ✅ | ✅ | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, deployment names |
| AWS Bedrock | ✅ | ✅ | AWS keys + region |
| Google (AI Studio / Vertex) | ✅ | ✅ | `GOOGLE_API_KEY` or GCP credentials |
| Anthropic | ✅ | n/a | `ANTHROPIC_API_KEY` |
| HuggingFace (local) | n/a | ✅ | none (runs on CPU) |

Pick the model per user in Settings; set the default for everyone with
`DEFAULT_LLM_PROVIDER` / `DEFAULT_LLM_MODEL` and `DEFAULT_EMBED_PROVIDER` /
`DEFAULT_EMBED_MODEL`.

---

## 4.16 Chat command reference

| Command | Who | Does |
|---|---|---|
| `/help` | all | List commands for your role |
| `/clear` | all | Delete your indexed documents and reset the conversation |
| `/reset` | all | Reset settings to defaults |
| `/sidebar` | all | Refresh the session panel |
| `/golden` | all | Run golden-dataset evaluation (upload CSV first) |
| `/export` | all | Download evaluations as Excel |
| `/logs` | engineer, admin | Query audit logs |
| `/stats` | admin | 24 h usage stats |
| `/create_user email=… password=… role=… name=…` | admin | Create a user |
| `/usage from=… to=… user=…` | admin | Usage and cost summary |
| `/set_quota uid=… tokens=… cost=… requests=…` | admin | Set a monthly quota |

---

## 4.17 Which features need which services

The app runs with **only OpenAI + Firebase**. Other features light up as
you add services:

| Feature | Local mode (Chroma, no DB, no Redis) | + Postgres / Supabase (`AUDIT_DB_URL`) | + Redis / Upstash | + Qdrant |
|---|---|---|---|---|
| Upload, ask, cite, rerank, OCR, vision | ✅ | ✅ | ✅ | ✅ |
| Semantic cache | ✅ local JSON | | ✅ shared across instances | |
| RAGAS per answer | ✅ | | ✅ survives profile switches | |
| Audit logs, `/stats`, `/logs` | file log only | ✅ | | |
| Conversation memory (server-side) | ❌ | ✅ | | |
| Usage & FinOps, quotas | ❌ | ✅ | | |
| Vector store | Chroma on disk | | | ✅ cloud, multi-instance |
| Admin → Clear Qdrant collection | use `/clear` | | | ✅ |

All the cloud services have free tiers (Supabase, Upstash, Qdrant Cloud).
A later chapter on deployment switches them on one at a time.

Next: [Chapter 5: Code walkthrough](05-code-walkthrough.md)
