# Chapter 2: Setup

By the end of this chapter you will have two Python environments, a working
`.env`, an admin user, and a green test run. Commands are given for
**Windows (PowerShell)** and **macOS / Linux (bash)**.

> Every pitfall in the yellow "⚠️" boxes below was hit during real testing of
> this project. Read them; they will save you hours.

## 2.1 Prerequisites

| Tool | Version | Check | Get it |
|---|---|---|---|
| Python | **3.11.x** exactly | `python --version` / `py -3.11 --version` | python.org |
| Git | any | `git --version` | git-scm.com |
| Disk space | ~6 GB free | | PyTorch + models |
| OpenAI API key | | | platform.openai.com → API keys |
| Firebase project | free tier | | console.firebase.google.com |
| Google OAuth client | free | | console.cloud.google.com |

> ⚠️ **Python 3.12+ will not work.** `requirements.txt` pins `numpy==1.26.4`,
> which has no wheels for 3.12+. Use 3.11.

**Optional:** [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
(much faster than EasyOCR on CPU for scanned PDFs).

## 2.2 Get the code

```bash
git clone https://github.com/palkarchinmay/agentic-rag-langgraph.git
cd agentic-rag-langgraph
```

## 2.3 Create the backend environment

The backend has the heavy ML stack (PyTorch, transformers, EasyOCR,
sentence-transformers). The install takes **10 to 20 minutes**.

**Windows**
```powershell
py -3.11 -m venv .venv-api
.venv-api\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3.11 -m venv .venv-api
source .venv-api/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **PowerShell says "running scripts is disabled"?** Run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` once in
> that window, then activate again.

Verify:
```bash
python -c "import fastapi, langgraph, chromadb, qdrant_client; print('backend env OK')"
python -c "from sentence_transformers import CrossEncoder; print('reranker libs OK')"
```

> ⚠️ **`ImportError: pyarrow requires NumPy 2.0 or newer`**: pip pulled a
> too-new `pyarrow`. Fix: `pip install "pyarrow==21.0.0"`. (The repo's
> `requirements.txt` pins this; older copies may not.)

## 2.4 Create the UI environment

The UI only needs Chainlit and an HTTP client, so it installs in about a
minute. Open a **second terminal** in the project folder:

**Windows**
```powershell
py -3.11 -m venv .venv-ui
.venv-ui\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.chainlit.txt
```

**macOS / Linux**
```bash
python3.11 -m venv .venv-ui
source .venv-ui/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.chainlit.txt
```

**Why two environments?** The UI and API are separate services (chapter 1).
Separate environments keep the UI small, avoid version clashes (Chainlit and
ChromaDB want different FastAPI/Starlette versions), and mirror how they are
deployed as two containers.

## 2.5 Set up Firebase (login)

1. In the [Firebase console](https://console.firebase.google.com) create a project.
2. **Build → Authentication → Get started → Sign-in method**: enable
   **Email/Password** (and **Google** if you want the Google button).
3. **Project settings → General**: copy the **Web API key** →
   `FIREBASE_WEB_API_KEY`.
4. **Project settings → Service accounts → Generate new private key**:
   downloads a JSON file → `FIREBASE_SERVICE_ACCOUNT_JSON` (see 2.7 for how to
   put it in `.env`).

> 🔐 The service-account JSON is a **secret**: it grants admin access to your
> Firebase project. Never commit it.

## 2.6 Set up Google OAuth (required by the UI)

The UI registers a Google sign-in callback, and **Chainlit refuses to start
unless an OAuth provider is configured**, even if you only use
email/password.

1. [Google Cloud console](https://console.cloud.google.com) → select the
   project Firebase created → **APIs & Services → Credentials → Create
   credentials → OAuth client ID → Web application**.
2. Authorised redirect URI: `http://localhost:8080/auth/oauth/google/callback`
3. Copy **Client ID** → `OAUTH_GOOGLE_CLIENT_ID` and **Client secret** →
   `OAUTH_GOOGLE_CLIENT_SECRET`.

## 2.7 Create `.env`

Copy the template and fill it in:

```bash
cp .env.example .env          # Windows: Copy-Item .env.example .env
```

The minimum local configuration (no cloud databases needed):

```ini
# ── Mode: local-first ──
SECRETS_BACKEND=local
VECTOR_BACKEND=chroma
CACHE_BACKEND=local
STORAGE_BACKEND=local
LOG_BACKEND=both
ANONYMIZED_TELEMETRY=False

# ── Where the UI finds the API ──
RAG_API_URL=http://127.0.0.1:8000
CORS_ORIGINS=http://localhost:8080,http://127.0.0.1:8080
API_TIMEOUT_SECONDS=900

# ── PII masking at ingestion (chapter 4.13) ──
PII_MASKING_ENABLED=true
PII_SKIP_CATEGORIES=DATE_ISO,DATE_DMY,DATE,SALARY

# ── OpenAI ──
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_EMBED_PROVIDER=openai
DEFAULT_EMBED_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...

# ── Firebase ──
AUTH_PROVIDER=firebase
FIREBASE_WEB_API_KEY=AIza...
FIREBASE_SERVICE_ACCOUNT_JSON={"type":"service_account","project_id":"...", ... }
INTERNAL_SERVICE_SECRET=<any long random string>

# ── Chainlit ──
CHAINLIT_AUTH_SECRET=<run: chainlit create-secret>
OAUTH_GOOGLE_CLIENT_ID=...apps.googleusercontent.com
OAUTH_GOOGLE_CLIENT_SECRET=...
```

Generate the two random secrets:
```bash
chainlit create-secret                                   # in .venv-ui → CHAINLIT_AUTH_SECRET
python -c "import secrets; print(secrets.token_urlsafe(48))"   # → INTERNAL_SERVICE_SECRET
```

> ⚠️ **Service-account JSON must be on one line, unquoted**, with the
> `\n` sequences inside `private_key` left as literal backslash-n. Wrapping it
> in double quotes makes python-dotenv turn `\n` into real newlines and the
> JSON stops parsing (symptom: login returns **500**).
> One-liner to flatten the downloaded file:
> ```bash
> python -c "import json,sys; print('FIREBASE_SERVICE_ACCOUNT_JSON=' + json.dumps(json.load(open(sys.argv[1]))))" path/to/service-account.json >> .env
> ```

> ⚠️ **Use `127.0.0.1`, not `localhost`, in `RAG_API_URL`.** On Windows,
> resolving `localhost` from Chainlit's worker threads was observed to hang
> forever: uploads and questions froze while login still worked.

> ⚠️ **No placeholders left behind.** A key set to a literal `<...>` value
> passes "is it set?" checks but fails at runtime ("API key not valid").

Check your `.env` without printing secrets:
```bash
python -c "import json; from dotenv import dotenv_values as d; v=d('.env'); req=['OPENAI_API_KEY','FIREBASE_WEB_API_KEY','FIREBASE_SERVICE_ACCOUNT_JSON','CHAINLIT_AUTH_SECRET','OAUTH_GOOGLE_CLIENT_ID','OAUTH_GOOGLE_CLIENT_SECRET','INTERNAL_SERVICE_SECRET']; print({k:('MISSING' if not v.get(k) else 'PLACEHOLDER' if v[k].startswith('<') else 'set') for k in req}); json.loads(v['FIREBASE_SERVICE_ACCOUNT_JSON']); print('service account JSON parses OK')"
```

## 2.8 Sync your clock (Windows especially)

Firebase rejects tokens if your computer's clock is off by more than 60
seconds ("Token used too early"). Windows: **Settings → Time & language →
Date & time → Set time automatically → Sync now**.

## 2.9 Create the first admin user

In the backend environment:
```bash
python scripts/create_first_admin.py
```
It asks for an email, password (min 6 characters) and name, and creates a
Firebase user with the `admin` role.

## 2.10 Run the tests

```bash
python -m pytest -q tests -p no:cacheprovider
```

Expected: **all pass**. The tests are fully
mocked, so they make no paid API calls. (The first run downloads tiktoken's
encoding files, so it needs internet once.)

## ✅ Checkpoint

- [ ] `backend env OK` and `reranker libs OK` printed in `.venv-api`
- [ ] `requirements.chainlit.txt` installed in `.venv-ui`
- [ ] The `.env` check prints only `set` and `service account JSON parses OK`
- [ ] An admin user exists in Firebase → Authentication → Users
- [ ] Tests pass

Next: [Chapter 3: First run](03-first-run.md)
