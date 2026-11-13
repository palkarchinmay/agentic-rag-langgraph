# Chapter 9: Troubleshooting

Every problem in this chapter was hit for real while setting up and testing
this project, mostly on Windows. They are grouped by when they happen. Each
entry gives the **symptom** you see, the **cause**, and the **fix**.

> **First, find the trace id.** Every API request logs a short id in square
> brackets, such as `[a1b2c3d4] index request: …`. Search the API log
> (terminal 1, or `logs/rag_app.log`) for that id to see everything that
> request did. If the API log shows **nothing** for your action, the request
> never arrived: jump to 9.4.

## 9.1 Installing

### `ImportError: pyarrow requires NumPy 2.0 or newer`

- **Seen as:** at startup, or as a misleading `Reranker load failed … pip
  install sentence-transformers` warning, even though sentence-transformers
  *is* installed.
- **Cause:** pip installed a recent `pyarrow` built for NumPy 2. The project
  pins `numpy==1.26.4`. The reranker's real ImportError is reported with the
  wrong message.
- **Fix:** `pip install "pyarrow==21.0.0"` in `.venv-api`, then restart the
  API.

### numpy fails to build / "no matching distribution"

- **Cause:** Python 3.12 or newer. `numpy==1.26.4` has no wheels for it.
- **Fix:** create the venvs with Python 3.11: `py -3.11 -m venv .venv-api`.

### PowerShell: "running scripts is disabled on this system"

- **Fix:** run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy
  RemoteSigned` once in that window, then activate the venv again.

### Installing a package "breaks" the other service

- **Cause:** installing UI and API requirements into one environment.
  Chainlit and ChromaDB need different FastAPI/Starlette versions.
- **Fix:** keep the two venvs (chapter 2): `.venv-api` for uvicorn and
  `.venv-ui` for chainlit.

## 9.2 Starting the API

### `/health` hangs, or the API restarts in a loop

- **Cause:** `uvicorn … --reload` watches the whole folder, including
  `.venv-*`. Anything pip or Python writes there triggers a restart.
- **Fix:** run without `--reload`, or restrict it:
  `--reload --reload-dir app --reload-dir api`.

### "Address already in use" / port 8000 busy, or requests vanish

- **Cause:** an old uvicorn process (often an orphaned `--reload` worker
  whose parent has died) still holds port 8000. New requests go to the old,
  stuck process.
- **Diagnose (Windows):**
  ```powershell
  netstat -ano | findstr ":8000" | findstr LISTENING          # note the PID
  Get-CimInstance Win32_Process -Filter "ProcessId=<PID>" |
      Select-Object ProcessId, ParentProcessId, CommandLine
  ```
  If `Get-Process -Id <PID>` says "cannot find a process", that PID was the
  dead parent. Find its live children:
  ```powershell
  Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq <PID> } |
      Select-Object ProcessId, CommandLine
  ```
- **Fix:** `Stop-Process -Id <child PID> -Force`. Run `netstat` again; it
  must print nothing before you start uvicorn.
- **macOS / Linux:** `lsof -i :8000`, then `kill <PID>`.

### `CONFIG:` warnings at startup

- **Cause:** `validate_config()` found a missing or inconsistent setting, for
  example `CACHE_BACKEND=redis` without `REDIS_URL`. The app falls back to a
  safe default.
- **Fix:** read the warning. It names the variable.

## 9.3 Signing in

| Symptom | Cause | Fix |
|---|---|---|
| Login returns **500** | `FIREBASE_SERVICE_ACCOUNT_JSON` doesn't parse: wrapped in quotes, split over lines, or truncated | Put it on **one line, unquoted** (chapter 2.7 has a one-liner). Check with the `.env` check command in 2.7 |
| Login returns **400 "API key not valid"** | `FIREBASE_WEB_API_KEY` is a placeholder (`<…>`) or the wrong project's key | Copy the Web API key from Firebase → Project settings → General |
| **500 "Token used too early"** | Your PC clock is more than ~60 s off; Firebase rejects tokens "from the future" | Windows: Settings → Time & language → Sync now |
| Chainlit won't start: OAuth provider error | Chainlit requires an OAuth provider to be configured | Set `OAUTH_GOOGLE_CLIENT_ID` and `_SECRET` (chapter 2.6) |
| Google sign-in: `redirect_uri_mismatch` | The callback URL isn't registered | Add `http://localhost:8080/auth/oauth/google/callback` (or your deployed URL) to the OAuth client |
| Login hangs, then the UI times out | The UI can't reach the API | See 9.4 |
| Signed in, but no admin tabs | The user's role claim isn't `admin` | `python scripts/promote_to_admin.py`, then log out and back in, since roles are read from a fresh token |

### "Session expired, please log in again" after about an hour

- **Cause:** Firebase ID tokens last 1 hour, and the UI keeps the token from
  when you signed in. A page refresh reuses the same old token.
- **Fix:** log out (avatar → Logout) and log in again, or use a private
  window.

## 9.4 Uploads or questions never reach the API

- **Symptom:** the UI shows "Sending to API for processing…" or a spinner
  forever, and the API log shows **no** `index request` or `query` line.
  Login may still work.

Check these in order.

1. **Is the right API alive?**
   ```powershell
   Measure-Command { Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing }
   ```
   It should take well under a second. If it hangs, see "port 8000 busy" in
   9.2.

2. **Use `127.0.0.1`, not `localhost`.** On Windows, resolving `localhost`
   from Chainlit's worker threads was observed to hang forever. Set
   `RAG_API_URL=http://127.0.0.1:8000` in `.env`. Then check that the value
   was actually **saved** and is what Chainlit sees:
   ```powershell
   Select-String -Path .env -Pattern "RAG_API_URL"
   ```
   Restart Chainlit after changing it.

3. **CORS is not the cause here.** `CORS_ORIGINS` affects browsers calling
   the API directly. The UI calls the API from Python, server to server, so
   CORS doesn't apply. It should still list the **UI's** origin
   (`http://localhost:8080`), not port 8000.

4. **See exactly where Python is stuck.** Install `py-spy` in `.venv-ui` and
   dump the running Chainlit process:
   ```powershell
   pip install py-spy
   Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -like "*chainlit*" } |
       Select-Object ProcessId, ParentProcessId, CommandLine
   py-spy dump --pid <the child ProcessId>
   ```
   Use the **child** process. On Windows a venv's `python.exe` is a small
   launcher that starts the real interpreter as a child; py-spy on the
   launcher fails with "Failed to find python version". A stack ending in
   `socket.create_connection` means the TCP connection to the API never
   completes: steps 1 and 2.

With the patched `api_client.py`, a connection that can't be made fails
after `API_CONNECT_TIMEOUT_SECONDS` (default 10 s) with a clear error,
instead of hanging.

## 9.5 Indexing

| Symptom | Cause | Fix |
|---|---|---|
| UI says **"timed out"** but the API log later says `indexing complete` | Scanned PDFs take ~30 s per page with EasyOCR on CPU; the UI stopped waiting | Set `API_TIMEOUT_SECONDS=900` in `.env` and restart Chainlit. Use Tesseract or a cloud OCR engine for speed |
| `Expected collection name … [a-zA-Z0-9._-]` (Chroma) | Collection `user_<email>` contains `@`, which Chroma rejects | Apply the `chroma_safe_name` patch (`vector_store.py`, `retriever_service.py`) |
| Log says `scanned → easyocr` although you chose Tesseract or Document AI | The UI's engine name (`tesseract`, `google_docai`) didn't match the internal key, and unknown names fell back to EasyOCR | Apply the `ocr.py` / `index.py` patch (`normalize_method`) |
| `TesseractNotFoundError`, or a fallback-to-EasyOCR warning | The Tesseract program isn't installed or not on PATH | Install it (chapter 4.4) and set `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` |
| "75 chunks" but "67 upserted" | Not an error: the quality filter dropped short or garbled chunks | Nothing to fix. Very noisy scans lose more chunks; use a better OCR engine |
| `BM25 … total=134` after uploading the same file again | Uploading adds chunks; it doesn't replace them, so every chunk is now duplicated | Run `/clear`, then upload once |
| Answers miss text you can see in the PDF | Scanned page with poor OCR, or the fact is masked (`[SALARY]`, `[DATE]`) | Open 🔍 Diagnostics and read the chunks. Change OCR engine, or adjust `PII_SKIP_CATEGORIES` and re-index |

## 9.6 Answers

### "This information was not found in your uploaded document" for something that *is* in it

Work through the pipeline in order, using 🔍 Diagnostics:

1. **Is the text in any chunk?** If not, it's an ingestion problem: OCR
   quality, PII masking, or the file wasn't indexed. Run `/clear` and upload
   again.
2. **Is it in a chunk, but not in the retrieved top-k?** It's a retrieval
   problem:
   - raise `top_k` in ⚙️ Settings;
   - try retrieval mode `hybrid`;
   - rephrase with the document's own words.
3. **Retrieved, but the answer still failed validation?** The rerank score
   was low and the answer didn't match the sources closely enough (Tier 2 <
   0.60). That's common with very noisy OCR text. Better OCR fixes it.

### Diagnostics shows "rerank not applied"

- **Cause:** a display bug. The API didn't pass `rerank_score` to the UI,
  although reranking did run.
- **Fix:** apply `api-fixes-2026-11-13.zip`.

### A general question ("capital of France") is routed as RAG

- **This is by design.** Once you have documents indexed, the router sends
  most questions to retrieval. The validation node then sees the answer isn't
  in your document and answers from general knowledge, with a label. The cost
  is a little extra time.

### A question returns **403 "You can only access your own documents"**

- **Cause:** the request named another user's collection. This usually
  happens with an old browser session after switching accounts.
- **Fix:** log out and back in.

### **429** "quota exceeded" or "Rate limit reached"

- **Cause:** either the user's monthly quota (💰 Usage & FinOps), or the
  per-minute LLM limit (`LLM_RPM_LIMIT`, `LLM_TPM_LIMIT`).
- **Fix:** an admin raises or removes the quota, or you wait a minute.

### "LLM service unavailable"

- **Cause:** the circuit breaker opened after 5 consecutive LLM failures. It
  usually means a bad API key, an exhausted OpenAI balance, or a provider
  outage.
- **Fix:** check the API log for the first error, fix it, and wait 60 s.

### The same old answer keeps coming back

- **Cause:** a semantic-cache hit (the footer shows ⚡ cached).
- **Fix:** the cache clears automatically when you upload or `/clear`. To
  force it, rephrase the question substantially or clear the cache from the
  Admin panel.

## 9.7 Evaluation

| Symptom | Cause | Fix |
|---|---|---|
| No **📊 Evaluate** button | Cached answer, general or fallback answer, no chunks, or role `user` | Chapter 7.2, "When the button does not appear" |
| "RAGAS evaluation failed" | Usually an OpenAI error inside RAGAS (key, quota), or the stored data expired (2 h) | Check the API log for the eval key; ask the question again |
| Faithfulness low on a correct answer | The judge can't verify claims against noisy OCR text | Better OCR; compare with answer correctness on a golden set |
| `/golden` says "No CSV uploaded yet" | The CSV must be uploaded in the same chat session before `/golden` | Upload with 📎, then type `/golden` |
| "CSV must have a `question` column" | Header is missing or the file isn't comma-separated | Save as "CSV UTF-8" with a `question` header |
| Evaluations disappear when switching tabs | Local mode keeps them per session | Configure Redis (`CACHE_BACKEND=redis`) |

## 9.8 Tests

| Failing test | Cause | Status |
|---|---|---|
| `test_step5.py::test_query_router` | It imports `_heuristic_classify`, which was renamed to `_classify_query` | Stale test, fixed in the GitHub prep |
| `test_llm_guard.py::…test_timeout_kills_slow_llm` | Passes alone, fails in the full run: another test imports `llm_resilience` first with a 60 s timeout | Test-order bug, fixed in the GitHub prep |
| `test_conversation_service.py::TestTokenCounting…` | tiktoken downloads its encoding on first use; fails without internet | Run once with internet access |

Run a single test with its output visible:

```bash
python -m pytest -q tests/test_step5.py::test_query_router -p no:cacheprovider -s
```

## 9.9 Docker and Cloud Run

| Symptom | Cause | Fix |
|---|---|---|
| `docker compose up` fails mounting `users.json` | The file doesn't exist | `echo {} > users.json`, or remove that volume line |
| Build is huge and slow | No `.dockerignore`; the venvs are sent as build context | Add the `.dockerignore` from chapter 8.1 |
| `deploy_gcp.sh`: "failed to read dockerfile" | It builds with `Dockerfile`, which doesn't exist | Add `--file Dockerfile.api` |
| On Cloud Run, documents disappear after a while | Local Chroma on a container disk | `VECTOR_BACKEND=qdrant` (chapter 8.3) |
| Exact-term searches get worse on Cloud Run | The BM25 index is on local disk and is lost on new instances | Known limitation (chapter 8.3) |
| Indexing fails after exactly 300 s on Cloud Run | The service's request timeout | `gcloud run services update … --timeout=900` |
| Google login works locally but not deployed | The deployed callback URL isn't registered, or `CHAINLIT_URL` is unset | Chapter 8.5 |
| CI deploy: "Permission denied" or "Unable to acquire impersonated credentials" | Workload Identity binding is for a different repo, or the variables are wrong | Recheck step 5 of chapter 8.4, especially `OWNER/REPO` and `PROJECT_NUMBER` |

## 9.10 Getting help effectively

When you ask for help, in an issue or a chat, include:

1. what you did (the exact command, or what you clicked);
2. what you expected, and what happened instead;
3. the API log lines for that **trace id**;
4. OS, Python version (`python --version`), and whether you applied the
   patches.

**Never paste** `.env` contents, API keys, service-account JSON, ID tokens or
passwords. If one leaks, rotate it immediately:

- **OpenAI:** platform → API keys → revoke and create a new key.
- **Firebase password:** reset it in Authentication → Users.
- **Firebase service account:** delete the key and generate a new one.

Back to the [course index](README.md).
