# Codegen Studio — User Manual

Codegen Studio is a local web app for a coding agent. It works like a terminal-style coding assistant in your browser. It combines the three Sprint 5 notebooks into one app with three **modes**:

| Mode | Based on | What it does |
|---|---|---|
| **chat** | Notebook 1 — *Code generator with tools* | A conversational agent that reads, writes, lists and searches files in the workspace, and can optionally run shell commands. It remembers the conversation. |
| **bugbot** | Notebook 2 — *Self-correcting code agent* | Generates a self-contained Python program, **runs it**, fixes errors, and has an AI reviewer check it. It retries until the program works or it runs out of attempts. It can also edit code you paste in. |
| **agent** | Notebook 3 — *Production coding agent* | Works on a whole project folder. It plans the change using codebase search, writes the code, has an AI reviewer check it, **asks for your approval** with a diff, then writes the files and tests them. |

The backend uses **FastAPI**, **LangChain** and **LangGraph**. All models are called through **OpenRouter**.

---

## 1. Setup

### Requirements
- Python 3.10+ (the code uses `str | None` type hints)
- An [OpenRouter](https://openrouter.ai) API key

### Install
```bash
cd sprint5/codegen
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Dependencies: `fastapi`, `uvicorn`, `python-dotenv`, `pydantic>=2`, `langchain-core`, `langchain-openai`, `langchain-community`, `langchain-text-splitters`, `langgraph`, `faiss-cpu`.

### Configure the API key
Create a `.env` file **one level above** `codegen/`, at `sprint5/.env`, not inside `codegen/`:

```env
OPENROUTER_API_KEY=sk-or-...

# Optional
CODEGEN_MODEL=openai/gpt-4o-mini                       # default chat model
CODEGEN_EMBEDDING_MODEL=openai/text-embedding-3-small  # model for codebase search
```

| Variable | Default | Used for |
|---|---|---|
| `OPENROUTER_API_KEY` | — (required) | Every LLM and embedding call |
| `CODEGEN_MODEL` | `openai/gpt-4o-mini` | Default model in all modes. You can override it per session with `/model`. |
| `CODEGEN_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | FAISS semantic search over the workspace |

If the key is missing, the UI shows a red warning when it loads, and every request fails with `OPENROUTER_API_KEY not found`.

### Run
```bash
python server.py
```
Then open **http://127.0.0.1:8765**.

To use a different port, pass `--port`:
```bash
python server.py --port 9000
```
Then open **http://127.0.0.1:9000**.

### Two interfaces, same backend
| Address | Interface |
|---|---|
| `http://127.0.0.1:<port>/studio` | **Studio**: a friendly chat-style interface with a conversation sidebar, a mode picker, collapsible steps and an approval card (see §3a) |
| `http://127.0.0.1:<port>/` | **Classic**: the terminal-style interface with slash commands (see §3) |

Both use the same API, workspace and rules, so you can switch between them freely.

The server listens on **localhost only**. This is on purpose, because the agents can write files and run code and shell commands on your machine.

---

## 2. Folder layout

```
codegen/
├── server.py                    FastAPI app: API routes + serves the UI
├── rules.md                     Global coding rules applied to every mode
├── requirements.txt
├── agents/
│   ├── common.py                Config, LLM factory, workspace sandbox, code/shell runners
│   ├── chat_agent.py            chat mode   (NB1)
│   ├── bugbot.py                bugbot mode (NB2)
│   ├── orchestrator.py          agent mode  (NB3)
│   └── codebase.py              FAISS / keyword search over workspace files
├── static/index.html            Classic terminal-style UI (served at /)
├── static/studio.html           Friendly chat-style UI (served at /studio)
├── sample_project_template/     Pristine copy of the demo project
│   ├── app.py                   Streamlit "ChatGPT clone" UI
│   ├── chat.py                  OpenRouter streaming client
│   └── config.py                Title, icon, model, base URL
└── workspace/                   ← the agents' sandbox (created automatically, git-ignored)
    ├── sample_project/          Demo project, seeded on first run
    └── .scratch/                Temp files used by bugbot when running code
```

### The workspace sandbox
- All file operations are limited to `codegen/workspace/`. A path that resolves outside it is rejected with `Path '...' is outside the workspace sandbox`.
- Paths are always **relative to the workspace root**, for example `sample_project/app.py`. A leading `workspace/` added by the model is removed automatically.
- The first time the server starts with an empty workspace, it copies `sample_project_template/` to `workspace/sample_project/`. It never re-creates the project on its own after that. Use `/reset-sample` to restore it, or `/reset-sample <folder>` to empty any other project folder.
- `.git`, `__pycache__`, `.venv`, `venv`, `node_modules` and `.scratch` are hidden from file listings and search.
- Code that the agents run does **not** receive `OPENROUTER_API_KEY` in its environment.

---

## 3. Using the classic interface (`/`)

The UI looks like a terminal: a scrolling log, a `>` prompt at the bottom, and a status line below the prompt.

**Status line** shows: current mode · model · *(agent mode)* project picker · mode flags, such as `shell off`, `review on · attempts 3`, or `sequential`.

### Keyboard shortcuts
| Key | Action |
|---|---|
| `Enter` | Send |
| `Shift+Enter` | New line |
| `Shift+Tab` | Switch mode: chat → bugbot → agent → chat |
| `/` | Open the command menu. Use `↑`/`↓` to choose and `Tab` to complete. |
| `Esc` | Interrupt the running request |
| `1` / `2` | Approve or reject when agent mode is waiting for your decision and the prompt is empty |

### Files panel
The **files** button in the top-right corner, or `/files`, opens a panel listing every file in the workspace. Click a file to show its contents in the log. The panel refreshes after every run.

### Slash commands
| Command | Description |
|---|---|
| `/help` | List all commands |
| `/clear` | Clear the log and start a new chat conversation (new memory thread) |
| `/mode chat\|bugbot\|agent` | Switch mode |
| `/model <id>` | Set the OpenRouter model id, e.g. `/model anthropic/claude-sonnet-4.5`. Leave it blank to go back to the default. |
| `/rules <text>` | Set **session rules** that are added to every prompt. `/rules clear` removes them, and `/rules` on its own shows the current ones. |
| `/shell on\|off` | *(chat)* Allow the chat agent to run shell commands. Off by default. |
| `/review on\|off` | *(bugbot)* Turn the AI reviewer step on or off |
| `/attempts N` | *(bugbot)* Maximum generate→run attempts, from 1 to 6. The default is 3. |
| `/edit` | *(bugbot)* Show or hide the inline-edit box for pasting existing code. This also switches to bugbot mode. |
| `/parallel on\|off` | *(agent)* Write the files in parallel instead of one after another |
| `/project <dir>` | *(agent)* Choose which folder in `workspace/` the agent works on. A new name creates the folder. |
| `/history` | *(agent)* "Time travel": list every checkpoint of the last agent run |
| `/files` | Show or hide the workspace file panel |
| `/cat <path>` | Print a workspace file in the log |
| `/reset-sample [folder]` | With a folder name, e.g. `/reset-sample try1`, **delete everything inside** `workspace/<folder>/` and keep the empty folder. The UI asks you to confirm first. With no folder, restore `workspace/sample_project` from `sample_project_template/`. |

The command toggles (`/shell`, `/review`, `/parallel`) work without an argument and flip the current value.

**Saved settings:** mode, model, session rules, shell, parallel, review, attempts and project are saved in the browser's `localStorage`, so they are kept after a page reload. Conversation memory is not kept, because it lives in server memory (see §8).

---

## 3a. Using the Studio interface (`/studio`)

Studio does the same things as the classic interface through buttons and menus instead of slash commands.

**Layout**
- **Left sidebar**: *New conversation*, your recent conversations (click to reopen, hover to delete), and links to *Workspace files*, *Settings* and the *Classic terminal UI*. The sidebar button at the top left hides or shows it.
- **Centre**: the conversation. A new conversation starts with three mode cards and some example prompts you can click.
- **Message box** at the bottom:
  - **＋** opens a box where you paste existing code for Bugbot to change. This is the same as `/edit`, and it switches to Bugbot.
  - The **mode pill** (💬 Chat / 🐞 Bugbot / 🛠️ Agent) picks the mode.
  - In Agent mode, a **folder pill** picks the project folder or creates a new one.
  - The current model is shown on the right. **Enter** sends, **Shift+Enter** adds a new line. While a reply is running, the send button becomes **Stop** (or press **Esc**).

**What you see in each mode**
- **Chat**: the answer streams in as formatted text, and code blocks have a **Copy** button. Tool calls (read, write, list, search, shell) are grouped in a collapsible *Used N tools* box. Each one can show its output.
- **Bugbot**: a collapsible *How it got there* timeline (wrote → ran → fixed → reviewed), then a result card with the final code, its output, **Save to workspace…** and **Keep editing this code**.
- **Agent**: a step timeline (plan → code → AI review), then a **Review these changes** card with the plan, the reviewer's notes and a colour-coded diff for each file. Click **Apply changes** to save and test, or write what should be different and click **Ask for changes**. Typing feedback in the message box while a review card is waiting also asks for changes.

**Settings panel** (top right, or in the sidebar)
| Setting | Same as |
|---|---|
| Model | `/model` |
| Extra rules for this browser | `/rules` |
| Allow shell commands | `/shell` |
| AI review / Maximum attempts | `/review` / `/attempts` |
| Project folder / Write files in parallel | `/project` / `/parallel` |
| Restore sample_project / Empty project folder… | `/reset-sample` / `/reset-sample <folder>` |
| Theme (system, light or dark) | Studio only |

**Files panel** (top right): a folder tree of the workspace. Click a file to preview it, with a Copy button.

**Saved in your browser:** settings and up to 40 recent conversations are kept in `localStorage`. After a server restart, old conversations can still be read, but you can't continue them: chat memory and waiting agent runs live in server memory.

---

## 4. Chat mode (NB1)

Use this mode for general coding help inside the workspace: "create a FastAPI hello-world in `demo/main.py`", "explain `sample_project/chat.py`", "add a docstring to every function in …".

**How it works:** a LangGraph `agent ⇄ tools` loop over `MessagesState`. The model is given these tools:

| Tool | What it does |
|---|---|
| `read_file(filepath)` | Read a workspace file |
| `write_file(filepath, content)` | Write a complete file, creating folders as needed |
| `list_directory(directory=".")` | List files recursively |
| `search_codebase(query)` | Semantic search over workspace code (see §7) |
| `execute_shell(command)` | Run a shell command in the workspace. **Only when `/shell on`.** Timeout 30 s. |

- The system prompt asks for type hints, docstrings, PEP 8 and modern Python. It also asks the model to look at existing files before writing, and to save code to files rather than only printing it.
- Answers **stream** token by token. Each tool call is shown with its input and up to 4,000 characters of output.
- **Memory:** each conversation is a LangGraph thread backed by a `MemorySaver` checkpointer, so follow-up questions see earlier turns. `/clear` starts a new thread.
- The loop is limited to 50 graph steps per message.

---

## 5. Bugbot mode (NB2)

Use this mode for **standalone Python programs** that you want to know actually run: algorithms, utilities, data-processing snippets.

### The loop
```
generate ──► execute ──┬─ ok ─────► review ──┬─ approved ──► done
   ▲                   │                     │
   └── error (retry) ──┘                     └── rejected (revise) ──┐
   ▲                                                                 │
   └─────────────────────────────────────────────────────────────────┘
               (stops when attempts reach max_attempts)
```

1. **Generate** — the LLM returns structured output `{code, explanation}`. The code must be self-contained, run with `python script.py`, use no `input()` and no network, use the standard library unless the task says otherwise, and print its own demo or test results.
2. **Execute** — the code is written to `workspace/.scratch/`, run in a subprocess with a **10-second timeout**, and the temp file is deleted afterwards.
3. **Review** *(if `/review on`)* — an LLM reviewer returns `{approved, feedback}`. It only rejects for real bugs, wrong behaviour or rule violations, not small style issues.
4. On an execution error, the error and the previous code go back to *generate* ("fix the code"). On a rejected review, the reviewer's feedback goes back ("improve the code").

### Result
The final card shows one of these outcomes:
- **Success — executed and approved**, or **Success — executed** when review is off
- **Runs, but reviewer still has concerns**
- **Gave up — still failing**

It also shows the number of attempts, the explanation and the final code, with three buttons:
- **copy** — copy the code to the clipboard
- **save to workspace…** — save it under a path you choose (default `generated/solution.py`)
- **edit further (/edit)** — load the code into the inline-edit box for another round

### Inline edit (like Cmd+K)
Type `/edit` to open the "Existing code to edit" box. Paste code there, then describe the change in the prompt. Bugbot modifies that code instead of writing new code, and still runs it and reviews it.

---

## 6. Agent mode (NB3)

Use this mode for **feature work on a multi-file project** in the workspace, for example: "add a sidebar dropdown to pick the model" in `sample_project`.

### Pick a project
Use the **project** dropdown in the status line or `/project <dir>`. The agent only plans and writes inside `workspace/<project>/`. Choose **+ new folder…** to start an empty project. The folder is created when the agent writes its first file.

### The pipeline
```
plan ─► dispatch ─► code (sequential)   ─┐
            │                            ├─► AI review ──approved──► HUMAN REVIEW ──approve──► apply ─► test ─► done
            └─► code_file × N (parallel) ┘        │                       │
                     ▲                            │ needs revision        │ reject (+ feedback)
                     └────────────────────────────┴───────────────────────┘
```

1. **Plan** — searches the project with codebase search (§7), lists its files, and asks the LLM for a structured `Plan`: a one-line summary plus a list of file tasks `{filepath, description, action}`. The planner prefers changing existing files over creating new ones. Each path is moved under the project folder, and `create` / `modify` is set by checking whether the file already exists.
2. **Code**
   - **Sequential (default):** files are written one after another. Each coder sees the files already written in this pass, so names and APIs stay consistent.
   - **Parallel (`/parallel on`):** one coder per file runs at the same time using LangGraph's `Send` API. Each coder is told what the other files are changing.
   - **New files** are generated in full.
   - **Existing files** are changed with small **SEARCH/REPLACE** edit blocks, so code that isn't part of the task stays exactly as it was. Each SEARCH block must match exactly one place in the file. Differences in trailing whitespace are allowed.
   - Every Python result is checked with `compile()`. If parsing fails, or a SEARCH block doesn't match, the coder gets up to **2 repair rounds**. If it still fails, the run stops with an error.
3. **AI review** — the LLM approves or asks for revision. It rejects only for serious bugs, wrong behaviour or rule violations, such as wrong API URLs or model ids, or hardcoded secrets. A revision goes back to *dispatch* with the reviewer's feedback. After **2 rejections** the code is auto-approved and passed to you.
4. **Human review** — the graph **pauses** (LangGraph `interrupt`). You see the plan, the AI reviewer's comments and a **unified diff** for each file. Choose:
   - **1. Yes, apply changes and run tests**, or
   - **2. No, revise.** You can add feedback in the small box or just type it in the main prompt and press Enter. Typing in the prompt while a decision is waiting always counts as *reject with this feedback*. Rejecting resets the AI-review budget and re-codes with your feedback.
5. **Apply** — writes the approved files to disk.
6. **Test** — for each changed `.py` file:
   - `python -m py_compile` (**PASS/FAIL compile**)
   - `import <module>` from the file's folder (**PASS/FAIL import**). This step is skipped for entry-point or UI scripts that would start running when imported, which it detects by `__main__`, `streamlit`, `gradio`, `app.run(`, `uvicorn.run(`, `input(` or `mainloop(`.

   The run ends with `tests_passed` or `tests_failed`. It does **not** run your app or its unit tests. Use chat mode with `/shell on` for that.

### Time travel
Every step is checkpointed. `/history` shows a table of all checkpoints for the last agent run: step, status, next node, number of files, review attempts and plan.

---

## 7. Codebase search ("codebase brain")

Used by chat's `search_codebase` tool and by agent mode's planner (`agents/codebase.py`).

- Indexes text files with these extensions: `.py .md .txt .js .ts .tsx .html .css .json .toml .yaml .yml .cfg .ini`
- Splits them into chunks of 500 characters with 50 characters of overlap (`RecursiveCharacterTextSplitter`)
- Embeds the chunks with `CODEGEN_EMBEDDING_MODEL` through OpenRouter and stores them in an in-memory **FAISS** index. It returns the top 4 chunks.
- The index is cached per folder and only rebuilt when a file's modification time or size changes.
- **Fallback:** if embeddings fail (no key, model unavailable, network error), search switches to a simple keyword-count score, so search keeps working. The plan step shows which method was used: `faiss` or `keyword`.

---

## 8. Rules (like `.cursorrules`)

Rules are added to the prompts of **every mode** (chat system prompt, bugbot generator and reviewer, agent planner, coders and reviewer).

There are two layers:

1. **Global rules — `codegen/rules.md`.** The file is re-read on every request, so edits apply without a restart. Lines starting with `# ` (or a lone `#`) are comments and are stripped, but `##` headings are kept. The current file covers:
   - **Secrets:** never hardcode keys. Load them with `python-dotenv` and add it to `requirements.txt`.
   - Checked base URLs, endpoints, auth headers, key variable names and current model ids for **OpenRouter, Hugging Face Inference Providers, OpenAI, Google Gemini, Groq and DeepSeek**, plus a list of retired models and endpoints to avoid.
   - Prefer the `openai` SDK with a `base_url` for any OpenAI-compatible provider.
   - For unknown APIs, state the assumed endpoint and model in one constant with a comment.
2. **Session rules — `/rules <text>`.** Stored in the browser and added under a `## Session rules` heading.

Keep `rules.md` up to date. It is the main way to stop the models from generating outdated API URLs or model names.

---

## 9. HTTP API

The UI is a thin client over these endpoints. Streaming endpoints return **Server-Sent Events** (`data: {json}\n\n`). Errors arrive as `{"type": "error", "message": ...}`.

| Method & path | Body / params | Returns |
|---|---|---|
| `GET /` | — | The UI (`static/index.html`) |
| `GET /api/config` | — | `{model, has_key, workspace, projects}` |
| `GET /api/files` | — | `{files: [...]}`, all workspace files |
| `GET /api/file?path=` | `path` | `{path, content}` |
| `POST /api/save` | `{path, content}` | `{path}` |
| `POST /api/chat` | `{message, thread_id, model?, rules?, allow_shell?}` | SSE: `token`, `message_end`, `tool_start`, `tool_end`, `done` |
| `POST /api/bugbot` | `{task, existing_code?, rules?, max_attempts?=3, review?=true, model?}` | SSE: `node` (generate/execute/review), then `done` with `{status, attempts, code, explanation, output, error, review_feedback}` |
| `POST /api/agent/start` | `{feature_request, thread_id, project?="sample_project", parallel?=false, model?, rules?}` | SSE: `node` events, then `interrupt` with `{plan, review_result, review_attempts, files[{filepath, action, code, explanation, diff}]}` |
| `POST /api/agent/resume` | `{thread_id, decision: "approve"\|"reject", feedback?}` | SSE: continues the run. Ends with `done` `{status, test_output, files}`, or another `interrupt`. |
| `GET /api/agent/history/{thread_id}` | — | `{history: [checkpoints]}` |
| `POST /api/workspace/reset-sample` | `{folder?}` | `{ok, action, folder}`. Blank `folder` restores `sample_project` (`action: "restored"`). Otherwise it empties that folder (`action: "cleared"`). Returns 400 if the folder is missing, is the workspace root, or is outside the workspace. |

---

## 10. The sample project

`sample_project_template/` is a small **Streamlit "ChatGPT clone"** used to try out agent mode:
- `config.py` — page title and icon, `MODEL = "openai/gpt-4o-mini"`, OpenRouter `BASE_URL`
- `chat.py` — `get_client(api_key)` and `stream_response(client, messages)` using the `openai` SDK
- `app.py` — Streamlit chat UI with history in `st.session_state`. It reads `OPENROUTER_API_KEY` from the environment or from a sidebar input.

To run it yourself, which needs `pip install streamlit openai`:
```bash
cd workspace/sample_project
streamlit run app.py
```

Feature requests to try in agent mode:
- "Add a sidebar dropdown to choose between three OpenRouter models"
- "Add a 'Clear chat' button"
- "Add a system prompt setting in the sidebar"

---

## 11. Limits and good to know

- **Memory is in-process.** Chat threads and agent checkpoints use `MemorySaver`, so restarting the server loses them. A page reload starts a new chat thread.
- **Esc only stops the browser's stream.** The server may finish the current LLM call in the background. If you interrupt an agent run partway through, start a new one.
- **Bugbot runs Python only.** It allows 10 s per run and doesn't work with programs that need input or the network.
- **Agent tests are shallow:** they only check compiling and importing. They don't run your app or a test suite.
- **Safety:** generated code and shell commands run with your user's permissions. Only the *file tools* are limited to the workspace. Shell commands in chat mode, and the code bugbot runs, can reach anything your user can. Keep the server on localhost and leave `/shell` off unless you need it.
- Model output can vary between models. If a model has trouble with the structured output or edit format, try a stronger one, for example `/model anthropic/claude-sonnet-4.5`.

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| Red banner "OPENROUTER_API_KEY not found" | Put the key in `sprint5/.env` (one level above `codegen/`) and restart `python server.py` |
| "Backend not reachable" | The server isn't running, or the page wasn't opened at `http://127.0.0.1:<port>` (8765 unless you passed `--port`) |
| Search says `keyword` instead of `faiss` | The embedding call failed. Check the key and `CODEGEN_EMBEDDING_MODEL`. Search still works. |
| `Coder could not produce valid code for …` | The model's edits didn't match the file or had syntax errors 3 times in a row. Try again, rephrase, or use a stronger model. |
| `Path '…' is outside the workspace sandbox` | Use paths relative to `workspace/` |
| `sample_project` got messed up | `/reset-sample` |
| Generated code uses old URLs or model names | Add or correct the provider's section in `rules.md` (no restart needed) |
