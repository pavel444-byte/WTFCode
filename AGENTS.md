# AGENTS.md — Guide for AI Agents

This file gives AI coding agents everything needed to understand and work on **WTFCode**.

## Project Overview

WTFCode is a CLI-based AI coding assistant (v1.0.6) powered by LLMs. It has two chat modes
(**Agent** with autonomous tool use, **Ask** for plain Q&A), an optional Streamlit web UI,
and a customtkinter desktop GUI. It supports 6 LLM providers: `openai`, `anthropic`,
`openrouter`, `gemini`, `azure_openai`, `llama` (Ollama).

- Language: Python (requires **>= 3.13**, see `.python-version`)
- Package/env manager: **uv** — always use `uv sync`, `uv run`, `uv add` (never bare pip)
- Entry points (`pyproject.toml`): `wtfcode` / `WTFcode` → `main:start_cli`, `WTFdesktop` → `main:start_desktop`
- No test suite exists. The only CI quality gate is **pylint** (`.github/workflows/pylint.yml`,
  runs `pylint $(git ls-files '*.py')` on Python 3.9–3.13 for every push)

## Repository Layout

Flat module layout — no package directory:

| File | Purpose |
|---|---|
| `main.py` | Everything core: tool functions, `CodeAssist` class, CLI REPL (`start_cli`), slash commands (~1650 lines) |
| `ya_config.py` | YAML config management (`~/.wtfcode/config.yml`) + `.env` mirroring for LSP settings |
| `theme_manager.py` | Rich console theming (`dark`, `light`, `matrix`, `dracula`) |
| `web.py` | Streamlit chat UI reusing `CodeAssist` |
| `dekstop.py` | customtkinter desktop GUI (**filename typo is intentional — do not rename**) |
| `example.env` | Documents every supported env var |
| `CHANGELOG.md` | Release notes; README "Latest Release" section mirrors only the newest entry |

## Architecture (main.py)

### CodeAssist class (~L708)
The central assistant. Key facts:
- `__init__(provider, model)` builds the right SDK client per provider; exits if the required
  API key env var is missing. OpenRouter and Llama reuse the OpenAI client with a custom `base_url`.
- Keeps **separate conversation histories per provider family**: `openai_history`,
  `anthropic_history`, `gemini_history`. Use `add_context_message()` to inject context,
  `clear_context()` / `reset_history()` to reset.
- `run_agent(prompt, render=True)` — the agentic loop: sends history + tools, executes tool
  calls, loops until a text-only reply, returns the final text. UI callers (web/desktop) pass
  `render=False` and render the returned string themselves.
- `ask_only(prompt, render=True)` — stateless single-turn Q&A, no tools.
- Image context is **one-shot**: attached to the next message then stripped from history.
- Default models: openai/azure → `gpt-4o`, anthropic → `claude-3-5-sonnet-20241022`,
  openrouter → `openai/gpt-4o`, gemini → `gemini-1.5-flash`, llama → `llama3.2`.

### Agent tools (module-level functions, declared in `TOOL_SPECS`)
| Tool | Behavior |
|---|---|
| `read_file(path)` | Reads with `NNNN \|` line-number prefixes |
| `write_file(path, content)` | Create/overwrite; renders a unified diff panel |
| `edit_file(path, old_str, new_str)` | Exact single-occurrence replacement; errors on 0 or >1 matches |
| `execute_command(command)` | No shell — `shlex.split`, **shell metacharacters are rejected**; interactive y/n confirm; 120s timeout |
| `glob_search(pattern)` | `rglob` from project root |
| `mcp_call(server, tool, arguments)` | Spawns configured MCP server subprocess, stdio JSON-RPC |
| `git_commit(message)` | `git add .` + commit |

**Security invariant:** all file tool paths go through `_resolve_project_path()` which sandboxes
them inside `PROJECT_ROOT` (cwd at startup). Preserve this when modifying tools.

### CLI REPL (`start_cli`, ~L1405)
Slash commands: `/exit`, `/theme`, `/mode`, `/help`, `/init`, `/config {reload|create}`,
`/context clear`, `/context image {list|add|remove}`, `/lsp install|on|off`,
`/mcp {enable|disable|restart|install}`, `/commit`, `/web`, `/add {file}`, `/models`,
`/multiinput`. Anything else is sent to the assistant.

Input handling (`_read_user_query`): multi-line mode is default (`MULTILINE_INPUT=true`);
the prompt renders identically to single-line mode (`agent >: `) and submits on an empty line.
There is **no TUI mode** — it was removed; do not reintroduce `/tui`, `TUI_MODE`, or `tui.py`.

## Configuration

- Persistent config: `~/.wtfcode/config.yml` (auto-created on Linux/macOS, explicit on Windows
  via `/config create`). Managed exclusively through `ya_config.py`; the module-level `config`
  dict is imported everywhere.
- Env vars (see `example.env`): provider API keys, `PROVIDER`, `MODEL`, `WEB_MODE`,
  `MULTILINE_INPUT`, `THEME`, `AZURE_OPENAI_*`, `LLAMA_BASE_URL`, plus JSON blobs
  `LSP_SERVERS` / `LSP_SERVER_STATES`.
- Precedence: env vars win; config values are copied into env at import time only when unset.
- When adding a new persisted setting, update all of: `get_default_config()` in `ya_config.py`,
  the env-seeding block at the top of `main.py`, and `example.env`.

## Development Workflow

```bash
uv sync                 # install dependencies (use uv, not pip)
uv run wtfcode          # run the CLI
uv run streamlit run web.py   # run web UI
uv run python -m py_compile main.py ya_config.py  # quick syntax check
uvx pylint $(git ls-files '*.py')                 # what CI runs
```

## Conventions & Gotchas

- `dekstop.py` filename is misspelled on purpose; imports and `py-modules` in `pyproject.toml`
  reference it as-is.
- Local imports use a try/except pattern (`from ya_config import ...` falling back to
  `from .ya_config import ...`) — keep this for both script and package execution.
- Keep `requirements.txt` (used by CI) and `pyproject.toml` dependencies in sync; Windows-only
  deps use `sys_platform == "win32"` markers.
- If you add/remove a top-level module, update `py-modules` in `pyproject.toml`.
- Rich styling colors come from `theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]`;
  don't hardcode colors in new panels.
- New slash commands must be added to the `/help` panel text in `start_cli`.
- Version bumps touch: `pyproject.toml`, `__init__.py`, README release header, `CHANGELOG.md`,
  and `uv.lock`.
- README's "Latest Release" section must contain **only the single newest** CHANGELOG entry.
- Commit messages: include `Co-authored-by: openhands <openhands@all-hands.dev>` when committed
  by an agent.
