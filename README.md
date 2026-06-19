#   WTFCode
                                          

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/pavel444-byte/WTFCode)

                                 
A powerful CLI-based coding assistant that helps you write, edit, and manage code autonomously.


## Latest Release

**WTFCode 1.0.6** refreshes release metadata for the latest package build:
- bumped packaging metadata to 1.0.6;
- updated runtime MCP client version reporting;
- refreshed release documentation and lock metadata.

See [`CHANGELOG.md`](CHANGELOG.md) for full release notes.

## Features
- **Agent Mode**: Full autonomous tool use (Read, Write, Edit, Bash, Glob).
- **Ask Mode**: Quick Q&A for your codebase.
- **Auto Code Edit**: Precise file modifications using search and replace.
- **Auto Command Execute**: Runs shell commands and tests.
- **Optional TUI Mode**: OpenCode-style terminal workspace that can be enabled with `/tui on`, `.env`, or config.

## Tools Included
WTFCode uses the following tools to interact with your environment:
- `read_file`: Reads file content with line numbers.
- `write_file`: Creates or overwrites files.
- `edit_file`: Performs surgical text replacements in existing files.
- `execute_command`: Runs bash/shell commands.
- `glob_search`: Finds files using pattern matching.
- `mcp_call`: Calls a configured MCP server tool over stdio.

## Installation

1. **Install it**:

For linux/macos:

```bash
 uv pip install git+https://github.com/pavel444-byte/WTFCode.git
```
For windows:
```bash
 uv pip install git+https://github.com/pavel444-byte/WTFCode.git[windows]
```

3. **Configure Environment**:
   Create a `.env` file in the root directory and add your API key:
   ```env
   OPENAI_API_KEY=your_actual_key_here
   ```

   For Azure OpenAI, set these additional variables:
   ```env
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   PROVIDER=azure_openai
   ```

   For Llama (via Ollama), install [Ollama](https://ollama.com) and set:
   ```env
   LLAMA_BASE_URL=http://localhost:11434
   LLAMA_API_KEY=ollama
   PROVIDER=llama
   ```

   Optional MCP server configuration in `~/.wtfcode/config.yml`:
   ```yaml
   mcp_servers:
     filesystem:
       command: "npx"
       args: ["-y", "@modelcontextprotocol/server-filesystem", "."]
       env: {}
   ```

## Usage

Run CLI mode:
```bash
wtfcode
```

Follow the on-screen prompts to switch between **Agent** and **Ask** modes.

TUI mode is optional and is not enabled by default; the classic `main.py` CLI remains the default interface. Enable it interactively or at startup:
```bash
/tui on
/tui off
```

Startup configuration options:
```env
TUI_MODE=false
```

Or in `~/.wtfcode/config.yml`:
```yaml
settings:
  tui_mode: false
```

MCP management command:
```bash
/mcp enable <server>
/mcp disable <server>
/mcp restart <server>
/mcp install <server> <package_or_link> [extra_args...]
```

LSP management commands:
```bash
/lsp install <package_or_url> [args...]
/lsp on <lsp> [args...]
/lsp off <lsp> [args...]
```

`/lsp install` stores an LSP entry in `~/.wtfcode/config.yml`, enables it, and mirrors the JSON configuration into `.env` when that file exists. Package names use `npx -y`; URL-style entries use `uvx`. You can also manage `lsp_servers` and `lsp_server_states` manually in config, or with `LSP_SERVERS` and `LSP_SERVER_STATES` JSON values in `.env`.

Context management commands:
```bash
/context clear
/context image add /path/to/image.png
/context image list
/context image remove {index|all|/path/to/image.png}
```

Image context supports PNG, JPEG, GIF, and WebP files. Attached images are sent with the next Agent or Ask request, then automatically removed from context after that message.

Run Web mode (module-style invocation compatible with `web.py` imports):
```bash
uv run python -m streamlit run web.py
```
