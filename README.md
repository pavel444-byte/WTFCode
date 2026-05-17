# WTFCode
Thank you, Devin for making this code more cool!

A powerful CLI-based coding assistant that helps you write, edit, and manage code autonomously.


## Latest Release

**WTFCode 1.0.4** focuses on stability for CLI/Web/Desktop usage:
- safer command streaming with timeout handling;
- fixed Streamlit chat history and Agent Mode rendering;
- assistant methods now return generated text for UI integrations;
- config/theme writes are now explicit instead of happening during imports.

See [`CHANGELOG.md`](CHANGELOG.md) for full release notes.

## Features
- **Agent Mode**: Full autonomous tool use (Read, Write, Edit, Bash, Glob).
- **Ask Mode**: Quick Q&A for your codebase.
- **Auto Code Edit**: Precise file modifications using search and replace.
- **Auto Command Execute**: Runs shell commands and tests.

## Tools Included
WTFCode uses the following tools to interact with your environment:
- `read_file`: Reads file content with line numbers.
- `write_file`: Creates or overwrites files.
- `edit_file`: Performs surgical text replacements in existing files.
- `execute_command`: Runs bash/shell commands.
- `glob_search`: Finds files using pattern matching.

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

## Usage

Run the main script to start the assistant:
```bash
wtfcode
```

Follow the on-screen prompts to switch between **Agent** and **Ask** modes.
