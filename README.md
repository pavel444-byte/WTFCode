# WTFCode

A powerful CLI-based coding assistant that helps you write, edit, and manage code autonomously.

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
```bash
 uv pip install git+https://github.com/pavel444-byte/WTFCode.git
```
3. **Configure Environment**:
   Create a `.env` file in the root directory and add your API key:
   ```env
   OPENAI_API_KEY=your_actual_key_here
   ```

## Usage

Run the main script to start the assistant:
```bash
wtfcode
```

Follow the on-screen prompts to switch between **Agent** and **Ask** modes.
