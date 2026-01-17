import os
import sys
import subprocess
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import time

try:
    import openai
    import anthropic
    import google.genai as genai
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.live import Live
    from rich.prompt import Prompt
    from dotenv import load_dotenv
except ImportError:
    print("Error: Missing dependencies. Run 'uv sync'")
    sys.exit(1)

load_dotenv()

console = Console()

def fetch_available_models(provider: str) -> List[str]:
    """Fetch available models for the given provider."""
    try:
        if provider == "openai":
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            models = client.models.list()
            return [m.id for m in models.data]
        elif provider == "openrouter":
            import requests
            api_key = os.getenv("OPENROUTER_API_KEY")
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
            if response.status_code == 200:
                return [m["id"] for m in response.json().get("data", [])]
            return []
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                client = anthropic.Anthropic(api_key=api_key)
                return [m.id for m in client.models.list().data]
            return ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20250219"]
        elif provider == "gemini":
            import requests            
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get("https://generativelanguage.googleapis.com/v1beta/models", headers=headers)
                if response.status_code == 200:
                    return [m["name"] for m in response.json().get("models", [])]
            return []
        return []
    except Exception as e:
        console.print(f"[red]Error fetching models for {provider}: {str(e)}[/red]")
        return []

def read_file(path: str) -> str:
    abs_path = os.path.abspath(path)
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return "".join([f"{i+1:4} | {line}" for i, line in enumerate(lines)])
    except Exception as e:
        return f"Error reading file {path}: {str(e)}"

def write_file(path: str, content: str) -> str:
    abs_path = os.path.abspath(path)
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file {path}: {str(e)}"

def edit_file(path: str, old_str: str, new_str: str) -> str:
    abs_path = os.path.abspath(path)
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if old_str not in content:
            return f"Error: The exact string to replace was not found in {path}. Ensure indentation matches."
        if content.count(old_str) > 1:
            return f"Error: Multiple occurrences of the search string found in {path}. Please provide more context."
        new_content = content.replace(old_str, new_str)
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return f"Successfully updated {path}"
    except Exception as e:
        return f"Error editing file {path}: {str(e)}"

def execute_command(command: str) -> str:
    try:
        console.print(Panel(f"[bold yellow]Command:[/bold yellow] {command}", title="Executing Bash", border_style="yellow"))
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        output = result.stdout
        if result.stderr:
            output += f"\n--- Errors ---\n{result.stderr}"
        return output if output.strip() else "Command executed successfully (no output)."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 120 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

def glob_search(pattern: str) -> str:
    try:
        files = list(Path(".").rglob(pattern))
        return "\n".join([str(f) for f in files if f.is_file()]) or "No files found matching pattern."
    except Exception as e:
        return f"Error during glob search: {str(e)}"

def get_latest_github_version(repo_url: str = "https://github.com/pavel444-byte/WTFcode.git") -> str:
    """Fetch the latest version/tag from the GitHub repository."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--tags", repo_url],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                # Get the latest tag (last line typically has the latest tag)
                for line in reversed(lines):
                    if line and 'refs/tags/' in line:
                        tag = line.split('refs/tags/')[-1].replace('^{}', '')
                        return tag
        return "unknown"
    except Exception as e:
        return "unavailable"

def get_config_or_prompt(env_key: str, prompt_text: str, choices: Optional[List[str]] = None, default: Optional[str] = None) -> str:
    """Get configuration value from .env, or prompt user if not found."""
    value = os.getenv(env_key)
    if value:
        return value
    
    if choices:
        return Prompt.ask(prompt_text, choices=choices, default=default or choices[0])
    else:
        return Prompt.ask(prompt_text, default=default or "")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file with line numbers for context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path to the file to read."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or overwrite an existing one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path where the file should be written."},
                    "content": {"type": "string", "description": "The content to write into the file."}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a specific block of text in a file. Very precise.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path to the file to edit."},
                    "old_str": {"type": "string", "description": "The exact text to find and replace."},
                    "new_str": {"type": "string", "description": "The text to replace it with."}
                },
                "required": ["path", "old_str", "new_str"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Run a bash/shell command. Useful for tests, builds, or git.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "glob_search",
            "description": "Find files in the project using glob patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g., 'src/**/*.py')."}
                },
                "required": ["pattern"]
            }
        }
    }
]

class CodeAssist:
    def __init__(self, provider: str = "openai", model: Optional[str] = None) -> None:
        self.provider = provider
        if model is None:
            if provider == "openai":
                model = "gpt-4o"
            elif provider == "anthropic":
                model = "claude-3-5-sonnet-20241022"
            elif provider == "openrouter":
                model = "openai/gpt-4o"
            elif provider == "gemini":
                model = "gemini-1.5-flash"
        self.model: str = model  # type: ignore
        
        # Validate model exists in provider
        self._validate_model()

        if provider in ["openai", "openrouter"]:
            api_key_env = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
            api_key = os.getenv(api_key_env)
            if not api_key:
                console.print(f"[bold red]Error:[/bold red] {api_key_env} not found in environment or .env file.")
                sys.exit(1)
            base_url = "https://openrouter.ai/api/v1" if provider == "openrouter" else None
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY not found in environment or .env file.")
                sys.exit(1)
            self.client = anthropic.Anthropic(api_key=api_key)
        elif provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                console.print("[bold red]Error:[/bold red] GOOGLE_API_KEY not found in environment or .env file.")
                sys.exit(1)
            self.client = genai.Client(api_key=api_key)  # type: ignore
        self.history = [
            {"role": "system", "content": """You are 'CodeAssist', a high-performance AI coding agent.
You help users by modifying code, running commands, and answering questions.
Guidelines:
1. When asked to fix/add features, use 'glob_search' to find files, 'read_file' to understand them, and 'edit_file' or 'write_file' to apply changes.
2. Always verify your work by running tests or the code using 'execute_command' if applicable.
3. Be concise and professional.
4. If a command is dangerous, warn the user first (though in this CLI, they are auto-executed)."""}
        ]

    def _validate_model(self) -> None:
        """Validate that the selected model exists in the provider."""
        available_models = fetch_available_models(self.provider)
        if available_models and self.model not in available_models:
            console.print(f"[bold yellow]Warning:[/bold yellow] Model '{self.model}' not found in {self.provider} provider.")
            console.print(f"Available models: {', '.join(available_models[:5])}{'...' if len(available_models) > 5 else ''}")
        elif not available_models:
            console.print(f"[bold yellow]Warning:[/bold yellow] Could not validate model availability for {self.provider}.")
        else:
            console.print(f"[bold green]Model validated:[/bold green] {self.model}")

    def process_tool_calls(self, tool_calls: list) -> list:
        results = []
        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            
            with console.status(f"[bold cyan]Tool Call: {name}({list(args.values())[0] if args else ''})..."):
                if name == "read_file": res = read_file(**args)
                elif name == "write_file": res = write_file(**args)
                elif name == "edit_file": res = edit_file(**args)
                elif name == "execute_command": res = execute_command(**args)
                elif name == "glob_search": res = glob_search(**args)
                else: res = f"Unknown tool: {name}"
            
            results.append({
                "tool_call_id": tc.id,
                "role": "tool",
                "name": name,
                "content": res
            })
        return results

    def run_agent(self, prompt: str) -> None:
        self.history.append({"role": "user", "content": prompt})
        
        while True:
            msg: Any = None
            if self.provider in ["openai", "openrouter"]:
                response = self.client.chat.completions.create(  # type: ignore
                    model=self.model,
                    messages=self.history,  # type: ignore
                    tools=TOOLS,  # type: ignore
                    tool_choice="auto"
                )
                msg = response.choices[0].message
            elif self.provider == "anthropic":
                response = self.client.messages.create(  # type: ignore
                    model=self.model,
                    max_tokens=4096,
                    messages=self.history,  # type: ignore
                    tools=TOOLS  # type: ignore
                )
                msg = response
            elif self.provider == "gemini":
                console.print("[red]Gemini provider does not support agent mode with tools yet.[/red]")
                return
            
            self.history.append(msg)  # type: ignore

            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_results = self.process_tool_calls(msg.tool_calls)
                self.history.extend(tool_results)
                continue

            content: str = ""
            if self.provider in ["openai", "openrouter"]:
                content = msg.content or ""
            elif self.provider == "anthropic":
                content = msg.content[0].text if msg.content else ""
            elif self.provider == "gemini":
                content = ""
            if content:
                console.print(Panel(Markdown(content), title="CodeAssist", border_style="green"))
            break

    def ask_only(self, prompt: str) -> None:
        """Standard Q&A mode without tool access for speed."""
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant. Answer the question directly."},
            {"role": "user", "content": prompt}
        ]
        content: str = ""
        if self.provider in ["openai", "openrouter"]:
            response = self.client.chat.completions.create(  # type: ignore
                model=self.model,
                messages=messages  # type: ignore
            )
            content = response.choices[0].message.content or ""
        elif self.provider == "anthropic":
            response = self.client.messages.create(  # type: ignore
                model=self.model,
                max_tokens=4096,
                messages=messages  # type: ignore
            )
            content = response.content[0].text if response.content else ""  # type: ignore
        elif self.provider == "gemini":
            response = self.client.generate_content(prompt)  # type: ignore
            content = response.text if hasattr(response, 'text') else ""
        console.print(Panel(Markdown(content), title="Ask Mode", border_style="blue"))

def start_cli() -> None:

    # Fetch latest version from GitHub
    with console.status("[bold cyan]Fetching latest version from GitHub..."):
        latest_version = get_latest_github_version()
    with console.status("[bold cyan]Starting WTFcode CLI..."):
        time.sleep(1.8)
    console.print(Panel.fit(
        "[bold green]WTFcode[/bold green]\n"
        "Auto Code Edit | Agent Mode | Ask Mode | Auto Bash\n"
        f"[dim]Latest: {latest_version}[/dim]",
    ))
    
    # Try to get provider from .env, otherwise prompt user
    provider = get_config_or_prompt(
        "PROVIDER",
        "[bold white]Provider[/bold white] ([cyan]openai[/cyan]/[green]anthropic[/green]/[yellow]openrouter[/yellow]/[blue]gemini[/blue])",
        choices=["openai", "anthropic", "openrouter", "gemini"],
        default="openai"
    )
    
    # Try to get model from .env, otherwise use default logic in CodeAssist
    model = os.getenv("MODEL")
    
    # Auto-fetch model info if specified in .env
    if model:
        with console.status("[bold cyan]Fetching model information..."):
            available_models = fetch_available_models(provider)
        if available_models and model in available_models:
            console.print(f"[bold green]Model loaded from .env:[/bold green] {model}")
        elif available_models:
            console.print(f"[bold yellow]Warning:[/bold yellow] Model '{model}' from .env not found in {provider} provider.")
    
    assistant: CodeAssist = CodeAssist(provider=provider, model=model)
    mode = "agent"
    
    while True:
        try:
            query = Prompt.ask(f"[bold {('cyan' if mode == 'agent' else 'blue')}]{mode}[/bold {('cyan' if mode == 'agent' else 'blue')}] [green]>").strip()
            
            if query.lower() == '/exit':
                time.sleep(1.6)
            with console.status("[bold magenta]Exiting WTFcode...[/bold magenta]"):
                time.sleep(1.6)
            with console.status("[bold magenta]Saving all...[/bold magenta]"):
                time.sleep(1.5)
            with console.status("[bold magenta]Exiting...[/bold magenta]"):
                time.sleep(1.4)
                exit()
            
            if query == '/mode':
                mode = Prompt.ask("\n[bold white]Switch Mode[/bold white] ([cyan]agent[/cyan]/[blue]ask[/blue])", choices=["agent", "ask"], default=mode).lower()
                console.print(f"[bold green]Mode switched to:[/bold green] {mode}")
                continue
            if query == '/help':
                console.print(Panel(
                    "[bold cyan]/mode[/bold cyan] - Switch between Agent and Ask modes\n"
                    "[bold cyan]/models[/bold cyan] - List and select available models for the current provider\n"
                    "[bold cyan]/exit[/bold cyan] - Exit the application\n"
                    "[bold cyan]/help[/bold cyan] - Show this help message",
                    title="Help", border_style="cyan"
                ))
                continue
            if query.startswith('/models'):
                console.print(f"\n[bold yellow]Fetching available models for {provider}...[/bold yellow]")
                available_models = fetch_available_models(provider)
                if available_models:
                    console.print(f"\n[bold green]Available Models for {provider}:[/bold green]")
                    for idx, model in enumerate(available_models[:20], 1):
                        console.print(f"  {idx}. {model}")
                    if len(available_models) > 20:
                        console.print(f"  ... and {len(available_models) - 20} more")
                    
                    # Prompt for model selection
                    selected_model = Prompt.ask("\n[bold white]Enter model name (or press Enter to skip)[/bold white]", default="")
                    if selected_model:
                        if selected_model in available_models:
                            assistant.model = selected_model
                            assistant._validate_model()
                            console.print(f"[bold green]Model updated to:[/bold green] {selected_model}")
                        else:
                            console.print(f"[bold red]Error:[/bold red] Model '{selected_model}' not found.")
                else:
                    console.print("[bold red]Could not fetch available models.[/bold red]")
                continue

            if not query:
                continue

            if mode == 'agent':
                assistant.run_agent(query)
            else:
                assistant.ask_only(query)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type '/exit' to quit.[/yellow]")
            continue
        except Exception as e:
            console.print(f"[bold red]Exception:[/bold red] {str(e)}")

if __name__ == "__main__":
    start_cli()