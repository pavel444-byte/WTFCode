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
    from win11toast import toast
    import pygetwindow as gw
    from ya_config import config, init_config
except ImportError:
    print("Error: Missing dependencies. Run 'uv sync'")
    sys.exit(1)

load_dotenv()

# Update environment variables from config if they are not already set
if config.get("api_keys"):
    for provider, key in config["api_keys"].items():
        env_var = f"{provider.upper()}_API_KEY"
        if key and not os.getenv(env_var):
            os.environ[env_var] = key

console = Console()

def is_app_in_background() -> bool:
    """Check if the current terminal window is in the background."""
    try:
        active_window = gw.getActiveWindow()
        if not active_window:
            return True
        # Check if "WTFcode" or the current terminal title is in the active window title
        # This is a bit heuristic as terminal titles vary
        title = active_window.title.lower()
        return not ("wtfcode" in title or "powershell" in title or "cmd" in title or "terminal" in title)
    except Exception:
        return True

def send_notification(title: str, message: str):
    """Send a Windows notification if the app is in the background."""
    if is_app_in_background():
        try:
            toast(title, message, duration='short')
        except Exception:
            pass

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
        old_content = ""
        if os.path.exists(abs_path):
            with open(abs_path, 'r', encoding='utf-8') as f:
                old_content = f.read()
        
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        
        # Show diff if file exists
        if old_content:
            import difflib
            diff = list(difflib.unified_diff(
                old_content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}"
            ))
            if diff:
                console.print(Panel("".join(diff), title=f"Changes in {path}", border_style="blue"))
        else:
            console.print(Panel(f"New file created: {path}", border_style="green"))

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
        
        # Show diff before writing
        import difflib
        diff = list(difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}"
        ))
        if diff:
            console.print(Panel("".join(diff), title=f"Changes in {path}", border_style="blue"))

        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return f"Successfully updated {path}"
    except Exception as e:
        return f"Error editing file {path}: {str(e)}"

def execute_command(command: str, silent: bool = False) -> str:
    try:
        if not silent:
            console.print(Panel(f"[bold yellow]Command:[/bold yellow] {command}", title="Executing Bash", border_style="yellow"))
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        
        if silent:
            return (result.stdout + result.stderr).strip()

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

def git_commit(message: str) -> str:
    """Commit all changes in the repository with the given message."""
    try:
        # Check if it's a git repo
        if not os.path.exists(".git"):
            return "Error: Not a git repository."
        
        # Add all changes
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        
        # Commit
        result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
        
        if result.returncode == 0:
            return f"Successfully committed changes: {message}"
        elif "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            return "Nothing to commit, working tree clean."
        else:
            return f"Error committing changes: {result.stderr or result.stdout}"
    except Exception as e:
        return f"Error during git commit: {str(e)}"

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
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Commit all changes in the repository with a descriptive message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The commit message."}
                },
                "required": ["message"]
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
3. After making changes to the code, use 'git_commit' to commit your changes with a descriptive message.
4. Be concise and professional.
5. If a command is dangerous, warn the user first (though in this CLI, they are auto-executed)."""}
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
                elif name == "git_commit": res = git_commit(**args)
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
                # Prepare request parameters
                params = {
                    "model": self.model,
                    "messages": self.history,  # type: ignore
                    "tools": TOOLS,  # type: ignore
                    "tool_choice": "auto"
                }
                
                # Add reasoning for OpenRouter if applicable
                if self.provider == "openrouter":
                    params["extra_body"] = {"include_reasoning": True}

                response = self.client.chat.completions.create(**params)  # type: ignore
                msg = response.choices[0].message
                
                # Handle thinking/reasoning content for OpenRouter/OpenAI
                reasoning = getattr(msg, 'reasoning_content', None) or (msg.extra_body.get('reasoning') if hasattr(msg, 'extra_body') and msg.extra_body else None)
                
                # Check for reasoning in the message object itself (some SDK versions)
                if not reasoning and hasattr(msg, 'reasoning'):
                    reasoning = msg.reasoning

                # Fallback for some OpenRouter models that put reasoning in the content with <thought> tags
                if not reasoning and msg.content and ("<thought>" in msg.content or "<think>" in msg.content):
                    import re
                    thought_match = re.search(r'<(thought|think)>(.*?)</\1>', msg.content, re.DOTALL)
                    if thought_match:
                        reasoning = thought_match.group(2).strip()
                        # Remove thought from content so it's not displayed twice
                        msg.content = msg.content.replace(thought_match.group(0), "").strip()

                if reasoning:
                    console.print(Panel(Markdown(reasoning), title="Thinking Process", border_style="dim cyan"))
            elif self.provider == "anthropic":
                response = self.client.messages.create(  # type: ignore
                    model=self.model,
                    max_tokens=4096,
                    messages=self.history,  # type: ignore
                    tools=TOOLS  # type: ignore
                )
                msg = response
                # Handle Anthropic thinking (if supported by the model/API version)
                for content_block in getattr(msg, 'content', []):
                    if getattr(content_block, 'type', None) == 'thinking':
                        console.print(Panel(Markdown(content_block.thinking), title="Thinking Process", border_style="dim cyan"))
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
                console.print(Panel(Markdown(content), title="WTFCode", border_style="green"))
                send_notification("WTFcode: AI Answered", content[:100] + "..." if len(content) > 100 else content)
            break

    def ask_only(self, prompt: str) -> None:
        """Standard Q&A mode without tool access for speed."""
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant. Answer the question directly."},
            {"role": "user", "content": prompt}
        ]
        content: str = ""
        if self.provider in ["openai", "openrouter"]:
            # Prepare request parameters
            params = {
                "model": self.model,
                "messages": messages,  # type: ignore
            }
            
            # Add reasoning for OpenRouter if applicable
            if self.provider == "openrouter":
                params["extra_body"] = {"include_reasoning": True}

            response = self.client.chat.completions.create(**params)  # type: ignore
            msg = response.choices[0].message
            reasoning = getattr(msg, 'reasoning_content', None) or (msg.extra_body.get('reasoning') if hasattr(msg, 'extra_body') and msg.extra_body else None)
            
            # Check for reasoning in the message object itself (some SDK versions)
            if not reasoning and hasattr(msg, 'reasoning'):
                reasoning = msg.reasoning

            # Fallback for some OpenRouter models that put reasoning in the content with <thought> tags
            if not reasoning and msg.content and ("<thought>" in msg.content or "<think>" in msg.content):
                import re
                thought_match = re.search(r'<(thought|think)>(.*?)</\1>', msg.content, re.DOTALL)
                if thought_match:
                    reasoning = thought_match.group(2).strip()
                    # Remove thought from content so it's not displayed twice
                    msg.content = msg.content.replace(thought_match.group(0), "").strip()

            if reasoning:
                console.print(Panel(Markdown(reasoning), title="Thinking Process", border_style="dim cyan"))
            content = msg.content or ""
        elif self.provider == "anthropic":
            response = self.client.messages.create(  # type: ignore
                model=self.model,
                max_tokens=4096,
                messages=messages  # type: ignore
            )
            # Handle Anthropic thinking
            for content_block in getattr(response, 'content', []):
                if getattr(content_block, 'type', None) == 'thinking':
                    console.print(Panel(Markdown(content_block.thinking), title="Thinking Process", border_style="dim cyan"))
            content = response.content[0].text if response.content else ""  # type: ignore
        elif self.provider == "gemini":
            response = self.client.generate_content(prompt)  # type: ignore
            content = response.text if hasattr(response, 'text') else ""
        console.print(Panel(Markdown(content), title="Ask Mode", border_style="blue"))
        send_notification("WTFcode: AI Answered", content[:100] + "..." if len(content) > 100 else content)

    def generate_commit_message(self) -> str:
        """Generate a commit message based on git diff."""
        diff = execute_command("git diff --cached", silent=True)
        if not diff or not diff.strip():
            diff = execute_command("git diff", silent=True)
        
        if not diff or not diff.strip():
            return "Update"

        prompt = f"Generate a concise, professional git commit message for these changes:\n\n{diff[:4000]}"
        
        # Use a simplified internal call to get just the message
        messages = [
            {"role": "system", "content": "You are a git commit message generator. Return ONLY the message, no quotes or extra text."},
            {"role": "user", "content": prompt}
        ]
        
        content = ""
        if self.provider in ["openai", "openrouter"]:
            response = self.client.chat.completions.create(model=self.model, messages=messages) # type: ignore
            content = response.choices[0].message.content or "Update"
        elif self.provider == "anthropic":
            response = self.client.messages.create(model=self.model, max_tokens=100, messages=messages) # type: ignore
            content = response.content[0].text if response.content else "Update" # type: ignore
        elif self.provider == "gemini":
            response = self.client.generate_content(f"System: You are a git commit message generator. Return ONLY the message.\nUser: {prompt}") # type: ignore
            content = response.text if hasattr(response, 'text') else "Update"
            
        return content.strip().strip('"').strip("'")

def start_cli() -> None:

    # Check for WEB_MODE in .env
    if os.getenv("WEB_MODE", "").lower() == "true":
        console.print("[bold green]WEB_MODE detected in .env. Starting Streamlit...[/bold green]")
        try:
            subprocess.run(["streamlit", "run", "web.py"])
        except KeyboardInterrupt:
            console.print("\n[bold magenta]Exiting WTFcode...[/bold magenta]")
            sys.exit(0)
        return

    # Fetch latest version from GitHub
    with console.status("[bold cyan]Fetching latest version from GitHub..."):
        latest_version = get_latest_github_version()
    with console.status("[bold cyan]Starting WTFcode CLI..."):
        time.sleep(1.8)
    console.print(Panel.fit(
        "[bold green]WTFcode[/bold green]\n"
        "Auto Code Edit | Agent Mode | Ask Mode | Auto Bash",
        subtitle=f"[dim]{latest_version}[/dim]",
        border_style="green"
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
            
            if query == '/exit':
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
                    "[bold cyan]/web[/bold cyan] - Start the Streamlit web interface\n"
                    "[bold cyan]/add {file}[/bold cyan] - Add a file's content to the conversation context\n"
                    "[bold cyan]/commit[/bold cyan] - Generate a commit message and commit all changes\n"
                    "[bold cyan]/init[/bold cyan] - Initialize config folder or AGENTS.md\n"
                    "[bold cyan]/exit[/bold cyan] - Exit the application\n"
                    "[bold cyan]/help[/bold cyan] - Show this help message",
                    title="Help", border_style="cyan"
                ))
                continue
            if query == '/init':
                choice = Prompt.ask("\n[bold white]Initialize[/bold white] ([cyan]config[/cyan]/[green]agents[/green])", choices=["config", "agents"], default="config")
                if choice == "config":
                    with console.status("[bold cyan]Initializing config..."):
                        result = init_config()
                    console.print(f"[bold green]{result}[/bold green]")
                else:
                    agents_path = Path("AGENTS.md")
                    if agents_path.exists():
                        console.print("[bold yellow]AGENTS.md already exists.[/bold yellow]")
                    else:
                        content = "### Agents Instructions\nUse uv for installing, running, other (python)\n"
                        with open(agents_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        console.print("[bold green]AGENTS.md created successfully.[/bold green]")
                continue
            if query == '/commit':
                with console.status("[bold cyan]Generating commit message..."):
                    commit_msg = assistant.generate_commit_message()
                
                console.print(f"[bold green]Generated message:[/bold green] {commit_msg}")
                confirm = Prompt.ask("Commit with this message?", choices=["y", "n"], default="y")
                if confirm == "y":
                    with console.status("[bold cyan]Committing..."):
                        result = git_commit(commit_msg)
                    console.print(f"[bold green]{result}[/bold green]")
                continue
            if query == '/web':
                console.print("[bold green]Starting Streamlit web interface...[/bold green]")
                try:
                    subprocess.run(["streamlit", "run", "web.py"])
                except KeyboardInterrupt:
                    console.print("\n[bold magenta]Exiting Streamlit...[/bold magenta]")
                continue
            if query.startswith('/add '):
                file_to_add = query[5:].strip()
                if os.path.exists(file_to_add):
                    with console.status(f"[bold cyan]Reading {file_to_add}..."):
                        content = read_file(file_to_add)
                        assistant.history.append({
                            "role": "user",
                            "content": f"Context from file `{file_to_add}`:\n\n{content}"
                        })
                    console.print(f"[bold green]Added {file_to_add} to context.[/bold green]")
                else:
                    console.print(f"[bold red]Error:[/bold red] File '{file_to_add}' not found.")
                continue
            if query.startswith('/models') or query.startswith('/model'):
                console.print(f"\n[bold yellow]Fetching available models for {provider}...[/bold yellow]")
                available_models = fetch_available_models(provider)
                if available_models:
                    console.print(f"\n[bold green]Available Models for {provider}:[/bold green]")
                    for idx, model in enumerate(available_models[:20], 1):
                        console.print(f"  {idx}. {model}")
                    if len(available_models) > 20:
                        console.print(f"  ... and {len(available_models) - 20} more")
                    
                    # Prompt for model selection
                    selected_model = Prompt.ask("\n[bold white]Enter model name (or press Enter to skip to default)[/bold white]", default="deepseek/deepseek-v3.2")
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