import os
import sys
import subprocess
import json
import shlex
import uuid
import base64
import mimetypes
import queue
import threading
from typing import List, Dict, Any, Optional, Union, cast
from dataclasses import dataclass
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path.cwd().resolve()
SHELL_METACHARACTERS = frozenset("|&;<>()$`\\!*?[]{}~\n")
COMMAND_TIMEOUT_SECONDS = 120
COMMAND_TERMINATION_GRACE_SECONDS = 2

try:
    import openai
    import anthropic
    import google.genai as genai
    from anthropic.types import MessageParam, ToolResultBlockParam, ToolUnionParam
    from openai.types.chat import (
        ChatCompletionAssistantMessageParam,
        ChatCompletionMessageParam,
        ChatCompletionMessageToolCallParam,
        ChatCompletionToolMessageParam,
        ChatCompletionToolParam,
    )
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.live import Live
    from rich.prompt import Prompt
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: [Missing dependencies: {e}]. Run 'uv sync'")
    sys.exit(1)

# Optional Windows-only dependencies
toast: Any = None
gw: Any = None

try:
    from win11toast import toast as _toast
    toast = _toast
    _HAS_TOAST = True
except ImportError:
    _HAS_TOAST = False

try:
    import pygetwindow as _gw
    gw = _gw
    _HAS_PYGETWINDOW = True
except ImportError:
    _HAS_PYGETWINDOW = False

# Import local modules with fallback for package imports
try:
    from ya_config import config, init_config, reload_config, get_config_path, set_mcp_server_state, set_lsp_server_state, set_tui_mode, upsert_mcp_server, upsert_lsp_server
    from theme_manager import ThemeManager
except ImportError:
    try:
        from .ya_config import config, init_config, reload_config, get_config_path, set_mcp_server_state, set_lsp_server_state, set_tui_mode, upsert_mcp_server, upsert_lsp_server
        from .theme_manager import ThemeManager
    except ImportError as e:
        # If both fail, provide helpful error message
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Error: Could not import local modules (ya_config, theme_manager).")
        print(f"Make sure they are in: {current_dir}")
        print(f"Original error: {e}")
        sys.exit(1)

load_dotenv()
def start_desktop() -> None:
    """Start the desktop application."""
    try:
        from dekstop import WTFCodeDesktop
    except ImportError:
        console.print("[bold red]Error:[/bold red] Missing dependencies for desktop mode. Run 'uv sync'")
        sys.exit(1)
    app = WTFCodeDesktop()
    app.mainloop()
# Update environment variables from config if they are not already set
if config.get("api_keys"):
    for provider, key in config["api_keys"].items():
        env_var = f"{provider.upper()}_API_KEY"
        if key and not os.getenv(env_var):
            os.environ[env_var] = key

# Update Azure OpenAI settings from config
if config.get("azure_openai"):
    azure_cfg = config["azure_openai"]
    if azure_cfg.get("endpoint") and not os.getenv("AZURE_OPENAI_ENDPOINT"):
        os.environ["AZURE_OPENAI_ENDPOINT"] = azure_cfg["endpoint"]
    if azure_cfg.get("api_version") and not os.getenv("AZURE_OPENAI_API_VERSION"):
        os.environ["AZURE_OPENAI_API_VERSION"] = azure_cfg["api_version"]

# Update Llama/Ollama settings from config
if config.get("llama"):
    llama_cfg = config["llama"]
    if llama_cfg.get("base_url") and not os.getenv("LLAMA_BASE_URL"):
        os.environ["LLAMA_BASE_URL"] = llama_cfg["base_url"]

# Update other settings from config
if not os.getenv("PROVIDER") and config.get("provider"):
    os.environ["PROVIDER"] = config["provider"]
if not os.getenv("MODEL") and config.get("model"):
    os.environ["MODEL"] = config["model"]
if config.get("settings"):
    if "theme" in config["settings"] and not os.getenv("THEME"):
        os.environ["THEME"] = config["settings"]["theme"]
    if "multi_line_input" in config["settings"] and not os.getenv("MULTILINE_INPUT"):
        os.environ["MULTILINE_INPUT"] = str(config["settings"]["multi_line_input"]).lower()
    if "tui_mode" in config["settings"] and not os.getenv("TUI_MODE"):
        os.environ["TUI_MODE"] = str(config["settings"]["tui_mode"]).lower()

console = Console()
theme_manager = ThemeManager(console)

def is_app_in_background() -> bool:
    """Check if the current terminal window is in the background."""
    if not _HAS_PYGETWINDOW:
        return False
    try:
        active_window = gw.getActiveWindow()
        if not active_window:
            return True
        title = active_window.title.lower()
        return not ("wtfcode" in title or "powershell" in title or "cmd" in title or "terminal" in title)
    except Exception:
        return True

def send_notification(title: str, message: str):
    """Send a desktop notification if the app is in the background."""
    if not _HAS_TOAST:
        return
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
        elif provider == "azure_openai":
            client = openai.AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            )
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
        elif provider == "llama":
            import requests
            base_url = os.getenv("LLAMA_BASE_URL", "http://localhost:11434")
            try:
                response = requests.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    return [m["name"] for m in response.json().get("models", [])]
            except requests.ConnectionError:
                console.print(f"[bold yellow]Warning:[/bold yellow] Could not connect to Ollama at {base_url}. Is Ollama running?")
            return []
        return []
    except Exception as e:
        console.print(f"[{theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['error']}]Error fetching models for {provider}: {str(e)}[/{theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['error']}]")
        return []

def _resolve_project_path(path: str) -> Path:
    """Resolve a user-supplied path and ensure it stays inside the current project."""
    requested_path = Path(path).expanduser()
    if requested_path.is_absolute():
        resolved_path = requested_path.resolve()
    else:
        resolved_path = (PROJECT_ROOT / requested_path).resolve()

    try:
        resolved_path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path '{path}' is outside the project root: {PROJECT_ROOT}") from exc

    return resolved_path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _parse_safe_command(command: str) -> List[str]:
    """Parse a command for subprocess execution without invoking a shell."""
    if not command.strip():
        raise ValueError("Command cannot be empty.")

    if any(char in command for char in SHELL_METACHARACTERS):
        raise ValueError(
            "Shell metacharacters are not allowed in tool commands. "
            "Run a direct executable with arguments instead."
        )

    args = shlex.split(command, posix=os.name != "nt")
    if not args:
        raise ValueError("Command cannot be empty.")
    return args


def read_file(path: str) -> str:
    try:
        safe_path = _resolve_project_path(path)
        if not safe_path.is_file():
            return f"Error reading file {path}: file does not exist or is not a regular file."
        with safe_path.open('r', encoding='utf-8') as f:
            lines = f.readlines()
        return "".join([f"{i+1:4} | {line}" for i, line in enumerate(lines)])
    except Exception as e:
        return f"Error reading file {path}: {str(e)}"

def write_file(path: str, content: str) -> str:
    try:
        safe_path = _resolve_project_path(path)
        old_content = ""
        if safe_path.exists():
            if not safe_path.is_file():
                return f"Error writing file {path}: target exists and is not a regular file."
            with safe_path.open('r', encoding='utf-8') as f:
                old_content = f.read()
        
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        display_path = _display_path(safe_path)
        
        # Show diff if file exists
        if old_content:
            import difflib
            diff = list(difflib.unified_diff(
                old_content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{display_path}",
                tofile=f"b/{display_path}"
            ))
            if diff:
                console.print(Panel("".join(diff), title=f"Changes in {display_path}", border_style=theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['panel.border']))
        else:
            console.print(Panel(f"New file created: {display_path}", border_style=theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['success']))

        with safe_path.open('w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {display_path}"
    except Exception as e:
        return f"Error writing file {path}: {str(e)}"

def edit_file(path: str, old_str: str, new_str: str) -> str:
    try:
        safe_path = _resolve_project_path(path)
        if not safe_path.is_file():
            return f"Error editing file {path}: file does not exist or is not a regular file."
        with safe_path.open('r', encoding='utf-8') as f:
            content = f.read()
        if old_str not in content:
            return f"Error: The exact string to replace was not found in {path}. Ensure indentation matches."
        if content.count(old_str) > 1:
            return f"Error: Multiple occurrences of the search string found in {path}. Please provide more context."
        
        new_content = content.replace(old_str, new_str)
        display_path = _display_path(safe_path)
        
        # Show diff before writing
        import difflib
        diff = list(difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{display_path}",
            tofile=f"b/{display_path}"
        ))
        if diff:
            console.print(Panel("".join(diff), title=f"Changes in {display_path}", border_style=theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['panel.border']))

        with safe_path.open('w', encoding='utf-8') as f:
            f.write(new_content)
        return f"Successfully updated {display_path}"
    except Exception as e:
        return f"Error editing file {path}: {str(e)}"

def _format_command_output(output: str) -> str:
    return output.strip() or "Command executed successfully (no output)."


def _timeout_error() -> str:
    return f"Error: Command timed out after {COMMAND_TIMEOUT_SECONDS} seconds."


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=COMMAND_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def execute_command(command: str, silent: bool = False) -> str:
    try:
        args = _parse_safe_command(command)
        if not silent:
            theme_color = theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['warning']
            console.print(Panel(f"[bold {theme_color}]Command:[/bold {theme_color}] {command}", title="Executing Bash", border_style=theme_color))
            confirm = cast(str, Prompt.ask(
                "[bold yellow]Run this command?[/bold yellow]",
                choices=["y", "n"],
                default="y"
            ))
            if confirm == "n":
                return "Command execution skipped by user."

        if not silent:
            output_lines: List[str] = []
            output_queue: queue.Queue[str] = queue.Queue()

            with Live(console=console, refresh_per_second=4) as live:
                process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=PROJECT_ROOT,
                    bufsize=1,
                )

                if process.stdout is None:
                    return "Error: Failed to capture command output."
                stdout = process.stdout

                def read_output() -> None:
                    try:
                        for line in stdout:
                            output_queue.put(line)
                    finally:
                        stdout.close()

                reader_thread = threading.Thread(target=read_output, daemon=True)
                reader_thread.start()

                start_time = time.monotonic()
                timed_out = False

                while True:
                    try:
                        line = output_queue.get(timeout=0.1)
                        output_lines.append(line)
                        live.update(Panel("".join(output_lines), title="Command Output", border_style=theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['panel.border']))
                    except queue.Empty:
                        pass

                    if process.poll() is not None and output_queue.empty():
                        break

                    if time.monotonic() - start_time > COMMAND_TIMEOUT_SECONDS:
                        timed_out = True
                        _terminate_process(process)
                        break

                while not output_queue.empty():
                    output_lines.append(output_queue.get())

                reader_thread.join(timeout=COMMAND_TERMINATION_GRACE_SECONDS)

                if output_lines:
                    live.update(Panel("".join(output_lines), title="Command Output", border_style=theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['panel.border']))

                if timed_out:
                    return _timeout_error()

            return _format_command_output("".join(output_lines))

        result = subprocess.run(args, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS, cwd=PROJECT_ROOT)
        output = result.stdout
        if result.stderr:
            output += f"\n--- Errors ---\n{result.stderr}"
        return _format_command_output(output)
    except subprocess.TimeoutExpired:
        return _timeout_error()
    except Exception as e:
        return f"Error executing command: {str(e)}"


def _read_user_query(mode: str) -> str:
    mode_color = theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['prompt']
    multi_input = os.getenv("MULTILINE_INPUT", "true").lower() == "true"

    if not multi_input:
        return cast(str, Prompt.ask(f"[{mode_color}]{mode}[/{mode_color}] [green]>"))

    console.print(f"[{mode_color}]{mode}[/{mode_color}] [green]>(multi-line enabled, submit with empty line)[/green]")
    lines: List[str] = []
    while True:
        line = console.input("")
        if not line:
            break
        lines.append(line)

    return "\n".join(lines)

def glob_search(pattern: str) -> str:
    try:
        files = list(Path(".").rglob(pattern))
        return "\n".join([str(f) for f in files if f.is_file()]) or "No files found matching pattern."
    except Exception as e:
        return f"Error during glob search: {str(e)}"

def mcp_call(server: str, tool: str, arguments: Optional[Dict[str, Any]] = None) -> str:
    """Call a tool from a configured MCP server over stdio JSON-RPC."""
    states = config.get("mcp_server_states", {})
    if isinstance(states, dict) and states.get(server) is False:
        return f"MCP server '{server}' is disabled. Run /mcp enable {server} to use it."
    servers = config.get("mcp_servers", {})
    if not isinstance(servers, dict) or server not in servers:
        return f"MCP server '{server}' not found in config['mcp_servers']."
    server_cfg = servers.get(server) or {}
    command = server_cfg.get("command")
    args = server_cfg.get("args", [])
    env = server_cfg.get("env", {})
    if not command:
        return f"MCP server '{server}' is missing required 'command' in configuration."

    proc_env = os.environ.copy()
    if isinstance(env, dict):
        proc_env.update({str(k): str(v) for k, v in env.items()})

    try:
        process = subprocess.Popen(
            [str(command), *[str(a) for a in args]],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=proc_env,
        )
    except Exception as e:
        return f"Failed to start MCP server '{server}': {e}"

    def _rpc(method: str, params: Optional[Dict[str, Any]] = None, expect_response: bool = True) -> Optional[Dict[str, Any]]:
        if not process.stdin or not process.stdout:
            raise RuntimeError("MCP process pipes are unavailable.")
        request: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        req_id = str(uuid.uuid4())
        if params is not None:
            request["params"] = params
        if expect_response:
            request["id"] = req_id
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        if not expect_response:
            return None
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed the stream unexpectedly.")
            msg = json.loads(line)
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise RuntimeError(str(msg["error"]))
                return cast(Dict[str, Any], msg.get("result", {}))

    try:
        _rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "wtfcode", "version": "1.0.6"},
        })
        _rpc("notifications/initialized", {}, expect_response=False)
        result = _rpc("tools/call", {"name": tool, "arguments": arguments or {}}) or {}
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"Error calling MCP tool '{tool}' on server '{server}': {e}"
    finally:
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            process.kill()

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
        return cast(str, Prompt.ask(prompt_text, choices=choices, default=default or choices[0]))
    else:
        return cast(str, Prompt.ask(prompt_text, default=default or ""))

SYSTEM_PROMPT = """You are 'CodeAssist', a high-performance AI coding agent.
You help users by modifying code, running commands, and answering questions.
Guidelines:
1. When asked to fix/add features, use 'glob_search' to find files, 'read_file' to understand them, and 'edit_file' or 'write_file' to apply changes.
2. Always verify your work by running tests or the code using 'execute_command' if applicable.
3. After making changes to the code, use 'git_commit' to commit your changes with a descriptive message.
4. Be concise and professional.
5. If a command is dangerous, warn the user first (though in this CLI, they are auto-executed)."""

OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openrouter", "azure_openai", "llama"}
SUPPORTED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


@dataclass(frozen=True)
class ContextImage:
    path: Path
    mime_type: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, object]


TOOL_SPECS = [
    ToolSpec(
        name="read_file",
        description="Read the contents of a file with line numbers for context.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path to the file to read."}
            },
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="write_file",
        description="Create a new file or overwrite an existing one.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path where the file should be written."},
                "content": {"type": "string", "description": "The content to write into the file."},
            },
            "required": ["path", "content"],
        },
    ),
    ToolSpec(
        name="edit_file",
        description="Replace a specific block of text in a file. Very precise.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path to the file to edit."},
                "old_str": {"type": "string", "description": "The exact text to find and replace."},
                "new_str": {"type": "string", "description": "The text to replace it with."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    ),
    ToolSpec(
        name="execute_command",
        description="Run a bash/shell command. Useful for tests, builds, or git.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute."}
            },
            "required": ["command"],
        },
    ),
    ToolSpec(
        name="glob_search",
        description="Find files in the project using glob patterns.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g., 'src/**/*.py')."}
            },
            "required": ["pattern"],
        },
    ),
    ToolSpec(
        name="git_commit",
        description="Commit all changes in the repository with a descriptive message.",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The commit message."}
            },
            "required": ["message"],
        },
    ),
    ToolSpec(
        name="mcp_call",
        description="Call a configured MCP server tool over stdio.",
        parameters={
            "type": "object",
            "properties": {
                "server": {"type": "string", "description": "MCP server name from config.mcp_servers."},
                "tool": {"type": "string", "description": "Tool name exposed by the MCP server."},
                "arguments": {"type": "object", "description": "Arguments object passed to the MCP tool."},
            },
            "required": ["server", "tool"],
        },
    ),
]


def _build_openai_tool(spec: ToolSpec) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _build_anthropic_tool(spec: ToolSpec) -> ToolUnionParam:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.parameters,
    }


OPENAI_TOOLS: List[ChatCompletionToolParam] = [_build_openai_tool(spec) for spec in TOOL_SPECS]
ANTHROPIC_TOOLS: List[ToolUnionParam] = [_build_anthropic_tool(spec) for spec in TOOL_SPECS]

# Cache for validated model lists per provider
_model_cache: Dict[str, List[str]] = {}


class CodeAssist:
    client: Union[openai.OpenAI, openai.AzureOpenAI, "anthropic.Anthropic", "genai.Client"]

    def __init__(self, provider: str = "openai", model: Optional[str] = None) -> None:
        self.provider = provider
        self.system_prompt = SYSTEM_PROMPT
        self.openai_history: List[ChatCompletionMessageParam] = []
        self.anthropic_history: List[MessageParam] = []
        self.gemini_history: List[Dict[str, str]] = []
        self.context_images: List[ContextImage] = []
        if model is None:
            if provider == "openai":
                model = "gpt-4o"
            elif provider == "azure_openai":
                model = "gpt-4o"
            elif provider == "anthropic":
                model = "claude-3-5-sonnet-20241022"
            elif provider == "openrouter":
                model = "openai/gpt-4o"
            elif provider == "gemini":
                model = "gemini-1.5-flash"
            elif provider == "llama":
                model = "llama3.2"
        if model is None:
            raise ValueError(f"Unsupported provider: {provider}")
        self.model = model
        
        # Validate model exists in provider (uses cache to avoid repeated API calls)
        self._validate_model()

        if provider in ["openai", "openrouter"]:
            api_key_env = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
            api_key = os.getenv(api_key_env)
            if not api_key:
                console.print(f"[bold red]Error:[/bold red] {api_key_env} not found in environment or .env file.")
                sys.exit(1)
            base_url = "https://openrouter.ai/api/v1" if provider == "openrouter" else None
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        elif provider == "azure_openai":
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not api_key:
                console.print("[bold red]Error:[/bold red] AZURE_OPENAI_API_KEY not found in environment or .env file.")
                sys.exit(1)
            if not endpoint:
                console.print("[bold red]Error:[/bold red] AZURE_OPENAI_ENDPOINT not found in environment or .env file.")
                sys.exit(1)
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            self.client = openai.AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version
            )
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
            self.client = genai.Client(api_key=api_key)
        elif provider == "llama":
            base_url = os.getenv("LLAMA_BASE_URL", "http://localhost:11434")
            api_key = os.getenv("LLAMA_API_KEY", "ollama")
            self.client = openai.OpenAI(api_key=api_key, base_url=f"{base_url}/v1")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.reset_history()

    def reset_history(self) -> None:
        """Reset conversation history for the active provider."""
        if self.provider in OPENAI_COMPATIBLE_PROVIDERS:
            self.openai_history.clear()
            self.openai_history.append({"role": "system", "content": self.system_prompt})
        elif self.provider == "anthropic":
            self.anthropic_history.clear()
        elif self.provider == "gemini":
            self.gemini_history.clear()
            self.gemini_history.append({"role": "user", "content": f"System instructions:\n{self.system_prompt}"})

    def clear_context(self) -> None:
        """Clear conversation history and all attached image context."""
        self.context_images.clear()
        self.reset_history()

    def _resolve_image_path(self, path: str) -> Path:
        image_path = Path(path).expanduser()
        if not image_path.is_absolute():
            image_path = (PROJECT_ROOT / image_path).resolve()
        else:
            image_path = image_path.resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image '{path}' not found.")
        if not image_path.is_file():
            raise ValueError(f"Image '{path}' is not a file.")
        return image_path

    def _detect_image_mime_type(self, path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_MIME_TYPES))
            raise ValueError(f"Unsupported image type for '{path}'. Supported types: {supported}.")
        return mime_type

    def add_context_image(self, path: str) -> str:
        """Attach an image file to the next AI request."""
        image_path = self._resolve_image_path(path)
        mime_type = self._detect_image_mime_type(image_path)
        if any(item.path == image_path for item in self.context_images):
            return f"Image already in context: {image_path}"
        self.context_images.append(ContextImage(path=image_path, mime_type=mime_type))
        return f"Added image to context: {image_path}"

    def remove_context_image(self, target: str) -> str:
        """Remove an image from the next AI request by path or 1-based list index."""
        if not self.context_images:
            return "No images are currently attached to context."
        target = target.strip()
        if target.lower() in {"all", "*"}:
            removed_count = len(self.context_images)
            self.context_images.clear()
            return f"Removed {removed_count} image(s) from context."
        if target.isdigit():
            index = int(target) - 1
            if 0 <= index < len(self.context_images):
                removed = self.context_images.pop(index)
                return f"Removed image from context: {removed.path}"
            return f"Image index out of range: {target}"
        image_path = self._resolve_image_path(target)
        for index, image in enumerate(self.context_images):
            if image.path == image_path:
                removed = self.context_images.pop(index)
                return f"Removed image from context: {removed.path}"
        return f"Image is not currently in context: {image_path}"

    def list_context_images(self) -> str:
        """List images attached to the next AI request."""
        if not self.context_images:
            return "No images are currently attached to context."
        lines = ["Images attached to context:"]
        for index, image in enumerate(self.context_images, start=1):
            lines.append(f"{index}. {image.path} ({image.mime_type})")
        return "\n".join(lines)

    def _build_context_text(self, prompt: str) -> str:
        if not self.context_images:
            return prompt
        image_list = "\n".join(f"- {image.path}" for image in self.context_images)
        return f"{prompt}\n\nAttached image context for this message only:\n{image_list}"

    def _clear_used_context_images(self, had_context_images: bool) -> None:
        """Remove one-shot image attachments after a message has used them."""
        if had_context_images:
            self.context_images.clear()

    def _encode_context_image(self, image: ContextImage) -> str:
        return base64.b64encode(image.path.read_bytes()).decode("ascii")

    def _build_openai_user_message(self, prompt: str) -> ChatCompletionMessageParam:
        if not self.context_images:
            return {"role": "user", "content": prompt}
        content: List[Dict[str, Any]] = [{"type": "text", "text": self._build_context_text(prompt)}]
        for image in self.context_images:
            encoded = self._encode_context_image(image)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"},
            })
        return cast(ChatCompletionMessageParam, {"role": "user", "content": content})

    def _build_anthropic_user_message(self, prompt: str) -> MessageParam:
        if not self.context_images:
            return {"role": "user", "content": prompt}
        content: List[Dict[str, Any]] = [{"type": "text", "text": self._build_context_text(prompt)}]
        for image in self.context_images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.mime_type,
                    "data": self._encode_context_image(image),
                },
            })
        return cast(MessageParam, {"role": "user", "content": content})

    def _build_gemini_user_message(self, prompt: str) -> Dict[str, Any]:
        if not self.context_images:
            return {"role": "user", "content": prompt}
        parts: List[Dict[str, Any]] = [{"text": self._build_context_text(prompt)}]
        for image in self.context_images:
            parts.append({
                "inline_data": {
                    "mime_type": image.mime_type,
                    "data": self._encode_context_image(image),
                }
            })
        return {"role": "user", "parts": parts, "content": self._build_context_text(prompt)}

    def _strip_openai_image_content(self, message: ChatCompletionMessageParam) -> ChatCompletionMessageParam:
        if message.get("role") != "user" or not isinstance(message.get("content"), list):
            return message
        return cast(ChatCompletionMessageParam, {"role": "user", "content": self._extract_text_from_content(message.get("content"))})

    def _strip_anthropic_image_content(self, message: MessageParam) -> MessageParam:
        if message.get("role") != "user" or not isinstance(message.get("content"), list):
            return message
        return cast(MessageParam, {"role": "user", "content": self._extract_text_from_content(message.get("content"))})

    def _extract_text_from_content(self, content: Any) -> str:
        """Extract text from provider-specific content payloads."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                    continue
                if isinstance(block, dict):
                    block_text = block.get("text") or block.get("content")
                    if isinstance(block_text, str):
                        text_parts.append(block_text)
                    continue
                block_text = getattr(block, "text", None) or getattr(block, "content", None)
                if isinstance(block_text, str):
                    text_parts.append(block_text)
            return "\n".join(part for part in text_parts if part).strip()
        return str(content)

    def get_last_assistant_text(self) -> str:
        """Return the last textual assistant response for the active provider."""
        if self.provider in OPENAI_COMPATIBLE_PROVIDERS:
            for message in reversed(self.openai_history):
                if message.get("role") == "assistant":
                    return self._extract_text_from_content(message.get("content"))
            return ""
        if self.provider == "anthropic":
            for message in reversed(self.anthropic_history):
                if message.get("role") == "assistant":
                    return self._extract_text_from_content(message.get("content"))
            return ""
        if self.provider == "gemini":
            for message in reversed(self.gemini_history):
                if message.get("role") == "model":
                    return self._extract_text_from_content(message.get("content"))
            return ""
        return ""

    def _validate_model(self) -> None:
        """Validate that the selected model exists in the provider. Uses a cache to avoid repeated API calls."""
        if self.provider in _model_cache:
            available_models = _model_cache[self.provider]
        else:
            available_models = fetch_available_models(self.provider)
            if available_models:
                _model_cache[self.provider] = available_models

        if available_models and self.model not in available_models:
            console.print(f"[bold yellow]Warning:[/bold yellow] Model '{self.model}' not found in {self.provider} provider.")
            console.print(f"Available models: {', '.join(available_models[:5])}{'...' if len(available_models) > 5 else ''}")
        elif not available_models:
            console.print(f"[bold yellow]Warning:[/bold yellow] Could not validate model availability for {self.provider}.")
        else:
            console.print(f"[bold green]Model validated:[/bold green] {self.model}")

    def add_context_message(self, content: str, include_images: bool = False) -> None:
        if self.provider in OPENAI_COMPATIBLE_PROVIDERS:
            message = self._build_openai_user_message(content) if include_images else {"role": "user", "content": content}
            self.openai_history.append(message)
        elif self.provider == "anthropic":
            message = self._build_anthropic_user_message(content) if include_images else {"role": "user", "content": content}
            self.anthropic_history.append(message)
        elif self.provider == "gemini":
            message = self._build_gemini_user_message(content) if include_images else {"role": "user", "content": content}
            self.gemini_history.append(cast(Dict[str, str], message))

    def _run_local_tool(self, name: str, args: Dict[str, Any]) -> str:
        info_color = theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['info']
        with console.status(f"[bold {info_color}]Tool Call: {name}({list(args.values())[0] if args else ''})..."):
            if name == "read_file":
                return read_file(**args)
            if name == "write_file":
                return write_file(**args)
            if name == "edit_file":
                return edit_file(**args)
            if name == "execute_command":
                return execute_command(**args)
            if name == "glob_search":
                return glob_search(**args)
            if name == "git_commit":
                return git_commit(**args)
            if name == "mcp_call":
                return mcp_call(**args)
            return f"Unknown tool: {name}"

    def _build_openai_assistant_message(self, msg: Any) -> ChatCompletionAssistantMessageParam:
        assistant_message: ChatCompletionAssistantMessageParam = {"role": "assistant"}
        if msg.content is not None:
            assistant_message["content"] = msg.content
        elif not getattr(msg, "tool_calls", None):
            assistant_message["content"] = ""

        if getattr(msg, "tool_calls", None):
            tool_calls: List[ChatCompletionMessageToolCallParam] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in msg.tool_calls
            ]
            assistant_message["tool_calls"] = tool_calls
        return assistant_message

    def process_tool_calls(self, tool_calls: list[Any]) -> list[ChatCompletionToolMessageParam]:
        results: list[ChatCompletionToolMessageParam] = []
        for tc in tool_calls:
            name = tc.function.name
            args = cast(Dict[str, Any], json.loads(tc.function.arguments))
            res = self._run_local_tool(name, args)
            results.append({
                "tool_call_id": tc.id,
                "role": "tool",
                "content": res,
            })
        return results

    def process_anthropic_tool_uses(self, content_blocks: list[Any]) -> list[ToolResultBlockParam]:
        results: list[ToolResultBlockParam] = []
        for block in content_blocks:
            if getattr(block, "type", None) != "tool_use":
                continue
            args = cast(Dict[str, Any], block.input)
            res = self._run_local_tool(block.name, args)
            results.append({
                "tool_use_id": block.id,
                "type": "tool_result",
                "content": res,
            })
        return results

    def _extract_anthropic_text(self, msg: Any) -> str:
        text_parts = [
            block.text
            for block in getattr(msg, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        return "\n".join(text_parts).strip()

    def _extract_reasoning(self, msg: Any) -> Optional[str]:
        """Extract reasoning/thinking content from a provider response message."""
        reasoning = None
        if self.provider in ["openai", "openrouter", "azure_openai", "llama"]:
            reasoning = getattr(msg, 'reasoning_content', None) or (
                msg.extra_body.get('reasoning') if hasattr(msg, 'extra_body') and msg.extra_body else None
            )
            if not reasoning and hasattr(msg, 'reasoning'):
                reasoning = msg.reasoning
            if not reasoning and msg.content and ("<thought>" in msg.content or "<think>" in msg.content):
                import re
                thought_match = re.search(r'<(thought|think)>(.*?)</\1>', msg.content, re.DOTALL)
                if thought_match:
                    reasoning = thought_match.group(2).strip()
                    msg.content = msg.content.replace(thought_match.group(0), "").strip()
        elif self.provider == "anthropic":
            for content_block in getattr(msg, 'content', []):
                if getattr(content_block, 'type', None) == 'thinking':
                    reasoning = content_block.thinking
                    break
        return reasoning

    def _display_reasoning(self, reasoning: Optional[str]) -> None:
        """Display reasoning/thinking content if present."""
        if reasoning:
            console.print(Panel(
                Markdown(reasoning),
                title="Thinking Process",
                border_style=theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['thinking']
            ))

    def run_agent(self, prompt: str, render: bool = True) -> str:
        """Run agent mode with tool access and return the final assistant text."""
        had_context_images = bool(self.context_images)
        openai_user_message_index = len(self.openai_history) if self.provider in OPENAI_COMPATIBLE_PROVIDERS else -1
        anthropic_user_message_index = len(self.anthropic_history) if self.provider == "anthropic" else -1
        self.add_context_message(prompt, include_images=True)
        final_content = ""
        
        while True:
            msg: Any = None
            try:
                if self.provider in OPENAI_COMPATIBLE_PROVIDERS:
                    client = cast(Union[openai.OpenAI, openai.AzureOpenAI], self.client)
                    if self.provider == "openrouter":
                        response = client.chat.completions.create(
                            model=self.model,
                            messages=self.openai_history,
                            tools=OPENAI_TOOLS,
                            tool_choice="auto",
                            extra_body={"include_reasoning": True},
                        )
                    else:
                        response = client.chat.completions.create(
                            model=self.model,
                            messages=self.openai_history,
                            tools=OPENAI_TOOLS,
                            tool_choice="auto",
                        )
                    msg = response.choices[0].message

                elif self.provider == "anthropic":
                    client = cast(anthropic.Anthropic, self.client)
                    response = client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        system=self.system_prompt,
                        messages=self.anthropic_history,
                        tools=ANTHROPIC_TOOLS,
                    )
                    msg = response

                elif self.provider == "gemini":
                    client = cast(genai.Client, self.client)
                    gemini_messages = []
                    for m in self.gemini_history:
                        role = "user" if m["role"] == "user" else "model"
                        if "parts" in m:
                            gemini_messages.append({"role": role, "parts": cast(Any, m["parts"])})
                        else:
                            gemini_messages.append({"role": role, "parts": [m["content"]]})
                    response = client.models.generate_content(
                        model=self.model,
                        contents=cast(Any, gemini_messages),
                    )
                    msg = response

            except Exception as e:
                logger.exception("API call failed in agent loop")
                if render:
                    console.print(f"[bold red]API Error:[/bold red] {e}")
                    console.print("[yellow]Retrying may help. The failed response was not added to history.[/yellow]")
                break

            reasoning = self._extract_reasoning(msg)
            if render:
                self._display_reasoning(reasoning)

            if self.provider in OPENAI_COMPATIBLE_PROVIDERS:
                self.openai_history.append(self._build_openai_assistant_message(msg))
            elif self.provider == "anthropic":
                self.anthropic_history.append({"role": "assistant", "content": msg.content})
            elif self.provider == "gemini":
                gemini_text = cast(str, getattr(msg, "text", "") or "")
                self.gemini_history.append({"role": "model", "content": gemini_text})

            if self.provider in OPENAI_COMPATIBLE_PROVIDERS and getattr(msg, "tool_calls", None):
                tool_results = self.process_tool_calls(msg.tool_calls)
                self.openai_history.extend(tool_results)
                continue
            if self.provider == "anthropic":
                tool_results = self.process_anthropic_tool_uses(msg.content)
                if tool_results:
                    self.anthropic_history.append({"role": "user", "content": tool_results})
                    continue
            if self.provider == "gemini" and not cast(str, getattr(msg, "text", "") or ""):
                continue

            content: str = ""
            if self.provider in OPENAI_COMPATIBLE_PROVIDERS:
                content = msg.content or ""
            elif self.provider == "anthropic":
                content = self._extract_anthropic_text(msg)
            elif self.provider == "gemini":
                content = cast(str, getattr(msg, 'text', '') or '')
            if content:
                final_content = content
                if render:
                    console.print(Panel(Markdown(content), title="WTFCode", border_style=theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['success']))
                    send_notification("WTFcode: AI Answered", content[:100] + "..." if len(content) > 100 else content)
            break
        if self.provider in OPENAI_COMPATIBLE_PROVIDERS and openai_user_message_index >= 0 and openai_user_message_index < len(self.openai_history):
            self.openai_history[openai_user_message_index] = self._strip_openai_image_content(self.openai_history[openai_user_message_index])
        if self.provider == "anthropic" and anthropic_user_message_index >= 0 and anthropic_user_message_index < len(self.anthropic_history):
            self.anthropic_history[anthropic_user_message_index] = self._strip_anthropic_image_content(self.anthropic_history[anthropic_user_message_index])
        if self.provider == "gemini":
            for message in self.gemini_history:
                if "parts" in message:
                    message.pop("parts", None)
        self._clear_used_context_images(had_context_images)
        return final_content

    def ask_only(self, prompt: str, render: bool = True) -> str:
        """Standard Q&A mode without tool access for speed; return the answer text."""
        had_context_images = bool(self.context_images)
        ask_system_prompt = "You are a helpful coding assistant. Answer the question directly."
        openai_messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": ask_system_prompt},
            self._build_openai_user_message(prompt),
        ]
        anthropic_messages: List[MessageParam] = [self._build_anthropic_user_message(prompt)]
        content: str = ""
        try:
            if self.provider in OPENAI_COMPATIBLE_PROVIDERS:
                client = cast(Union[openai.OpenAI, openai.AzureOpenAI], self.client)
                if self.provider == "openrouter":
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=openai_messages,
                        extra_body={"include_reasoning": True},
                    )
                else:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=openai_messages,
                    )
                msg = response.choices[0].message
                reasoning = self._extract_reasoning(msg)
                if render:
                    self._display_reasoning(reasoning)
                content = msg.content or ""
            elif self.provider == "anthropic":
                client = cast(anthropic.Anthropic, self.client)
                response = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=ask_system_prompt,
                    messages=anthropic_messages,
                )
                reasoning = self._extract_reasoning(response)
                if render:
                    self._display_reasoning(reasoning)
                content = self._extract_anthropic_text(response)
            elif self.provider == "gemini":
                client = cast(genai.Client, self.client)
                gemini_user = self._build_gemini_user_message(f"System: {ask_system_prompt}\nUser: {prompt}")
                contents = cast(Any, [{"role": "user", "parts": gemini_user.get("parts", [gemini_user["content"]])}])
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                )
                content = cast(str, getattr(response, 'text', '') or '')
        except Exception as e:
            logger.exception("API call failed in ask mode")
            if render:
                console.print(f"[bold red]API Error:[/bold red] {e}")
            self._clear_used_context_images(had_context_images)
            return ""
        if render:
            console.print(Panel(Markdown(content), title="Ask Mode", border_style=theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['panel.border']))
            send_notification("WTFcode: AI Answered", content[:100] + "..." if len(content) > 100 else content)
        self._clear_used_context_images(had_context_images)
        return content

    def generate_commit_message(self) -> str:
        """Generate a commit message based on git diff."""
        diff = execute_command("git diff --cached", silent=True)
        if not diff or not diff.strip():
            diff = execute_command("git diff", silent=True)
        
        if not diff or not diff.strip():
            return "Update"

        prompt = f"Generate a concise, professional git commit message for these changes:\n\n{diff[:4000]}"
        
        # Use a simplified internal call to get just the message
        commit_system_prompt = "You are a git commit message generator. Return ONLY the message, no quotes or extra text."
        openai_messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": commit_system_prompt},
            {"role": "user", "content": prompt},
        ]
        anthropic_messages: List[MessageParam] = [{"role": "user", "content": prompt}]
        
        content = ""
        if self.provider in OPENAI_COMPATIBLE_PROVIDERS:
            client = cast(Union[openai.OpenAI, openai.AzureOpenAI], self.client)
            response = client.chat.completions.create(model=self.model, messages=openai_messages)
            content = response.choices[0].message.content or "Update"
        elif self.provider == "anthropic":
            client = cast(anthropic.Anthropic, self.client)
            response = client.messages.create(
                model=self.model,
                max_tokens=100,
                system=commit_system_prompt,
                messages=anthropic_messages,
            )
            content = self._extract_anthropic_text(response) or "Update"
        elif self.provider == "gemini":
            client = cast(genai.Client, self.client)
            response = client.models.generate_content(
                model=self.model,
                contents=f"System: {commit_system_prompt}\nUser: {prompt}",
            )
            content = cast(str, getattr(response, 'text', '') or 'Update')
            
        return content.strip().strip('"').strip("'")



def _derive_lsp_server_name(package_or_url: str) -> str:
    """Derive a readable LSP config key from a package name or URL."""
    normalized = package_or_url.rstrip("/")
    if normalized.startswith("@") and "/" in normalized:
        scope, package = normalized.split("/", 1)
        value = scope.lstrip("@") if package in {"language-server", "lsp"} else f"{scope.lstrip('@')}-{package}"
    else:
        value = normalized.split("/")[-1]
    if value.endswith(".git"):
        value = value[:-4]
    value = value.lstrip("@").replace("@", "-").replace("/", "-")
    for suffix in ("-language-server", "-lsp", "_language_server", "_lsp"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value or "lsp"


def _build_lsp_server_config(package_or_url: str, arguments: List[str]) -> Dict[str, Any]:
    """Build an executable config for an LSP server installed from a package or URL."""
    if package_or_url.startswith(("http://", "https://", "git+", "git@")):
        return {"command": "uvx", "args": [package_or_url, *arguments], "env": {}}
    return {"command": "npx", "args": ["-y", package_or_url, *arguments], "env": {}}


def handle_lsp_command(command: str) -> str:
    """Handle /lsp install/on/off commands for LSP server configuration."""
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return f"Invalid /lsp command: {exc}"

    if len(parts) < 2:
        return "Usage: /lsp install <package_or_url> [args...] | /lsp {on|off} <lsp> [args...]"

    action = parts[1].lower()
    if action == "install":
        if len(parts) < 3:
            return "Usage: /lsp install <package_or_url> [args...]"
        package_or_url = parts[2]
        arguments = parts[3:]
        server = _derive_lsp_server_name(package_or_url)
        server_config = _build_lsp_server_config(package_or_url, arguments)
        install_msg = upsert_lsp_server(server, server_config)
        enable_msg = set_lsp_server_state(server, True)
        return f"{install_msg} {enable_msg}"

    if action not in {"on", "off"}:
        return "Usage: /lsp install <package_or_url> [args...] | /lsp {on|off} <lsp> [args...]"

    if len(parts) < 3:
        return "Usage: /lsp {on|off} <lsp> [args...]"
    server = parts[2]
    servers = config.get("lsp_servers", {})
    if not isinstance(servers, dict) or server not in servers:
        return f"LSP server '{server}' not found in config.lsp_servers. Run /lsp install <package_or_url> [args...] first."
    arguments = parts[3:] or None
    return set_lsp_server_state(server, action == "on", arguments)

def handle_context_command(assistant: CodeAssist, command: str) -> str:
    """Handle /context commands for clearing and managing image context."""
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return f"Invalid /context command: {exc}"

    if len(parts) >= 2 and parts[1].lower() == "clear":
        assistant.clear_context()
        return "Cleared AI conversation and image context."

    if len(parts) >= 3 and parts[1].lower() == "image":
        action = parts[2].lower()
        try:
            if action == "list":
                return assistant.list_context_images()
            if action == "add":
                if len(parts) < 4:
                    return "Usage: /context image add /path/to/image"
                return assistant.add_context_image(" ".join(parts[3:]))
            if action == "remove":
                if len(parts) < 4:
                    return "Usage: /context image remove {index|all|/path/to/image}"
                return assistant.remove_context_image(" ".join(parts[3:]))
        except (OSError, ValueError) as exc:
            return f"Error: {exc}"

    return "Usage: /context clear | /context image {list|add|remove} [/path/to/image]"


def _start_tui(assistant: CodeAssist, mode: str = "agent") -> str:
    """Start the separate TUI implementation and return its exit reason."""
    try:
        from tui import WTFCodeTUI
    except ImportError:
        try:
            from .tui import WTFCodeTUI
        except ImportError as exc:
            console.print(f"[bold red]Error:[/bold red] Could not import TUI interface: {exc}")
            return "import_error"

    return WTFCodeTUI(assistant=assistant, mode=mode, console=console).run()


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
    info_color = theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['info']
    with console.status(f"[bold {info_color}]Fetching latest version from GitHub..."):
        latest_version = get_latest_github_version()
    with console.status(f"[bold {info_color}]Starting WTFcode CLI..."):
        time.sleep(1.8)
    
    success_color = theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['success']
    console.print(Panel.fit(
        f"[bold {success_color}]WTFcode[/bold {success_color}]\n"
        "Auto Code Edit | Agent Mode | Ask Mode | Auto Bash",
        subtitle=f"[dim]{latest_version}[/dim]",
        border_style=success_color
    ))
    
    # Try to get provider from .env, otherwise prompt user
    provider_prompt = f"[bold white]Provider[/bold white] ([cyan]openai[/cyan]/[green]anthropic[/green]/[yellow]openrouter[/yellow]/[blue]gemini[/blue]/[magenta]azure_openai[/magenta]/[red]llama[/red])"
    provider = get_config_or_prompt(
        "PROVIDER",
        provider_prompt,
        choices=["openai", "anthropic", "openrouter", "gemini", "azure_openai", "llama"],
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

    if os.getenv("TUI_MODE", "").lower() == "true":
        console.print("[bold green]TUI_MODE detected. Starting TUI interface...[/bold green]")
        result = _start_tui(assistant, mode)
        if result == "tui_off":
            console.print(f"[bold green]{set_tui_mode(False)}[/bold green]")
            os.environ["TUI_MODE"] = "false"
        if result == "exit":
            return
    
    while True:
        try:
            query = _read_user_query(mode).strip()
            
            if query == '/exit':
                time.sleep(1.6)
                with console.status("[bold magenta]Exiting WTFcode...[/bold magenta]"):
                    time.sleep(1.6)
                with console.status("[bold magenta]Saving all...[/bold magenta]"):
                    time.sleep(1.5)
                with console.status("[bold magenta]Exiting...[/bold magenta]"):
                    time.sleep(1.4)
                    exit()
            
            if query == '/theme':
                themes = theme_manager.list_themes()
                theme_choice = cast(str, Prompt.ask(f"\n[bold white]Select Theme[/bold white] ({'/'.join(themes)})", choices=themes, default=theme_manager.current_theme_name))
                theme_manager.set_theme(theme_choice)
                console.print(f"[bold {theme_manager.DEFAULT_THEMES[theme_choice]['success']}]Theme switched to: {theme_choice}[/bold {theme_manager.DEFAULT_THEMES[theme_choice]['success']}]")
                continue

            if query == '/mode':
                mode = cast(str, Prompt.ask("\n[bold white]Switch Mode[/bold white] ([cyan]agent[/cyan]/[blue]ask[/blue])", choices=["agent", "ask"], default=mode)).lower()
                console.print(f"[bold green]Mode switched to:[/bold green] {mode}")
                continue
            if query == '/help':
                help_color = theme_manager.DEFAULT_THEMES[theme_manager.current_theme_name]['info']
                console.print(Panel(
                    "[bold cyan]/mode[/bold cyan] - Switch between Agent and Ask modes\n"
                    "[bold cyan]/theme[/bold cyan] - Change terminal theme\n"
                    "[bold cyan]/models[/bold cyan] - List and select available models for the current provider\n"
                    "[bold cyan]/web[/bold cyan] - Start the Streamlit web interface\n"
                    "[bold cyan]/tui {on|off}[/bold cyan] - Toggle the OpenCode-style TUI interface\n"
                    "[bold cyan]/add {file}[/bold cyan] - Add a file's content to the conversation context\n"
                    "[bold cyan]/commit[/bold cyan] - Generate a commit message and commit all changes\n"
                    "[bold cyan]/init[/bold cyan] - Initialize AGENTS.md\n"
                    "[bold cyan]/config {option}[/bold cyan] - Manage config (reload, create)\n"
                    "[bold cyan]/mcp {enable|disable|restart} {server}[/bold cyan] - Manage MCP server state\n"
                    "[bold cyan]/mcp install {server} {package_or_link} [extra_args...][/bold cyan] - Install and enable MCP server\n"
                    "[bold cyan]/lsp install {package_or_url} [args...][/bold cyan] - Install and enable LSP server config\n"
                    "[bold cyan]/lsp {on|off} {lsp} [args...][/bold cyan] - Toggle an installed LSP server\n"
                    "[bold cyan]/context clear[/bold cyan] - Clear AI conversation and image context\n"
                    "[bold cyan]/context image {list|add|remove}[/bold cyan] - Manage image context attachments\n"
                    "[bold cyan]/exit[/bold cyan] - Exit the application\n"
                    "[bold cyan]/multiinput[/bold cyan] - Toggle multi-line input mode\n"
                    "[bold cyan]/help[/bold cyan] - Show this help message",
                    title="Help", border_style=help_color
                ))
                continue
            if query == '/init':
                agents_path = Path("AGENTS.md")
                if agents_path.exists():
                    console.print("[bold yellow]AGENTS.md already exists.[/bold yellow]")
                else:
                    content = "### Agents Instructions\nUse uv for installing, running, other (python)\n"
                    with open(agents_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    console.print("[bold green]AGENTS.md created successfully.[/bold green]")
                continue
            if query.startswith('/config'):
                parts = query.split()
                option = parts[1] if len(parts) > 1 else cast(str, Prompt.ask("\n[bold white]Config Option[/bold white] ([cyan]reload[/cyan]/[green]create[/green])", choices=["reload", "create"], default="reload"))
                
                if option == "create":
                    with console.status("[bold cyan]Initializing config..."):
                        result = init_config()
                    console.print(f"[bold green]{result}[/bold green]")
                elif option == "reload":
                    with console.status("[bold cyan]Reloading config..."):
                        reload_config()
                    console.print(f"[bold green]Config reloaded from {get_config_path()}[/bold green]")
                continue
            if query.startswith('/context'):
                result = handle_context_command(assistant, query)
                console.print(f"[bold green]{result}[/bold green]")
                continue
            if query.startswith('/lsp'):
                result = handle_lsp_command(query)
                console.print(f"[bold green]{result}[/bold green]")
                continue
            if query.startswith('/mcp'):
                parts = query.split()
                if len(parts) < 3:
                    console.print("[bold yellow]Usage:[/bold yellow] /mcp {enable|disable|restart} {server} | /mcp install {server} {package_or_link} [extra_args...]")
                    continue
                action = parts[1].lower()
                server = parts[2]

                if action == "install":
                    if len(parts) < 4:
                        console.print("[bold yellow]Usage:[/bold yellow] /mcp install {server} {package_or_link} [extra_args...]")
                        continue
                    package_or_link = parts[3]
                    extra_args = parts[4:]
                    server_config = {
                        "command": "npx",
                        "args": ["-y", package_or_link, *extra_args],
                        "env": {}
                    }
                    install_msg = upsert_mcp_server(server, server_config)
                    enable_msg = set_mcp_server_state(server, True)
                    console.print(f"[bold green]{install_msg} {enable_msg}[/bold green]")
                    continue

                servers = config.get("mcp_servers", {})
                if not isinstance(servers, dict) or server not in servers:
                    console.print(f"[bold red]Error:[/bold red] MCP server '{server}' not found in config.mcp_servers")
                    continue
                if action == "enable":
                    console.print(f"[bold green]{set_mcp_server_state(server, True)}[/bold green]")
                elif action == "disable":
                    console.print(f"[bold green]{set_mcp_server_state(server, False)}[/bold green]")
                elif action == "restart":
                    disable_msg = set_mcp_server_state(server, False)
                    enable_msg = set_mcp_server_state(server, True)
                    console.print(f"[bold green]{disable_msg} {enable_msg} Restart requested.[/bold green]")
                else:
                    console.print("[bold yellow]Usage:[/bold yellow] /mcp {enable|disable|restart} {server} | /mcp install {server} {package_or_link} [extra_args...]")
                continue
            if query == '/commit':
                with console.status("[bold cyan]Generating commit message..."):
                    commit_msg = assistant.generate_commit_message()
                
                console.print(f"[bold green]Generated message:[/bold green] {commit_msg}")
                confirm = cast(str, Prompt.ask("Commit with this message?", choices=["y", "n"], default="y"))
                if confirm == "y":
                    with console.status("[bold cyan]Committing..."):
                        result = git_commit(commit_msg)
                    console.print(f"[bold green]{result}[/bold green]")
                continue
            if query.startswith('/tui'):
                parts = query.split()
                if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
                    console.print("[bold yellow]Usage:[/bold yellow] /tui {on|off}")
                    continue
                enabled = parts[1].lower() == "on"
                console.print(f"[bold green]{set_tui_mode(enabled)}[/bold green]")
                os.environ["TUI_MODE"] = str(enabled).lower()
                if enabled:
                    result = _start_tui(assistant, mode)
                    if result == "tui_off":
                        console.print(f"[bold green]{set_tui_mode(False)}[/bold green]")
                        os.environ["TUI_MODE"] = "false"
                    if result == "exit":
                        return
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
                        assistant.add_context_message(f"Context from file `{file_to_add}`:\n\n{content}")
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
                    selected_model = cast(str, Prompt.ask("\n[bold white]Enter model name (or press Enter to skip to default)[/bold white]", default="deepseek/deepseek-v3.2"))
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

            if query == '/multiinput':
                current = os.getenv("MULTILINE_INPUT", "true").lower() == "true"
                new_val = not current
                os.environ["MULTILINE_INPUT"] = str(new_val).lower()
                console.print(f"[bold green]Multi-line input {'enabled' if new_val else 'disabled'}.[/bold green]")
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
