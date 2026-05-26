import os
import yaml
from pathlib import Path
from typing import Any, Dict

def get_config_path() -> Path:
    """Get the path to the config file in the user's home directory."""
    home = Path.home()
    config_dir = home / ".wtfcode"
    return config_dir / "config.yml"

def get_default_config() -> Dict[str, Any]:
    """Return the default configuration without touching the filesystem."""
    return {
        "provider": "openai",
        "model": "gpt-4o",
        "api_keys": {
            "openai": os.getenv("OPENAI_API_KEY", ""),
            "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
            "google": os.getenv("GOOGLE_API_KEY", ""),
            "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
            "azure_openai": os.getenv("AZURE_OPENAI_API_KEY", ""),
            "llama": os.getenv("LLAMA_API_KEY", "ollama")
        },
        "azure_openai": {
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        },
        "llama": {
            "base_url": os.getenv("LLAMA_BASE_URL", "http://localhost:11434")
        },
        "mcp_servers": {},
        "settings": {
            "notifications": True,
            "theme": "dark",
            "multi_line_input": True,
            "web_mode": False
        }
    }

def load_config(create_if_missing: bool = False) -> Dict[str, Any]:
    """Load configuration from disk, optionally creating the file explicitly."""
    config_path = get_config_path()
    default_config = get_default_config()
    
    if not config_path.exists():
        if create_if_missing:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False)
        return default_config
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
            if user_config is None:
                return default_config
            
            # Merge user config with defaults to ensure all keys exist
            # Simple merge for top-level keys
            merged_config = default_config.copy()
            for key, value in user_config.items():
                if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
            return merged_config
    except Exception as e:
        print(f"Error loading config: {e}")
        return default_config

def init_config() -> str:
    """Explicitly initialize the config directory and file."""
    config_path = get_config_path()
    if config_path.exists():
        return f"Configuration already exists at {config_path}"
    
    load_config(create_if_missing=True)
    return f"Configuration initialized at {config_path}"

def reload_config() -> Dict[str, Any]:
    """Reload the global config dict in-place from disk."""
    global config
    new_config = load_config()
    config.clear()
    config.update(new_config)
    return config

def set_mcp_server_state(server: str, enabled: bool) -> str:
    """Set an MCP server enabled/disabled state in memory and on disk (if config exists)."""
    config.setdefault("mcp_server_states", {})
    config["mcp_server_states"][server] = enabled

    config_path = get_config_path()
    if not config_path.exists():
        state_text = "enabled" if enabled else "disabled"
        return f"MCP server '{server}' {state_text} in memory. Config file not found at {config_path}."

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            current_config = yaml.safe_load(f) or {}
        current_config.setdefault("mcp_server_states", {})
        current_config["mcp_server_states"][server] = enabled
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(current_config, f, default_flow_style=False)
        state_text = "enabled" if enabled else "disabled"
        return f"MCP server '{server}' {state_text}."
    except Exception as e:
        return f"MCP server '{server}' updated in memory, but failed to persist config: {e}"


def upsert_mcp_server(server: str, server_config: Dict[str, Any]) -> str:
    """Create or update an MCP server config in memory and on disk (if config exists)."""
    config.setdefault("mcp_servers", {})
    existing = server in config["mcp_servers"] if isinstance(config.get("mcp_servers"), dict) else False
    if not isinstance(config.get("mcp_servers"), dict):
        config["mcp_servers"] = {}
    config["mcp_servers"][server] = server_config

    config_path = get_config_path()
    action = "updated" if existing else "installed"
    if not config_path.exists():
        return f"MCP server '{server}' {action} in memory. Config file not found at {config_path}."

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            current_config = yaml.safe_load(f) or {}
        if not isinstance(current_config.get("mcp_servers"), dict):
            current_config["mcp_servers"] = {}
        current_config["mcp_servers"][server] = server_config
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(current_config, f, default_flow_style=False)
        return f"MCP server '{server}' {action}."
    except Exception as e:
        return f"MCP server '{server}' {action} in memory, but failed to persist config: {e}"

# Global config instance
config = load_config()
