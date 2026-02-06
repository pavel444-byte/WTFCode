import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

def get_config_path() -> Path:
    """Get the path to the config file in the user's home directory."""
    home = Path.home()
    config_dir = home / ".wtfcode"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.yml"

def load_config() -> Dict[str, Any]:
    """Load the configuration from the YAML file."""
    config_path = get_config_path()
    
    # Default configuration
    default_config = {
        "provider": "openai",
        "model": "gpt-4o",
        "api_keys": {
            "openai": os.getenv("OPENAI_API_KEY", ""),
            "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
            "google": os.getenv("GOOGLE_API_KEY", ""),
            "openrouter": os.getenv("OPENROUTER_API_KEY", "")
        },
        "settings": {
            "notifications": True,
            "theme": "dark",
            "multi_line_output": True,
            "web_mode": False
        }
    }
    
    if not config_path.exists():
        # Create default config if it doesn't exist
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
    
    load_config() # This creates the file if it doesn't exist
    return f"Configuration initialized at {config_path}"

# Global config instance
config = load_config()
