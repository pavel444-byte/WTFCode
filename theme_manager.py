from typing import Dict, Any
from rich.theme import Theme
from rich.console import Console
import yaml

# Import local modules with fallback for package imports
try:
    from ya_config import config, get_config_path
except ImportError:
    try:
        from .ya_config import config, get_config_path
    except ImportError as e:
        print(f"Error: Could not import ya_config module. {e}")
        raise

class ThemeManager:
    """Manages terminal themes using Rich."""
    
    DEFAULT_THEMES = {
        "dark": {
            "info": "cyan",
            "warning": "yellow",
            "error": "red",
            "success": "green",
            "panel.title": "bold green",
            "panel.border": "green",
            "thinking": "dim cyan",
            "prompt": "bold cyan",
            "user_input": "green"
        },
        "light": {
            "info": "blue",
            "warning": "magenta",
            "error": "red",
            "success": "green",
            "panel.title": "bold blue",
            "panel.border": "blue",
            "thinking": "dim blue",
            "prompt": "bold blue",
            "user_input": "black"
        },
        "matrix": {
            "info": "green",
            "warning": "bright_green",
            "error": "red",
            "success": "bright_green",
            "panel.title": "bold green",
            "panel.border": "green",
            "thinking": "dim green",
            "prompt": "bold green",
            "user_input": "bright_green"
        },
        "dracula": {
            "info": "#8be9fd",
            "warning": "#ffb86c",
            "error": "#ff5555",
            "success": "#50fa7b",
            "panel.title": "bold #bd93f9",
            "panel.border": "#6272a4",
            "thinking": "dim #6272a4",
            "prompt": "bold #ff79c6",
            "user_input": "#f8f8f2"
        }
    }

    def __init__(self, console: Console):
        self.console = console
        self.current_theme_name = config.get("settings", {}).get("theme", "dark")
        self.apply_theme(self.current_theme_name)

    def apply_theme(self, theme_name: str):
        """Apply a theme by name."""
        if theme_name not in self.DEFAULT_THEMES:
            theme_name = "dark"
        
        theme_data = self.DEFAULT_THEMES[theme_name]
        rich_theme = Theme(theme_data)
        self.console.push_theme(rich_theme)
        self.current_theme_name = theme_name
        
        # Update config
        self._update_config_theme(theme_name)

    def _update_config_theme(self, theme_name: str):
        """Update the theme in the config file."""
        config_path = get_config_path()
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                current_config = yaml.safe_load(f) or {}
            
            if "settings" not in current_config:
                current_config["settings"] = {}
            
            current_config["settings"]["theme"] = theme_name
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(current_config, f, default_flow_style=False)
        except Exception as e:
            # Silently fail if config can't be updated
            pass

    def list_themes(self):
        """Return a list of available theme names."""
        return list(self.DEFAULT_THEMES.keys())
