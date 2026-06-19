"""Rich-based TUI mode for WTFCode.

This module keeps the alternate terminal interface separate from ``main.py``.
It intentionally uses the project's existing Rich dependency instead of adding
an additional TUI framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text


class AssistantProtocol(Protocol):
    """Subset of CodeAssist used by the TUI."""

    provider: str
    model: str

    def run_agent(self, prompt: str, render: bool = True) -> str: ...

    def ask_only(self, prompt: str, render: bool = True) -> str: ...


@dataclass
class TUIMessage:
    """A chat transcript entry rendered in the TUI."""

    role: str
    content: str


@dataclass
class WTFCodeTUI:
    """OpenCode-inspired terminal UI for WTFCode.

    The layout uses a header, conversation panel, compact status sidebar, and a
    command bar.  It can be entered from the classic CLI with ``/tui on`` and
    exited back to the classic CLI with ``/tui off``.
    """

    assistant: AssistantProtocol
    mode: str = "agent"
    console: Console = field(default_factory=Console)
    history: list[TUIMessage] = field(default_factory=list)
    max_visible_messages: int = 8

    def run(self) -> str:
        """Run the TUI loop until the user exits or switches TUI off."""
        self.history.append(
            TUIMessage(
                "system",
                "TUI mode enabled. Use /tui off to return to the classic CLI, /mode to switch agent/ask, or /exit to quit.",
            )
        )
        while True:
            self._render()
            query = Prompt.ask("[bold cyan]wtfcode[/bold cyan]").strip()
            if not query:
                continue
            command_result = self._handle_command(query)
            if command_result in {"tui_off", "exit"}:
                return command_result
            if command_result == "handled":
                continue
            self._send_to_assistant(query)

    def _handle_command(self, query: str) -> str:
        if query == "/tui off":
            self.history.append(TUIMessage("system", "TUI mode disabled."))
            return "tui_off"
        if query == "/exit":
            return "exit"
        if query == "/help":
            self.history.append(
                TUIMessage(
                    "system",
                    "Commands: /tui off, /mode, /help, /exit. Use the classic CLI for setup commands like /config, /mcp, /lsp, /models, /web, and /commit.",
                )
            )
            return "handled"
        if query == "/mode":
            self.mode = "ask" if self.mode == "agent" else "agent"
            self.history.append(TUIMessage("system", f"Mode switched to {self.mode}."))
            return "handled"
        if query.startswith("/tui"):
            self.history.append(TUIMessage("system", "Usage: /tui on | /tui off"))
            return "handled"
        return "unhandled"

    def _send_to_assistant(self, query: str) -> None:
        self.history.append(TUIMessage("user", query))
        with self.console.status("[bold cyan]Thinking...[/bold cyan]"):
            if self.mode == "agent":
                response = self.assistant.run_agent(query, render=False)
            else:
                response = self.assistant.ask_only(query, render=False)
        self.history.append(TUIMessage("assistant", response or "Done."))

    def _render(self) -> None:
        self.console.clear()
        self.console.print(self._header())
        body = Table.grid(expand=True)
        body.add_column(ratio=4)
        body.add_column(ratio=1)
        body.add_row(self._conversation_panel(), self._status_panel())
        self.console.print(body)
        self.console.print(self._command_bar())

    def _header(self) -> Panel:
        title = Text("WTFCode TUI", style="bold cyan")
        subtitle = Text("OpenCode-style focused coding workspace", style="dim")
        return Panel(Group(title, subtitle), border_style="cyan")

    def _conversation_panel(self) -> Panel:
        visible = self.history[-self.max_visible_messages :]
        if not visible:
            content: Group | Text = Text("No messages yet.", style="dim")
        else:
            rendered = []
            for message in visible:
                style = {"user": "bold green", "assistant": "bold magenta", "system": "yellow"}.get(message.role, "white")
                rendered.append(Text(f"{message.role.upper()}", style=style))
                rendered.append(Markdown(message.content))
                rendered.append(Text(""))
            content = Group(*rendered)
        return Panel(content, title="Conversation", border_style="magenta", height=24)

    def _status_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold cyan")
        table.add_column(style="white")
        table.add_row("Mode", self.mode)
        table.add_row("Provider", self.assistant.provider)
        table.add_row("Model", self.assistant.model)
        table.add_row("Messages", str(len(self.history)))
        return Panel(table, title="Status", border_style="blue", height=24)

    def _command_bar(self) -> Panel:
        return Panel(
            "[cyan]/tui off[/cyan] classic CLI  •  [cyan]/mode[/cyan] toggle agent/ask  •  [cyan]/help[/cyan] commands  •  [cyan]/exit[/cyan] quit",
            title="Command Bar",
            border_style="cyan",
        )
