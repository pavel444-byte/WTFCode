"""Focused tests for Plan Mode's hard read-only boundary."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class PlanModeTests(unittest.TestCase):
    def setUp(self):
        self.assistant = main.CodeAssist.__new__(main.CodeAssist)
        self.assistant._allowed_tool_names = main.PLAN_TOOL_NAMES

    def test_plan_tool_schemas_only_expose_read_operations(self):
        openai_names = {tool["function"]["name"] for tool in main.OPENAI_PLAN_TOOLS}
        anthropic_names = {tool["name"] for tool in main.ANTHROPIC_PLAN_TOOLS}

        self.assertEqual(openai_names, {"read_file", "glob_search"})
        self.assertEqual(anthropic_names, {"read_file", "glob_search"})

    def test_dispatch_rejects_mutating_tool_even_if_model_requests_it(self):
        result = self.assistant._run_local_tool(
            "write_file", {"path": "should-not-exist", "content": "unsafe"}
        )

        self.assertIn("unavailable in Plan Mode", result)

    def test_project_instructions_are_loaded_from_workspace_root(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "AGENTS.md").write_text("Always test first.", encoding="utf-8")
            with patch.object(main, "PROJECT_ROOT", Path(directory)):
                result = main.load_project_instructions()

        self.assertIn("Project instructions from AGENTS.md", result)
        self.assertIn("Always test first.", result)


if __name__ == "__main__":
    unittest.main()
