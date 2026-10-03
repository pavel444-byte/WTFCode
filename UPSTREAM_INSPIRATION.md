# Upstream inspiration

WTFCode remains a Python application rather than vendoring the TypeScript and Rust source trees
of other agents. Its interaction design and safety model are informed by these projects:

- [Pi](https://github.com/earendil-works/pi) (MIT): compact, extensible terminal-agent UX.
- [OpenAI Codex](https://github.com/openai/codex) (Apache-2.0): explicit execution modes,
  bounded agent loops, and defense-in-depth around tool permissions.

No upstream source files are copied into this repository. This keeps installation lightweight,
preserves WTFCode's six-provider architecture, and avoids coupling its Python runtime to Node.js
or Rust. Plan Mode applies the shared design principles directly: the model only sees read tools,
and the dispatcher independently rejects mutation attempts.
