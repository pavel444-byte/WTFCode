# Changelog

## WTFCode 1.0.5 - 2026-05-26

### Changed
- Bumped project version metadata from 1.0.4 to 1.0.5 across packaging metadata, runtime client info, and README release header.

## WTFCode 1.0.4 - 2026-05-17

### Fixed
- Fixed the Streamlit web chat history reset and Agent Mode response rendering by using provider-aware assistant history helpers instead of a missing shared `history` attribute.
- Fixed command streaming so long-running commands can safely emit both stdout and stderr without deadlocking, while still enforcing a timeout.
- Fixed assistant responses for UI clients by returning generated content from `run_agent()` and `ask_only()` instead of requiring callers to scrape console output.
- Fixed configuration side effects during imports: config file creation is now explicit, and applying a theme no longer writes to disk unless the user changes the theme.

### Added
- Added reusable assistant history utilities for resetting provider-specific histories and reading the latest assistant response.
- Added safer release-ready command execution behavior with combined output streaming and clearer timeout cleanup.
- Added explicit theme persistence via `set_theme()` so CLI settings changes can be saved intentionally.
- Added release documentation for version 1.0.4.
