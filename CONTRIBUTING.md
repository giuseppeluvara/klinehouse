# Contributing

Thank you for your interest in improving this project.

## How to contribute

1. **Open an issue first** for non-trivial changes so the approach can be discussed before
   implementation work begins.
2. **Fork** the repository and create a branch from `main`.
3. **Write clear, focused commits.** One logical change per commit.
4. **All code, comments, and commit messages must be in English.**
5. **Keep dependencies minimal.** Adding a new dependency requires justification in the PR
   description.
6. **Test your changes** end-to-end using `docker compose up -d` before opening a PR.
7. **Run the linter** before submitting:
   ```bash
   pip install ruff
   ruff check src/
   ```
8. Open a **Pull Request** against `main` with a concise description of what changed and why.

## Code style

- Python 3.12, `asyncio`, type-annotated.
- Formatted and linted with [ruff](https://docs.astral.sh/ruff/).
- No secrets, API keys, or personal data in the repository.

## Reporting bugs

Open a GitHub issue and include:
- The exact `docker compose` version and host OS/architecture.
- Relevant log output (`docker compose logs collector`).
- Steps to reproduce.

## License

By contributing you agree that your contributions will be licensed under the MIT License.
