# Repository Guidelines

## Project Structure & Module Organization
- `agents/`: Python teaching implementations (`s01`-`s12`) plus `s_full.py`.
- `docs/{en,zh,ja}/`: session documentation in three languages.
- `skills/`: skill definitions and helper scripts (for example `skills/agent-builder/scripts/init_agent.py`).
- `web/`: Next.js 16 + TypeScript learning site; app code is in `web/src`, static files in `web/public`.
- `.github/workflows/`: CI pipelines for web build/type-check and Python test jobs.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install Python dependencies.
- `cp .env.example .env`: create local config (`ANTHROPIC_API_KEY`, `MODEL_ID`).
- `python agents/s01_agent_loop.py`: run the first session.
- `python agents/s12_worktree_task_isolation.py`: run the advanced session.
- `python agents/s_full.py`: run the full-capability reference agent.
- `cd web && npm ci && npm run dev`: start the web app at `http://localhost:3000`.
- `cd web && npm run build`: production build (includes content extraction via `prebuild`).
- `cd web && npx tsc --noEmit`: strict type-check used in CI.

## Coding Style & Naming Conventions
- Python: 4-space indentation, snake_case functions, and UPPER_SNAKE_CASE constants.
- Keep session filenames in `agents/` as `sNN_topic.py` (example: `s08_background_tasks.py`).
- TypeScript/React: strict typing is enabled; use PascalCase component exports and kebab-case filenames in `web/src/components`.
- Prefer the `@/*` path alias in `web` imports over deep relative paths.

## Testing Guidelines
- This checkout currently has no tracked `tests/` directory, so use session runs as smoke tests for Python changes.
- Before opening a PR, run:
  - `cd web && npx tsc --noEmit`
  - `cd web && npm run build`
  - at least one relevant `python agents/sXX_*.py` script you touched.
- If you add Python tests, place them under `tests/` and use `test_*.py` naming.

## Commit & Pull Request Guidelines
- Prefer concise, imperative commit subjects; this repo commonly uses `feat:`, `fix:`, and `fix(scope): ...`.
- Keep commits focused to one logical change.
- PRs should include a short summary, affected paths, validation commands run, and screenshots for UI updates.
- Link related issues/PRs when applicable (example: `(#39)`).
