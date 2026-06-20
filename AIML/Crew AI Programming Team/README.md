# Crew AI Programming Team

A [CrewAI](https://docs.crewai.com) crew that acts as a small engineering team. Give it
high-level requirements and it produces a working Python module, a Gradio demo UI, and
unit tests.

## The team

| Agent | Role | LLM |
|-------|------|-----|
| `engineering_lead` | Turns requirements into a detailed design | `gpt-4o` |
| `backend_engineer` | Implements the design in one self-contained module | `claude-3-7-sonnet-latest` |
| `frontend_engineer` | Builds a simple Gradio UI (`app.py`) for the backend | `claude-3-7-sonnet-latest` |
| `test_engineer` | Writes unit tests for the backend module | `claude-3-7-sonnet-latest` |

The agents run **sequentially**: design → code → (frontend + tests both use the code as context).
All generated files land in `output/`.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) (CrewAI's recommended tool), then install deps:

   ```bash
   uv pip install -e .
   # or, with plain pip:
   pip install -e .
   ```

2. Copy `.env.example` to `.env` and add your API keys:

   ```bash
   cp .env.example .env
   ```

   You need **both** an `ANTHROPIC_API_KEY` and an `OPENAI_API_KEY` (the lead uses gpt-4o).

3. The backend and test engineers run with `code_execution_mode="safe"`, which uses
   **Docker** to execute code. Make sure Docker Desktop is running. (If you don't want
   Docker, set `allow_code_execution=False` on those agents in `src/engineering_team/crew.py`.)

## Run

```bash
crewai run
# or
run_crew
# or
python -m engineering_team.main
```

## Customize what gets built

Edit `requirements`, `module_name`, and `class_name` in
[`src/engineering_team/main.py`](src/engineering_team/main.py). The default example builds a
trading-account management system.

After a run, try the generated UI:

```bash
cd output
python app.py
```

## Project layout

```
.
├── pyproject.toml
├── .env.example
├── output/                         # generated files land here
└── src/engineering_team/
    ├── main.py                     # requirements + kickoff
    ├── crew.py                     # agents, tasks, crew wiring
    └── config/
        ├── agents.yaml
        └── tasks.yaml
```
