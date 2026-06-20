"""
Crew AI Programming Team — Gradio front-end.

A lightweight dashboard that hands plain-English requirements to an AI
engineering team (Lead -> Backend -> Frontend -> Tests) and shows the
generated design, code, demo UI, and unit tests.

Run with:  python gradio_app.py
           (then open the printed http://127.0.0.1:7860 URL)

Cost note: the crew uses the cheapest viable model per role (Claude Haiku for
the Gradio UI, Sonnet for code/tests, gpt-4o only for the short design step),
so a typical run is a few cents. Nothing is called until you click "Build it!".
"""
import os
import sys
import time
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Make the crew importable and load API keys
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVELOPER_NAME = "Triaz Malik"

AGENTS = [
    ("🧠", "Engineering Lead", "gpt-4o",
     "Turns your requirements into a detailed technical design."),
    ("⚙️", "Backend Engineer", "claude-sonnet-4-5",
     "Implements the design as one self-contained Python module."),
    ("🎨", "Frontend Engineer", "claude-haiku-4-5",
     "Builds a simple Gradio demo UI for the backend."),
    ("🧪", "Test Engineer", "claude-sonnet-4-5",
     "Writes unit tests that exercise the backend module."),
]

DEFAULT_REQ = """A simple account management system for a trading simulation platform.
The system should allow the user to create an account, deposit funds, and withdraw funds.
The system should allow the user to record that they have bought or sold shares, providing a quantity.
The system should calculate the total value of the user's portfolio, and the profit or loss from the initial deposit.
The system should report holdings, profit/loss, and the list of transactions at any time.
Prevent withdrawing more than the balance, buying more than affordable, or selling unowned shares.
There is a function get_share_price(symbol) returning prices, with test values for AAPL, TSLA, GOOGL."""


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def _paths(module_name: str):
    stem = Path(module_name).stem
    return {
        "design": OUTPUT_DIR / f"{stem}_design.md",
        "backend": OUTPUT_DIR / module_name,
        "frontend": OUTPUT_DIR / "app.py",
        "tests": OUTPUT_DIR / f"test_{module_name}",
    }


def _strip_code_fences(path: Path) -> None:
    """Remove stray leading/trailing markdown ``` fences so .py files run."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip() == "":
        lines.pop()
    if lines and lines[-1].lstrip().startswith("```"):
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing(module_name: str):
    """Populate the output tabs from whatever is already in output/."""
    p = _paths(module_name)
    return (
        _read(p["design"]) or "_No design generated yet._",
        _read(p["backend"]) or "# No backend module yet.",
        _read(p["frontend"]) or "# No frontend UI yet.",
        _read(p["tests"]) or "# No tests yet.",
    )


def build(requirements: str, module_name: str, class_name: str):
    """Run the crew, then return (status, design, backend, frontend, tests)."""
    ok_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    ok_openai = bool(os.environ.get("OPENAI_API_KEY"))
    blank = ("", "", "", "")

    if not (ok_anthropic and ok_openai):
        return ("❌ Both ANTHROPIC_API_KEY and OPENAI_API_KEY must be set in .env.",
                *blank)
    if not requirements.strip():
        return ("⚠️ Please enter some requirements first.", *blank)

    from engineering_team.crew import EngineeringTeam

    inputs = {
        "requirements": requirements,
        "module_name": module_name,
        "class_name": class_name,
    }

    start = time.time()
    try:
        EngineeringTeam().crew().kickoff(inputs=inputs)
    except Exception as e:  # noqa: BLE001
        return (f"❌ The crew hit an error: {e}", *load_existing(module_name))

    for fname in (module_name, "app.py", f"test_{module_name}"):
        _strip_code_fences(OUTPUT_DIR / fname)

    status = f"✅ Done in {time.time() - start:.0f}s — see the tabs below."
    return (status, *load_existing(module_name))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Crew AI Programming Team") as demo:
    gr.Markdown(
        f"""
        # 🤖 Crew AI Programming Team
        Describe what you want built. An AI engineering team **designs it, writes the
        code, builds a demo UI, and adds unit tests** — automatically.
        *👨‍💻 Built & orchestrated by {DEVELOPER_NAME}*
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 📋 What should the team build?")
            requirements = gr.Textbox(
                label="Requirements", value=DEFAULT_REQ, lines=10,
            )
            with gr.Row():
                module_name = gr.Textbox(label="Module file name", value="accounts.py")
                class_name = gr.Textbox(label="Main class name", value="Account")
            run_btn = gr.Button("🚀 Build it!", variant="primary")
            status = gr.Markdown("Enter requirements and click **Build it!**")

        with gr.Column(scale=1):
            gr.Markdown("### 👥 The Team")
            team_md = "\n".join(
                f"**{e} {n}**  \n`{m}`  \n{d}\n" for e, n, m, d in AGENTS
            )
            gr.Markdown(team_md)
            ok_a = "✅" if os.environ.get("ANTHROPIC_API_KEY") else "❌"
            ok_o = "✅" if os.environ.get("OPENAI_API_KEY") else "❌"
            gr.Markdown(f"**🔑 Keys**  \n{ok_a} Anthropic  \n{ok_o} OpenAI")

    gr.Markdown("### 📦 Generated deliverables")
    with gr.Tabs():
        with gr.Tab("📐 Design"):
            design_out = gr.Markdown()
        with gr.Tab("⚙️ Backend"):
            backend_out = gr.Code(language="python")
        with gr.Tab("🎨 Frontend UI"):
            frontend_out = gr.Code(language="python")
        with gr.Tab("🧪 Tests"):
            tests_out = gr.Code(language="python")

    outputs = [design_out, backend_out, frontend_out, tests_out]

    run_btn.click(
        fn=build,
        inputs=[requirements, module_name, class_name],
        outputs=[status, *outputs],
    )
    # Show any previously generated files on load.
    demo.load(fn=load_existing, inputs=[module_name], outputs=outputs)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
