"""
Crew AI Programming Team — Streamlit front-end.

A branded dashboard that lets you hand requirements to an AI engineering team
(Lead → Backend → Frontend → Tests) and watch the generated code appear.

Run with:  streamlit run streamlit_app.py
"""
import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Make the crew importable and load API keys
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 👇 Change this to your own name.
DEVELOPER_NAME = "Triaz Malik"

AGENTS = [
    {"emoji": "🧠", "name": "Engineering Lead", "model": "gpt-4o",
     "desc": "Turns your requirements into a detailed technical design."},
    {"emoji": "⚙️", "name": "Backend Engineer", "model": "claude-sonnet-4-5",
     "desc": "Implements the design as one self-contained Python module."},
    {"emoji": "🎨", "name": "Frontend Engineer", "model": "claude-haiku-4-5",
     "desc": "Builds a simple Gradio demo UI for the backend."},
    {"emoji": "🧪", "name": "Test Engineer", "model": "claude-sonnet-4-5",
     "desc": "Writes unit tests that exercise the backend module."},
]

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crew AI Programming Team",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#0f1117 0%, #161a23 100%); }
    .hero {
        background: linear-gradient(120deg,#6366f1 0%,#8b5cf6 45%,#ec4899 100%);
        padding: 2.2rem 2.4rem; border-radius: 20px; color:white;
        box-shadow: 0 12px 40px rgba(99,102,241,.35); margin-bottom: 1.5rem;
    }
    .hero h1 { font-size: 2.4rem; margin: 0; font-weight: 800; letter-spacing:-1px; }
    .hero p  { font-size: 1.05rem; margin:.4rem 0 0; opacity:.95; }
    .byline  { display:inline-block; margin-top:.9rem; padding:.3rem .9rem;
               background:rgba(255,255,255,.18); border-radius:999px;
               font-weight:600; font-size:.9rem; }
    .agent-card {
        background:#1e2330; border:1px solid #2c3344; border-radius:14px;
        padding:1rem 1.1rem; margin-bottom:.8rem;
    }
    .agent-card .nm { font-weight:700; font-size:1.05rem; color:#e5e7eb; }
    .agent-card .md { color:#8b93a7; font-size:.78rem; font-family:monospace; }
    .agent-card .ds { color:#aeb6c7; font-size:.86rem; margin-top:.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>🤖 Crew AI Programming Team</h1>
        <p>Describe what you want built. An AI engineering team designs it,
           writes the code, builds a demo UI, and adds unit tests — automatically.</p>
        <span class="byline">👨‍💻 Built &amp; orchestrated by {DEVELOPER_NAME}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — the team + key status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("👥 The Team")
    for a in AGENTS:
        st.markdown(
            f"""<div class="agent-card">
                   <div class="nm">{a['emoji']} {a['name']}</div>
                   <div class="md">{a['model']}</div>
                   <div class="ds">{a['desc']}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("🔑 API key status")
    ok_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    ok_openai = bool(os.environ.get("OPENAI_API_KEY"))
    st.write(("✅" if ok_anthropic else "❌") + " Anthropic (Claude)")
    st.write(("✅" if ok_openai else "❌") + " OpenAI (gpt-4o)")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
DEFAULT_REQ = """A simple account management system for a trading simulation platform.
The system should allow the user to create an account, deposit funds, and withdraw funds.
The system should allow the user to record that they have bought or sold shares, providing a quantity.
The system should calculate the total value of the user's portfolio, and the profit or loss from the initial deposit.
The system should report holdings, profit/loss, and the list of transactions at any time.
Prevent withdrawing more than the balance, buying more than affordable, or selling unowned shares.
There is a function get_share_price(symbol) returning prices, with test values for AAPL, TSLA, GOOGL."""

st.subheader("📋 What should the team build?")
col1, col2 = st.columns([3, 1])
with col1:
    requirements = st.text_area("Requirements", value=DEFAULT_REQ, height=210)
with col2:
    module_name = st.text_input("Module file name", value="accounts.py")
    class_name = st.text_input("Main class name", value="Account")
    st.caption("Generated files are saved to the `output/` folder.")

run_clicked = st.button("🚀 Build it!", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run the crew
# ---------------------------------------------------------------------------
if run_clicked:
    if not (ok_anthropic and ok_openai):
        st.error("Both ANTHROPIC_API_KEY and OPENAI_API_KEY must be set in your .env file.")
        st.stop()
    if not requirements.strip():
        st.warning("Please enter some requirements first.")
        st.stop()

    from engineering_team.crew import EngineeringTeam

    inputs = {
        "requirements": requirements,
        "module_name": module_name,
        "class_name": class_name,
    }

    start = time.time()
    with st.status("🏗️ The engineering team is working… this can take a few minutes.",
                   expanded=True) as status:
        st.write("🧠 Engineering Lead is designing the solution…")
        try:
            EngineeringTeam().crew().kickoff(inputs=inputs)
            status.update(label=f"✅ Done in {time.time() - start:.0f}s!",
                          state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001
            status.update(label="❌ The crew hit an error.", state="error")
            st.exception(e)
            st.stop()
    st.balloons()
    st.session_state["ran"] = True

# ---------------------------------------------------------------------------
# Show results (from output/) — persists across reruns
# ---------------------------------------------------------------------------
def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


design_p = OUTPUT_DIR / f"{Path(module_name).stem}_design.md"
backend_p = OUTPUT_DIR / module_name
app_p = OUTPUT_DIR / "app.py"
test_p = OUTPUT_DIR / f"test_{module_name}"

if any(p.exists() for p in (design_p, backend_p, app_p, test_p)):
    st.divider()
    st.subheader("📦 Generated deliverables")
    t1, t2, t3, t4 = st.tabs(["📐 Design", "⚙️ Backend", "🎨 Frontend UI", "🧪 Tests"])

    with t1:
        d = _read(design_p)
        st.markdown(d) if d else st.info("No design file yet.")
        if d:
            st.download_button("⬇️ Download design", d, file_name=design_p.name)
    with t2:
        c = _read(backend_p)
        st.code(c, language="python") if c else st.info("No backend module yet.")
        if c:
            st.download_button("⬇️ Download " + backend_p.name, c, file_name=backend_p.name)
    with t3:
        c = _read(app_p)
        st.code(c, language="python") if c else st.info("No UI file yet.")
        if c:
            st.download_button("⬇️ Download app.py", c, file_name="app.py")
    with t4:
        c = _read(test_p)
        st.code(c, language="python") if c else st.info("No tests yet.")
        if c:
            st.download_button("⬇️ Download " + test_p.name, c, file_name=test_p.name)
else:
    st.info("👆 Enter your requirements and click **Build it!** to see the team in action.")
