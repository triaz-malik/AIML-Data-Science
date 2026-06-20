"""
Generate a PDF report describing the Crew AI Programming Team project:
what it is, its multi-agent architecture, and one worked end-to-end example.

Run with:  python make_report.py
Output:    Crew_AI_Programming_Team_Report.pdf

Pure reportlab — no API calls, no cost.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Preformatted, HRFlowable,
)

ROOT = Path(__file__).parent
OUT = ROOT / "Crew_AI_Programming_Team_Report.pdf"

INDIGO = colors.HexColor("#6366f1")
PURPLE = colors.HexColor("#8b5cf6")
DARK = colors.HexColor("#1e2330")
GREY = colors.HexColor("#5b6473")
LIGHT = colors.HexColor("#eef0f6")

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], textColor=INDIGO, fontSize=26,
                    spaceAfter=6, leading=30)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], textColor=GREY, fontSize=12,
                     alignment=TA_CENTER, spaceAfter=18)
H2 = ParagraphStyle("H2", parent=ss["Heading1"], textColor=PURPLE, fontSize=16,
                    spaceBefore=16, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=ss["Heading2"], textColor=DARK, fontSize=12,
                    spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=10.5, leading=15,
                      spaceAfter=8, textColor=colors.HexColor("#222a33"))
CODE = ParagraphStyle("CODE", parent=ss["Code"], fontSize=8, leading=10.5,
                      backColor=LIGHT, borderPadding=6, textColor=DARK)
CAP = ParagraphStyle("CAP", parent=ss["Normal"], fontSize=8.5, textColor=GREY,
                     spaceAfter=12)


def code_block(text: str, max_lines: int = 26):
    lines = text.strip("\n").splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["    # … (truncated for the report)"]
    return Preformatted("\n".join(lines), CODE)


def read(p: Path, max_lines: int = 26) -> str:
    if not p.exists():
        return "# (file not found)"
    return "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     .splitlines()[:max_lines])


def hr():
    return HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#d8dbe6"),
                      spaceBefore=4, spaceAfter=10)


story = []

# ---- Cover -----------------------------------------------------------------
story += [
    Spacer(1, 30 * mm),
    Paragraph("Crew AI Programming Team", H1),
    Paragraph("An autonomous AI engineering team that turns plain-English "
              "requirements into working software", SUB),
    hr(),
    Paragraph("Project report &mdash; architecture &amp; worked example", H3),
    Paragraph("Built &amp; orchestrated by <b>Triaz Malik</b>", BODY),
    Paragraph("Stack: CrewAI &bull; Claude (Sonnet 4.5, Haiku 4.5) &bull; "
              "OpenAI gpt-4o &bull; Gradio / Streamlit", BODY),
    Spacer(1, 8 * mm),
]

# ---- 1. What it is ---------------------------------------------------------
story += [
    Paragraph("1. What is it?", H2),
    Paragraph(
        "The <b>Crew AI Programming Team</b> is a multi-agent system built on "
        "<b>CrewAI</b>. You describe what you want built in plain English; a small "
        "team of specialised AI agents then collaborates to deliver it &mdash; a "
        "technical design, a self-contained Python module, a Gradio demo UI, and a "
        "matching suite of unit tests. The agents run as a pipeline, each one "
        "building on the previous agent's output, mirroring how a real engineering "
        "team hands work down the line.", BODY),
    Paragraph(
        "Everything the team produces is written to the <font face='Courier'>output/</font> "
        "folder, ready to run. Two front-ends are provided to drive the team: a "
        "<b>Gradio</b> app and a <b>Streamlit</b> dashboard.", BODY),
]

# ---- 2. Architecture -------------------------------------------------------
story += [
    Paragraph("2. Architecture", H2),
    Paragraph(
        "The system follows a <b>sequential</b> process. The Engineering Lead designs "
        "the solution; the Backend Engineer implements it; then the Frontend and Test "
        "engineers both consume the finished code as context to build a demo UI and "
        "unit tests respectively.", BODY),
    code_block(
        "Requirements (plain English)\n"
        "        |\n"
        "        v\n"
        "  [Engineering Lead]  --gpt-4o-->  design.md\n"
        "        |\n"
        "        v\n"
        "  [Backend Engineer]  --Sonnet 4.5-->  accounts.py\n"
        "        |\n"
        "        +----------------------+\n"
        "        v                      v\n"
        "  [Frontend Eng.]        [Test Engineer]\n"
        "   --Haiku 4.5-->          --Sonnet 4.5-->\n"
        "      app.py             test_accounts.py", max_lines=30),
    Paragraph("The team", H3),
]

team_data = [
    ["Agent", "Role", "Model", "Output"],
    ["Engineering\nLead", "Turns requirements into a detailed design", "gpt-4o",
     "*_design.md"],
    ["Backend\nEngineer", "Implements the design in one self-contained module",
     "claude-sonnet-4-5", "accounts.py"],
    ["Frontend\nEngineer", "Builds a simple Gradio demo UI", "claude-haiku-4-5",
     "app.py"],
    ["Test\nEngineer", "Writes unit tests for the backend module",
     "claude-sonnet-4-5", "test_accounts.py"],
]
tbl = Table(team_data, colWidths=[28*mm, 62*mm, 42*mm, 36*mm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), INDIGO),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("FONTNAME", (2, 1), (2, -1), "Courier"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9cde0")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story += [
    tbl,
    Spacer(1, 4 * mm),
    Paragraph(
        "<b>Cost control.</b> Each role uses the cheapest model that does the job "
        "well: Claude <b>Haiku 4.5</b> for the boilerplate-heavy Gradio UI, Claude "
        "<b>Sonnet 4.5</b> for code and tests, and gpt-4o only for the short design "
        "step. Code-writing agents get a large output budget (max_tokens=16000) so "
        "long files are never truncated. A typical full run costs a few cents.", BODY),
]

# ---- 3. Worked example -----------------------------------------------------
story += [
    Paragraph("3. Worked example", H2),
    Paragraph("3.1 &nbsp; What was requested", H3),
    Paragraph(
        "A trading-simulation <b>account management system</b>: create an account, "
        "deposit / withdraw funds, buy / sell shares, and report portfolio value, "
        "profit/loss, holdings, and transaction history &mdash; with guardrails that "
        "prevent overdrawing, over-buying, or selling shares the user doesn't own. A "
        "<font face='Courier'>get_share_price(symbol)</font> helper returns fixed "
        "prices for AAPL, TSLA and GOOGL.", BODY),
]

story += [
    Paragraph("3.2 &nbsp; Design produced by the Engineering Lead", H3),
    Paragraph("The lead returns a markdown design listing every class and method "
              "signature &mdash; signatures only, no implementation:", BODY),
    code_block(
        "class Account:\n"
        "    def __init__(self, account_id: str, initial_deposit: float): ...\n"
        "    def deposit(self, amount: float) -> None: ...\n"
        "    def withdraw(self, amount: float) -> None: ...\n"
        "    def buy_shares(self, symbol: str, quantity: int) -> None: ...\n"
        "    def sell_shares(self, symbol: str, quantity: int) -> None: ...\n"
        "    def calculate_portfolio_value(self) -> float: ...\n"
        "    def calculate_profit_or_loss(self) -> float: ...\n"
        "    def get_holdings(self) -> dict: ...\n"
        "    def get_transactions(self) -> list: ...\n\n"
        "def get_share_price(symbol: str) -> float: ...", max_lines=20),
    Paragraph("Source: output/accounts.py_design.md", CAP),
]

story += [
    Paragraph("3.3 &nbsp; Backend code (Backend Engineer)", H3),
    code_block(read(ROOT / "output" / "accounts.py", 24)),
    Paragraph("Source: output/accounts.py", CAP),
]

story += [
    Paragraph("3.4 &nbsp; Gradio demo UI (Frontend Engineer)", H3),
    code_block(read(ROOT / "output" / "app.py", 24)),
    Paragraph("Source: output/app.py", CAP),
]

story += [
    Paragraph("3.5 &nbsp; Unit tests (Test Engineer)", H3),
    code_block(read(ROOT / "output" / "test_accounts.py", 24)),
    Paragraph("Source: output/test_accounts.py", CAP),
]

# ---- 4. How to run ---------------------------------------------------------
story += [
    Paragraph("4. How to run", H2),
    code_block(
        "# 1. Install\n"
        "pip install -e .\n\n"
        "# 2. Add API keys to .env (ANTHROPIC_API_KEY + OPENAI_API_KEY)\n\n"
        "# 3a. Drive the team from a web UI\n"
        "python gradio_app.py            # Gradio\n"
        "streamlit run streamlit_app.py  # or Streamlit\n\n"
        "# 3b. …or from the CLI\n"
        "crewai run\n\n"
        "# 4. Try the generated app\n"
        "cd output && python app.py", max_lines=20),
]


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(20 * mm, 12 * mm, "Crew AI Programming Team — project report")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=20 * mm, rightMargin=20 * mm,
    topMargin=18 * mm, bottomMargin=20 * mm,
    title="Crew AI Programming Team — Report", author="Triaz Malik",
)
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print(f"Wrote {OUT}")
