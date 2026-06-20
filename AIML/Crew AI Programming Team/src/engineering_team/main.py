#!/usr/bin/env python
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

from engineering_team.crew import EngineeringTeam

# Load API keys from the project-root .env (two levels up from this file).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Make sure the output directory exists for the generated files.
os.makedirs('output', exist_ok=True)

# ----------------------------------------------------------------------------
# Define the high-level requirements for the engineering team to build.
# Swap these out for whatever you want the crew to implement.
# ----------------------------------------------------------------------------
requirements = """
A simple account management system for a trading simulation platform.
The system should allow the user to create an account, deposit funds, and withdraw funds.
The system should allow the user to record that they have bought or sold shares, providing a quantity.
The system should calculate the total value of the user's portfolio, and the profit or loss from the initial deposit.
The system should be able to report the holdings of the user at any point in time.
The system should be able to report the profit or loss of the user at any point in time.
The system should be able to list the transactions that the user has made over time.
The system should prevent the user from withdrawing funds that would leave them with a negative balance,
or from buying more shares than they can afford, or selling shares that they don't have.
The system has access to a function get_share_price(symbol) which returns the current price of a share,
and includes a test implementation that returns fixed prices for AAPL, TSLA, GOOGL.
"""

module_name = "accounts.py"
class_name = "Account"


def _strip_code_fences(path: Path) -> None:
    """Remove leading/trailing markdown ``` fences that LLMs sometimes add,
    so generated .py files are directly runnable."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    # Drop a leading ```lang line and a trailing ``` line if present.
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip() == "":
        lines.pop()
    if lines and lines[-1].lstrip().startswith("```"):
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    """Run the engineering team crew."""
    inputs = {
        'requirements': requirements,
        'module_name': module_name,
        'class_name': class_name,
    }
    result = EngineeringTeam().crew().kickoff(inputs=inputs)

    # Clean any markdown fences from the generated Python files.
    for fname in (module_name, "app.py", f"test_{module_name}"):
        _strip_code_fences(Path("output") / fname)

    print("\n\n=== CREW FINISHED ===")
    print(result)


if __name__ == "__main__":
    run()
