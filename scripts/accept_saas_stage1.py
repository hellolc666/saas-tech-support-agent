"""Run stage 1 SaaS acceptance locally without paid model calls."""
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
os.environ["SAAS_SUPPORT_DB_PATH"] = str(root / "data" / "stage1_acceptance.db")
subprocess.run([sys.executable, "-m", "pytest", "tests/technical_support", "-q"], check=True)
subprocess.run([sys.executable, "scripts/evaluate_saas_rag.py", "--report",
                "data/evaluation/saas_stage1_rag_report.json"], check=True)
