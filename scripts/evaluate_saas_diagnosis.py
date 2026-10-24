"""Reproducible offline diagnosis baseline, WITHOUT Qwen or embedding downloads.

Usage: python scripts/evaluate_saas_diagnosis.py
The synthetic fixture labels are accessed only by this evaluation script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from technical_support import diagnosis, incident_scenarios as data  # noqa: E402


def evaluate() -> dict:
    cases = []
    with TemporaryDirectory(prefix="saas-diagnosis-") as location:
        db = Path(location) / "isolated.db"
        for case_id, expected in sorted(data.GROUND_TRUTH.items()):
            app_id = data.SCENARIOS[case_id]["app_id"]
            code = "DEMO-ALPHA-4821" if app_id == "APP_DEMO_001" else "DEMO-BETA-7394"
            observed = []
            views = []
            for _ in range(2):
                state = diagnosis.collected_views(observed, case_id)
                view = diagnosis.next_evidence_view(state)
                if view is None:
                    break
                payload = data.get_incident_evidence(case_id, app_id, code, view, db)
                if payload.get("ok") is not True:
                    raise RuntimeError(f"Evidence lookup failed for {case_id}/{view}")
                observed.append(SimpleNamespace(name="get_incident_evidence", content=json.dumps(payload)))
                views.append(view)
            result = diagnosis.assess(diagnosis.collected_views(observed, case_id))
            prediction = result["root_cause"] if result else "UNDETERMINED"
            cases.append({
                "case_id": case_id,
                "expected": expected,
                "predicted": prediction,
                "correct": expected == prediction,
                "views": views,
                "verdict": result["verdict"] if result else "insufficient",
            })
    total = len(cases)
    return {
        "scope": "deterministic offline baseline; NOT a Qwen end-to-end evaluation",
        "cases": total,
        "correct": sum(row["correct"] for row in cases),
        "root_cause_accuracy": sum(row["correct"] for row in cases) / total if total else 0,
        "mean_evidence_calls": sum(len(row["views"]) for row in cases) / total if total else 0,
        "details": cases,
    }


if __name__ == "__main__":
    report = evaluate()
    print(json.dumps(report, ensure_ascii=False, indent=2))
