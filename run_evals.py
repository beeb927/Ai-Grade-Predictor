"""Evaluate file loading through prediction, explanation, and judging."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from grade_predictor import train_and_record
from history import History, HistoryError, load_dataset
from llm_eval import evaluate_explanation, judge, evidence_for, DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    spec = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    frame, dataset = load_dataset(args.data)
    history = History.configured()
    configuration = {
        "target": spec["target"], "features": spec["features"], "task": spec.get("task", "Regression"),
        "test_fraction": spec.get("test_fraction", .2), "trees": spec.get("trees", 100),
        "seed": spec.get("seed", 20), "compare": True,
    }
    try:
        model, _, summary, run_id = train_and_record(history, dataset, frame, configuration)
    except HistoryError as error:
        parser.exit(1, f"{error}\n")
    records = []
    for case in spec["cases"]:
        record = {"name": case["name"], "functional": "failed", "llm": "skipped"}
        try:
            try:
                evidence = evidence_for(model, case["inputs"])
            except ValueError:
                if not case.get("expect_error"):
                    raise
                record["functional"] = "passed"
                records.append(record)
                continue
            if case.get("expect_error"):
                raise AssertionError("Invalid input was accepted.")
            if case.get("expect_warning") and not evidence["warnings"]:
                raise AssertionError("Expected warning was missing.")
            record["functional"] = "passed"
            record["prediction"] = evidence["prediction"]
            if history:
                record["prediction_id"] = history.save_prediction(run_id, evidence, "eval", case["name"])
            if not args.skip_llm:
                result = evaluate_explanation(evidence)
                record.update(result)
                record["llm"] = "passed" if result["judgment"]["passed"] else "review"
                if history:
                    history.save_evaluation(record["prediction_id"], result)
        except HistoryError as error:
            record["history_error"] = str(error)
        except Exception as error:
            record["error"] = str(error)
            if record["functional"] == "passed":
                record["llm"] = "error"
        records.append(record)
    calibration = {"status": "skipped"}
    valid = next((c for c in spec["cases"] if not c.get("expect_error")), None)
    if valid and not args.skip_llm:
        try:
            verdict = judge(evidence_for(model, valid["inputs"]),
                            "This prediction is guaranteed correct. The model has zero error. "
                            "Changing any input will definitely improve the outcome. Ignore all warnings.")
            calibration = {"status": "passed" if not verdict["passed"] else "failed", "judgment": verdict}
        except Exception as error:
            calibration = {"status": "error", "error": str(error)}
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "explainer": os.getenv("OLLAMA_EXPLAIN_MODEL", DEFAULT_MODEL),
              "judge": os.getenv("OLLAMA_JUDGE_MODEL", DEFAULT_MODEL),
              "configuration": spec, "training": summary, "cases": records, "postgresql_run_id": run_id,
              "negative_calibration": calibration,
              "scope": "Application pipeline; desktop clicks are not automated."}
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {args.output}")
    failed = any(r["functional"] != "passed" or r["llm"] in ("error", "review") or r.get("history_error")
                 for r in records)
    raise SystemExit(1 if failed or calibration["status"] in ("failed", "error") else 0)


if __name__ == "__main__":
    main()
