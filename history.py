"""Persist grade predictor experiments to a configured PostgreSQL database."""
import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
from io import BytesIO
import os
from pathlib import Path
import sys

import pandas as pd


def load_dataset(path):
    path = Path(path)
    payload = path.read_bytes()
    stream = BytesIO(payload)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(stream)
    else:
        frame = pd.read_excel(stream, engine="xlrd" if path.suffix.lower() == ".xls" else "openpyxl")
    frame.columns = frame.columns.map(str)
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("Choose a nonempty dataset with at least two columns.")
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique.")
    dataset = (path.name, hashlib.sha256(payload).hexdigest(), len(frame), list(frame.columns))
    return frame, dataset


class HistoryError(RuntimeError):
    """Database error safe to show in the UI."""


class History:
    def __init__(self, dsn):
        self.dsn = dsn

    @classmethod
    def configured(cls):
        dsn = os.getenv("GRADE_PREDICTOR_DSN", "").strip()
        return cls(dsn) if dsn else None

    @contextmanager
    def connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as error:
            raise HistoryError("Install the Python dependencies from requirements.txt.") from error
        try:
            with psycopg.connect(self.dsn, connect_timeout=5, row_factory=dict_row) as connection:
                yield connection
        except psycopg.errors.UndefinedTable as error:
            raise HistoryError("Initialize PostgreSQL history with python history.py --init.") from error
        except psycopg.OperationalError as error:
            raise HistoryError("PostgreSQL is unavailable. Verify the server and GRADE_PREDICTOR_DSN.") from error
        except psycopg.Error as error:
            raise HistoryError(f"PostgreSQL history operation failed (SQLSTATE {error.sqlstate or 'unknown'}).") from error

    def initialize(self):
        with self.connect() as connection:
            connection.execute(Path(__file__).with_name("schema.sql").read_text(encoding="utf-8"))

    def save_run(self, dataset, configuration, model):
        import joblib
        from psycopg.types.json import Jsonb

        artifact = BytesIO()
        joblib.dump(model, artifact, compress=3)
        versions = {name: version(name) for name in ("scikit-learn", "pandas", "numpy", "scipy", "joblib")}
        versions["python"] = ".".join(map(str, sys.version_info[:3]))
        filename, digest, row_count, columns = dataset
        with self.connect() as connection:
            dataset_id = connection.execute(
                """INSERT INTO grade_datasets (filename, sha256, row_count, columns)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (sha256, filename) DO UPDATE SET filename = EXCLUDED.filename
                   RETURNING id""",
                (filename, digest, row_count, Jsonb(columns)),
            ).fetchone()["id"]
            return connection.execute(
                """INSERT INTO grade_training_runs
                   (dataset_id, configuration, selected_model, metrics, comparison, artifact, versions)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (dataset_id, Jsonb(configuration), model.selected_name_, Jsonb(model.metrics_),
                 Jsonb(model.comparison_.to_dict("records")), artifact.getvalue(), Jsonb(versions)),
            ).fetchone()["id"]

    def save_prediction(self, run_id, evidence, source="desktop", case_name=None):
        from psycopg.types.json import Jsonb

        with self.connect() as connection:
            return connection.execute(
                """INSERT INTO grade_predictions (run_id, source, case_name, evidence)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (run_id, source, case_name, Jsonb(evidence)),
            ).fetchone()["id"]

    def save_evaluation(self, prediction_id, result):
        from psycopg.types.json import Jsonb

        with self.connect() as connection:
            connection.execute(
                """INSERT INTO grade_llm_evaluations
                   (prediction_id, explainer_model, judge_model, explanation, judgment)
                   VALUES (%s, %s, %s, %s, %s)""",
                (prediction_id, result["explainer_model"], result["judge_model"],
                 result["explanation"], Jsonb(result["judgment"])),
            )

    def recent_runs(self):
        with self.connect() as connection:
            return connection.execute(
                """SELECT r.id, d.filename, r.selected_model, r.created_at,
                          r.metrics, r.configuration, r.comparison, r.versions
                   FROM grade_training_runs r JOIN grade_datasets d ON d.id = r.dataset_id
                   ORDER BY r.id DESC LIMIT 100"""
            ).fetchall()

    def recent_predictions(self, run_id):
        with self.connect() as connection:
            return connection.execute(
                """SELECT p.id, p.source, p.case_name, p.evidence, p.created_at,
                          e.explainer_model, e.judge_model, e.explanation, e.judgment
                   FROM grade_predictions p LEFT JOIN grade_llm_evaluations e ON e.prediction_id = p.id
                   WHERE p.run_id = %s ORDER BY p.id DESC LIMIT 100""",
                (run_id,),
            ).fetchall()

    def model_artifact(self, run_id):
        with self.connect() as connection:
            row = connection.execute(
                "SELECT artifact FROM grade_training_runs WHERE id = %s", (run_id,)
            ).fetchone()
        if row is None:
            raise HistoryError("The training run could not be found.")
        return bytes(row["artifact"])


def main():
    parser = argparse.ArgumentParser(description="Initialize the PostgreSQL history tables.")
    parser.add_argument("--init", action="store_true", required=True)
    parser.parse_args()
    history = History.configured()
    if history is None:
        parser.exit(1, "Set GRADE_PREDICTOR_DSN first. See README.md.\n")
    try:
        history.initialize()
    except HistoryError as error:
        parser.exit(1, f"{error}\n")
    print("PostgreSQL history tables are ready.")


if __name__ == "__main__":
    main()
