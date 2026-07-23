"""Airflow DAG — scheduled ingestion of the copilot corpus.

A thin PythonOperator wrapper around ``scripts/ingest.py::main``, which is the same
code path verified end-to-end against live Postgres + OpenSearch. Because
``DocumentStore.upsert_document`` replaces a document's rows instead of appending,
re-runs, retries, and backfills are safe.

Runs inside an Airflow environment; not executed by the offline test suite.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Make the service's src/ and scripts/ importable when Airflow parses this DAG.
SERVICE_ROOT = Path(__file__).resolve().parents[2]
for _path in (SERVICE_ROOT / "src", SERVICE_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def run_ingest() -> None:
    from ingest import main  # scripts/ingest.py

    main()


default_args = {
    "owner": "platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="platform_copilot_ingest",
    description="Ingest platform runbooks/ADRs into Postgres + OpenSearch for the copilot",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["platform-copilot", "rag", "ingestion"],
) as dag:
    PythonOperator(
        task_id="ingest_corpus",
        python_callable=run_ingest,
    )
