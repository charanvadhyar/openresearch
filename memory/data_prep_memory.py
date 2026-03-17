"""
Persistent storage for DataPrepReport snapshots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from autoresearch.schemas import DataPrepReport


class DataPrepMemory:
    """
    Save and load DataPrepReport objects per dataset signature.
    Stored as JSON in autoresearch_memory/data_prep/.
    """

    def __init__(self, base_dir: Path | str = "autoresearch_memory/data_prep"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_report(self, dataset_signature: str, report: DataPrepReport) -> Path:
        payload = report.model_dump()
        payload["dataset_signature"] = dataset_signature
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        ts = payload["timestamp"].replace(":", "-")
        fname = f"{ts}_{dataset_signature}_{uuid4().hex[:8]}.json"
        path = self.base_dir / fname
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load_reports(self, dataset_signature: str) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(data.get("dataset_signature", "")) == dataset_signature:
                reports.append(data)
        return reports
