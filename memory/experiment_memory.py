"""
Offline experiment memory backed by JSON files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_METHODS_BY_TASK = {
    "classification": ["xgboost", "lightgbm", "random_forest"],
    "regression": ["xgboost_regression", "lightgbm_regression", "ridge_regression"],
    "nlp": ["tfidf_logreg", "distilbert", "xgboost"],
    "computer_vision": ["resnet50_transfer", "efficientnet_transfer", "xgboost"],
    "time_series": ["xgboost_timeseries", "prophet", "lstm_timeseries"],
    "clustering": ["kmeans", "hdbscan", "random_forest"],
}


class ExperimentMemory:
    """
    Save and query past experiment outcomes to bias future method selection.
    """

    def __init__(self, base_dir: Path | str = "autoresearch_memory/experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_dataset_signature(dataset_profile: dict[str, Any], task_type: str) -> str:
        dataset_type = str(dataset_profile.get("dataset_type", "tabular"))
        rows = int(dataset_profile.get("rows", 0) or 0)
        if rows < 1_000:
            size = "small"
        elif rows <= 100_000:
            size = "medium"
        else:
            size = "large"
        return f"{dataset_type}_{size}_{task_type}"

    def save_experiment(self, record: dict[str, Any]) -> None:
        """
        Persist a single experiment record as a JSON file.
        """
        payload = dict(record)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        ts = payload["timestamp"].replace(":", "-")
        method = str(payload.get("method_id", "unknown"))
        fname = f"{ts}_{method}_{uuid4().hex[:8]}.json"
        path = self.base_dir / fname
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def recommend_methods(self, dataset_profile: dict[str, Any], task_type: str) -> list[str]:
        """
        Return top-3 methods for similar datasets by average historical score.
        """
        signature = self.compute_dataset_signature(dataset_profile, task_type)
        ranked = self._rank_methods(signature=signature, task_type=task_type)
        if not ranked:
            return DEFAULT_METHODS_BY_TASK.get(task_type, ["xgboost", "lightgbm", "random_forest"])[:3]
        return [m for m, _avg, _count in ranked[:3]]

    def recommendation_stats(self, dataset_profile: dict[str, Any], task_type: str) -> dict[str, Any]:
        """
        Companion stats for printing/debugging memory influence.
        """
        signature = self.compute_dataset_signature(dataset_profile, task_type)
        ranked = self._rank_methods(signature=signature, task_type=task_type)
        total_similar = sum(c for _m, _avg, c in ranked)
        return {
            "dataset_signature": signature,
            "similar_datasets_found": total_similar,
            "top_methods": [
                {"method_id": m, "avg_score": avg, "count": c}
                for m, avg, c in ranked[:3]
            ],
        }

    def _rank_methods(self, signature: str, task_type: str) -> list[tuple[str, float, int]]:
        records = self._load_records()
        if not records:
            return []

        similar = [
            r for r in records
            if str(r.get("dataset_signature", "")) == signature
            and str(r.get("task_type", "")) == task_type
        ]
        if not similar:
            # Fallback by task type only.
            similar = [r for r in records if str(r.get("task_type", "")) == task_type]
        if not similar:
            return []

        scores: dict[str, list[float]] = {}
        for rec in similar:
            method_id = str(rec.get("method_id", "")).strip()
            if not method_id:
                continue
            try:
                score = float(rec.get("score"))
            except Exception:
                continue
            scores.setdefault(method_id, []).append(score)

        ranked = [
            (method_id, sum(vals) / len(vals), len(vals))
            for method_id, vals in scores.items()
            if vals
        ]
        ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return ranked

    def _load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return records
