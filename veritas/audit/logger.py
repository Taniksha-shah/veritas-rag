"""
Enterprise Compliance Audit Logger.
Persists auditable, structured logs of all retrieval, draft generation,
recursive verification loops, and citation attribution actions.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class AuditRecord:
    execution_id: str
    timestamp: str
    query: str
    retrieved_chunk_ids: List[str]
    iterations_run: int
    converged: bool
    initial_draft: str
    final_output: str
    initial_hallucination_rate: float
    final_hallucination_rate: float
    final_attestation_rate: float
    citations_count: int
    iteration_traces: List[Dict[str, Any]]
    execution_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """Manages appending and retrieving compliance audit records."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "veritas_audit_trail.jsonl"

    def log_execution(self, record: AuditRecord) -> str:
        """Write an audit record to the persistent JSONL log."""
        data = record.to_dict()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
        return record.execution_id

    def get_recent_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent audit records for UI and compliance inspection."""
        if not self.log_file.exists():
            return []

        records = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

        return records[-limit:]
