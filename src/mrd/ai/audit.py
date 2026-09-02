import json
from datetime import datetime, timezone
from pathlib import Path


class AIAuditLogger:

    def __init__(
        self,
        path: str = "ai_audit.jsonl",
    ) -> None:

        self.path = Path(path)

    def log(
        self,
        event: dict,
    ) -> None:

        record = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            **event,
        }

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    record,
                    default=str,
                )
                + "\n"
            )