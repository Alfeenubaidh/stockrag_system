"""
versioning.py — manages embedding + schema version lock
"""

import json
from pathlib import Path


class VersionManager:
    VERSION_FILE = Path("data/embedding_version.json")

    CURRENT_VERSION = {
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_schema_version": "v1",
    }

    def check_or_initialize(self):
        # first run → create file
        if not self.VERSION_FILE.exists():
            self._write()
            return

        # existing → validate
        existing = json.loads(self.VERSION_FILE.read_text())

        if existing != self.CURRENT_VERSION:
            raise RuntimeError(
                "Version mismatch detected.\n"
                "You must delete embeddings and reprocess.\n"
                f"Expected: {self.CURRENT_VERSION}\n"
                f"Found: {existing}"
            )

    def _write(self):
        self.VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.VERSION_FILE.write_text(
            json.dumps(self.CURRENT_VERSION, indent=2),
            encoding="utf-8",
        )