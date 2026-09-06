"""Validate corpus files before they are admitted to authoritative retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


@dataclass
class CorpusIssue:
    path: str
    issue: str


def validate_corpus(root: str | Path) -> Dict[str, object]:
    """Return a machine-readable validation report without mutating the corpus."""
    root_path = Path(root)
    issues: List[CorpusIssue] = []
    hashes: Dict[str, List[str]] = {}
    files = [path for path in root_path.rglob("*") if path.is_file()]
    for path in files:
        relative = str(path.relative_to(root_path))
        size = path.stat().st_size
        if size == 0:
            issues.append(CorpusIssue(relative, "zero_byte_file"))
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(relative)
        if path.suffix.lower() == ".pdf" and path.read_bytes()[:4] != b"%PDF":
            issues.append(CorpusIssue(relative, "invalid_pdf_signature"))
    for digest, paths in hashes.items():
        if len(paths) > 1:
            for path in paths[1:]:
                issues.append(CorpusIssue(path, f"duplicate_content:{paths[0]}"))
    return {
        "root": str(root_path),
        "file_count": len(files),
        "valid": not issues,
        "issues": [asdict(issue) for issue in issues],
    }
