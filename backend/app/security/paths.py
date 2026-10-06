from pathlib import Path


def resolve_path_under(base: Path, *parts: str) -> Path:
    """Resolve a path and ensure it stays under base (prevents traversal)."""
    base_resolved = base.resolve()
    target = (base_resolved / Path(*parts)).resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise ValueError("Path escapes allowed directory")
    return target
