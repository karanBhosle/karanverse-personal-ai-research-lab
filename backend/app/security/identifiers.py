import re
import uuid

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


def is_safe_identifier(value: str) -> bool:
    """Reject path segments and traversal in user-supplied ids."""
    if not value or value in {".", ".."}:
        return False
    if "/" in value or "\\" in value or "\x00" in value:
        return False
    return bool(_SAFE_ID_RE.match(value))


def validate_safe_identifier(value: str, *, field: str = "id") -> str:
    if not is_safe_identifier(value):
        raise ValueError(f"Invalid {field}")
    return value


def is_uuid_like(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False
