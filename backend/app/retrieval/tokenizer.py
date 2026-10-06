import re

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer for BM25 (lowercased alphanumeric tokens)."""
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]
