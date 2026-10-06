import re
from pathlib import Path

_PDF_MAGIC = b"%PDF-"
_UNSAFE_FILENAME_RE = re.compile(r"[^\w.\- ]+", re.UNICODE)


def sanitize_filename(filename: str, *, fallback: str = "upload.pdf", require_extension: str | None = None) -> str:
    name = Path(filename).name.strip()
    if not name or name in {".", ".."}:
        return fallback
    cleaned = _UNSAFE_FILENAME_RE.sub("_", name)
    if require_extension and not cleaned.lower().endswith(require_extension.lower()):
        cleaned = f"{Path(cleaned).stem}{require_extension}"
    return cleaned[:200]


def validate_pdf_upload(content: bytes, *, max_bytes: int) -> None:
    if not content:
        raise ValueError("Empty file")
    if len(content) > max_bytes:
        raise ValueError(f"File exceeds maximum size of {max_bytes} bytes")
    if not content.startswith(_PDF_MAGIC):
        raise ValueError("File is not a valid PDF (missing %PDF- header)")


def validate_text_ingest(text: str, *, max_chars: int) -> None:
    if not text or not text.strip():
        raise ValueError("Text content is empty")
    if len(text) > max_chars:
        raise ValueError(f"Text exceeds maximum length of {max_chars} characters")
