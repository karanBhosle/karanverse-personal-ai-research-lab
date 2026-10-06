from urllib.parse import urlparse


def validate_http_service_base_url(url: str, *, allowed_hosts: frozenset[str]) -> str:
    """
    Ensure configured outbound API base URLs cannot be pointed at arbitrary hosts (SSRF via env).
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"https", "http"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme or 'missing'}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("URL host is required")
    if host not in allowed_hosts:
        raise ValueError(f"URL host not allowed: {host}")
    return url.rstrip("/")
