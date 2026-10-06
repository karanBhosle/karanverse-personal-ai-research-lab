class ResearchSourceError(Exception):
    """Base error for external literature sources."""


class UnknownResearchSourceError(ResearchSourceError):
    def __init__(self, source: str) -> None:
        self.source = source
        super().__init__(f"Unknown research source: {source!r}")
