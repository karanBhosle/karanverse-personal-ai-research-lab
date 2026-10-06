from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = _BACKEND_ROOT.parent


def get_project_root() -> Path:
    return _PROJECT_ROOT


def get_data_dir() -> Path:
    return _PROJECT_ROOT / "data"


def get_processed_dir() -> Path:
    return get_data_dir() / "processed"


def get_raw_dir() -> Path:
    return get_data_dir() / "raw"


def get_index_dir() -> Path:
    return get_data_dir() / "index"


def get_bm25_index_dir() -> Path:
    return get_index_dir() / "bm25"


def get_openalex_cache_dir() -> Path:
    return get_data_dir() / "cache" / "openalex"


def get_arxiv_cache_dir() -> Path:
    return get_data_dir() / "cache" / "arxiv"


def get_research_history_dir() -> Path:
    return get_data_dir() / "research_history"
