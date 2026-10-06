import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config.settings import Settings


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Optional shared-secret gate for deployments exposed beyond localhost."""

    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(self, request: Request, call_next):
        if not self._settings.api_auth_enabled:
            return await call_next(request)
        if request.url.path in {"/health", "/docs", "/openapi.json", "/redoc"}:
            return await call_next(request)
        expected = self._settings.api_key
        if not expected:
            return JSONResponse(
                status_code=503,
                content={"detail": "API authentication is enabled but API_KEY is not configured"},
            )
        provided = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix(
            "Bearer ",
        ).strip()
        if not provided or not secrets.compare_digest(provided, expected):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)
