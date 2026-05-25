from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from starlette.datastructures import Headers

from .auth import build_principal_dependency
from .backends import ServiceContainer
from .config import Settings
from .logging import configure_logging, new_request_id, request_id_var
from .models import HealthResponse, InteractionRequest, InteractionResponse
from .service import MemoryService
from .worker import ConsolidationWorker

logger = logging.getLogger(__name__)


class RequestContextMiddleware:
    def __init__(self, app, request_id_header: str) -> None:
        self.app = app
        self.request_id_header = request_id_header

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = headers.get(self.request_id_header, new_request_id())
        token = request_id_var.set(request_id)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((self.request_id_header.lower().encode("latin-1"), request_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            request_id_var.reset(token)


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    configure_logging()
    principal_dependency = build_principal_dependency(runtime_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = ServiceContainer.from_settings(runtime_settings)
        await container.ensure_ready()
        app.state.container = container
        app.state.service = MemoryService(container)
        app.state.worker = ConsolidationWorker(app.state.service)
        try:
            yield
        finally:
            await container.close()

    app = FastAPI(title=runtime_settings.app_name, lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware, request_id_header=runtime_settings.request_id_header)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok", service=runtime_settings.app_name, details={"environment": runtime_settings.environment})

    @app.get("/readyz", response_model=HealthResponse)
    async def readyz(request: Request) -> HealthResponse:
        container: ServiceContainer = request.app.state.container
        details = await container.ensure_ready()
        return HealthResponse(status="ready", service=runtime_settings.app_name, details=details)

    @app.post("/v1/interactions", response_model=InteractionResponse)
    async def interact(
        payload: InteractionRequest,
        request: Request,
        principal=Depends(principal_dependency),
    ) -> InteractionResponse:
        container: ServiceContainer = request.app.state.container
        service = MemoryService(container)
        request_id = request.headers.get(runtime_settings.request_id_header, new_request_id())
        response = await service.interact(
            tenant_id=principal.tenant_id,
            user_id=payload.user_id,
            prompt=payload.prompt,
            request_id=request_id,
        )
        return response

    return app


app = create_app()
