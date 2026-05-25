from __future__ import annotations

from .backends import GistLatticeContainer
from .config import Settings
from .service import GistLatticeService


def build_default_container(settings: Settings | None = None) -> GistLatticeContainer:
    runtime_settings = settings or Settings.from_env()
    return GistLatticeContainer.from_settings(runtime_settings)


def build_default_service(settings: Settings | None = None) -> GistLatticeService:
    container = build_default_container(settings)
    return GistLatticeService(container)
