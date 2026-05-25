from __future__ import annotations

from fastapi import Header, HTTPException, status

from .config import Settings
from .models import Principal


def build_principal_dependency(settings: Settings):
    async def require_principal(
        authorization: str | None = Header(default=None, alias="Authorization"),
        tenant_id: str | None = Header(default=None, alias=settings.tenant_header),
    ) -> Principal:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
        token = authorization.removeprefix("Bearer ").strip()
        if token != settings.api_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")
        if not tenant_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing tenant header")
        return Principal(subject="authenticated-client", tenant_id=tenant_id)

    return require_principal
