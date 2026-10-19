"""Access boundary for a single, explicitly approved accounting workspace."""

from dataclasses import dataclass
from typing import Callable, Mapping

from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


DEMO_EMAILS = frozenset({"demo@fatura.tn", "admin@fatura.tn", "user@fatura.tn"})


@dataclass(frozen=True)
class WorkspaceAccessPolicy:
    required: bool
    allowed_emails: frozenset[str]

    @classmethod
    def from_environment(cls, env: Mapping[str, str]):
        # PostgreSQL deployments cannot opt out via AUTH_REQUIRED=false.
        required = bool(env.get("DATABASE_URL", "").strip()) or any(
            env.get(key, "").strip().lower() in {"1", "true", "yes", "on"}
            for key in ("REQUIRE_POSTGRES", "AUTH_REQUIRED")
        )
        emails = frozenset(
            value.strip().lower()
            for value in env.get("WORKSPACE_ALLOWED_EMAILS", "").split(",")
            if value.strip()
        ) - DEMO_EMAILS
        return cls(required=required, allowed_emails=emails)

    def denial(self, user):
        if not self.required:
            return None
        if not user:
            return 401, "Connexion requise."
        if not self.allowed_emails:
            return 503, "Acces a cet espace non configure. Contactez son administrateur."
        if str(user.get("email", "")).strip().lower() not in self.allowed_emails:
            return 403, "Ce compte n'est pas autorise dans cet espace."
        return None

    @staticmethod
    def public_route(method: str, path: str) -> bool:
        if method in {"GET", "HEAD"}:
            return path in {"/", "/health", "/supported-formats"}
        return method == "POST" and path in {"/auth/login", "/auth/signup"}


class WorkspaceAccessMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, policy: WorkspaceAccessPolicy, authenticate: Callable):
        super().__init__(app)
        self.policy = policy
        self.authenticate = authenticate

    async def dispatch(self, request, call_next):
        if not self.policy.required or self.policy.public_route(request.method, request.url.path):
            return await call_next(request)

        # Query-string tokens must not grant access to documents or leak in logs.
        user = await run_in_threadpool(self.authenticate, request.headers.get("authorization"))
        denial = self.policy.denial(user)
        if denial:
            status, message = denial
            headers = {"Cache-Control": "no-store"}
            if status == 401:
                headers["WWW-Authenticate"] = "Bearer"
            return JSONResponse({"detail": message}, status_code=status, headers=headers)
        request.state.workspace_user = user
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response
