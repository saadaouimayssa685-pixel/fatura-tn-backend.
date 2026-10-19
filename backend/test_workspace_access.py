import ast
from pathlib import Path
import unittest
from unittest.mock import MagicMock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from workspace_access import WorkspaceAccessMiddleware, WorkspaceAccessPolicy


SOURCE = ast.parse(Path(__file__).with_name("main.py").read_text(encoding="utf-8-sig"))
ORIGIN = "https://fatura.example.test"
OWNER = "owner@example.test"


def make_app(emails=OWNER, required=True):
    policy = WorkspaceAccessPolicy.from_environment({
        "AUTH_REQUIRED": str(required), "WORKSPACE_ALLOWED_EMAILS": emails,
    })
    app = FastAPI()
    handler = MagicMock()

    async def private_data():
        handler()
        return {"private": "invoice"}

    # Cover every existing route, not just a manually maintained sensitive list.
    for node in SOURCE.body:
        for decorator in getattr(node, "decorator_list", []):
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                if isinstance(decorator.func.value, ast.Name) and decorator.func.value.id == "app":
                    method = decorator.func.attr.upper()
                    if method in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                        app.add_api_route(decorator.args[0].value, private_data, methods=[method])

    def authenticate(header):
        return {
            "Bearer owner-session": {"email": OWNER},
            "Bearer other-session": {"email": "unapproved@example.test"},
            "Bearer demo-session": {"email": "demo@fatura.tn"},
        }.get(header)

    app.add_middleware(WorkspaceAccessMiddleware, policy=policy, authenticate=authenticate)
    app.add_middleware(CORSMiddleware, allow_origins=[ORIGIN], allow_credentials=True,
                       allow_headers=["*"], allow_methods=["*"])
    return app, handler


class WorkspacePolicyTests(unittest.TestCase):
    def test_hosted_database_always_enforces_access(self):
        self.assertTrue(WorkspaceAccessPolicy.from_environment({
            "DATABASE_URL": "postgresql://example", "AUTH_REQUIRED": "false",
        }).required)
        self.assertTrue(WorkspaceAccessPolicy.from_environment({"REQUIRE_POSTGRES": "true"}).required)

    def test_local_default_is_unchanged(self):
        self.assertFalse(WorkspaceAccessPolicy.from_environment({}).required)

    def test_empty_workspace_is_fail_closed(self):
        policy = WorkspaceAccessPolicy.from_environment({"AUTH_REQUIRED": "true"})
        self.assertEqual(policy.denial({"email": OWNER})[0], 503)

    def test_known_demo_accounts_cannot_be_authorized(self):
        policy = WorkspaceAccessPolicy.from_environment({
            "AUTH_REQUIRED": "true",
            "WORKSPACE_ALLOWED_EMAILS": "demo@fatura.tn,admin@fatura.tn,user@fatura.tn," + OWNER,
        })
        for email in ("demo@fatura.tn", "admin@fatura.tn", "user@fatura.tn"):
            self.assertEqual(policy.denial({"email": email})[0], 403)

    def test_exact_email_only_with_case_normalization(self):
        policy = WorkspaceAccessPolicy.from_environment({"AUTH_REQUIRED": "true", "WORKSPACE_ALLOWED_EMAILS": " OWNER@example.test "})
        self.assertIsNone(policy.denial({"email": OWNER.upper()}))
        self.assertEqual(policy.denial({"email": "not-" + OWNER})[0], 403)


class WorkspaceRouteTests(unittest.TestCase):
    def setUp(self):
        self.app, self.handler = make_app()
        self.client = TestClient(self.app)

    def test_every_non_public_route_requires_authentication(self):
        for route in self.app.routes:
            for method in route.methods or []:
                if WorkspaceAccessPolicy.public_route(method, route.path):
                    continue
                path = route.path.replace("{invoice_id}", "1").replace("{document_id}", "1")
                with self.subTest(method=method, path=path):
                    self.assertEqual(self.client.request(method, path).status_code, 401)
        self.handler.assert_not_called()

    def test_registered_but_unapproved_account_has_no_workspace_access(self):
        for path in ("/database/stats", "/invoices/catalog", "/invoices/1", "/invoices/catalog/download", "/rag/documents"):
            self.assertEqual(self.client.get(path, headers={"Authorization": "Bearer other-session"}).status_code, 403)
        self.handler.assert_not_called()

    def test_approved_account_can_read_and_write(self):
        for method, path in (("GET", "/invoices/1"), ("GET", "/invoices/catalog/download"), ("PUT", "/invoices/1"), ("POST", "/invoices/1/validate")):
            response = self.client.request(method, path, headers={"Authorization": "Bearer owner-session"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["cache-control"], "no-store")

    def test_query_token_does_not_grant_document_access(self):
        self.assertEqual(self.client.get("/invoices/catalog/download?token=owner-session").status_code, 401)
        self.handler.assert_not_called()

    def test_invalid_or_expired_session_denied(self):
        response = self.client.get("/database/stats", headers={"Authorization": "Bearer expired"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")
        self.handler.assert_not_called()

    def test_missing_configuration_does_not_unlock_existing_sessions(self):
        app, handler = make_app(emails="")
        response = TestClient(app).get("/database/stats", headers={"Authorization": "Bearer owner-session"})
        self.assertEqual(response.status_code, 503)
        handler.assert_not_called()

    def test_demo_session_denied_even_if_previously_created(self):
        self.assertEqual(self.client.get("/invoices", headers={"Authorization": "Bearer demo-session"}).status_code, 403)

    def test_health_remains_public(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_public_path_does_not_unlock_other_methods(self):
        self.assertEqual(self.client.post("/health").status_code, 401)

    def test_cors_preflight_and_authentication_error(self):
        response = self.client.options("/database/stats", headers={
            "Origin": ORIGIN, "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        })
        self.assertEqual(response.status_code, 200)
        response = self.client.get("/database/stats", headers={"Origin": ORIGIN})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["access-control-allow-origin"], ORIGIN)
        self.assertEqual(response.headers["access-control-allow-credentials"], "true")

    def test_unknown_origin_not_approved(self):
        response = self.client.get("/database/stats", headers={"Origin": "https://untrusted.example.test"})
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_local_legacy_mode_remains_available(self):
        app, _ = make_app(required=False)
        self.assertEqual(TestClient(app).get("/database/stats").status_code, 200)


class RealAuthHandlerTests(unittest.TestCase):
    """Execute the actual auth handlers without loading OCR, LLM or databases."""
    def setUp(self):
        self.app = FastAPI()
        self.session = MagicMock(return_value="new-session")
        self.db = MagicMock()
        self.db.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = (1, "valid-hash", "salt")
        env = {
            "app": self.app, "BaseModel": BaseModel, "Field": Field,
            "HTTPException": HTTPException,
            "WORKSPACE_ACCESS": WorkspaceAccessPolicy(True, frozenset({OWNER})),
            "normalize_email": lambda email: email.strip().lower(),
            "auth_connection": self.db, "create_session": self.session,
            "hash_password": lambda password, salt: "valid-hash" if password == "correct-password" else "wrong",
            "get_user_from_token": lambda token: {"email": OWNER},
            "secrets": __import__("secrets"),
        }
        selected = [node for node in SOURCE.body if getattr(node, "name", "") in {"SignupRequest", "LoginRequest", "signup", "login"}]
        exec(compile(ast.Module(body=selected, type_ignores=[]), "main_auth_handlers", "exec"), env)
        self.client = TestClient(self.app)

    def test_hosted_signup_is_disabled_before_database_write(self):
        response = self.client.post("/auth/signup", json={"full_name": "Other User", "company": "Other", "email": "other@example.test", "phone": "00000000", "password": "strong-test-password"})
        self.assertEqual(response.status_code, 403)
        self.db.assert_not_called()
        self.session.assert_not_called()

    def test_login_does_not_issue_session_to_unapproved_account(self):
        response = self.client.post("/auth/login", json={"email": "other@example.test", "password": "correct-password"})
        self.assertEqual(response.status_code, 403)
        self.session.assert_not_called()

    def test_login_keeps_password_check(self):
        response = self.client.post("/auth/login", json={"email": OWNER, "password": "incorrect"})
        self.assertEqual(response.status_code, 401)
        self.session.assert_not_called()

    def test_approved_login_issues_session(self):
        response = self.client.post("/auth/login", json={"email": OWNER, "password": "correct-password"})
        self.assertEqual(response.status_code, 200)
        self.session.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
