from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import aiofiles
import hashlib
import io
import mimetypes
import os
import re
import secrets
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, Field

from models import OCRResponse, InvoiceData
from invoice_store import InvoiceStore
from auth_database import connect_auth
from document_store import DocumentStore
from fatura_dataset_store import FaturaDatasetStore
from postgres_invoice_store import PostgresInvoiceStore
from postgres_document_store import PostgresDocumentStore
from postgres_file_store import PostgresFileStore
from ocr_extractor import OptimizedOCRExtractor
from pdf_extractor import PDFExtractor
from rag import RAGService
from ollama_analyzer import OllamaInvoiceAnalyzer
from gemini_cloud import GeminiCloud
from invoice_rules import extract_invoice_table_text, filter_invoice_line_items
from dotenv import load_dotenv


# Charger les variables d'environnement
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")


def project_path(env_name: str, default_relative_path: str) -> str:
    configured_path = Path(os.getenv(env_name, default_relative_path))
    if configured_path.is_absolute():
        return str(configured_path)
    return str(PROJECT_ROOT / configured_path)


# Configuration
UPLOAD_FOLDER = project_path("UPLOAD_FOLDER", "data/uploads")
INVOICE_STORAGE_FOLDER = project_path("INVOICE_STORAGE_FOLDER", "data/invoice_storage/documents")
DOCUMENT_STORAGE_FOLDER = project_path("DOCUMENT_STORAGE_FOLDER", "data/document_ai/documents")
RAG_STORAGE_FOLDER = project_path("RAG_STORAGE_FOLDER", "data/rag_storage/documents")
YOLO_MODEL_PATH = project_path("YOLO_MODEL_PATH", "models/extraction/best.pt")
AUTH_DB_PATH = project_path("AUTH_DB_PATH", "data/auth.sqlite3")
INVOICE_DB_PATH = project_path("INVOICE_DB_PATH", "data/invoice_storage/invoices.sqlite3")
RAG_DB_PATH = project_path("RAG_DB_PATH", "data/rag_storage/rag.sqlite3")
FATURA_DATASET_ZIP = os.getenv("FATURA_DATASET_ZIP", r"C:\Users\maiss\Downloads\FATURA.zip")
FATURA_DATASET_DB_PATH = project_path("FATURA_DATASET_DB_PATH", "data/fatura_dataset.sqlite3")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))  # 10MB par défaut
ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "png,jpg,jpeg,tiff,bmp,webp,pdf").split(",")
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://0.0.0.0:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
]
ALLOWED_ORIGINS = sorted(set(
    DEFAULT_ALLOWED_ORIGINS
    + [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]
))
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
REQUIRE_POSTGRES = os.getenv("REQUIRE_POSTGRES", "false").lower() == "true"
SEED_DEMO_USERS = os.getenv("SEED_DEMO_USERS", "true").lower() == "true"
if REQUIRE_POSTGRES and not DATABASE_URL.startswith(("postgresql://", "postgres://")):
    raise RuntimeError("Configure DATABASE_URL with a persistent PostgreSQL database.")
ENABLE_PADDLEOCR = os.getenv("ENABLE_PADDLEOCR", "false").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_GEMMA = os.getenv("ENABLE_GEMMA", "false").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_YOLO = os.getenv("ENABLE_YOLO", "true").strip().lower() in {"1", "true", "yes", "on"}

app = FastAPI(
    title="Optimized Invoice OCR API",
    description="API optimisée pour l'extraction et l'analyse de données de factures avec OCR et IA",
    version="2.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Créer le dossier de téléchargement
@app.middleware("http")
async def check_detection_availability(request, call_next):
    if (REQUIRE_POSTGRES and request.method not in {"GET", "HEAD", "OPTIONS"}
            and request.url.path not in {"/auth/signup", "/auth/login"}):
        from starlette.concurrency import run_in_threadpool
        user = await run_in_threadpool(user_from_authorization, request.headers.get("authorization"))
        if not user:
            headers = {}
            origin = request.headers.get("origin")
            if origin in ALLOWED_ORIGINS:
                headers = {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
            return JSONResponse(status_code=401, content={"detail": "Connexion requise."}, headers=headers)
    if request.url.path in {"/extract-entities", "/extract-entities-with-ocr"} and not ENABLE_YOLO:
        return JSONResponse(
            status_code=503,
            content={"detail": "Detection YOLO indisponible sur ce serveur."},
        )
    return await call_next(request)


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INVOICE_STORAGE_FOLDER, exist_ok=True)
os.makedirs(DOCUMENT_STORAGE_FOLDER, exist_ok=True)
os.makedirs(RAG_STORAGE_FOLDER, exist_ok=True)


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question à poser aux documents indexés")
    document_id: Optional[int] = Field(None, description="Limiter la recherche à un document")
    top_k: int = Field(5, ge=1, le=20, description="Nombre de passages à retourner")


class InvoiceValidationRequest(BaseModel):
    normalized_data: Optional[dict] = Field(None, description="Champs corrigés par l'utilisateur avant validation")
    user_email: Optional[str] = Field(None, description="Utilisateur ayant valide la facture")


class InvoiceUpdateRequest(BaseModel):
    normalized_data: dict = Field(..., description="Champs de facture corrigés")
    status: str = Field("draft", description="Statut de la facture")


class SignupRequest(BaseModel):
    full_name: str = Field(..., min_length=2)
    company: str = Field(..., min_length=2)
    email: str = Field(..., min_length=5)
    phone: str = Field(..., min_length=6)
    password: str = Field(..., min_length=8)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=1)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 180000)
    return digest.hex()


def create_password_hash(password: str) -> tuple[str, str]:
    salt = secrets.token_hex(16)
    return salt, hash_password(password, salt)


def auth_connection():
    return connect_auth(DATABASE_URL, AUTH_DB_PATH)


def public_user(row: tuple) -> dict:
    return {
        "id": row[0],
        "email": row[1],
        "full_name": row[2],
        "company": row[3],
        "phone": row[4],
        "role": row[5],
    }


def init_auth_database():
    Path(AUTH_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with auth_connection() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT NOT NULL,
                company TEXT NOT NULL,
                phone TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'comptable',
                created_at REAL NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id ON auth_sessions(user_id)")
        if not DATABASE_URL:
            db.execute("PRAGMA optimize")
        if not SEED_DEMO_USERS:
            return
        demo_email = "demo@fatura.tn"
        existing = db.execute("SELECT id FROM users WHERE email = ?", (demo_email,)).fetchone()
        if not existing:
            salt, password_hash = create_password_hash("fatura-demo")
            db.execute(
                """
                INSERT INTO users (email, password_hash, salt, full_name, company, phone, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (demo_email, password_hash, salt, "Maissa Saadaoui", "YASMEEN ENGINEERING SYSTEMS", "+216 20 000 000", "Agent de saisie", time.time()),
            )
        else:
            db.execute(
                "UPDATE users SET full_name = ?, company = ?, role = ? WHERE email = ?",
                ("Maissa Saadaoui", "YASMEEN ENGINEERING SYSTEMS", "Agent de saisie", demo_email),
            )
        for email, password, role, full_name in (
            ("admin@fatura.tn", "admin123", "admin", "Administrateur FATURA"),
            ("user@fatura.tn", "user123", "user", "Operateur FATURA"),
        ):
            existing = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if not existing:
                salt, password_hash = create_password_hash(password)
                db.execute(
                    """
                    INSERT INTO users (email, password_hash, salt, full_name, company, phone, role, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (email, password_hash, salt, full_name, "FATURA.tn", "+216 20 000 001", role, time.time()),
                )


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with auth_connection() as db:
        db.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (now,))
        db.execute(
            "INSERT INTO auth_sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now, now + 60 * 60 * 24 * 7),
        )
    return token


def get_user_from_token(token: str) -> Optional[dict]:
    if not token:
        return None
    with auth_connection() as db:
        row = db.execute(
            """
            SELECT users.id, users.email, users.full_name, users.company, users.phone, users.role
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token = ? AND auth_sessions.expires_at > ?
            """,
            (token, time.time()),
        ).fetchone()
    return public_user(row) if row else None


def user_from_authorization(authorization: Optional[str]) -> Optional[dict]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    return get_user_from_token(authorization[7:].strip())


init_auth_database()

# Initialiser les services
ocr_extractor = OptimizedOCRExtractor()
pdf_extractor = PDFExtractor(ocr_extractor)
try:
    if not DATABASE_URL.lower().startswith(("postgresql://", "postgres://")):
        raise RuntimeError("DATABASE_URL PostgreSQL absent ou invalide")
    invoice_store = PostgresInvoiceStore(database_url=DATABASE_URL, documents_dir=INVOICE_STORAGE_FOLDER)
    document_store = PostgresDocumentStore(database_url=DATABASE_URL, documents_dir=DOCUMENT_STORAGE_FOLDER)
    rag_service = RAGService(database_url=DATABASE_URL, storage_dir=RAG_STORAGE_FOLDER)
    ACTIVE_DATABASE_BACKEND = "postgresql"
except Exception as database_error:
    if REQUIRE_POSTGRES or DATABASE_URL:
        raise RuntimeError("PostgreSQL unavailable; refusing temporary SQLite fallback.") from database_error
    print(f"PostgreSQL indisponible, demarrage en SQLite local: {database_error}")
    invoice_store = InvoiceStore(db_path=INVOICE_DB_PATH, documents_dir=INVOICE_STORAGE_FOLDER)
    document_store = DocumentStore(db_path=str(PROJECT_ROOT / "data" / "document_ai.sqlite3"), documents_dir=DOCUMENT_STORAGE_FOLDER)
    rag_service = RAGService(database_url="", storage_dir=RAG_STORAGE_FOLDER, sqlite_db_path=RAG_DB_PATH)
    ACTIVE_DATABASE_BACKEND = "sqlite"
ollama_analyzer = OllamaInvoiceAnalyzer()
ai_analyzer = GeminiCloud()
fatura_dataset_store = FaturaDatasetStore(db_path=FATURA_DATASET_DB_PATH, zip_path=FATURA_DATASET_ZIP)
gemma_analyzer = None
model = None
classes = {}
if ENABLE_YOLO:
    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL_PATH)
    classes = model.names

# Initialiser PaddleOCR seulement si demandé. Son chargement peut être très long
# sur un poste local et bloquer le démarrage de l'API.
paddle_pipeline = None
if ENABLE_PADDLEOCR:
    try:
        from paddleocr import PPChatOCRv4Doc

        paddle_pipeline = PPChatOCRv4Doc()
        print("✅ Pipeline PaddleOCR initialisé avec succès")
    except Exception as e:
        print(f"⚠️ Erreur initialisation PaddleOCR: {e}")


def get_gemma_analyzer():
    """Charge Gemma uniquement à la demande pour ne pas bloquer le démarrage."""
    global gemma_analyzer
    if not ENABLE_GEMMA:
        raise RuntimeError("Gemma est désactivé. Active ENABLE_GEMMA=true pour charger le modèle local.")
    if gemma_analyzer is None:
        from gemma_analyzer import GemmaAnalyzer

        gemma_analyzer = GemmaAnalyzer()
    return gemma_analyzer


def validate_file(file: UploadFile) -> bool:
    """Valide le fichier téléchargé"""
    file_extension = file.filename.split('.')[-1].lower() if file.filename else ""
    return file_extension in [ext.strip().lower() for ext in ALLOWED_EXTENSIONS]

def is_pdf_file(file_path: str) -> bool:
    """Vérifie si le fichier est un PDF."""
    return Path(file_path).suffix.lower() == ".pdf"

def extract_document_text(file_path: str):
    """Extrait le texte d'une image ou d'un PDF avec les métadonnées associées."""
    if is_pdf_file(file_path):
        return pdf_extractor.extract_text(file_path)

    return ocr_extractor.extract_text(file_path)

def classify_invoice_text(text: str):
    """Classe un document comme facture à partir du texte extrait."""
    normalized_text = text.lower()
    strong_keywords = ["facture", "invoice", "bon de livraison", "avoir"]
    weak_keywords = [
        "bill",
        "receipt",
        "total",
        "amount",
        "montant",
        "prix",
        "price",
        "tva",
        "ht",
        "ttc",
        "subtotal",
        "tax",
        "net a payer",
        "net à payer",
        "due date",
        "terms and conditions",
    ]
    negative_keywords = [
        "curriculum vitae",
        "profil",
        "experiences professionnelles",
        "expériences professionnelles",
        "formation",
        "langues",
        "competences",
        "compétences",
        "certifications",
        "projets academiques",
        "projets académiques",
        "linkedin.com",
    ]

    invoice_label_match = bool(re.search(r"\bfactu\s*re\b|\binvoice\b", normalized_text, flags=re.IGNORECASE))
    strong_matches = sum(
        1
        for keyword in strong_keywords
        if keyword != "facture" and keyword in normalized_text
    ) + int(invoice_label_match)
    weak_matches = sum(1 for keyword in weak_keywords if keyword in normalized_text)
    negative_matches = sum(1 for keyword in negative_keywords if keyword in normalized_text)
    invoice_number_match = bool(
        re.search(
            r"\b(?:factu\s*re|invoice)\s*(?:id|number|no|n[°o.]*)?\s*[:#-]?\s*[a-z0-9][a-z0-9\-\/]{2,}",
            normalized_text,
            flags=re.IGNORECASE,
        )
    )
    fiscal_match = bool(re.search(r"\b(?:mf|matricule\s+fiscal|identifiant\s+fiscal)\b", normalized_text, flags=re.IGNORECASE))
    amount_labels = sum(
        1
        for pattern in [
            r"\btotal\b",
            r"\btotal\s+ht\b",
            r"\btotal\s+ttc\b",
            r"\bmontant\s+ht\b",
            r"\bmontant\s+ttc\b",
            r"\bamount\b",
            r"\bprice\b",
            r"\bsubtotal\b",
            r"\bnet\s+[aà]\s+payer\b",
            r"\btva\b",
            r"\btax\b",
            r"\bdiscount\b",
            r"\btimbre\b",
        ]
        if re.search(pattern, normalized_text, flags=re.IGNORECASE)
    )
    date_match = bool(re.search(r"\b\d{1,2}[-\/.]\d{1,2}[-\/.]\d{2,4}\b", normalized_text) or re.search(r"\b\d{1,2}-[a-z]{3}-\d{4}\b", normalized_text))
    table_terms = sum(
        1
        for keyword in [
            "designation",
            "désignation",
            "quantite",
            "quantité",
            "prix unitaire",
            "pu ht",
            "items",
            "quantity",
            "price",
            "amount",
        ]
        if keyword in normalized_text
    )
    party_terms = sum(1 for keyword in ["buyer", "seller", "client", "customer", "fournisseur", "supplier"] if keyword in normalized_text)

    structure_score = 0
    structure_score += 2 if invoice_number_match else 0
    structure_score += 1 if fiscal_match else 0
    structure_score += min(amount_labels, 3)
    structure_score += 1 if date_match else 0
    structure_score += 1 if table_terms >= 2 else 0
    structure_score += 1 if party_terms >= 1 else 0

    has_invoice_structure = structure_score >= 4 and amount_labels >= 1
    # Hosted OCR can read the invoice header while missing a pale or distant
    # totals table. A literal invoice label together with number, date and MF
    # is still sufficient document evidence to enter extraction. It is kept
    # deliberately narrower than the generic "bon de livraison" signal.
    header_identity_evidence = invoice_label_match and invoice_number_match and fiscal_match and date_match
    is_invoice = (
        (strong_matches > 0 and has_invoice_structure and negative_matches < 3)
        or (weak_matches >= 4 and has_invoice_structure and negative_matches == 0)
        or (header_identity_evidence and negative_matches == 0)
    )

    confidence_score = 0.0
    if is_invoice:
        confidence_score = min(0.55 + (structure_score * 0.08) + (weak_matches * 0.03), 0.97)
        if header_identity_evidence:
            confidence_score = max(confidence_score, 0.82)
    elif strong_matches > 0 or weak_matches > 0:
        confidence_score = max(0.2, min(0.55 + (structure_score * 0.04) - (negative_matches * 0.12), 0.68))
    else:
        confidence_score = 0.1 if negative_matches > 0 else 0.0

    return is_invoice, confidence_score, {
        "strong_keyword_matches": strong_matches,
        "weak_keyword_matches": weak_matches,
        "negative_keyword_matches": negative_matches,
        "invoice_number_match": invoice_number_match,
        "header_identity_evidence": header_identity_evidence,
        "fiscal_match": fiscal_match,
        "amount_label_matches": amount_labels,
        "date_match": date_match,
        "table_term_matches": table_terms,
        "party_term_matches": party_terms,
        "structure_score": structure_score,
        "strong_keywords": strong_keywords,
        "weak_keywords": weak_keywords,
        "negative_keywords": negative_keywords,
    }

def safe_float(value):
    """Convertit une valeur texte/numérique en float exploitable."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d,\.\-]", "", value.replace(" ", ""))
        if "," in cleaned and "." in cleaned:
            # Format tunisien: 1.603,641 ou 1,603.641.
            decimal_separator = "," if cleaned.rfind(",") > cleaned.rfind(".") else "."
            thousands_separator = "." if decimal_separator == "," else ","
            cleaned = cleaned.replace(thousands_separator, "").replace(decimal_separator, ".")
        elif cleaned.count(".") > 1:
            # OCR des montants TND: 5.534.100 = 5 534,100.
            parts = cleaned.split(".")
            if all(len(part) == 3 for part in parts[1:]):
                cleaned = "".join(parts[:-1]) + "." + parts[-1]
            else:
                cleaned = "".join(parts)
        else:
            cleaned = cleaned.replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0

def normalize_money_amount(value, fallback_value=0.0):
    """Normalise les montants OCR/LLM, notamment les factures TND a 3 decimales.

    Certains LLM suppriment la virgule tunisienne: 6064,634 devient 6064634.
    Si un fallback regex plausible existe, il est prioritaire face a une valeur
    LLM manifestement 1000x trop grande. Sinon on applique une correction
    prudente sur les tres grands montants entiers.
    """
    amount = safe_float(value)
    fallback_amount = safe_float(fallback_value)

    if fallback_amount > 0 and amount > 0:
        ratio = amount / fallback_amount
        if 900 <= ratio <= 1100:
            return fallback_amount
        # Un LLM peut retourner le timbre ou une quantité (1, 7) au lieu du
        # total. Le montant OCR/RAG est préférable lorsqu'il est nettement
        # supérieur et que la valeur LLM est manifestement trop petite.
        if fallback_amount >= 10 and amount < fallback_amount * 0.1:
            return fallback_amount

    if amount >= 100000 and float(amount).is_integer():
        return amount / 1000

    return amount

def normalize_invoice_data(
    invoice_data: Optional[InvoiceData],
    extracted_text: str,
    rule_candidates: Optional[Dict[str, Any]] = None,
):
    """Convertit InvoiceData interne vers le format plat exploitable par la saisie."""
    fallback = (rule_candidates or {}).get("fields") or extract_invoice_fields_rag_assisted(extracted_text)[0]
    invoice_data = invoice_data or InvoiceData()

    fallback_invoice_number = str(fallback.get("invoice_number", "") or "").strip()
    invoice_number = fallback_invoice_number if len(fallback_invoice_number) >= 3 else str(invoice_data.invoice_number or "").strip()
    if invoice_number.upper() in {"A", "S", "SS"}:
        invoice_number = ""

    def candidate_text(key: str, model_value: Any) -> str:
        candidate = str(fallback.get(key, "") or "").strip()
        return candidate or str(model_value or "").strip()

    def candidate_money(key: str, model_value: Any) -> Optional[float]:
        candidate = safe_float(fallback.get(key))
        if candidate > 0:
            return candidate
        amount = normalize_money_amount(model_value, 0.0)
        return amount if amount > 0 else None

    # Candidate rows are produced by OCR table rules.  Do not expose raw LLM
    # rows: a model may be useful for deciding fields, never for inventing rows.
    if rule_candidates is not None:
        items = list(rule_candidates.get("items") or [])
    else:
        items = list(fallback.get("items") or [])

    normalized = {
        "invoice_number": invoice_number,
        "date": candidate_text("date", invoice_data.invoice_date),
        "due_date": invoice_data.due_date or "",
        "vendor_name": candidate_text("vendor_name", invoice_data.supplier.name if invoice_data.supplier else ""),
        "vendor_address": invoice_data.supplier.address if invoice_data.supplier and invoice_data.supplier.address else "",
        "vendor_tax_id": (
            candidate_text("vendor_tax_id", invoice_data.supplier.vat_number)
            if invoice_data.supplier and invoice_data.supplier.vat_number
            else candidate_text("vendor_tax_id", invoice_data.supplier.registration_number if invoice_data.supplier else "")
        ),
        "tax_id": (
            candidate_text("vendor_tax_id", invoice_data.supplier.vat_number)
            if invoice_data.supplier and invoice_data.supplier.vat_number
            else candidate_text("vendor_tax_id", invoice_data.supplier.registration_number if invoice_data.supplier else "")
        ),
        "vendor_phone": candidate_text("vendor_phone", invoice_data.supplier.phone if invoice_data.supplier else ""),
        "customer_name": candidate_text("customer_name", invoice_data.customer.name if invoice_data.customer else ""),
        "customer_address": invoice_data.customer.address if invoice_data.customer and invoice_data.customer.address else "",
        "customer_tax_id": (
            candidate_text("customer_tax_id", invoice_data.customer.vat_number)
            if invoice_data.customer and invoice_data.customer.vat_number
            else candidate_text("customer_tax_id", "")
        ),
        "customer_phone": candidate_text("customer_phone", invoice_data.customer.phone if invoice_data.customer else ""),
        "subtotal": candidate_money("subtotal", invoice_data.subtotal),
        "tax_amount": candidate_money("tax_amount", invoice_data.tax_amount),
        "tax_rate": fallback.get("tax_rate", ""),
        "stamp_duty": candidate_money("stamp_duty", None),
        "total_amount": candidate_money("total_amount", invoice_data.total_amount),
        "currency": invoice_data.currency or fallback.get("currency") or "TND",
        "payment_method": fallback.get("payment_method", ""),
        "rib": fallback.get("rib", ""),
        "amount_in_words": getattr(invoice_data, "amount_in_words", "") or fallback.get("amount_in_words", ""),
        "notes": getattr(invoice_data, "notes", "") or fallback.get("notes", "") or fallback.get("amount_in_words", ""),
        "items": items,
    }
    return normalized


EVIDENCE_GUARDED_FIELDS = {
    "invoice_number", "date", "vendor_name", "vendor_tax_id", "customer_name",
    "customer_tax_id", "vendor_phone", "customer_phone", "subtotal", "tax_amount",
    "stamp_duty", "total_amount", "currency",
}


def _canonical_identifier(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _canonical_date(value: Any) -> str:
    match = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", str(value or ""))
    if not match:
        return ""
    return "/".join(match.groups())


def _money_values_in_text(text: str) -> List[float]:
    values = []
    for token in re.findall(r"\b\d+(?:[ .]\d{3})*(?:[,.]\d{1,3})?\b", text or ""):
        amount = safe_float(token)
        if amount > 0:
            values.append(amount)
    return values


def value_has_ocr_support(field_name: str, value: Any, text: str) -> bool:
    """Require a proposed high-risk field to be traceable to OCR text."""
    if not field_has_value(field_name, value):
        return False
    if field_name in {"subtotal", "tax_amount", "stamp_duty", "total_amount"}:
        expected = safe_float(value)
        return any(abs(candidate - expected) <= 0.005 for candidate in _money_values_in_text(text))
    if field_name == "date":
        expected = _canonical_date(value)
        return bool(expected and expected in re.sub(r"[.\-]", "/", text or ""))
    if field_name == "currency":
        return bool(re.search(rf"\b{re.escape(str(value).upper())}\b", text or "", flags=re.IGNORECASE))
    expected = _canonical_identifier(value)
    observed = _canonical_identifier(text)
    if field_name in {"vendor_name", "customer_name"}:
        tokens = [token for token in re.findall(r"[a-z0-9]{3,}", str(value).lower())]
        return bool(tokens and all(token in observed for token in tokens))
    return bool(expected and expected in observed)


def enforce_evidence_guard(
    normalized_fields: Dict[str, Any],
    extracted_text: str,
    field_evidence: Optional[Dict[str, Any]] = None,
    rule_candidates: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Reject LLM/fallback values that cannot be found in OCR and return auditable proofs."""
    fallback = (rule_candidates or {}).get("fields") or extract_invoice_fields_fallback(extracted_text)
    guarded = dict(normalized_fields)
    proofs: Dict[str, Any] = {}

    for field_name in EVIDENCE_GUARDED_FIELDS:
        value = guarded.get(field_name)
        supported = value_has_ocr_support(field_name, value, extracted_text)
        # A missing currency marker on a Tunisian invoice is common. Keep the
        # explicit workflow default while recording that it is not OCR-derived.
        workflow_default_currency = field_name == "currency" and str(value or "").upper() == "TND"
        if workflow_default_currency:
            supported = True
        if not supported:
            fallback_value = fallback.get(field_name, "")
            if value_has_ocr_support(field_name, fallback_value, extracted_text):
                guarded[field_name] = fallback_value
                value = fallback_value
                supported = True
            else:
                guarded[field_name] = None if field_name in {"subtotal", "tax_amount", "stamp_duty", "total_amount"} else ""
                value = guarded[field_name]
        matching_chunk = next(
            (
                chunk for chunk in (field_evidence or {}).get(field_name, [])
                if supported and value_has_ocr_support(field_name, value, chunk.get("text", ""))
            ),
            None,
        )
        proofs[field_name] = {
            "value": value,
            "supported": supported,
            "source": "workflow_default_tnd" if workflow_default_currency else ({
                "type": "rag_chunk",
                "chunk_index": matching_chunk.get("chunk_index"),
                "page_number": matching_chunk.get("page_number"),
                "preview": matching_chunk.get("preview"),
            } if matching_chunk else ("ocr_text" if supported else None)),
        }

    table_text = extract_invoice_table_text(extracted_text)
    proposed_items = (
        list((rule_candidates or {}).get("items") or [])
        if rule_candidates is not None
        else list(guarded.get("items") or [])
    )
    verified_items, rejected_items = filter_invoice_line_items(
        proposed_items,
        table_text,
        guarded.get("total_amount"),
    )
    guarded["items"] = verified_items
    proofs["items"] = {
        "value": len(verified_items),
        "supported": bool(verified_items),
        "source": "ocr_table_rules" if verified_items else None,
        "table_detected": bool(table_text),
        "rejected": rejected_items,
    }
    return guarded, proofs


FIELD_RETRIEVAL_QUERIES = {
    "invoice_number": ["facture", "numero facture", "invoice number", "n facture"],
    "date": ["date facture", "invoice date", "date"],
    "vendor_name": ["fournisseur", "supplier", "vendeur"],
    "customer_name": ["raison sociale", "client", "customer", "acheteur"],
    "subtotal": ["montant ht", "total ht", "sous total", "droits et taxes", "subtotal"],
    "tax_amount": ["tva", "tax amount", "vat"],
    "stamp_duty": ["timbre", "timbre fiscal", "stamp duty"],
    "total_amount": ["montant ttc", "total ttc", "net a payer", "total amount"],
    "payment_method": ["mode paiement", "payment method", "paiement"],
    "rib": ["rib", "iban", "banque"],
}


def tokenize_for_retrieval(value: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[\wÀ-ÿ]+", value.lower())
        if len(token) > 1
    ]


def rank_chunks_for_field(text: str, field_name: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retourne les passages les plus utiles pour extraire un champ donne.

    C'est une recherche locale type RAG avant indexation: elle utilise les memes
    chunks que le RAG SQL, mais evite un appel LLM pour chaque champ.
    """
    chunks = rag_service.chunker.split(text)
    queries = FIELD_RETRIEVAL_QUERIES.get(field_name, [field_name])
    query_tokens = set(tokenize_for_retrieval(" ".join(queries)))
    scored_chunks: List[Dict[str, Any]] = []

    for chunk in chunks:
        chunk_text_lower = chunk.text.lower()
        chunk_tokens = set(tokenize_for_retrieval(chunk.text))
        token_overlap = len(query_tokens.intersection(chunk_tokens))
        phrase_hits = sum(1 for query in queries if query.lower() in chunk_text_lower)
        if token_overlap == 0 and phrase_hits == 0:
            continue

        number_bonus = 0.2 if re.search(r"\d", chunk.text) else 0.0
        score = token_overlap + (phrase_hits * 3) + number_bonus
        evidence_text = chunk.text[:700] + ("..." if len(chunk.text) > 700 else "")

        scored_chunks.append(
            {
                "field": field_name,
                "score": round(float(score), 3),
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "text": evidence_text,
                "preview": re.sub(r"\s+", " ", chunk.text).strip()[:260],
            }
        )

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    return scored_chunks[:top_k]


def field_has_value(field_name: str, value: Any) -> bool:
    if field_name in {"subtotal", "tax_amount", "stamp_duty", "total_amount"}:
        return safe_float(value) > 0
    if field_name == "items":
        return bool(value)
    return bool(str(value or "").strip())


def extract_invoice_fields_rag_assisted(text: str):
    """Extrait les champs en ciblant d'abord les passages pertinents par champ."""
    full_fallback = extract_invoice_fields_fallback(text)
    assisted = {
        "invoice_number": "",
        "date": "",
        "vendor_name": "",
        "customer_name": "",
        "subtotal": 0.0,
        "tax_amount": 0.0,
        "stamp_duty": 0.0,
        "total_amount": 0.0,
        "currency": full_fallback.get("currency", "TND"),
        "payment_method": "",
        "rib": "",
        "items": [],
    }
    evidence: Dict[str, Any] = {}

    for field_name in FIELD_RETRIEVAL_QUERIES:
        ranked_chunks = rank_chunks_for_field(text, field_name)
        evidence[field_name] = ranked_chunks
        if not ranked_chunks:
            continue

        field_context = "\n".join(chunk["text"] for chunk in ranked_chunks)
        candidate = extract_invoice_fields_fallback(field_context)
        candidate_value = candidate.get(field_name)
        if field_has_value(field_name, candidate_value):
            assisted[field_name] = candidate_value

    for field_name, fallback_value in full_fallback.items():
        if not field_has_value(field_name, assisted.get(field_name)) and field_has_value(field_name, fallback_value):
            assisted[field_name] = fallback_value

    return assisted, evidence


def get_invoice_field_evidence(extracted_text: str) -> Dict[str, Any]:
    _, evidence = extract_invoice_fields_rag_assisted(extracted_text)
    return evidence


def extract_invoice_line_items_from_text(text: str) -> List[Dict[str, Any]]:
    """Extrait des lignes de facture simples depuis le texte OCR.

    Cette extraction reste prudente: elle garde seulement les lignes qui
    ressemblent a un tableau article/quantite/prix/total, et ignore les totaux
    de bas de page pour ne pas inventer des articles.
    """
    table_text = extract_invoice_table_text(text)
    if not table_text:
        table_start = re.search(r"\b(?:commande\s+client|bon\s+de\s+livraison\s+client)\b", text, flags=re.IGNORECASE)
        table_text = text[table_start.start():] if table_start else ""
    if not table_text:
        return []
    table_end = re.search(r"\b(?:base\s+tva|total\s+tva|total\s+ttc|droit\s+de\s+timbre|arretee\s+la\s+presente)\b", table_text, flags=re.IGNORECASE)
    if table_end:
        table_text = table_text[: table_end.start()]

    columnar_items = extract_columnar_invoice_items(table_text)
    if columnar_items:
        return filter_invoice_line_items(columnar_items, table_text)[0]

    items: List[Dict[str, Any]] = []
    header_terms = {
        "code",
        "article",
        "designation",
        "désignation",
        "quantite",
        "quantité",
        "prix",
        "unitaire",
        "montant",
        "tva",
        "htva",
        "ttc",
    }
    meta_terms = {
        "facture",
        "client",
        "vendeur",
        "date",
        "mode",
        "reglement",
        "règlement",
        "bon de livraison",
        "commande client",
        "tel",
        "fax",
        "site",
        "email",
        "page",
        "capital",
    }
    summary_terms = re.compile(
        r"\b(?:total|tva|taxe|timbre|net\s*(?:à|a)?\s*payer|montant\s+(?:ht|ttc)|base\s+tva)\b",
        flags=re.IGNORECASE,
    )
    amount_pattern = r"\d+(?:[\s.]\d{3})*(?:[,.]\d{1,3})?"
    pending_description = ""

    for raw_line in re.split(r"[\n\r]+", table_text):
        line = re.sub(r"\s+", " ", raw_line).strip(" |;:")
        if len(line) < 8:
            if 3 <= len(line) <= 40 and re.search(r"[A-Za-zÀ-ÿ]{3,}", line):
                pending_description = line
            continue
        lower_line = line.lower()
        if any(term in lower_line for term in meta_terms):
            continue
        if summary_terms.search(line):
            continue

        number_matches = list(re.finditer(amount_pattern, line))
        words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", line)
        if len(number_matches) == 0 and words:
            if not all(word.lower() in header_terms for word in words):
                pending_description = line[:180]
            continue
        if len(number_matches) < 1:
            continue

        first_number = number_matches[0]
        inline_description = line[: first_number.start()].strip(" -:|")
        description = inline_description if len(inline_description) >= 3 else pending_description
        if inline_description and not re.search(r"[A-Za-zÀ-ÿ]{3,}", inline_description) and pending_description:
            description = pending_description
        if re.fullmatch(r"[A-Za-zÀ-ÿ]{1,3}", inline_description or "") and pending_description:
            description = pending_description
        if pending_description and inline_description and len(inline_description.split()) == 1:
            description = pending_description
        if len(description) < 3 or not re.search(r"[A-Za-zÀ-ÿ]{3,}", description):
            continue
        if any(term in description.lower() for term in meta_terms):
            continue

        parsed_numbers = [(match, safe_float(match.group(0))) for match in number_matches]
        if len(parsed_numbers) < 2:
            continue

        # Read monetary columns from the right. Product references such as
        # `N6/5` occur before the actual quantity and must never win this tie.
        total = parsed_numbers[-1][1]
        possible_tax_rate = parsed_numbers[-2][1]
        has_tax_rate_column = possible_tax_rate in {1.0, 5.0, 7.0, 13.0, 19.0}
        unit_price_index = -3 if has_tax_rate_column and len(parsed_numbers) >= 3 else -2
        unit_price = parsed_numbers[unit_price_index][1]
        quantity_candidates = parsed_numbers[:unit_price_index]
        quantity = 1.0
        quantity_match = None
        for candidate_match, candidate in reversed(quantity_candidates):
            if 0 < candidate <= 10000:
                quantity = candidate
                quantity_match = candidate_match
                break
        if total <= 0 or unit_price <= 0:
            continue
        if quantity_match is not None:
            description = line[:quantity_match.start()].strip(" -:|")
        elif unit_price_index < 0:
            description = line[:parsed_numbers[unit_price_index][0].start()].strip(" -:|")
        if len(description) < 3 or not re.search(r"[A-Za-zÀ-ÿ]{3,}", description):
            continue

        items.append(
            {
                "description": description[:180],
                "quantity": quantity,
                "unit_price": unit_price,
                "total": total,
                "confidence": 0.62 if len(number_matches) >= 3 else 0.52,
                "source": "ocr_table_fallback",
            }
        )
        pending_description = ""

    # Deduplicate OCR echoes while preserving order.
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        key = (
            re.sub(r"\W+", "", item["description"].lower())[:50],
            round(float(item["total"]), 3),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return filter_invoice_line_items(deduped, table_text)[0]


def extract_columnar_invoice_items(table_text: str) -> List[Dict[str, Any]]:
    """Reconciles OCR tables whose labels and amount columns are separate.

    Some scans are read column by column: the OCR returns several descriptions,
    then their right-hand amount column. We only activate this conservative path
    when at least four descriptions and a matching monetary sequence exist.
    """
    # Do not treat the space between two adjacent OCR cells as a thousands
    # separator; otherwise `1.400.000 1400.000` becomes one corrupted amount.
    amount_pattern = r"(?:\d+(?:\.\d{3})+|\d+(?:,\d{3})+)"
    header_or_summary = re.compile(
        r"\b(?:code|article|designation|désignation|quantite|quantité|prix|unitaire|montant|"
        r"total|tva|timbre|net\s*(?:à|a)?\s*payer|client|vendeur|facture|date|page)\b",
        flags=re.IGNORECASE,
    )
    inline_items: List[Dict[str, Any]] = []
    pending_descriptions: List[str] = []
    standalone_amounts: List[float] = []

    # Tesseract often returns a vertical table as one OCR cell per line.  A
    # reference followed by designation, quantity, unit price and line amount
    # is much stronger evidence than pairing arbitrary text with later digits.
    reference_pattern = r"(?:[A-Z0-9]{1,4}-)?\d{4,}(?:[-/]\d+){0,3}"
    money_pattern = r"\d+(?:[ .]\d{3})*(?:[,.]\d{1,3})?"
    vertical_row_pattern = re.compile(
        rf"^\s*{reference_pattern}\s*$\s*"
        rf"^\s*(?P<description>[A-Za-zÀ-ÿ][^\r\n]{{2,180}})\s*$\s*"
        rf"^\s*(?P<quantity>\d+(?:[,.]\d+)?)\s*$\s*"
        rf"^\s*(?P<unit_price>{money_pattern})\s*$\s*"
        rf"^\s*(?P<total>{money_pattern})\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    vertical_items = [
        {
            "description": match.group("description").strip(),
            "quantity": safe_float(match.group("quantity")),
            "unit_price": safe_float(match.group("unit_price")),
            "total": safe_float(match.group("total")),
            "confidence": 0.84,
            "source": "ocr_vertical_table",
        }
        for match in vertical_row_pattern.finditer(table_text)
    ]
    if vertical_items:
        return vertical_items

    for raw_line in re.split(r"[\n\r]+", table_text):
        line = re.sub(r"\s+", " ", raw_line).strip(" |;:")
        if not line or header_or_summary.search(line):
            continue
        if re.fullmatch(amount_pattern, line):
            amount = safe_float(line)
            if amount > 0:
                standalone_amounts.append(amount)
            continue

        matches = list(re.finditer(amount_pattern, line))
        letters = re.search(r"[A-Za-zÀ-ÿ]{3,}", line)
        if not letters:
            continue
        if matches:
            description = line[: matches[0].start()].strip(" -:=|")
            description = re.sub(r"\s*\|\s*", " ", description)
            description = re.sub(rf"^(?:{reference_pattern})\s*", "", description, flags=re.IGNORECASE)
            # The OCR cell immediately before a monetary column is usually
            # quantity; it belongs to the numeric columns, not the label.
            description = re.sub(r"\s+\d+(?:[,.]\d+)?$", "", description).strip()
            if len(description) < 3:
                continue
            amounts = [safe_float(match.group(0)) for match in matches]
            total = amounts[-1]
            if total <= 0:
                continue
            inline_items.append({
                "description": description[:180],
                "quantity": 1,
                "unit_price": amounts[-2] if len(amounts) > 1 else total,
                "total": total,
                "confidence": 0.72,
                "source": "ocr_columnar_table",
            })
        else:
            pending_descriptions.append(line[:180])

    # When a complete row is already on one OCR line, its two monetary cells
    # are stronger evidence than attempting a second column reconciliation.
    if len(inline_items) >= 2:
        return inline_items
    if len(pending_descriptions) < 2 or len(standalone_amounts) < len(pending_descriptions):
        return []

    # OCR often repeats the monetary column. The first complete sequence is the
    # source order corresponding to the pending descriptions.
    reconciled = [*inline_items]
    for description, amount in zip(pending_descriptions, standalone_amounts):
        reconciled.append({
            "description": description,
            "quantity": 1,
            "unit_price": amount,
            "total": amount,
            "confidence": 0.68,
            "source": "ocr_columnar_table",
        })
    return reconciled


def extract_amount_in_words(text: str) -> str:
    """Extrait le montant en lettres quand il apparait dans le pied de facture."""
    if not text:
        return ""

    normalized_text = re.sub(r"\r\n?", "\n", text)
    patterns = [
        r"(?:arr[êe]t[ée]e?\s+la\s+pr[ée]sente\s+facture\s+(?:à|a)\s+la\s+somme\s+de|arretee\s+la\s+presente\s+facture\s+a\s+la\s+somme\s+de)\s*:?\s*(.+)",
        r"(?:montant\s+en\s+lettres|somme\s+de)\s*:?\s*(.+)",
    ]
    stop_pattern = re.compile(
        r"\b(?:retour\s+de\s+marchandises|page\s+\d+|code\s+tva|base\s+tva|total\s+tva|total\s+ttc|cachet|signature|sa\.?\s+au\s+capital|matricule\s+fiscal)\b",
        flags=re.IGNORECASE,
    )

    for pattern in patterns:
        match = re.search(pattern, normalized_text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue

        candidate_lines: List[str] = []
        for raw_line in match.group(1).split("\n"):
            line = re.sub(r"\s+", " ", raw_line).strip(" :-;\t")
            if not line:
                continue
            stop_match = stop_pattern.search(line)
            if stop_match:
                line = line[: stop_match.start()].strip(" :-;\t")
            if not line:
                break
            if len(candidate_lines) >= 3:
                break
            candidate_lines.append(line)
            if stop_match:
                break

        candidate = " ".join(candidate_lines)
        candidate = re.sub(r"\s+", " ", candidate).strip(" .:-;")
        amount_words_match = re.search(
            r"([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s'\-]+Dinar(?:\(s\)|s)?(?:\s+et\s+[A-Za-zÀ-ÿ\s'\-]+Millime(?:\(s\)|s)?)?)",
            candidate,
            flags=re.IGNORECASE,
        )
        if amount_words_match:
            candidate = amount_words_match.group(1)
            candidate = re.sub(r"\s+", " ", candidate).strip(" .:-;")
        elif re.search(r"\b(?:Dinar|Millime)s?\b", candidate, flags=re.IGNORECASE):
            return ""
        if amount_words_match and len(candidate) >= 8 and re.search(r"[A-Za-zÀ-ÿ]{3,}", candidate):
            return candidate[:240]

    return ""


def extract_labeled_money(text: str, labels: List[str], prefer: str = "last") -> float:
    """Find a money amount close to an invoice total label.

    OCR often splits totals over two lines, for example:
    "Total HT\n4650.000" or "5534.100\nTOTAL TTC". We inspect nearby
    lines instead of taking the first number after a loose regex.
    """
    if not text:
        return 0.0

    lines = [re.sub(r"\s+", " ", line).strip(" |:;") for line in re.split(r"[\n\r]+", text)]
    label_patterns = [re.compile(label, flags=re.IGNORECASE) for label in labels]
    amount_pattern = re.compile(
        r"(?:\d{1,3}(?:[ .]\d{3})+(?:,\d{1,3})?|\d+(?:[,.]\d{1,3})?)"
    )
    table_header_pattern = re.compile(
        r"\b(?:code|reference|référence|article|designation|désignation|quantit[ée]|prix|unitaire|montant)\b",
        flags=re.IGNORECASE,
    )
    candidates: List[float] = []

    for index, line in enumerate(lines):
        if not line:
            continue
        if re.search(r"\bcode\s+t\.?\s*v\.?\s*a\.?\b|\bt\.?\s*v\.?\s*a\.?\s*:", line, flags=re.IGNORECASE):
            continue
        label_match = next((pattern.search(line) for pattern in label_patterns if pattern.search(line)), None)
        if not label_match:
            continue
        # A column title such as "Désignation | Prix unitaire | Total HT" is
        # evidence of a table, not evidence of the invoice subtotal.
        if table_header_pattern.search(line):
            continue

        scoped_candidates: List[float] = []
        after_label = line[label_match.end():]
        for match in amount_pattern.finditer(after_label):
            if match.end() < len(after_label) and after_label[match.end(): match.end() + 1] == "%":
                continue
            amount = safe_float(match.group(0))
            if amount > 0:
                scoped_candidates.append(amount)

        if not scoped_candidates and index + 1 < len(lines):
            next_line = lines[index + 1]
            next_amount = amount_pattern.fullmatch(re.sub(r"\s*(?:TND|DT)\s*$", "", next_line, flags=re.IGNORECASE))
            if next_amount:
                amount = safe_float(next_amount.group(0))
                if amount > 0:
                    scoped_candidates.append(amount)

        if not scoped_candidates and index > 0 and re.search(r"t\.?\s*t\.?\s*c\.?|net\s+(?:à|a)\s+payer", line, flags=re.IGNORECASE):
            for match in amount_pattern.finditer(lines[index - 1]):
                amount = safe_float(match.group(0))
                if amount > 0:
                    scoped_candidates.append(amount)

        if scoped_candidates:
            candidates.append(scoped_candidates[-1])

    if not candidates:
        return 0.0

    return max(candidates) if prefer == "max" else candidates[-1]


def extract_tax_rate(text: str) -> str:
    rates = re.findall(r"(?:tva|taxe|vat)[^0-9\n\r%]{0,20}([0-9]{1,2}(?:[,.][0-9]+)?)\s*%", text or "", flags=re.IGNORECASE)
    if not rates:
        rates = re.findall(r"\b([0-9]{1,2}(?:[,.][0-9]+)?)\s*%\b", text or "")
    numeric_rates = [safe_float(rate) for rate in rates]
    numeric_rates = [rate for rate in numeric_rates if 0 < rate <= 30]
    if not numeric_rates:
        return ""
    selected = max(numeric_rates)
    return f"{selected:g}%".replace(".", ",")


def extract_invoice_number(text: str) -> str:
    patterns = [
        r"\b(?:factu\s*re|invoice)\s*(?:n[°o.]*)?\s*[:#-]?\s*([A-Z0-9][A-Z0-9\-\/]{1,})",
        r"\bB\.?\s*L\.?\s*-\s*Facture\s*N[°o.]?\s*([A-Z0-9\-\/]+)",
        r"\bN[°o.]\s*(?:de\s+facture)?\s*[:#-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text or "", flags=re.IGNORECASE):
            value = match.group(1).strip(" .:-|")
            if value and any(char.isdigit() for char in value):
                return value[:80]
    return ""


def extract_invoice_date(text: str) -> str:
    patterns = [
        r"\bDate\s*(?:de\s+la\s+facture|de\s+Facturation|facture)?\s*[:\-]?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})",
        r"\b([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            return match.group(1).replace(".", "/")
    return ""


def extract_party_hints(text: str) -> Dict[str, str]:
    """Extract likely supplier/customer names and tax identifiers from OCR text."""
    hints = {
        "vendor_name": "",
        "vendor_tax_id": "",
        "customer_name": "",
        "customer_tax_id": "",
        "vendor_address": "",
        "customer_address": "",
    }
    if not text:
        return hints

    lines = [re.sub(r"\s+", " ", line).strip(" .:-|") for line in re.split(r"[\n\r]+", text) if line.strip()]
    tax_ids = re.findall(r"\b\d{5,8}\s*[A-Z]?(?:\s*/\s*[A-Z0-9]{1,3}){1,4}\b|\b\d{5,8}[A-Z]{1,3}\d{3}\b", text, flags=re.IGNORECASE)
    tax_ids = [clean_tax_identifier(value) for value in tax_ids if clean_tax_identifier(value)]

    customer_patterns = [
        r"Raison\s+Sociale?\s*[:;]\s*([^\n\r]{2,100})",
        r"Nom\s+du\s+Client\s*[:;]\s*([^\n\r]{2,100})",
        r"\bClient\s*[:;]\s*([^\n\r]{2,100})",
        r"\bSTE\s+DE\s+PROMOTION\b([^\n\r]*)",
        r"\b(BRIDGE\s+IMMOBILI[ÈE]RE[^\n\r]{0,50})",
    ]
    for pattern in customer_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = re.sub(r"\b(?:MF|TVA|ADRESSE|DATE)\b.*$", "", match.group(1), flags=re.IGNORECASE).strip(" .:-)")
            hints["customer_name"] = value[:120]
            break
    customer_tax_match = re.search(
        r"(?:Nom\s+du\s+Client|Client|Raison\s+Sociale?)\s*[:;][^\n\r]{0,160}(?:\r?\n[^\n\r]{0,160}){0,2}?\b(?:MF|TVA)\s*[:;]\s*(\d{5,8}\s*[A-Z]?(?:\s*/\s*[A-Z0-9]{1,3}){1,4})",
        text,
        flags=re.IGNORECASE,
    )
    if customer_tax_match:
        hints["customer_tax_id"] = clean_tax_identifier(customer_tax_match.group(1))
    if re.search(r"\bSTE\s+DE\s+PROMOTION\b", text, flags=re.IGNORECASE):
        hints["customer_name"] = "STE DE PROMOTION"
    if re.search(r"ami-commerciale|Commerce\s+de\s+Quincaillerie\s+en\s+Gros", text, flags=re.IGNORECASE):
        hints["vendor_name"] = "AMI Commerciale"

    header_window = "\n".join(lines[:12])
    vendor_patterns = [
        r"\b(COMPT[OI]R\s+HAMMAMI[^\n\r]{0,50})",
        r"\b(BEN\s+ABDELKADER)\b",
        r"\b(BUREAUTICA)\b",
        r"\b(?:ami|AMi)\s*\n?\s*(Commerciale)\b",
        r"\b(KAST\s+Events)\b",
        r"\bsoci[ée]t[ée]\s+(KAST)\b",
        r"\b(AZ\s*CUISINE)\b",
    ]
    if not hints["vendor_name"]:
        for pattern in vendor_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                if len(match.groups()) > 1 and match.group(1).lower() == "commerciale":
                    hints["vendor_name"] = "AMI Commerciale"
                else:
                    hints["vendor_name"] = re.sub(r"\s+", " ", match.group(1)).strip()[:120]
                break

    if not hints["vendor_name"]:
        for line in lines[:8]:
            if re.search(r"\b(facture|date|mf|tel|fax|adresse|client|page)\b", line, flags=re.IGNORECASE):
                continue
            if re.search(r"[A-ZÀ-Ý]{3,}", line) and len(line) <= 80:
                hints["vendor_name"] = line[:120]
                break

    if tax_ids:
        if hints["customer_name"] and re.search(r"Raison\s+Sociale?", text, flags=re.IGNORECASE) and len(tax_ids) >= 2:
            hints["customer_tax_id"] = tax_ids[0]
            hints["vendor_tax_id"] = tax_ids[-1]
        elif re.search(r"\bB\.?\s*L\.?\s*-\s*Facture\b", text, flags=re.IGNORECASE) and len(tax_ids) >= 2:
            hints["customer_tax_id"] = tax_ids[0]
            hints["vendor_tax_id"] = tax_ids[-1]
        else:
            hints["vendor_tax_id"] = tax_ids[0]

    return hints


def clean_phone_number(value: str) -> str:
    cleaned = re.sub(r"[^\d+]", " ", value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) not in {8, 11}:
        return ""
    return cleaned[:40]


def extract_phone_near_labels(text: str, labels: List[str], fallback_index: int = 0) -> str:
    if not text:
        return ""

    phone_pattern = r"(?:\+?\d[\d ().\-]{5,}\d)"
    label_pattern = "|".join(re.escape(label) for label in labels)
    for match in re.finditer(rf"(?:{label_pattern})[^\n]{{0,100}}?(?:t[ée]l(?:[ée]phone)?|phone|gsm|mobile)\s*[:\-]?\s*({phone_pattern})", text, flags=re.IGNORECASE):
        phone = clean_phone_number(match.group(1))
        if phone:
            return phone

    phones = []
    for match in re.finditer(rf"(?:t[ée]l(?:[ée]phone)?|tel|phone|gsm|mobile)\s*[:\-]?\s*({phone_pattern})", text, flags=re.IGNORECASE):
        phone = clean_phone_number(match.group(1))
        if phone and phone not in phones:
            phones.append(phone)

    if fallback_index < len(phones):
        return phones[fallback_index]
    return ""


def clean_tax_identifier(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z/]", " ", value or "").upper()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    compact = cleaned.replace(" ", "")
    slash_match = re.search(r"\d{5,8}[A-Z]?(?:/[A-Z0-9]{1,3}){1,4}", compact)
    if slash_match:
        return slash_match.group(0)
    compact_match = re.search(r"\d{5,8}[A-Z]{1,3}\d{3}", compact)
    if compact_match:
        return compact_match.group(0)
    grouped_match = re.search(r"\d{5,8}\s+[A-Z](?:\s+[A-Z0-9]{1,3}){1,4}", cleaned)
    if grouped_match:
        return grouped_match.group(0)
    return ""


def extract_tax_ids_by_party(text: str) -> Dict[str, str]:
    if not text:
        return {"vendor_tax_id": "", "customer_tax_id": ""}

    tax_pattern = r"\d{5,8}\s*[A-Za-z]?(?:\s*/\s*[A-Za-z0-9]{1,3}){1,4}\b|\b\d{5,8}[A-Za-z]{1,3}\d{3}\b"
    vendor_labels = ["fournisseur", "vendeur", "supplier", "seller", "société", "societe", "company"]
    customer_labels = ["client", "acheteur", "buyer", "customer", "facturé", "facturee", "facturée"]
    generic_tax_labels = ["mf", "matricule fiscal", "identifiant fiscal", "vat", "tax id"]

    candidates: List[Dict[str, Any]] = []
    for match in re.finditer(tax_pattern, text, flags=re.IGNORECASE):
        value = clean_tax_identifier(match.group(0))
        if not value:
            continue
        start = max(0, match.start() - 260)
        end = min(len(text), match.end() + 120)
        context = text[start:end].lower()
        before_context = text[start:match.start()].lower()
        has_tax_label = any(label in context for label in generic_tax_labels)
        if not has_tax_label and "/" not in value:
            continue
        vendor_score = sum(1 for label in vendor_labels if label in context)
        customer_score = sum(1 for label in customer_labels if label in context)
        if any(label in before_context[-120:] for label in customer_labels):
            customer_score += 2
        if any(label in before_context[-120:] for label in vendor_labels):
            vendor_score += 2
        candidates.append(
            {
                "value": value,
                "position": match.start(),
                "vendor_score": vendor_score,
                "customer_score": customer_score,
            }
        )

    unique_candidates: List[Dict[str, Any]] = []
    seen_tax_ids = set()
    for candidate in candidates:
        if candidate["value"] in seen_tax_ids:
            continue
        seen_tax_ids.add(candidate["value"])
        unique_candidates.append(candidate)

    vendor_tax_id = ""
    customer_tax_id = ""
    vendor_candidates = sorted(
        unique_candidates,
        key=lambda item: (item["vendor_score"] - item["customer_score"], -item["position"]),
        reverse=True,
    )
    customer_candidates = sorted(
        unique_candidates,
        key=lambda item: (item["customer_score"] - item["vendor_score"], item["position"]),
        reverse=True,
    )

    if vendor_candidates and vendor_candidates[0]["vendor_score"] > vendor_candidates[0]["customer_score"]:
        vendor_tax_id = vendor_candidates[0]["value"]
    if customer_candidates and customer_candidates[0]["customer_score"] > customer_candidates[0]["vendor_score"]:
        customer_tax_id = customer_candidates[0]["value"]

    remaining = [candidate["value"] for candidate in unique_candidates]
    if not vendor_tax_id and remaining:
        vendor_tax_id = remaining[0]
    return {"vendor_tax_id": vendor_tax_id, "customer_tax_id": customer_tax_id}


def extract_invoice_fields_fallback(text: str):
    """Extraction déterministe simple pour garder le flux fonctionnel si le LLM échoue."""
    data = {
        "invoice_number": "",
        "date": "",
        "vendor_name": "",
        "vendor_tax_id": "",
        "tax_id": "",
        "vendor_phone": "",
        "customer_name": "",
        "customer_tax_id": "",
        "customer_phone": "",
        "subtotal": 0.0,
        "tax_amount": 0.0,
        "stamp_duty": 0.0,
        "discount": 0.0,
        "total_amount": 0.0,
        "net_payable": 0.0,
        "currency": "TND",
        "payment_method": "",
        "rib": "",
        "amount_in_words": "",
        "notes": "",
        "items": [],
    }

    patterns = {
        "invoice_number": r"(?:facture|invoice)\s*(?:n[°o.]*)?\s*[:#-]?\s*([A-Z0-9\-\/]+)",
        "date": r"(?:date(?:\s+facture)?|invoice date)[^\n\r0-9]{0,80}([0-9]{1,2}[-\/.][0-9]{1,2}[-\/.][0-9]{2,4})",
        "vendor_name": r"(?:fournisseur|supplier)\s*[:\-]\s*(.+)",
        "customer_name": r"(?:client|customer|raison\s+sociale)\s*[:;\-]\s*(.+)",
        # Do not let an OCR amount consume the following table lines: ``\s``
        # includes newlines, while a monetary token may contain only spaces or
        # tabs within its own line.
        "subtotal": r"(?:sous[- ]?total(?:\s+ht)?|subtotal|total\s+ht|montant\s+ht|droits\s+et\s+taxes)\D{0,20}([0-9][0-9 \t.,]*)",
        "tax_amount": r"(?:tva|tax\s+amount)(?:\s+[0-9.,]+\s*%)?\D{0,15}([0-9][0-9 \t.,]*)",
        "stamp_duty": r"(?:timbre(?:\s+fiscal)?|stamp(?:\s+duty)?)\D{0,15}([0-9][0-9 \t.,]*)",
        "discount": r"(?:remise|discount|rabais)\D{0,15}([0-9][0-9 \t.,]*)",
        "total_amount": r"(?:total\s+ttc|montant\s+ttc|net\s+à\s+payer|net\s+a\s+payer|total amount)\D{0,20}([0-9][0-9 \t.,]*)",
        "net_payable": r"(?:net\s+à\s+payer|net\s+a\s+payer|montant\s+à\s+payer|montant\s+a\s+payer)\D{0,20}([0-9][0-9 \t.,]*)",
        "payment_method": r"(?:mode\s+paiement|payment method)\s*[:\-]\s*(.+)",
        "rib": r"(?:rib|iban)\s*[:\-]?\s*([A-Z0-9 ]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip()
        if key in {"subtotal", "tax_amount", "stamp_duty", "discount", "total_amount", "net_payable"}:
            data[key] = safe_float(value)
        else:
            data[key] = value[:160]

    party_hints = extract_party_hints(text)
    for key in ["vendor_name", "vendor_tax_id", "customer_name", "customer_tax_id", "vendor_address", "customer_address"]:
        if party_hints.get(key):
            data[key] = party_hints[key]
    if data["vendor_tax_id"]:
        data["tax_id"] = data["vendor_tax_id"]

    invoice_number = extract_invoice_number(text)
    if invoice_number:
        data["invoice_number"] = invoice_number
    elif str(data.get("invoice_number", "")).upper() in {"A", "S", "SS"}:
        data["invoice_number"] = ""
    if data["date"]:
        data["date"] = str(data["date"]).replace(".", "/")
    invoice_date = extract_invoice_date(text)
    if invoice_date:
        data["date"] = invoice_date

    subtotal = extract_labeled_money(
        text,
        [
            r"\btotal\s+(?:net\s+)?h\.?\s*t\.?\b",
            r"\bmontant\s+h\.?\s*t\.?\b",
            r"\bnet\s+h\.?\s*t\.?\b",
        ],
        prefer="max",
    )
    tax_amount = extract_labeled_money(text, [r"\bt\.?\s*v\.?\s*a\.?\b", r"\btax\s+amount\b"], prefer="last")
    stamp_duty = extract_labeled_money(text, [r"\btimbre(?:\s+fiscal)?\b", r"\bstamp\s+duty\b"], prefer="last")
    discount = extract_labeled_money(text, [r"\bremise\b", r"\bdiscount\b", r"\brabais\b"], prefer="last")
    total_amount = extract_labeled_money(
        text,
        [
            r"\btotal\s+t\.?\s*t\.?\s*c\.?\b",
            r"\bmontant\s+t\.?\s*t\.?\s*c\.?\b",
            r"\bmontant\s+total\s+de\s+la\s+facture\b",
            r"\bnet\s+(?:à|a)\s+payer\b",
        ],
        prefer="max",
    )
    net_payable = extract_labeled_money(text, [r"\bnet\s+(?:à|a)\s+payer\b", r"\bmontant\s+(?:à|a)\s+payer\b"], prefer="max")
    if subtotal > 0:
        data["subtotal"] = subtotal
    if tax_amount > 0:
        data["tax_amount"] = tax_amount
    if stamp_duty > 0:
        data["stamp_duty"] = stamp_duty
    if discount > 0:
        data["discount"] = discount
    if net_payable > 0:
        data["net_payable"] = net_payable
    if total_amount > 0:
        data["total_amount"] = total_amount
    expected_total = safe_float(data["subtotal"]) + safe_float(data["tax_amount"]) + safe_float(data["stamp_duty"])
    if safe_float(data["discount"]) > 0 and safe_float(data["subtotal"]) > safe_float(data["discount"]):
        expected_total = safe_float(data["subtotal"]) - safe_float(data["discount"]) + safe_float(data["tax_amount"]) + safe_float(data["stamp_duty"])
    if safe_float(data["net_payable"]) > 0:
        data["total_amount"] = safe_float(data["net_payable"])
    elif data.get("subtotal") and data.get("tax_amount") and (not data.get("total_amount") or safe_float(data["total_amount"]) < safe_float(data["subtotal"])):
        data["total_amount"] = expected_total
    data["tax_rate"] = extract_tax_rate(text)

    if not data["date"]:
        date_match = re.search(r"\b([0-9]{1,2}[-\/.][0-9]{1,2}[-\/.][0-9]{2,4})\b", text)
        if date_match:
            data["date"] = date_match.group(1).replace(".", "/")

    customer_multiline = re.search(
        r"raison\s+sociale\s*:\s*(.+?)\s*\n\s*[0-9]{1,2}[-\/][0-9]{1,2}[-\/][0-9]{2,4}\s+([^\n\r]{2,80})",
        text,
        flags=re.IGNORECASE,
    )
    if customer_multiline:
        first_line = customer_multiline.group(1).strip()
        second_line = re.split(
            r"\b(?:adresse|code\s+client|tel|designation)\b",
            customer_multiline.group(2).strip(),
            flags=re.IGNORECASE,
        )[0].strip(" :-")
        if first_line and second_line and second_line.lower() not in first_line.lower():
            data["customer_name"] = f"{first_line} {second_line}"[:160]

    data["vendor_phone"] = extract_phone_near_labels(text, ["fournisseur", "vendeur", "seller", "supplier"], 0)
    data["customer_phone"] = extract_phone_near_labels(text, ["client", "acheteur", "buyer", "customer"], 1)
    tax_ids = extract_tax_ids_by_party(text)
    if not data["vendor_tax_id"]:
        data["vendor_tax_id"] = tax_ids["vendor_tax_id"]
        data["tax_id"] = tax_ids["vendor_tax_id"]
    if not data["customer_tax_id"]:
        data["customer_tax_id"] = tax_ids["customer_tax_id"]
    if data["customer_tax_id"] and data["customer_tax_id"] == data["vendor_tax_id"]:
        duplicate_customer_tax = re.search(
            r"(?:Nom\s+du\s+Client|Client|Raison\s+Sociale?).{0,220}?\b(?:MF|TVA)\s*:\s*(\d{5,8}\s*[A-Z]?(?:\s*/\s*[A-Z0-9]{1,3}){1,4})",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        corrected_customer_tax = clean_tax_identifier(duplicate_customer_tax.group(1)) if duplicate_customer_tax else ""
        data["customer_tax_id"] = corrected_customer_tax if corrected_customer_tax != data["vendor_tax_id"] else ""

    if re.search(r"\b(eur|€)\b", text, flags=re.IGNORECASE):
        data["currency"] = "EUR"
    elif re.search(r"\b(usd|\$)\b", text, flags=re.IGNORECASE):
        data["currency"] = "USD"

    data["items"] = extract_invoice_line_items_from_text(text)
    data["amount_in_words"] = extract_amount_in_words(text)
    data["notes"] = data["amount_in_words"]

    return data


def build_rule_candidate_pack(extracted_text: str) -> Dict[str, Any]:
    """Build bounded OCR candidates before asking a model to resolve fields.

    Each candidate comes from a labelled regex/dictionary rule or an article
    table row that passed arithmetic and location checks. The LLM sees this
    package but cannot invent a new monetary value or product row.
    """
    fields = extract_invoice_fields_fallback(extracted_text)
    table_text = extract_invoice_table_text(extracted_text)
    accepted_items, rejected_items = filter_invoice_line_items(
        fields.get("items") or [],
        table_text,
        fields.get("total_amount"),
    )
    fields = dict(fields)
    fields["items"] = accepted_items
    field_values = {
        field_name: value
        for field_name, value in fields.items()
        if field_has_value(field_name, value)
    }
    return {
        "fields": fields,
        "llm_candidates": {
            "field_values": field_values,
            "article_rows": accepted_items,
            "article_table_detected": bool(table_text),
            "rejected_article_rows": rejected_items,
        },
        "items": accepted_items,
        "rejected_items": rejected_items,
    }

async def save_upload_file(upload_file: UploadFile) -> str:
    """Sauvegarde le fichier téléchargé et retourne le chemin"""
    timestamp = str(int(time.time()))
    original_name = Path(upload_file.filename or "upload.bin").name
    filename = f"{timestamp}_{original_name}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await upload_file.read()
        await f.write(content)
    
    return file_path


def detect_document_objects(file_path: str, confidence_threshold: float = 0.25) -> List[Dict[str, Any]]:
    """Detecte les objets/zones dans une image avec YOLO.

    Les PDF sont geres par OCR/PDF et RAG; la detection objet est limitee aux
    images car le modele YOLO courant attend une image.
    """
    if is_pdf_file(file_path) or model is None:
        return []

    detections: List[Dict[str, Any]] = []
    try:
        results = model(file_path, conf=confidence_threshold)
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0]) if box.cls is not None else -1
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                xyxy = box.xyxy[0].tolist() if box.xyxy is not None else []

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": classes.get(class_id, str(class_id)) if classes else str(class_id),
                        "confidence": round(confidence, 4),
                        "bbox": [round(float(value), 2) for value in xyxy],
                    }
                )
    except Exception as error:
        print(f"Detection objet ignoree: {error}")

    return detections


def analyze_invoice_fields(
    extracted_text: str,
    model_choice: str,
    field_evidence: Optional[Dict[str, Any]] = None,
    rule_candidates: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[InvoiceData], str, Dict[str, Any]]:
    """Analyse les champs de facture avec le modele choisi puis fallback regex."""
    invoice_data = None
    model_used = "fallback_regex"
    raw_data: Dict[str, Any] = {}

    try:
        if model_choice == "local":
            invoice_data = ollama_analyzer.analyze_invoice_text(
                extracted_text,
                field_evidence=field_evidence,
                rule_candidates=(rule_candidates or {}).get("llm_candidates"),
            )
            model_used = f"ollama:{ollama_analyzer.model_name}" if invoice_data else "fallback_regex"
        elif model_choice == "gemma":
            invoice_data = get_gemma_analyzer().analyze_document(extracted_text)
            model_used = "gemma"
        elif model_choice == "fallback":
            model_used = "fallback_regex"
        else:
            invoice_data = ai_analyzer.analyze_invoice_text(
                extracted_text,
                field_evidence=field_evidence,
                rule_candidates=(rule_candidates or {}).get("llm_candidates"),
            )
            model_used = "gemini" if ai_analyzer.model else "fallback_regex"
    except Exception as analysis_error:
        print(f"Analyse IA indisponible, fallback regex: {analysis_error}")
        raw_data["analysis_error"] = str(analysis_error)

    return invoice_data, model_used, raw_data


def add_agent_decision(
    decisions: List[Dict[str, Any]],
    step: str,
    decision: str,
    reason: str,
    status: str = "completed",
    confidence: Optional[float] = None,
    action: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a compact, auditable decision made by the document agent."""
    payload: Dict[str, Any] = {
        "step": step,
        "decision": decision,
        "reason": reason,
        "status": status,
    }
    if confidence is not None:
        payload["confidence"] = round(float(confidence), 4)
    if action:
        payload["action"] = action
    if metadata:
        payload["metadata"] = metadata
    decisions.append(payload)


def missing_required_invoice_fields(normalized_fields: Dict[str, Any]) -> List[str]:
    required_fields = ["invoice_number", "date", "vendor_name", "total_amount", "currency"]
    return [
        field_name
        for field_name in required_fields
        if not field_has_value(field_name, normalized_fields.get(field_name))
    ]


AGENT_POLICY = {
    "minimum_text_length": 80,
    "low_ocr_confidence": 0.45,
    "uncertain_classification": 0.70,
    "strong_classification": 0.85,
    "low_invoice_confidence": 0.65,
    "math_tolerance_tnd": 0.050,
    "required_invoice_fields": ["invoice_number", "date", "vendor_name", "total_amount", "currency"],
}


def add_agent_action(
    actions: List[Dict[str, Any]],
    name: str,
    trigger: str,
    outcome: str,
    status: str = "executed",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "name": name,
        "trigger": trigger,
        "outcome": outcome,
        "status": status,
    }
    if metadata:
        payload["metadata"] = metadata
    actions.append(payload)


def choose_agent_model(
    requested_model: str,
    extracted_text: str,
    extraction_confidence: float,
    classification_confidence: float,
) -> tuple[str, str]:
    """Route LLM extraction without forcing cloud use for every document."""
    normalized_choice = (requested_model or "local").strip().lower()
    if normalized_choice != "auto":
        return normalized_choice, "model_explicitly_requested"

    text_length = len(extracted_text or "")
    if text_length < AGENT_POLICY["minimum_text_length"]:
        return "fallback", "text_too_short_for_llm_extraction"

    difficult_document = (
        extraction_confidence < AGENT_POLICY["low_ocr_confidence"]
        or classification_confidence < AGENT_POLICY["uncertain_classification"]
        or text_length > 6000
    )
    if difficult_document and getattr(ai_analyzer, "model", None):
        return "gemini", "complex_or_uncertain_document_cloud_llm"

    return "local", "standard_document_local_llm"


def validate_normalized_invoice_fields(
    normalized_fields: Dict[str, Any],
    field_proofs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run deterministic business checks after extraction and normalization."""
    issues: List[Dict[str, Any]] = []

    for field_name in AGENT_POLICY["required_invoice_fields"]:
        if not field_has_value(field_name, normalized_fields.get(field_name)):
            issues.append(
                {
                    "code": f"missing_{field_name}",
                    "severity": "blocking",
                    "message": f"Required field '{field_name}' is missing.",
                }
            )

    for field_name in AGENT_POLICY["required_invoice_fields"]:
        proof = (field_proofs or {}).get(field_name, {})
        if field_has_value(field_name, normalized_fields.get(field_name)) and not proof.get("supported"):
            issues.append(
                {
                    "code": f"unproven_{field_name}",
                    "severity": "blocking",
                    "message": f"Required field '{field_name}' has no OCR evidence and must be reviewed.",
                }
            )

    currency = str(normalized_fields.get("currency") or "").strip().upper()
    if currency and currency != "TND":
        issues.append(
            {
                "code": "currency_not_tnd",
                "severity": "warning",
                "message": f"Currency is '{currency}', expected TND for the Tunisian workflow.",
            }
        )

    invoice_date = str(normalized_fields.get("date") or "").strip()
    if invoice_date and not re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", invoice_date):
        issues.append(
            {
                "code": "invoice_date_format_uncertain",
                "severity": "warning",
                "message": "Invoice date is present but not in a recognized dd/mm/yyyy-like format.",
            }
        )

    subtotal = safe_float(normalized_fields.get("subtotal"))
    tax_amount = safe_float(normalized_fields.get("tax_amount"))
    stamp_duty = safe_float(normalized_fields.get("stamp_duty"))
    total_amount = safe_float(normalized_fields.get("total_amount"))
    items = list(normalized_fields.get("items") or [])
    if total_amount > 0 and subtotal <= 0:
        issues.append(
            {
                "code": "missing_subtotal_for_total",
                "severity": "blocking",
                "message": "TTC was found, but no OCR-proven HT amount was found. Do not infer it from TTC.",
            }
        )
    if subtotal > 0 and tax_amount <= 0:
        issues.append(
            {
                "code": "tax_amount_not_detected",
                "severity": "warning",
                "message": "HT is available but no OCR-proven TVA amount was found.",
            }
        )
    if total_amount > 0 and (field_proofs or {}).get("items", {}).get("table_detected") and not items:
        issues.append(
            {
                "code": "no_verified_line_items",
                "severity": "blocking",
                "message": "An article table was detected but no row passed the OCR structure and arithmetic rules.",
            }
        )
    if items and total_amount > 0:
        items_total = sum(safe_float(item.get("total")) for item in items)
        if items_total > total_amount + AGENT_POLICY["math_tolerance_tnd"]:
            issues.append(
                {
                    "code": "line_items_exceed_total",
                    "severity": "blocking",
                    "message": "The verified item lines exceed TTC; the document requires human review.",
                    "metadata": {"items_total": round(items_total, 3), "total_amount": total_amount},
                }
            )
    if subtotal > 0 and total_amount > 0:
        expected_total = subtotal + tax_amount + stamp_duty
        delta = abs(expected_total - total_amount)
        if tax_amount > 0 and delta > AGENT_POLICY["math_tolerance_tnd"]:
            issues.append(
                {
                    "code": "invoice_total_math_mismatch",
                    "severity": "blocking",
                    "message": "HT + TVA + timbre does not match TTC within the configured TND tolerance.",
                    "metadata": {
                        "subtotal": subtotal,
                        "tax_amount": tax_amount,
                        "stamp_duty": stamp_duty,
                        "expected_total": round(expected_total, 3),
                        "total_amount": total_amount,
                        "delta": round(delta, 3),
                    },
                }
            )

    return issues


def choose_workflow_destination(
    is_invoice: bool,
    classification_confidence: float,
    extraction_confidence: float,
    validation_issues: List[Dict[str, Any]],
) -> str:
    if not is_invoice:
        return "non_invoice_review" if classification_confidence < AGENT_POLICY["strong_classification"] else "non_invoice_archive"
    if extraction_confidence < AGENT_POLICY["low_ocr_confidence"]:
        return "ocr_quality_review"
    if any(issue.get("severity") == "blocking" for issue in validation_issues):
        return "human_validation_required"
    return "ready_for_human_validation"


def build_agent_recommendation(
    is_invoice: bool,
    classification_confidence: float,
    extraction_confidence: float,
    normalized_fields: Dict[str, Any],
    rag_document: Optional[Dict[str, Any]],
    validation_issues: Optional[List[Dict[str, Any]]] = None,
    workflow_destination: Optional[str] = None,
) -> Dict[str, Any]:
    """Choose the next best action after the pipeline has run."""
    validation_issues = validation_issues or []
    if not is_invoice:
        return {
            "recommended_action": "confirm_non_invoice",
            "human_review_required": classification_confidence < 0.85,
            "reasons": ["document_classified_as_non_invoice"],
            "workflow_destination": workflow_destination or "non_invoice_review",
            "validation_issues": validation_issues,
        }

    reasons: List[str] = []
    if classification_confidence < 0.7:
        reasons.append("classification_confidence_low")
    if extraction_confidence < 0.45:
        reasons.append("ocr_confidence_low")

    missing_fields = missing_required_invoice_fields(normalized_fields)
    reasons.extend([f"missing_{field_name}" for field_name in missing_fields])
    for issue in validation_issues:
        issue_code = issue.get("code")
        if issue.get("severity") == "blocking" and issue_code and issue_code not in reasons:
            reasons.append(issue_code)

    if rag_document is None:
        reasons.append("rag_document_not_indexed")

    if reasons:
        return {
            "recommended_action": "send_to_human_review",
            "human_review_required": True,
            "reasons": reasons,
            "missing_fields": missing_fields,
            "workflow_destination": workflow_destination or "human_validation_required",
            "validation_issues": validation_issues,
        }

    return {
        "recommended_action": "ready_for_validation",
        "human_review_required": False,
        "reasons": ["classification_ocr_fields_and_rag_ok"],
        "missing_fields": [],
        "workflow_destination": workflow_destination or "ready_for_human_validation",
        "validation_issues": validation_issues,
    }


async def run_document_agent(
    file: UploadFile,
    model_choice: str = "local",
    index_for_rag: bool = True,
    detect_objects: bool = True,
) -> Dict[str, Any]:
    """Pipeline agentique: runs the document workflow and records decisions."""
    start_time = time.time()
    file_path = None
    agent_decisions: List[Dict[str, Any]] = []
    agent_actions: List[Dict[str, Any]] = []

    try:
        if not validate_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporte. Extensions autorisees: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Fichier trop volumineux. Taille maximale: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB",
            )

        await file.seek(0)
        file_path = await save_upload_file(file)
        filename = file.filename or Path(file_path).name
        file_type = Path(file_path).suffix.lower().lstrip(".")
        add_agent_decision(
            agent_decisions,
            step="upload",
            decision="accept_file",
            reason="Extension and file size are valid.",
            metadata={"filename": filename, "file_type": file_type, "size_bytes": len(content)},
        )

        from starlette.concurrency import run_in_threadpool
        extracted_text, extraction_confidence, extraction_metadata = await run_in_threadpool(extract_document_text, file_path)
        text_strategy = extraction_metadata.get("method_used", "unknown")
        add_agent_decision(
            agent_decisions,
            step="ocr_or_pdf_text",
            decision="extract_text",
            reason=f"Used {text_strategy} to obtain document text.",
            confidence=extraction_confidence,
            action="continue" if extraction_confidence >= 0.45 or extracted_text.strip() else "human_review_recommended",
            metadata={
                "method_used": text_strategy,
                "text_length": len(extracted_text or ""),
                "fallback_used": bool(extraction_metadata.get("fallback_error")),
            },
        )
        if len(extracted_text or "") < AGENT_POLICY["minimum_text_length"]:
            add_agent_action(
                agent_actions,
                name="text_quality_gate",
                trigger="text_length_below_minimum",
                outcome="route_document_to_human_review_if_no_better_signal",
                status="needs_attention",
                metadata={
                    "text_length": len(extracted_text or ""),
                    "minimum_text_length": AGENT_POLICY["minimum_text_length"],
                },
            )
        elif extraction_confidence < AGENT_POLICY["low_ocr_confidence"]:
            add_agent_action(
                agent_actions,
                name="ocr_confidence_gate",
                trigger="ocr_confidence_below_threshold",
                outcome="keep_processing_but_require_human_validation",
                status="needs_attention",
                metadata={
                    "confidence": extraction_confidence,
                    "threshold": AGENT_POLICY["low_ocr_confidence"],
                },
            )
        else:
            add_agent_action(
                agent_actions,
                name="text_quality_gate",
                trigger="text_available",
                outcome="continue_to_classification",
                metadata={"text_length": len(extracted_text or ""), "confidence": extraction_confidence},
            )

        is_invoice, classification_confidence, classification_details = classify_invoice_text(extracted_text)
        document_type = "invoice" if is_invoice else "other"
        add_agent_decision(
            agent_decisions,
            step="classification",
            decision="invoice" if is_invoice else "non_invoice",
            reason="Keyword and document-structure signals were evaluated.",
            confidence=classification_confidence,
            action="continue_to_invoice_extraction" if is_invoice else "store_as_non_invoice",
            metadata=classification_details,
        )

        if detect_objects and model is None:
            detections = []
            add_agent_decision(
                agent_decisions,
                step="yolo_visual_detection",
                decision="skip_yolo_unavailable",
                reason="Object detection is not available on this server.",
                action="continue",
            )
        elif detect_objects and not is_pdf_file(file_path):
            detections = detect_document_objects(file_path)
            add_agent_decision(
                agent_decisions,
                step="yolo_visual_detection",
                decision="run_yolo",
                reason="Input is an image; object detection can locate visual invoice evidence.",
                action="continue",
                metadata={"detections_count": len(detections)},
            )
        elif detect_objects:
            detections = []
            add_agent_decision(
                agent_decisions,
                step="yolo_visual_detection",
                decision="skip_yolo_for_pdf",
                reason="Current YOLO model expects image input; PDF is handled by text extraction and RAG.",
                action="continue",
            )
        else:
            detections = []
            add_agent_decision(
                agent_decisions,
                step="yolo_visual_detection",
                decision="skip_yolo_disabled",
                reason="Object detection was disabled by request.",
                action="continue",
            )

        invoice_record = None
        normalized_fields: Dict[str, Any] = {}
        model_used = "none"
        analysis_raw: Dict[str, Any] = {}
        field_evidence: Dict[str, Any] = {}
        field_proofs: Dict[str, Any] = {}
        rule_candidates: Dict[str, Any] = {}
        validation_issues: List[Dict[str, Any]] = []
        selected_model_choice = model_choice or "local"

        if is_invoice and extracted_text.strip():
            field_evidence = get_invoice_field_evidence(extracted_text)
            add_agent_decision(
                agent_decisions,
                step="rag_extraction",
                decision="retrieve_field_evidence",
                reason="Temporary chunks were ranked per target field before structured extraction.",
                action="continue_to_field_extraction",
                metadata={
                    "fields_with_evidence": [
                        field_name
                        for field_name, chunks in field_evidence.items()
                        if chunks
                    ],
                },
            )
            rule_candidates = build_rule_candidate_pack(extracted_text)
            add_agent_decision(
                agent_decisions,
                step="deterministic_candidates",
                decision="build_regex_dictionary_and_table_candidates",
                reason="Regex, label dictionaries and article-table arithmetic produced a bounded candidate set before the LLM call.",
                action="continue_to_llm_decision",
                metadata={
                    "candidate_fields": sorted(rule_candidates.get("llm_candidates", {}).get("field_values", {}).keys()),
                    "verified_article_rows": len(rule_candidates.get("items") or []),
                    "rejected_article_rows": rule_candidates.get("rejected_items") or [],
                },
            )

            selected_model_choice, model_route_reason = choose_agent_model(
                requested_model=model_choice or "local",
                extracted_text=extracted_text,
                extraction_confidence=extraction_confidence,
                classification_confidence=classification_confidence,
            )
            add_agent_decision(
                agent_decisions,
                step="model_routing",
                decision=f"use_{selected_model_choice}",
                reason=model_route_reason,
                action="run_structured_extraction",
                metadata={
                    "requested_model": model_choice,
                    "selected_model": selected_model_choice,
                    "auto_routing": (model_choice or "").lower() == "auto",
                },
            )
            add_agent_action(
                agent_actions,
                name="llm_model_routing",
                trigger=model_route_reason,
                outcome=f"selected_{selected_model_choice}",
                metadata={"requested_model": model_choice, "selected_model": selected_model_choice},
            )

            invoice_data, model_used, analysis_raw = await run_in_threadpool(
                analyze_invoice_fields, extracted_text, selected_model_choice, field_evidence, rule_candidates
            )
            add_agent_decision(
                agent_decisions,
                step="field_extraction",
                decision=f"use_{model_used}",
                reason="Structured invoice fields were extracted with the selected model and deterministic fallback.",
                action="normalize_fields",
                metadata={"requested_model": model_choice, "selected_model": selected_model_choice, "model_used": model_used},
            )

            normalized_fields = normalize_invoice_data(invoice_data, extracted_text, rule_candidates)
            normalized_fields, field_proofs = enforce_evidence_guard(
                normalized_fields, extracted_text, field_evidence, rule_candidates
            )
            missing_fields = missing_required_invoice_fields(normalized_fields)
            validation_issues = validate_normalized_invoice_fields(normalized_fields, field_proofs)
            blocking_validation_issues = [
                issue for issue in validation_issues if issue.get("severity") == "blocking"
            ]
            add_agent_decision(
                agent_decisions,
                step="normalization",
                decision="normalize_invoice_fields",
                reason="Amounts, currency, dates and line items were normalized for validation.",
                action="continue" if not blocking_validation_issues else "human_review_recommended",
                metadata={
                    "missing_required_fields": missing_fields,
                    "field_proofs": field_proofs,
                    "validation_issues": validation_issues,
                },
            )
            add_agent_action(
                agent_actions,
                name="business_rules_validation",
                trigger="normalized_fields_available",
                outcome="issues_found" if validation_issues else "passed",
                status="needs_attention" if any(issue.get("severity") == "blocking" for issue in validation_issues) else "executed",
                metadata={"issues": validation_issues},
            )

            invoice_confidence = extraction_confidence
            if invoice_data and invoice_data.confidence_score:
                invoice_confidence = (extraction_confidence + invoice_data.confidence_score) / 2

            invoice_record = invoice_store.create_invoice(
                filename=filename,
                file_type=file_type,
                source_file_path=file_path,
                extracted_text=extracted_text,
                normalized_data=normalized_fields,
                raw_data={
                    "classification": classification_details,
                    "extraction_metadata": extraction_metadata,
                    "object_detections": detections,
                    "field_evidence": field_evidence,
                    "field_proofs": field_proofs,
                    "rule_candidates": rule_candidates.get("llm_candidates", {}),
                    "rejected_article_rows": rule_candidates.get("rejected_items", []),
                    "agent_decisions": agent_decisions,
                    "agent_actions": agent_actions,
                    "agent_policy": AGENT_POLICY,
                    "validation_issues": validation_issues,
                    **analysis_raw,
                },
                model_used=model_used,
                confidence_score=invoice_confidence,
            )
            add_agent_decision(
                agent_decisions,
                step="draft_invoice",
                decision="create_invoice_draft",
                reason="The document is an invoice and structured fields are available for human validation.",
                action="continue_to_rag_document_indexing",
                confidence=invoice_confidence,
                metadata={"invoice_id": invoice_record["id"] if invoice_record else None},
            )
            duplicate_detection = (invoice_record or {}).get("duplicate_detection")
            if duplicate_detection:
                is_duplicate = bool(duplicate_detection.get("is_potential_duplicate"))
                add_agent_decision(
                    agent_decisions,
                    step="duplicate_detection",
                    decision="potential_duplicate_found" if is_duplicate else "no_duplicate_found",
                    reason="Stored invoices were compared after draft creation.",
                    status="needs_attention" if is_duplicate else "completed",
                    action="human_review_required" if is_duplicate else "continue_to_rag_document_indexing",
                metadata=duplicate_detection,
                )
                add_agent_action(
                    agent_actions,
                    name="duplicate_detection",
                    trigger="invoice_draft_created",
                    outcome="duplicate_candidates_found" if is_duplicate else "no_duplicate_candidates",
                    status="needs_attention" if is_duplicate else "executed",
                    metadata=duplicate_detection,
                )
            else:
                add_agent_decision(
                    agent_decisions,
                    step="duplicate_detection",
                    decision="not_available",
                    reason="The current invoice store did not return duplicate detection metadata.",
                    status="skipped",
                    action="continue_to_rag_document_indexing",
                )
                add_agent_action(
                    agent_actions,
                    name="duplicate_detection",
                    trigger="invoice_draft_created",
                    outcome="not_available_in_current_store",
                    status="skipped",
                )
        elif is_invoice:
            add_agent_decision(
                agent_decisions,
                step="field_extraction",
                decision="skip_no_text",
                reason="The document was classified as an invoice but no usable text was extracted.",
                status="needs_attention",
                action="human_review_required",
            )

        rag_document = None
        if index_for_rag and extracted_text.strip():
            try:
                rag_document = rag_service.index_document(
                    filename=filename,
                    file_type=file_type,
                    source_file_path=file_path,
                    extracted_text=extracted_text,
                    extraction_metadata={
                        "confidence": extraction_confidence,
                        "metadata": extraction_metadata,
                        "classification": classification_details,
                        "document_type": document_type,
                        "invoice_id": invoice_record["id"] if invoice_record else None,
                        "field_evidence": field_evidence,
                        "field_proofs": field_proofs,
                        "agent_decisions": agent_decisions,
                        "agent_actions": agent_actions,
                    },
                )
                add_agent_decision(
                    agent_decisions,
                    step="rag_documentary_indexing",
                    decision="index_document",
                    reason="Persistent RAG index was created for assistant search after extraction and normalization.",
                    action="continue_to_recommendation",
                    metadata={
                        "document_id": rag_document.get("document_id"),
                        "chunks_count": rag_document.get("chunks_count"),
                        "embedding_backend": rag_document.get("embedding_backend"),
                    },
                )
            except Exception as rag_error:
                print(f"Indexation RAG ignoree: {rag_error}")
                add_agent_decision(
                    agent_decisions,
                    step="rag_documentary_indexing",
                    decision="indexing_failed",
                    reason=str(rag_error),
                    status="needs_attention",
                    action="human_review_recommended",
                )
        elif not index_for_rag:
            add_agent_decision(
                agent_decisions,
                step="rag_documentary_indexing",
                decision="skip_disabled",
                reason="Persistent RAG indexing was disabled by request.",
                action="continue_to_recommendation",
            )

        workflow_destination = choose_workflow_destination(
            is_invoice=is_invoice,
            classification_confidence=classification_confidence,
            extraction_confidence=extraction_confidence,
            validation_issues=validation_issues,
        )
        add_agent_action(
            agent_actions,
            name="workflow_routing",
            trigger="pipeline_completed",
            outcome=workflow_destination,
            status="needs_attention" if "review" in workflow_destination or "required" in workflow_destination else "executed",
            metadata={"destination": workflow_destination},
        )

        recommendation = build_agent_recommendation(
            is_invoice=is_invoice,
            classification_confidence=classification_confidence,
            extraction_confidence=extraction_confidence,
            normalized_fields=normalized_fields,
            rag_document=rag_document,
            validation_issues=validation_issues,
            workflow_destination=workflow_destination,
        )
        add_agent_decision(
            agent_decisions,
            step="recommendation",
            decision=recommendation["recommended_action"],
            reason=", ".join(recommendation.get("reasons", [])),
            status="needs_attention" if recommendation["human_review_required"] else "completed",
            action=recommendation["recommended_action"],
            metadata=recommendation,
        )

        document_record = document_store.create_document(
            filename=filename,
            file_type=file_type,
            source_file_path=file_path,
            document_type=document_type,
            is_invoice=is_invoice,
            classification_confidence=classification_confidence,
            extraction_confidence=extraction_confidence,
            extracted_text=extracted_text,
            fields=normalized_fields,
            detections=detections,
            rag_document=rag_document,
            invoice_id=invoice_record["id"] if invoice_record else None,
            metadata={
                "model_used": model_used,
                "classification": classification_details,
                "extraction": extraction_metadata,
                "analysis": analysis_raw,
                "field_evidence": field_evidence,
                "agent_decisions": agent_decisions,
                "agent_actions": agent_actions,
                "agent_policy": AGENT_POLICY,
                "agent_recommendation": recommendation,
                "workflow_destination": workflow_destination,
                "validation_issues": validation_issues,
            },
            status="draft" if is_invoice else "classified",
        )

        return {
            "success": True,
            "message": "Document traite par l'agent IA documentaire",
            "processing_time": round(time.time() - start_time, 3),
            "document": document_record,
            "classification": {
                "document_type": document_type,
                "is_invoice": is_invoice,
                "confidence": classification_confidence,
                "details": classification_details,
            },
            "invoice": invoice_record,
            "rag_document": rag_document,
            "object_detections": detections,
            "agent": {
                "mode": "hybrid_document_ai_agent",
                "policy": AGENT_POLICY,
                "decisions": agent_decisions,
                "actions": agent_actions,
                "recommendation": recommendation,
                "workflow_destination": workflow_destination,
            },
        }

    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as error:
                print(f"Erreur suppression fichier temporaire: {error}")

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "Optimized Invoice OCR API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "OCR optimisé avec préprocessing avancé",
            "Analyse IA avec Gemini AI",
            "Analyse IA avec google/gemma-3-4b-it",
            "Extraction PDF native avec pdfplumber et fallback OCR",
            "Vérification de facture avec PaddleOCR layout analysis",
            "Extraction d'entités avec modèle YOLO personnalisé",
            "Support documents tunisiens",
            "JSON de réponse structuré"
        ],
        "endpoints": {
            "upload": "/extract-invoice",
            "upload_gemma": "/extract-invoice-gemma",
            "verify_invoice": "/verify-invoice",
            "extract_entities": "/extract-entities",
            "extract_entities_with_ocr": "/extract-entities-with-ocr",
            "model_info": "/model-info",
            "text_only": "/extract-text-only",
            "agent_process_document": "/agent/process-document",
            "agent_documents": "/agent/documents",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/auth/signup")
async def signup(request: SignupRequest):
    email = normalize_email(request.email)
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Email professionnel invalide")

    salt, password_hash = create_password_hash(request.password)
    try:
        with auth_connection() as db:
            cursor = db.execute(
                """
                INSERT INTO users (email, password_hash, salt, full_name, company, phone, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    email,
                    password_hash,
                    salt,
                    request.full_name.strip(),
                    request.company.strip(),
                    request.phone.strip(),
                    "Agent de saisie",
                    time.time(),
                ),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Un compte existe déjà avec cet email")

    token = create_session(int(user_id))
    user = get_user_from_token(token)
    return {"success": True, "token": token, "user": user}


@app.post("/auth/login")
async def login(request: LoginRequest):
    email = normalize_email(request.email)
    with auth_connection() as db:
        row = db.execute(
            "SELECT id, password_hash, salt FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    if not row or not secrets.compare_digest(row[1], hash_password(request.password, row[2])):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")

    token = create_session(int(row[0]))
    user = get_user_from_token(token)
    return {"success": True, "token": token, "user": user}


@app.get("/auth/me")
async def auth_me(token: str = "", authorization: Optional[str] = Header(None)):
    user = get_user_from_token(token) or user_from_authorization(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Session expirée ou invalide")
    return {"success": True, "user": user}


@app.post("/auth/logout")
async def logout(token: str = "", authorization: Optional[str] = Header(None)):
    token = token or (authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else "")
    with auth_connection() as db:
        db.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))
    return {"success": True}


@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "ocr": "optimized_tesseract",
            "pdf_extractor": "pdfplumber_with_ocr_fallback",
            "ai_analyzer": "available" if ai_analyzer.model else "fallback",
            "ollama_analyzer": "available" if ollama_analyzer.is_available() else "unavailable",
            "ollama_model": ollama_analyzer.model_name,
            "gemma_analyzer": "disabled" if not ENABLE_GEMMA else ("available" if gemma_analyzer and gemma_analyzer.model else "lazy"),
            "paddle_pipeline": "available" if paddle_pipeline is not None else "unavailable",
            "yolo_model": "available" if model else "unavailable",
            "database": ACTIVE_DATABASE_BACKEND,
            "auth_store": "postgresql" if DATABASE_URL else "sqlite",
            "invoice_store": "postgresql_jsonb" if ACTIVE_DATABASE_BACKEND == "postgresql" else "sqlite_json",
            "document_store": "postgresql_jsonb" if ACTIVE_DATABASE_BACKEND == "postgresql" else "sqlite_json",
            "rag_store": "postgresql_jsonb_embeddings" if ACTIVE_DATABASE_BACKEND == "postgresql" else "sqlite_json_embeddings",
            "file_store": "postgresql_bytea" if ACTIVE_DATABASE_BACKEND == "postgresql" else "local_disk",
            "upload_folder": "accessible" if os.path.exists(UPLOAD_FOLDER) else "error"
        },
        "version": "2.0.0"
    }


@app.post("/dataset/fatura/import")
async def import_fatura_dataset():
    """Indexe FATURA.zip dans une base SQLite locale."""
    try:
        return {
            "success": True,
            "dataset": fatura_dataset_store.import_from_zip(),
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/dataset/fatura/stats")
async def fatura_dataset_stats(auto_import: bool = False):
    """Statistiques de la base SQLite FATURA dataset."""
    try:
        if auto_import and fatura_dataset_store.needs_import():
            fatura_dataset_store.import_from_zip()
        return {
            "success": True,
            "dataset": fatura_dataset_store.stats(),
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/dataset/fatura/samples")
async def fatura_dataset_samples(split: Optional[str] = None, limit: int = 12):
    """Exemples du dataset FATURA indexe."""
    try:
        return {
            "success": True,
            "samples": fatura_dataset_store.samples(split=split, limit=limit),
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/dataset/fatura/annotation")
async def fatura_dataset_annotation(annotation_path: str):
    """Recupere une annotation JSON indexee depuis SQLite."""
    try:
        return {
            "success": True,
            "annotation": fatura_dataset_store.annotation(annotation_path),
        }
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

@app.post("/extract-invoice", response_model=OCRResponse)
async def extract_invoice_data(
    file: UploadFile = File(...),
    include_raw_text: Optional[bool] = True,
    return_gemini_json: Optional[bool] = True
):
    """
    Extrait et analyse les données d'une facture avec l'OCR et l'IA optimisés
    
    Args:
        file: Fichier image de la facture
        include_raw_text: Inclure le texte brut dans la réponse
        return_gemini_json: Inclure le JSON Gemini structuré dans la réponse
    
    Returns:
        OCRResponse avec les données extraites et structurées
    """
    start_time = time.time()
    file_path = None
    
    try:
        # Validation du fichier
        if not validate_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporté. Extensions autorisées: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Vérifier la taille du fichier
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Fichier trop volumineux. Taille maximale: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        # Remettre le pointeur au début pour la sauvegarde
        await file.seek(0)
        
        # Sauvegarder le fichier
        file_path = await save_upload_file(file)
        
        # Extraction texte optimisée (image OCR ou PDF natif/OCR)
        print(f"🔍 Début extraction texte optimisée...")
        extracted_text, ocr_confidence, ocr_metadata = extract_document_text(file_path)
        
        if not extracted_text.strip():
            return OCRResponse(
                success=False,
                message="Aucun texte détecté dans l'image",
                processing_time=time.time() - start_time,
                errors=["Aucun texte extrait par OCR"]
            )
        
        print(f"✅ Texte extrait: {len(extracted_text)} caractères (confiance: {ocr_confidence:.3f})")
        
        # Analyse IA optimisée
        print("🤖 Début analyse IA optimisée...")
        invoice_data = ai_analyzer.analyze_invoice_text(extracted_text)
        
        # Ajuster le score de confiance global
        if invoice_data.confidence_score:
            combined_confidence = (ocr_confidence + invoice_data.confidence_score) / 2
            invoice_data.confidence_score = combined_confidence
        else:
            invoice_data.confidence_score = ocr_confidence
        
        # Ajouter les métadonnées OCR
        invoice_data.additional_data.update({
            'ocr_metadata': ocr_metadata,
            'ocr_confidence': ocr_confidence,
            'processing_method': 'optimized'
        })
        
        processing_time = time.time() - start_time
        print(f"✅ Traitement terminé en {processing_time:.2f}s")
        
        # Construire la réponse
        response_data = {
            'success': True,
            'message': 'Extraction et analyse terminées avec succès',
            'processing_time': processing_time,
            'invoice_data': invoice_data
        }
        
        # Ajouter le texte brut si demandé
        if include_raw_text:
            response_data['extracted_text'] = extracted_text
        
        # Ajouter le JSON Gemini si disponible et demandé
        if return_gemini_json and 'gemini_result' in invoice_data.additional_data:
            response_data['gemini_structured_data'] = invoice_data.additional_data['gemini_result']
        
        return OCRResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return OCRResponse(
            success=False,
            message="Erreur lors du traitement du fichier",
            processing_time=time.time() - start_time,
            errors=[str(e)]
        )
    
    finally:
        # Nettoyer le fichier temporaire
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"⚠️ Erreur suppression fichier temporaire: {e}")

#     file: UploadFile = File(...),
#     include_raw_text: Optional[bool] = True,
# ):
#     """
#     
#     tout type de document (facture, reçu, attestation, etc.) et retourne un JSON structuré.
#     
#     Args:
#         file: Fichier image du document
#         include_raw_text: Inclure le texte brut OCR dans la réponse
#     
#     Returns:
#     """
#     start_time = time.time()
#     file_path = None
#     
#     try:
#         # Validation du fichier
#         if not validate_file(file):
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Type de fichier non supporté. Extensions autorisées: {', '.join(ALLOWED_EXTENSIONS)}"
#             )
#         
#         # Vérifier la taille du fichier
#         content = await file.read()
#         if len(content) > MAX_FILE_SIZE:
#             raise HTTPException(
#                 status_code=413,
#                 detail=f"Fichier trop volumineux. Taille maximale: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
#             )
#         
#         # Remettre le pointeur au début pour la sauvegarde
#         await file.seek(0)
#         
#         # Sauvegarder le fichier temporairement
#         file_path = await save_upload_file(file)
#         
#         # Extraction OCR
#         extracted_text, ocr_confidence, ocr_metadata = ocr_extractor.extract_text(file_path)
#         
#         if not extracted_text.strip():
#             return OCRResponse(
#                 success=False,
#                 message="Aucun texte détecté dans l'image",
#                 processing_time=time.time() - start_time,
#                 errors=["Aucun texte extrait par OCR"]
#             )
#         
#         print(f"✅ Texte extrait: {len(extracted_text)} caractères (confiance: {ocr_confidence:.3f})")
#         
#         
#         # Ajuster le score de confiance global
#         if invoice_data.confidence_score:
#             combined_confidence = (ocr_confidence + invoice_data.confidence_score) / 2
#             invoice_data.confidence_score = combined_confidence
#         else:
#             invoice_data.confidence_score = ocr_confidence
#         
#         # Ajouter les métadonnées OCR
#         invoice_data.additional_data.update({
#             'ocr_metadata': ocr_metadata,
#             'ocr_confidence': ocr_confidence,
#         })
#         
#         processing_time = time.time() - start_time
#         
#         # Construire la réponse
#         response_data = {
#             'success': True,
#             'processing_time': processing_time,
#             'invoice_data': invoice_data
#         }
#         
#         # Ajouter le texte brut si demandé
#         if include_raw_text:
#             response_data['extracted_text'] = extracted_text
#         
#         
#         return OCRResponse(**response_data)
#         
#     except HTTPException:
#         raise
#     except Exception as e:
#         return OCRResponse(
#             success=False,
#             processing_time=time.time() - start_time,
#             errors=[str(e)]
#         )
#     
#     finally:
#         # Nettoyer le fichier temporaire
#         if file_path and os.path.exists(file_path):
#             try:
#                 os.remove(file_path)
#             except Exception as e:
#                 print(f"⚠️ Erreur suppression fichier temporaire: {e}")

@app.post("/extract-invoice-gemma", response_model=OCRResponse)
async def extract_invoice_with_gemma(
    file: UploadFile = File(...),
    include_raw_text: Optional[bool] = True,
    include_gemma_result: Optional[bool] = True
):
    """
    Extrait et analyse les données d'un document avec OCR + google/gemma-3-4b-it
    
    Utilise le modèle google/gemma-3-4b-it via Hugging Face Transformers pour analyser 
    tout type de document (facture, reçu, attestation, etc.) et retourne un JSON structuré.
    
    Args:
        file: Fichier image du document
        include_raw_text: Inclure le texte brut OCR dans la réponse
        include_gemma_result: Inclure le résultat brut de Gemma dans la réponse
    
    Returns:
        OCRResponse avec les données extraites et analysées par Gemma
    """
    start_time = time.time()
    file_path = None
    
    try:
        # Validation du fichier
        if not validate_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporté. Extensions autorisées: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Vérifier la taille du fichier
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Fichier trop volumineux. Taille maximale: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        # Remettre le pointeur au début pour la sauvegarde
        await file.seek(0)
        
        # Sauvegarder le fichier temporairement
        file_path = await save_upload_file(file)
        
        # Extraction OCR
        print("🔍 Début extraction texte pour Gemma...")
        extracted_text, ocr_confidence, ocr_metadata = extract_document_text(file_path)
        
        if not extracted_text.strip():
            return OCRResponse(
                success=False,
                message="Aucun texte détecté dans l'image",
                processing_time=time.time() - start_time,
                errors=["Aucun texte extrait par OCR"]
            )
        
        print(f"✅ Texte extrait: {len(extracted_text)} caractères (confiance: {ocr_confidence:.3f})")
        
        # Analyse avec google/gemma-3-4b-it
        print("🤖 Début analyse google/gemma-3-4b-it...")
        invoice_data = get_gemma_analyzer().analyze_document(extracted_text)
        
        # Ajuster le score de confiance global
        if invoice_data.confidence_score:
            combined_confidence = (ocr_confidence + invoice_data.confidence_score) / 2
            invoice_data.confidence_score = combined_confidence
        else:
            invoice_data.confidence_score = ocr_confidence
        
        # Ajouter les métadonnées OCR
        invoice_data.additional_data.update({
            'ocr_metadata': ocr_metadata,
            'ocr_confidence': ocr_confidence,
            'processing_method': 'ocr_gemma'
        })
        
        processing_time = time.time() - start_time
        print(f"✅ Traitement Gemma terminé en {processing_time:.2f}s")
        
        # Construire la réponse
        response_data = {
            'success': True,
            'message': 'Extraction et analyse Gemma terminées avec succès',
            'processing_time': processing_time,
            'invoice_data': invoice_data
        }
        
        # Ajouter le texte brut si demandé
        if include_raw_text:
            response_data['extracted_text'] = extracted_text
        
        # Ajouter le résultat Gemma si disponible et demandé
        if include_gemma_result and 'gemma_result' in invoice_data.additional_data:
            response_data['gemma_structured_data'] = invoice_data.additional_data['gemma_result']
        
        return OCRResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur inattendue Gemma: {e}")
        return OCRResponse(
            success=False,
            message="Erreur lors du traitement du fichier avec Gemma",
            processing_time=time.time() - start_time,
            errors=[str(e)]
        )
    
    finally:
        # Nettoyer le fichier temporaire
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"⚠️ Erreur suppression fichier temporaire: {e}")

@app.post("/extract-text-only")
async def extract_text_only(file: UploadFile = File(...)):
    """
    Extrait uniquement le texte sans analyse IA
    
    Args:
        file: Fichier image
    
    Returns:
        Texte extrait avec métadonnées
    """
    start_time = time.time()
    file_path = None
    
    try:
        if not validate_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporté. Extensions autorisées: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        file_path = await save_upload_file(file)
        
        extracted_text, confidence, metadata = extract_document_text(file_path)
        
        return {
            "success": True,
            "text": extracted_text,
            "confidence": confidence,
            "metadata": metadata,
            "processing_time": time.time() - start_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "processing_time": time.time() - start_time
        }
    
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

@app.post("/verify-invoice")
async def verify_invoice(file: UploadFile = File(...)):
    """
    Vérifie si l'image fournie est une facture en utilisant l'analyse de layout PaddleOCR
    
    Args:
        file: Fichier image à analyser
    
    Returns:
        Résultat de la vérification avec confiance et détails
    """
    start_time = time.time()
    file_path = None
    
    try:
        # Validation du fichier
        if not validate_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporté. Extensions autorisées: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Vérifier la taille du fichier
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Fichier trop volumineux. Taille maximale: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        # Remettre le pointeur au début pour la sauvegarde
        await file.seek(0)
        
        # Sauvegarder le fichier temporairement
        file_path = await save_upload_file(file)

        if is_pdf_file(file_path):
            print("🔍 Début vérification PDF avec extraction texte...")
            extracted_text, text_confidence, text_metadata = extract_document_text(file_path)
            is_facture, confidence_score, keyword_details = classify_invoice_text(extracted_text)
            if is_facture:
                confidence_score = max(confidence_score, min(text_confidence, 0.95))
            processing_time = time.time() - start_time

            result = {
                "success": True,
                "is_invoice": is_facture,
                "confidence_score": round(confidence_score, 3),
                "detected_text": extracted_text[:500] if extracted_text else "",
                "processing_time": round(processing_time, 3),
                "analysis_method": "pdf_text_classification",
                "analysis_details": {
                    "text_extracted": bool(extracted_text.strip()),
                    "keyword_details": keyword_details,
                    "text_metadata": text_metadata,
                }
            }

            print(f"✅ Vérification PDF terminée: {'FACTURE' if is_facture else 'PAS FACTURE'} (confiance: {confidence_score:.3f})")
            return result

        # Fallback OCR/texte si PaddleOCR layout n'est pas disponible pour les images
        if paddle_pipeline is None:
            print("⚠️ PaddleOCR indisponible, fallback vérification texte/OCR...")
            extracted_text, text_confidence, text_metadata = extract_document_text(file_path)
            is_facture, confidence_score, keyword_details = classify_invoice_text(extracted_text)
            if is_facture:
                confidence_score = max(confidence_score, min(text_confidence, 0.95))
            processing_time = time.time() - start_time

            return {
                "success": True,
                "is_invoice": is_facture,
                "confidence_score": round(confidence_score, 3),
                "detected_text": extracted_text[:500] if extracted_text else "",
                "processing_time": round(processing_time, 3),
                "analysis_method": "text_ocr_classification_fallback",
                "analysis_details": {
                    "text_extracted": bool(extracted_text.strip()),
                    "keyword_details": keyword_details,
                    "text_metadata": text_metadata,
                }
            }
        
        print("🔍 Début analyse de layout pour détection de facture...")
        
        # Exécuter l'analyse de layout
        visual_results = paddle_pipeline.visual_predict(
            input=file_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_common_ocr=False,
            use_seal_recognition=True,
            use_table_recognition=False
        )
        
        # Analyser les résultats pour détecter une facture
        is_facture = False
        confidence_score = 0.0
        detected_text = ""
        analysis_details = {}
        
        if (
            isinstance(visual_results, list) and len(visual_results) > 0 and
            "visual_info" in visual_results[0] and
            "normal_text_dict" in visual_results[0]["visual_info"] and
            "words in other_text" in visual_results[0]["visual_info"]["normal_text_dict"]
        ):
            text = visual_results[0]["visual_info"]['normal_text_dict']['words in other_text']
            detected_text = str(text) if text else ""
            
            # Vérifier la présence du mot "facture"
            if isinstance(text, str):
                is_facture = 'facture' in text.lower()
                if is_facture:
                    confidence_score = 0.8  # Confiance élevée si mot trouvé directement
            elif isinstance(text, list):
                matching_items = [t for t in text if isinstance(t, str) and 'facture' in t.lower()]
                is_facture = len(matching_items) > 0
                if is_facture:
                    confidence_score = 0.8
            
            # Analyser d'autres mots-clés liés aux factures
            invoice_keywords = ['invoice', 'bill', 'receipt', 'total', 'montant', 'prix', 'tva', 'ht', 'ttc']
            if detected_text:
                keyword_matches = sum(1 for keyword in invoice_keywords if keyword.lower() in detected_text.lower())
                if keyword_matches > 0:
                    confidence_score = max(confidence_score, 0.3 + (keyword_matches * 0.1))
                    if not is_facture and keyword_matches >= 3:
                        is_facture = True
                        confidence_score = 0.6
            
            analysis_details = {
                "visual_results_available": True,
                "text_extracted": bool(detected_text),
                "keyword_matches": keyword_matches if 'keyword_matches' in locals() else 0,
                "raw_visual_info": visual_results[0].get("visual_info", {}) if visual_results else {}
            }
        else:
            analysis_details = {
                "visual_results_available": False,
                "error": "Impossible d'extraire les informations visuelles"
            }
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "is_invoice": is_facture,
            "confidence_score": round(confidence_score, 3),
            "detected_text": detected_text[:500] if detected_text else "",  # Limiter la taille
            "processing_time": round(processing_time, 3),
            "analysis_method": "paddleocr_layout_analysis",
            "analysis_details": analysis_details
        }
        
        print(f"✅ Vérification terminée: {'FACTURE' if is_facture else 'PAS FACTURE'} (confiance: {confidence_score:.3f})")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {e}")
        return {
            "success": False,
            "is_invoice": False,
            "confidence_score": 0.0,
            "error": str(e),
            "processing_time": round(time.time() - start_time, 3),
            "analysis_method": "paddleocr_layout_analysis"
        }
    
    finally:
        # Nettoyer le fichier temporaire
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"⚠️ Erreur suppression fichier temporaire: {e}")

@app.post("/extract-entities")
async def extract_entities(file: UploadFile = File(...)):
    """
    Extrait les entités d'un document en utilisant le modèle YOLO
    
    Args:
        file: Fichier image du document
    
    Returns:
        Entités détectées avec leurs coordonnées et confiances
    """
    start_time = time.time()
    file_path = None
    
    try:
        # Validation du fichier
        if not validate_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporté. Extensions autorisées: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Vérifier la taille du fichier
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Fichier trop volumineux. Taille maximale: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        # Remettre le pointeur au début pour la sauvegarde
        await file.seek(0)
        
        # Sauvegarder le fichier temporairement
        file_path = await save_upload_file(file)
        
        print("🔍 Début extraction d'entités avec YOLO...")
        
        # Effectuer la prédiction avec YOLO
        results = model.predict(source=file_path, conf=0.25, iou=0.45, verbose=False)
        
        # Traiter les résultats
        entities = []
        image_info = {}
        
        if results and len(results) > 0:
            result = results[0]  # Premier résultat
            
            # Informations sur l'image
            image_info = {
                "width": int(result.orig_shape[1]),
                "height": int(result.orig_shape[0]),
                "channels": len(result.orig_shape) if len(result.orig_shape) > 2 else 1
            }
            
            # Extraire les détections
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    # Coordonnées de la boîte englobante
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    
                    # Classe et confiance
                    class_id = int(boxes.cls[i].cpu().numpy())
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_name = classes.get(class_id, f"class_{class_id}")
                    
                    # Calculer les dimensions
                    width = float(x2 - x1)
                    height = float(y2 - y1)
                    center_x = float(x1 + width / 2)
                    center_y = float(y1 + height / 2)
                    
                    entity = {
                        "id": i,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": {
                            "x1": round(float(x1), 2),
                            "y1": round(float(y1), 2),
                            "x2": round(float(x2), 2),
                            "y2": round(float(y2), 2),
                            "width": round(width, 2),
                            "height": round(height, 2),
                            "center_x": round(center_x, 2),
                            "center_y": round(center_y, 2)
                        },
                        "normalized_bbox": {
                            "x1": round(float(x1) / image_info["width"], 4),
                            "y1": round(float(y1) / image_info["height"], 4),
                            "x2": round(float(x2) / image_info["width"], 4),
                            "y2": round(float(y2) / image_info["height"], 4),
                            "center_x": round(center_x / image_info["width"], 4),
                            "center_y": round(center_y / image_info["height"], 4)
                        }
                    }
                    entities.append(entity)
        
        processing_time = time.time() - start_time
        
        # Statistiques sur les entités détectées
        entity_stats = {}
        for entity in entities:
            class_name = entity["class_name"]
            if class_name not in entity_stats:
                entity_stats[class_name] = {
                    "count": 0,
                    "avg_confidence": 0,
                    "max_confidence": 0,
                    "min_confidence": 1
                }
            
            stats = entity_stats[class_name]
            stats["count"] += 1
            stats["max_confidence"] = max(stats["max_confidence"], entity["confidence"])
            stats["min_confidence"] = min(stats["min_confidence"], entity["confidence"])
            
            # Recalculer la moyenne
            total_confidence = sum(e["confidence"] for e in entities if e["class_name"] == class_name)
            stats["avg_confidence"] = round(total_confidence / stats["count"], 3)
        
        result = {
            "success": True,
            "entities_count": len(entities),
            "entities": entities,
            "image_info": image_info,
            "entity_statistics": entity_stats,
            "available_classes": list(classes.values()) if classes else [],
            "processing_time": round(processing_time, 3),
            "model_info": {
                "type": "YOLO",
                "model_path": YOLO_MODEL_PATH,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45
            }
        }
        
        print(f"✅ Extraction terminée: {len(entities)} entités détectées en {processing_time:.2f}s")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction d'entités: {e}")
        return {
            "success": False,
            "entities_count": 0,
            "entities": [],
            "error": str(e),
            "processing_time": round(time.time() - start_time, 3),
            "model_info": {
                "type": "YOLO",
                "model_path": YOLO_MODEL_PATH
            }
        }
    
    finally:
        # Nettoyer le fichier temporaire
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"⚠️ Erreur suppression fichier temporaire: {e}")

@app.post("/extract-entities-with-ocr")
async def extract_entities_with_ocr(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = 0.25,
    extract_text_from_entities: Optional[bool] = True
):
    """
    Extrait les entités avec YOLO et applique l'OCR sur chaque entité détectée
    
    Args:
        file: Fichier image du document
        confidence_threshold: Seuil de confiance pour la détection (0.0-1.0)
        extract_text_from_entities: Appliquer l'OCR sur chaque entité détectée
    
    Returns:
        Entités détectées avec leur texte extrait par OCR
    """
    start_time = time.time()
    file_path = None
    
    try:
        # Validation du fichier
        if not validate_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporté. Extensions autorisées: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Valider le seuil de confiance
        if not 0.0 <= confidence_threshold <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="Le seuil de confiance doit être entre 0.0 et 1.0"
            )
        
        # Vérifier la taille du fichier
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Fichier trop volumineux. Taille maximale: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        # Remettre le pointeur au début pour la sauvegarde
        await file.seek(0)
        
        # Sauvegarder le fichier temporairement
        file_path = await save_upload_file(file)
        
        print(f"🔍 Début extraction d'entités avec OCR (seuil: {confidence_threshold})...")
        
        # Effectuer la prédiction avec YOLO
        results = model.predict(source=file_path, conf=confidence_threshold, iou=0.45, verbose=False)
        
        # Traiter les résultats
        entities = []
        image_info = {}
        
        if results and len(results) > 0:
            result = results[0]  # Premier résultat
            
            # Informations sur l'image
            image_info = {
                "width": int(result.orig_shape[1]),
                "height": int(result.orig_shape[0]),
                "channels": len(result.orig_shape) if len(result.orig_shape) > 2 else 1
            }
            
            # Charger l'image pour l'OCR si nécessaire
            import cv2
            if extract_text_from_entities and result.boxes is not None:
                image = cv2.imread(file_path)
                
                # Extraire les détections
                boxes = result.boxes
                for i in range(len(boxes)):
                    # Coordonnées de la boîte englobante
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    
                    # Classe et confiance
                    class_id = int(boxes.cls[i].cpu().numpy())
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_name = classes.get(class_id, f"class_{class_id}")
                    
                    # Extraire la région d'intérêt
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    roi = image[y1:y2, x1:x2]
                    
                    # Appliquer l'OCR sur la région
                    extracted_text = ""
                    ocr_confidence = 0.0
                    
                    if roi.size > 0:
                        try:
                            print(f"🔍 [DEBUG] Extraction OCR pour entité {i} - ROI shape: {roi.shape}")
                            
                            # Utiliser la nouvelle méthode pour extraire directement depuis l'array
                            ocr_result = ocr_extractor.extract_full_text_from_array(roi)
                            extracted_text = ocr_result['text']
                            ocr_confidence = ocr_result['confidence']
                            
                            print(f"✅ [DEBUG] OCR entité {i} - Texte: '{extracted_text[:50]}...', Confiance: {ocr_confidence}")
                            
                        except Exception as ocr_error:
                            print(f"⚠️ [DEBUG] Erreur OCR sur entité {i}: {ocr_error}")
                            import traceback
                            print(f"🔍 [DEBUG] Traceback OCR: {traceback.format_exc()}")
                            extracted_text = ""
                            ocr_confidence = 0.0
                    
                    # Calculer les dimensions
                    width = float(x2 - x1)
                    height = float(y2 - y1)
                    center_x = float(x1 + width / 2)
                    center_y = float(y1 + height / 2)
                    
                    entity = {
                        "id": i,
                        "class_id": class_id,
                        "class_name": class_name,
                        "detection_confidence": round(confidence, 3),
                        "bbox": {
                            "x1": round(float(x1), 2),
                            "y1": round(float(y1), 2),
                            "x2": round(float(x2), 2),
                            "y2": round(float(y2), 2),
                            "width": round(width, 2),
                            "height": round(height, 2),
                            "center_x": round(center_x, 2),
                            "center_y": round(center_y, 2)
                        },
                        "extracted_text": extracted_text.strip(),
                        "ocr_confidence": round(ocr_confidence, 3),
                        "combined_confidence": round((confidence + ocr_confidence) / 2, 3) if extracted_text.strip() else round(confidence, 3)
                    }
                    entities.append(entity)
            else:
                # Juste la détection sans OCR
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        class_id = int(boxes.cls[i].cpu().numpy())
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_name = classes.get(class_id, f"class_{class_id}")
                        
                        width = float(x2 - x1)
                        height = float(y2 - y1)
                        center_x = float(x1 + width / 2)
                        center_y = float(y1 + height / 2)
                        
                        entity = {
                            "id": i,
                            "class_id": class_id,
                            "class_name": class_name,
                            "detection_confidence": round(confidence, 3),
                            "bbox": {
                                "x1": round(float(x1), 2),
                                "y1": round(float(y1), 2),
                                "x2": round(float(x2), 2),
                                "y2": round(float(y2), 2),
                                "width": round(width, 2),
                                "height": round(height, 2),
                                "center_x": round(center_x, 2),
                                "center_y": round(center_y, 2)
                            }
                        }
                        entities.append(entity)
        
        processing_time = time.time() - start_time
        
        # Statistiques
        entity_stats = {}
        text_extraction_stats = {
            "entities_with_text": 0,
            "entities_without_text": 0,
            "avg_text_length": 0,
            "avg_ocr_confidence": 0
        }
        
        for entity in entities:
            class_name = entity["class_name"]
            if class_name not in entity_stats:
                entity_stats[class_name] = {"count": 0, "avg_confidence": 0}
            
            entity_stats[class_name]["count"] += 1
            
            # Stats sur l'extraction de texte
            if extract_text_from_entities and "extracted_text" in entity:
                if entity["extracted_text"]:
                    text_extraction_stats["entities_with_text"] += 1
                    text_extraction_stats["avg_text_length"] += len(entity["extracted_text"])
                    text_extraction_stats["avg_ocr_confidence"] += entity["ocr_confidence"]
                else:
                    text_extraction_stats["entities_without_text"] += 1
        
        # Finaliser les moyennes
        if text_extraction_stats["entities_with_text"] > 0:
            text_extraction_stats["avg_text_length"] = round(
                text_extraction_stats["avg_text_length"] / text_extraction_stats["entities_with_text"], 1
            )
            text_extraction_stats["avg_ocr_confidence"] = round(
                text_extraction_stats["avg_ocr_confidence"] / text_extraction_stats["entities_with_text"], 3
            )
        
        result = {
            "success": True,
            "entities_count": len(entities),
            "entities": entities,
            "image_info": image_info,
            "entity_statistics": entity_stats,
            "text_extraction_statistics": text_extraction_stats if extract_text_from_entities else None,
            "processing_time": round(processing_time, 3),
            "parameters": {
                "confidence_threshold": confidence_threshold,
                "extract_text_from_entities": extract_text_from_entities
            },
            "model_info": {
                "type": "YOLO + Tesseract OCR",
                "model_path": YOLO_MODEL_PATH
            }
        }
        
        print(f"✅ Extraction terminée: {len(entities)} entités détectées en {processing_time:.2f}s")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction d'entités avec OCR: {e}")
        return {
            "success": False,
            "entities_count": 0,
            "entities": [],
            "error": str(e),
            "processing_time": round(time.time() - start_time, 3)
        }
    
    finally:
        # Nettoyer le fichier temporaire
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"⚠️ Erreur suppression fichier temporaire: {e}")

@app.post("/invoices/process")
async def process_invoice_for_data_entry(
    file: UploadFile = File(...),
    model_choice: Optional[str] = "gemini",
    index_for_rag: Optional[bool] = True,
):
    """Traite une facture image/PDF et crée une fiche de saisie en brouillon."""
    start_time = time.time()
    file_path = None

    try:
        if not validate_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporté. Extensions autorisées: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Fichier trop volumineux. Taille maximale: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )

        await file.seek(0)
        file_path = await save_upload_file(file)

        extracted_text, extraction_confidence, extraction_metadata = extract_document_text(file_path)
        if not extracted_text.strip():
            return {
                "success": False,
                "message": "Aucun texte détecté dans le document",
                "processing_time": round(time.time() - start_time, 3),
                "metadata": extraction_metadata,
            }

        model_used = "fallback_regex"
        invoice_data = None
        raw_data = {
            "extraction_metadata": extraction_metadata,
            "extracted_text_preview": extracted_text[:1000],
        }

        try:
            if model_choice == "local":
                invoice_data = ollama_analyzer.analyze_invoice_text(extracted_text)
                model_used = f"ollama:{ollama_analyzer.model_name}" if invoice_data else "fallback_regex"
            elif model_choice == "fallback":
                model_used = "fallback_regex"
            elif model_choice == "gemma":
                invoice_data = get_gemma_analyzer().analyze_document(extracted_text)
                model_used = "gemma"
            else:
                invoice_data = ai_analyzer.analyze_invoice_text(extracted_text)
                model_used = "gemini" if ai_analyzer.model else "fallback_regex"
        except Exception as analysis_error:
            print(f"⚠️ Analyse IA indisponible, fallback regex: {analysis_error}")
            raw_data["analysis_error"] = str(analysis_error)

        normalized_data = normalize_invoice_data(invoice_data, extracted_text)
        confidence_score = extraction_confidence
        if invoice_data and invoice_data.confidence_score:
            confidence_score = (extraction_confidence + invoice_data.confidence_score) / 2

        invoice_record = invoice_store.create_invoice(
            filename=file.filename or Path(file_path).name,
            file_type=Path(file_path).suffix.lower().lstrip("."),
            source_file_path=file_path,
            extracted_text=extracted_text,
            normalized_data=normalized_data,
            raw_data=raw_data,
            model_used=model_used,
            confidence_score=confidence_score,
        )

        rag_document = None
        if index_for_rag:
            try:
                rag_document = rag_service.index_document(
                    filename=file.filename or Path(file_path).name,
                    file_type=Path(file_path).suffix.lower().lstrip("."),
                    source_file_path=file_path,
                    extracted_text=extracted_text,
                    extraction_metadata={
                        "confidence": extraction_confidence,
                        "metadata": extraction_metadata,
                        "invoice_id": invoice_record["id"],
                    },
                )
            except Exception as rag_error:
                print(f"⚠️ Indexation RAG ignorée: {rag_error}")

        return {
            "success": True,
            "message": "Facture traitée et prête pour validation",
            "processing_time": round(time.time() - start_time, 3),
            "invoice": invoice_record,
            "rag_document": rag_document,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur traitement facture: {e}")
        return {
            "success": False,
            "message": "Erreur lors du traitement de la facture",
            "error": str(e),
            "processing_time": round(time.time() - start_time, 3),
        }
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"⚠️ Erreur suppression fichier temporaire: {e}")


@app.post("/agent/process-document")
async def agent_process_document(
    file: UploadFile = File(...),
    model_choice: Optional[str] = "local",
    index_for_rag: Optional[bool] = True,
    detect_objects: Optional[bool] = True,
    token: str = "",
    authorization: Optional[str] = Header(None),
):
    """Agent IA documentaire RAG pour image/PDF.

    Pipeline:
    1. Upload et controle fichier.
    2. Extraction texte: PDF natif/fallback OCR ou OCR image.
    3. Classification facture / non-facture.
    4. Detection visuelle YOLO pour les images.
    5. RAG d'extraction: chunks temporaires + evidence par champ.
    6. Routage LLM puis extraction des champs.
    7. Normalisation TND/dates/lignes + controles metier.
    8. Creation du brouillon et detection de doublons si disponible.
    9. Indexation RAG documentaire persistante.
    10. Routage workflow vers validation humaine, revue qualite ou archivage.
    """
    user = get_user_from_token(token) or user_from_authorization(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Connexion requise pour analyser une facture")
    try:
        return await run_document_agent(
            file=file,
            model_choice=model_choice or "local",
            index_for_rag=bool(index_for_rag),
            detect_objects=bool(detect_objects),
        )
    except HTTPException:
        raise
    except Exception as error:
        print(f"Erreur agent IA RAG: {error}")
        return {
            "success": False,
            "message": "Erreur lors du traitement agent IA RAG",
            "error": str(error),
        }


@app.get("/agent/documents")
async def agent_list_documents():
    """Liste les documents traites par l'agent IA RAG."""
    return {
        "success": True,
        "documents": document_store.list_documents(),
    }


@app.get("/agent/documents/{document_id}")
async def agent_get_document(document_id: int):
    """Recupere un document non structure traite par l'agent."""
    try:
        return {
            "success": True,
            "document": document_store.get_document(document_id),
        }
    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.get("/invoices")
async def list_invoices(
    search: str = "",
    invoice_name: str = "",
    supplier: str = "",
    company: str = "",
    tax_id: str = "",
    status: str = "",
    page: int = 1,
    page_size: int = 10,
):
    """Liste les factures traitées avec recherche métier et pagination."""
    result = invoice_store.search_invoices(
        search=search,
        invoice_name=invoice_name,
        supplier=supplier,
        company=company,
        tax_id=tax_id,
        status=status,
        page=page,
        page_size=page_size,
    )
    return {
        "success": True,
        "invoices": result["items"],
        "pagination": {
            "total": result["total"],
            "page": result["page"],
            "page_size": result["page_size"],
            "total_pages": result["total_pages"],
        },
    }


@app.get("/invoices/catalog")
async def invoice_catalog(
    source: str = "all",
    search: str = "",
    invoice_name: str = "",
    supplier: str = "",
    company: str = "",
    tax_id: str = "",
    status: str = "",
    page: int = 1,
    page_size: int = 10,
):
    """Catalogue paginé des factures traitées et des factures du dataset indexé."""
    normalized_source = source.lower().strip()
    if normalized_source == "processed":
        result = invoice_store.search_invoices(
            search=search,
            invoice_name=invoice_name,
            supplier=supplier,
            company=company,
            tax_id=tax_id,
            status=status,
            page=page,
            page_size=page_size,
        )
        items = result["items"]
    elif normalized_source == "dataset":
        result = fatura_dataset_store.invoice_catalog(
            search=search,
            invoice_name=invoice_name,
            supplier=supplier,
            company=company,
            tax_id=tax_id,
            page=page,
            page_size=page_size,
        )
        items = result["items"]
    else:
        processed = invoice_store.search_invoices(
            search=search,
            invoice_name=invoice_name,
            supplier=supplier,
            company=company,
            tax_id=tax_id,
            status=status,
            page=1,
            page_size=100,
        )
        dataset_page = max(1, page)
        dataset = fatura_dataset_store.invoice_catalog(
            search=search,
            invoice_name=invoice_name,
            supplier=supplier,
            company=company,
            tax_id=tax_id,
            page=dataset_page,
            page_size=page_size,
        )
        result = {
            **dataset,
            "total": int(processed["total"]) + int(dataset["total"]),
            "total_pages": max(1, (int(processed["total"]) + int(dataset["total"]) + int(dataset["page_size"]) - 1) // int(dataset["page_size"])),
        }
        items = processed["items"][:page_size] if page == 1 else []
        items.extend(dataset["items"][: max(0, page_size - len(items))])

    return {
        "success": True,
        "source": normalized_source,
        "invoices": items,
        "pagination": {
            "total": result["total"],
            "page": result["page"],
            "page_size": result["page_size"],
            "total_pages": result["total_pages"],
        },
    }


@app.get("/invoices/catalog/detail")
async def invoice_catalog_detail(source: str = "processed", invoice_id: str = ""):
    """Récupère le détail d'une facture traitée ou d'un élément dataset indexé."""
    if not invoice_id:
        raise HTTPException(status_code=400, detail="invoice_id est obligatoire")
    normalized_source = source.lower().strip()
    try:
        if normalized_source == "dataset" or invoice_id.startswith("dataset:"):
            return {
                "success": True,
                "source": "dataset",
                "invoice": fatura_dataset_store.get_catalog_invoice(invoice_id),
            }
        return {
            "success": True,
            "source": "processed",
            "invoice": invoice_store.get_invoice(int(invoice_id)),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/invoices/catalog/download")
async def invoice_catalog_download(source: str = "processed", invoice_id: str = "", inline: bool = False):
    """Télécharge le fichier réel lié à une facture ou à un élément dataset."""
    if not invoice_id:
        raise HTTPException(status_code=400, detail="invoice_id est obligatoire")
    normalized_source = source.lower().strip()
    try:
        if normalized_source == "dataset" or invoice_id.startswith("dataset:"):
            filename, data = fatura_dataset_store.read_image_bytes(invoice_id)
            media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            disposition = "inline" if inline else "attachment"
            headers = {"Content-Disposition": f'{disposition}; filename="{filename}"'}
            return StreamingResponse(io.BytesIO(data), media_type=media_type, headers=headers)

        invoice = invoice_store.get_invoice(int(invoice_id))
        reference = invoice.get("file_path", "")
        if reference.startswith(PostgresFileStore.PREFIX):
            from urllib.parse import quote
            content = invoice_store.files.get(reference)
            filename = invoice.get("filename") or "document"
            disposition = "inline" if inline else "attachment"
            return StreamingResponse(
                io.BytesIO(content),
                media_type=mimetypes.guess_type(filename)[0] or "application/octet-stream",
                headers={"Content-Disposition": f"{disposition}; filename*=UTF-8''{quote(filename, safe='')}"},
            )
        file_path = Path(reference)
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")
        return FileResponse(
            path=file_path,
            filename=invoice.get("filename") or file_path.name,
            media_type=mimetypes.guess_type(file_path.name)[0] or "application/octet-stream",
            content_disposition_type="inline" if inline else "attachment",
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/invoices/{invoice_id}")
async def get_invoice(invoice_id: int):
    """Récupère une facture traitée."""
    try:
        return {
            "success": True,
            "invoice": invoice_store.get_invoice(invoice_id),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put("/invoices/{invoice_id}")
async def update_invoice(invoice_id: int, request: InvoiceUpdateRequest):
    """Met à jour les champs corrigés d'une facture."""
    try:
        return {
            "success": True,
            "invoice": invoice_store.update_invoice_data(
                invoice_id=invoice_id,
                normalized_data=request.normalized_data,
                status=request.status,
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/invoices/{invoice_id}/validate")
async def validate_invoice_record(
    invoice_id: int,
    request: InvoiceValidationRequest,
    authorization: Optional[str] = Header(None),
):
    """Valide une facture après correction humaine."""
    user = user_from_authorization(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Connexion requise pour valider une facture")
    try:
        return {
            "success": True,
            "invoice": invoice_store.validate_invoice(
                invoice_id=invoice_id,
                normalized_data=request.normalized_data,
                user_email=user["email"],
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/database/stats")
async def database_stats():
    """Statistiques SQL metier pour dashboard."""
    return {
        "success": True,
        "stats": invoice_store.get_dashboard_stats(),
    }


@app.get("/database/suppliers")
async def database_suppliers():
    """Liste des fournisseurs/contreparties consolides depuis les factures."""
    return {
        "success": True,
        "suppliers": invoice_store.list_suppliers(),
    }


@app.get("/database/invoice-items")
async def database_invoice_items(invoice_id: Optional[int] = None):
    """Lignes de facture structurees."""
    return {
        "success": True,
        "items": invoice_store.get_invoice_items(invoice_id=invoice_id),
    }


@app.get("/database/validation-events")
async def database_validation_events(invoice_id: Optional[int] = None):
    """Historique des corrections et validations humaines."""
    return {
        "success": True,
        "events": invoice_store.get_validation_events(invoice_id=invoice_id),
    }


@app.post("/rag/index-document")
async def rag_index_document(file: UploadFile = File(...)):
    """Indexe un PDF ou une image dans la base RAG locale."""
    start_time = time.time()
    file_path = None

    try:
        if not validate_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporté. Extensions autorisées: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Fichier trop volumineux. Taille maximale: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )

        await file.seek(0)
        file_path = await save_upload_file(file)

        extracted_text, confidence, metadata = extract_document_text(file_path)
        if not extracted_text.strip():
            return {
                "success": False,
                "message": "Aucun texte détecté, impossible d'indexer le document",
                "processing_time": round(time.time() - start_time, 3),
                "metadata": metadata,
            }

        indexed = rag_service.index_document(
            filename=file.filename or Path(file_path).name,
            file_type=Path(file_path).suffix.lower().lstrip("."),
            source_file_path=file_path,
            extracted_text=extracted_text,
            extraction_metadata={
                "confidence": confidence,
                "metadata": metadata,
            },
        )

        return {
            "success": True,
            "message": "Document indexé avec succès",
            "processing_time": round(time.time() - start_time, 3),
            "document": indexed,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur indexation RAG: {e}")
        return {
            "success": False,
            "message": "Erreur lors de l'indexation RAG",
            "error": str(e),
            "processing_time": round(time.time() - start_time, 3),
        }
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"⚠️ Erreur suppression fichier temporaire: {e}")


@app.post("/rag/query")
async def rag_query(request: RAGQueryRequest):
    """Recherche les passages pertinents et prépare un prompt RAG."""
    try:
        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(rag_service.query,
            question=request.question,
            document_id=request.document_id,
            top_k=request.top_k,
        )
        result["generation"] = {"status": "unavailable", "model": None}
        if result["sources"] and ollama_analyzer.is_available():
            try:
                generated = await run_in_threadpool(ollama_analyzer.answer_sources, request.question, result["sources"])
                result.update(generated)
                result["generation"] = {"status": "completed", "model": f"ollama:{ollama_analyzer.model_name}"}
            except RuntimeError as error:
                result["generation"] = {"status": "failed", "message": str(error), "model": f"ollama:{ollama_analyzer.model_name}"}
        elif ai_analyzer.model and result["sources"]:
            try:
                generated = await run_in_threadpool(ai_analyzer.answer_sources, request.question, result["sources"])
                result.update(generated)
                result["generation"] = {"status": "completed", "model": ai_analyzer.model_name}
            except RuntimeError as error:
                result["generation"] = {"status": "failed", "message": str(error), "model": ai_analyzer.model_name}
        return {"success": True, **result}
    except Exception as e:
        print(f"❌ Erreur requête RAG: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@app.get("/rag/documents")
async def rag_documents():
    """Liste les documents indexés dans la base RAG locale."""
    return {
        "success": True,
        "documents": rag_service.list_documents(),
    }


@app.get("/supported-formats")
async def get_supported_formats():
    """Retourne les formats supportés et informations sur l'API"""
    return {
        "supported_extensions": ALLOWED_EXTENSIONS,
        "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024,
        "ocr_method": "Tesseract optimisé avec préprocessing",
        "ai_method": "Gemini avec prompt structuré",
        "specializations": [
            "Factures de location de véhicules",
            "Documents tunisiens (TND)",
            "Reçus et factures diverses"
        ],
        "version": "2.0.0"
    }

@app.get("/model-info")
async def get_model_info():
    """
    Retourne les informations sur le modèle YOLO chargé
    
    Returns:
        Informations sur le modèle et les classes disponibles
    """
    try:
        model_info = {
            "model_type": "YOLO",
            "model_path": YOLO_MODEL_PATH,
            "model_loaded": model is not None,
            "available_classes": {},
            "total_classes": len(classes) if classes else 0
        }
        
        if classes:
            model_info["available_classes"] = classes
            model_info["class_names"] = list(classes.values())
            model_info["class_ids"] = list(classes.keys())
        
        # Essayer d'obtenir plus d'informations sur le modèle
        if model is not None:
            try:
                model_info["model_details"] = {
                    "task": getattr(model, 'task', 'unknown'),
                    "device": str(getattr(model, 'device', 'unknown')),
                    "names": getattr(model, 'names', {}),
                }
            except Exception as e:
                model_info["model_details"] = {"error": f"Cannot access model details: {str(e)}"}
        
        return model_info
        
    except Exception as e:
        return {
            "model_type": "YOLO",
            "model_path": YOLO_MODEL_PATH,
            "model_loaded": False,
            "error": str(e),
            "available_classes": {},
            "total_classes": 0
        }

# Configuration de lancement du serveur
if __name__ == "__main__":
    import uvicorn
    print("🚀 Démarrage du serveur Optimized Invoice OCR API...")
    print("🔗 URL: http://localhost:8002")
    print("📚 Documentation: http://localhost:8002/docs")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )
