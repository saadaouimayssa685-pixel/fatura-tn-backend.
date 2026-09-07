"""Explicit hosted smoke test; creates a labelled QA account and invoice."""
import hashlib
import io
import json
import os
import secrets
import uuid
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


def main():
    base = os.getenv("QA_API_URL", "https://fatura-tn-backend.onrender.com")
    session = requests.Session()
    email = f"qa-{uuid.uuid4().hex[:12]}@example.com"
    response = session.post(base + "/auth/signup", json={
        "full_name": "QA Test", "company": "QA Synthetic", "phone": "00000000",
        "email": email, "password": secrets.token_urlsafe(24),
    }, timeout=90)
    response.raise_for_status()
    token = response.json()["token"]
    session.headers["Authorization"] = "Bearer " + token
    image = Image.new("RGB", (1600, 1800), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 38)
    lines = ["QA SYNTHETIC - FACTURE DE TEST", "FOURNISSEUR: QA FOURNITURES SARL",
             "Adresse: 10 rue du Test, Tunis", "MF: 1234567 A/M/000",
             "CLIENT: QA CLIENT SARL", "FACTURE N: QA-2026-1001", "Date: 07/09/2026",
             "Designation                    Quantite     PU HT       Montant HT",
             "Cahier QA                              2           10,000           20,000",
             "TOTAL HT: 20,000", "TVA 19%: 3,800", "TIMBRE: 1,000", "TOTAL TTC: 24,800"]
    for index, line in enumerate(lines):
        draw.text((60, 70 + index * 110), line, fill="black", font=font)
    output = io.BytesIO()
    image.save(output, format="PNG")
    content = output.getvalue()
    response = session.post(base + "/agent/process-document", params={
        "model_choice": "local", "index_for_rag": "true", "detect_objects": "true",
    }, files={"file": ("QA_SYNTHETIC.png", content, "image/png")}, timeout=300)
    response.raise_for_status()
    result = response.json()
    target = Path(__file__).resolve().parent.parent / "data" / "qa-hosted"
    target.mkdir(parents=True, exist_ok=True)
    (target / "extraction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    invoice = result.get("invoice") or {}
    print(json.dumps({"success": result.get("success"), "keys": list(result),
                      "invoice_id": invoice.get("id"), "normalized": invoice.get("normalized_data")}, ensure_ascii=True))
    if not invoice.get("id"):
        raise RuntimeError("No invoice persisted; inspect extraction.json")
    invoice_id = invoice["id"]
    download = session.get(base + "/invoices/catalog/download", params={"invoice_id": invoice_id}, timeout=90)
    download.raise_for_status()
    assert download.content == content, "Downloaded bytes differ"
    validation = session.post(base + f"/invoices/{invoice_id}/validate", json={}, timeout=90)
    print("Validation:", validation.status_code, validation.text[:500])
    rag = session.post(base + "/rag/query", json={"question": "Cahier QA", "top_k": 3}, timeout=90)
    rag.raise_for_status()
    print("RAG sources:", len(rag.json().get("sources", [])))
    (target / "persistence.json").write_text(json.dumps({"invoice_id": invoice_id,
        "sha256": hashlib.sha256(content).hexdigest()}), encoding="utf-8")
    session.post(base + "/auth/logout", timeout=90).raise_for_status()
    print("Download exact-byte comparison: PASS")


if __name__ == "__main__":
    main()
