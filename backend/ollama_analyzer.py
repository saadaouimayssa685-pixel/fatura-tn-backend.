import json
import os
import re
from typing import Any, Dict, Optional

import requests

from models import CustomerInfo, InvoiceData, SupplierInfo


class OllamaInvoiceAnalyzer:
    """Local LLM invoice analyzer using Ollama."""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.model_name = os.getenv("OLLAMA_INVOICE_MODEL", "phi3.5")
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(self._matches_model_name(model.get("name", "")) for model in models)
        except Exception:
            return False

    def _matches_model_name(self, installed_name: str) -> bool:
        return installed_name == self.model_name or installed_name.startswith(f"{self.model_name}:")

    def analyze_invoice_text(self, extracted_text: str) -> Optional[InvoiceData]:
        prompt = self._build_prompt(extracted_text)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.0,
                        "num_ctx": 8192,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw_response = response.json().get("response", "")
            parsed = self._parse_json(raw_response)
            if not parsed:
                return None
            return self._to_invoice_data(parsed, raw_response)
        except Exception as error:
            print(f"Ollama analysis unavailable: {error}")
            return None

    def _build_prompt(self, extracted_text: str) -> str:
        schema = {
            "invoice_number": None,
            "invoice_date": None,
            "due_date": None,
            "supplier_name": None,
            "supplier_address": None,
            "customer_name": None,
            "customer_address": None,
            "subtotal": None,
            "tax_amount": None,
            "stamp_duty": None,
            "total_amount": None,
            "currency": "TND",
            "payment_method": None,
            "rib": None,
            "items": [
                {
                    "description": None,
                    "quantity": None,
                    "unit_price": None,
                    "total": None,
                }
            ],
        }
        return (
            "Tu es un moteur d'extraction de factures. "
            "Retourne uniquement un JSON valide, sans markdown ni explication. "
            "Respecte exactement les cles du schema. Si une information est absente, utilise null. "
            "Les montants doivent etre des nombres.\n\n"
            f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"Texte OCR/PDF:\n{extracted_text[:12000]}"
        )

    def _parse_json(self, value: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", value, flags=re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    def _to_invoice_data(self, data: Dict[str, Any], raw_response: str) -> InvoiceData:
        supplier = None
        if data.get("supplier_name") or data.get("supplier_address"):
            supplier = SupplierInfo(
                name=data.get("supplier_name"),
                address=data.get("supplier_address"),
            )

        customer = None
        if data.get("customer_name") or data.get("customer_address"):
            customer = CustomerInfo(
                name=data.get("customer_name"),
                address=data.get("customer_address"),
            )

        return InvoiceData(
            invoice_number=data.get("invoice_number"),
            invoice_date=data.get("invoice_date"),
            due_date=data.get("due_date"),
            subtotal=self._safe_float(data.get("subtotal")),
            tax_amount=self._safe_float(data.get("tax_amount")),
            total_amount=self._safe_float(data.get("total_amount")),
            currency=data.get("currency") or "TND",
            supplier=supplier,
            customer=customer,
            confidence_score=0.82,
            additional_data={
                "ollama_result": data,
                "ollama_raw_response": raw_response,
                "analysis_method": "ollama_local",
                "model_version": self.model_name,
            },
        )

    def _safe_float(self, value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = re.sub(r"[^\d.,-]", "", value).replace(",", ".")
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None
