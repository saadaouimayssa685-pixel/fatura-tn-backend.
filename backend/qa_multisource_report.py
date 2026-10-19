"""Summarize the untouched multisource baseline, including failed and fallback runs."""
import argparse
import json
from pathlib import Path

from qa_archive_invoices import ROOT, evaluate, save_json


def summarize(output):
    evaluate(output)
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    rows, details = [], []
    replayed = False
    counts = {"documents_selected": len(manifest["selected"]), "completed": 0, "groq_completed": 0,
              "groq_partial": 0, "not_invoices": 0, "checks_passed": 0, "checks_total": 0}
    for item in manifest["selected"]:
        name = item["entry"]
        report = output / (Path(name).stem + ".result.json")
        result = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
        replayed = replayed or bool(result.get("replay"))
        comparison = evaluation.get(name, {})
        checks = comparison.get("checks", [])
        llm = (result.get("analysis_diagnostics") or {}).get("llm") or {}
        cls = result.get("classification", [None, None, {}])
        document_type = "Facture" if cls[0] is True else "Non-facture" if cls[0] is False else "Inconnu"
        status = llm.get("status", result.get("model_used", result.get("current_stage", "Non execute")))
        counts["completed"] += int(bool(result.get("completed")))
        counts["groq_completed"] += int(status == "completed")
        counts["groq_partial"] += int(status == "completed_partial")
        counts["not_invoices"] += int(cls[0] is False)
        counts["checks_passed"] += comparison.get("passed", 0)
        counts["checks_total"] += comparison.get("total", 0)
        total = (result.get("fields") or {}).get("total_amount")
        duration = result.get("replay", {}).get("llm_seconds", sum(result.get("timings", {}).values()))
        rows.append(f"| {name} | {document_type} | {status} | {comparison.get('passed', 0)}/{comparison.get('total', 0)} | {total} | {duration:.1f} s |")
        details.extend([f"## {name}", "", f"Origine : `{json.dumps(item['origin'], ensure_ascii=False)}`", "",
            f"Classification : {cls[2].get('analysis_method', 'inconnue')}. OCR : {result.get('ocr_metadata', {}).get('method_used', 'inconnu')}.",
            f"Recherche : {result.get('retrieval_backend', 'non executee')}. Modele : {result.get('model_used', 'non execute')}.",
            f"Durees : `{json.dumps(result.get('timings', {}))}`.", ""])
        if result.get("error"):
            details.append(f"Erreur de traitement : {result['error']}")
        if result.get("replay"):
            details.append("Rejeu Groq uniquement : OCR, classification et passages RAG repris du premier test; temps du tableau = nouvel appel LLM.")
        ocr_error = result.get("ocr_metadata", {}).get("glm_ocr_error")
        if ocr_error:
            details.append(f"Lecture GLM-OCR indisponible/incomplete : {ocr_error}; texte Tesseract utilise.")
        failed = [c for c in checks if not c["passed"]]
        if failed:
            details.extend(["| Champ | Attendu | Obtenu |", "|---|---|---|"])
            for check in failed:
                escape = lambda v: str(v).replace("|", "/").replace("\n", " ")
                details.append(f"| {check['field']} | {escape(check['expected'])} | {escape(check['actual'])} |")
        else:
            details.append("Aucun ecart sur les champs controles." if checks else "Pas de resultat evaluable.")
        rejected = {",".join(b["fields"]): b.get("rejected_fields", []) for b in llm.get("batches", []) if b.get("rejected_fields")}
        if rejected:
            details.extend(["", "Champs refuses par la verification des citations :", f"```json\n{json.dumps(rejected, ensure_ascii=False, indent=2)}\n```"])
        details.append("")
    summary = ["# Tests Groq sur plusieurs sources", "",
        ("Rejeu apres correction du schema Groq : memes images, OCR, classification et passages RAG que le premier test. Aucune ecriture dans la base de l'application."
         if replayed else "OCR relance sur chaque image, classification PaddleOCR, recherche hybride et extraction Groq. Aucune ecriture dans la base de l'application."), "",
        "Les annotations de reference sont reservees a l'evaluation; elles ne sont pas transmises au LLM. Les scores ci-dessous sont des controles de champs, pas un taux de fiabilite general.", "",
        "| Document | Classification | Generation | Controles | TTC final | Temps |", "|---|---|---|---|---|---|", *rows, "",
        "## Limites", "",
        "- Echantillon fixe de huit documents, pas une evaluation exhaustive des archives.",
        "- FATURA contient des factures synthetiques en anglais; les montants et dates imprimes peuvent etre incoherents. Ne pas les inventer ou les corriger automatiquement.",
        "- La page Hammami est la page 2/2 : cinq lignes visibles ne constituent pas necessairement toutes les lignes de la facture.",
        "- Les detections de logo/cachet/signature et les interactions de l'interface ne font pas partie de ce test.", "", *details]
    (output / "RAPPORT.md").write_text("\n".join(summary), encoding="utf-8")
    save_json(output / "summary.json", counts)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "qa-results" / "groq-multisource-20260912")
    summarize(parser.parse_args().output)
