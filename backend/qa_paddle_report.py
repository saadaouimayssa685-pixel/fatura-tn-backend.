"""Write comparison and recheck cached LLM outputs with final guards, offline."""
import json
import shutil
from pathlib import Path

from field_locations import locate_paddle_fields
from models import InvoiceData
from qa_archive_invoices import ROOT, evaluate, pipeline, save_json


baseline = ROOT / "qa-results/groq-multisource-20260912"
ocr_dir = ROOT / "qa-results/paddle-small-20260913"
output = ROOT / "qa-results/paddle-pipeline-groq-20260913-verified"
retry = ROOT / "qa-results/paddle-pipeline-groq-20260913-party-check/facture_00008.result.json"
if retry.exists():
    previous = output / "facture_00008.before-party-check.json"
    target = output / "facture_00008.result.json"
    if not previous.exists():
        shutil.copy2(target, previous)
    shutil.copy2(retry, target)
scope = pipeline()
for path in output.glob("*.result.json"):
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("llm"):
        continue
    data = InvoiceData.model_validate(result["llm"])
    text, rules, evidence = result["ocr_text"], result["rule_candidates"], result["evidence"]
    normalized = scope["normalize_invoice_data"](data, text, rules)
    fields, proofs = scope["enforce_evidence_guard"](normalized, text, evidence, rules)
    locations = locate_paddle_fields(result["ocr_metadata"].get("lines", []), fields, proofs)
    for key, location in locations.items():
        proofs.setdefault(key, {"value": fields.get(key)})["location"] = location
    result.setdefault("fields_before_final_guards", result["fields"])
    result.update(fields=fields, proofs=proofs, final_guards_rechecked_offline=True,
                  issues=scope["validate_normalized_invoice_fields"](fields, proofs))
    save_json(path, result)
evaluate(output)
evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
lines = ["# Comparatif OCR et corrections FATURA.tn", "",
         "Huit images figees, trois sources. Aucune ecriture dans SQLite et aucune modification du frontend.", "",
         "## OCR", "",
         "| Image | Ancien OCR Tesseract/GLM (s) | Paddle small FR/arabe (s) | Etat |", "|---|---:|---:|---|"]
for item in manifest["selected"]:
    stem = Path(item["entry"]).stem
    original = json.loads((baseline / (stem + ".result.json")).read_text(encoding="utf-8"))
    path = ocr_dir / (stem + ".ocr.json")
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    lines.append(f"| {Path(item['entry']).name} | {original['timings']['ocr']} | {current.get('seconds', 'non termine')} | {'Termine' if current.get('completed') else 'Non termine'} |")
lines += ["", "Les deux premiers temps Paddle comprennent deux demarrages a froid; le premier comprend aussi les telechargements et une passe arabe exhaustive ensuite remplacee par une passe ciblee. Ils ne representent pas le temps a chaud.",
          "Les six suivants utilisent le meme processus chaud. Ces durees concernent uniquement l'OCR, pas toute la requete HTTP. Le score moyen OCR n'est pas une mesure de justesse des champs.", "",
          "Un premier essai medium a ete interrompu faute de temps de reponse acceptable. Il n'est pas retenu.", "",
          "## Extraction RAG et Groq", "",
          "Trois cas ont ete rejoues avec le nouvel OCR, une nouvelle recherche hybride et des requetes Groq espacees. Les controles finaux ont ete reappliques hors ligne aux memes reponses, sans autre appel cloud. Les boites de classification proviennent du precedent test de la meme image.", "",
          "S.T.FOOM a fait l'objet d'un second appel cible apres renforcement des instructions fournisseur/client et TVA. Le resultat precedent est conserve separement.", "",
          "| Image | Controles reussis | Fournisseur | Lignes | HT | TVA | TTC |", "|---|---:|---|---:|---:|---:|---:|"]
for name, comparison in evaluation.items():
    result = json.loads((output / (Path(name).stem + ".result.json")).read_text(encoding="utf-8"))
    fields = result["fields"]
    lines.append(f"| {name} | {comparison['passed']}/{comparison['total']} | {fields.get('vendor_name')} | {len(fields.get('items') or [])} | {fields.get('subtotal')} | {fields.get('tax_amount')} | {fields.get('total_amount')} |")
lines += ["", "Ces controles de champs ne sont pas un taux de fiabilite general; ils comprennent les absences attendues.", "", "## Ecarts restants", ""]
for name, comparison in evaluation.items():
    for check in comparison["checks"]:
        if not check["passed"]:
            lines.append(f"- {name}, {check['field']} : attendu `{check['expected']}`, obtenu `{check['actual']}`.")
lines += ["", "## Perimetre et limites", "",
          "- L'interface et le parcours de sauvegarde n'ont pas ete retestes par navigateur.",
          "- Les sept factures n'ont pas toutes ete rejouees avec Groq : huit tests OCR, trois tests extraction RAG/LLM.",
          "- Pas de validation exhaustive sur factures exclusivement arabes ou PDF multipages.",
          "- PP-StructureV3 n'est pas installe : la reconstruction utilise les cellules/coordonnees PaddleOCR, pas un modele de structure de tableaux.",
          "- Les lignes illisibles, les conflits non resolus et les quotas cloud doivent encore demander une revue humaine.",
          "- La facture anglaise sans totaux conserve des totaux null; sa date d'echeance anterieure a la date de facture reste un probleme du document, pas une valeur a inventer.", ""]
report = ocr_dir / "RAPPORT.md"
report.write_text("\n".join(lines), encoding="utf-8")
print(str(report))
