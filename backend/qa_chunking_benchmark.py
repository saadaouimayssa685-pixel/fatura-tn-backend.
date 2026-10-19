"""Phase 2B: offline, source-annotated retrieval evaluation; no app/DB/LLM imports."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
from statistics import mean
import time
from unittest.mock import patch

from extraction_chunks import image_layout_frame
from extraction_retrieval import ExtractionRetriever, FIELD_QUERIES, source_passages


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "qa-results/chunking-phase2b"
MODES = ("legacy", "layout")
KS = (1, 3, 5)
QUERY_KEYS = {
    "invoice_number": "invoice_number", "invoice_date": "date",
    "supplier_name": "vendor_name", "supplier_tax_id": "vendor_tax_id",
    "customer_name": "customer_name", "total_ht": "subtotal", "vat": "tax_amount",
    "stamp": "stamp_duty", "total_ttc": "total_amount", "net_to_pay": "total_amount",
    "currency": "currency", "payment_method": "payment_method", "line_items": "items",
}
CAUSES = {
    "A": "Lignes source absentes des chunks",
    "B": "Preuve complete presente mais non retournee assez haut par la politique actuelle",
    "C": "Bloc PP-DocLayout incorrect (necessite une verite terrain de layout)",
    "D": "Lignes presentes mais mal regroupees",
    "E": "Corps de ligne de tableau fragmente",
    "F": "Corps de ligne present mais en-tete non associe dans le meme chunk",
    "G": "Probleme bbox / coordonnees",
    "H": "Requete insuffisante (hypothese non isolee par cette experience)",
    "I": "Autre / indetermine",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprints():
    paths = [p for p in ROOT.glob("*.py") if not p.name.startswith(("test_", "qa_"))]
    paths += [p for p in (ROOT / ".env", ROOT / ".env.example") if p.exists()]
    return {p.name: sha(p.read_bytes()) for p in sorted(paths)}


def configured_mode():
    # Inspect only this setting, never expose other .env values or load the app.
    path = ROOT / ".env"
    values = []
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key.strip() == "RAG_CHUNKING_MODE":
                values.append(value.split("#", 1)[0].strip().strip("\"'"))
    return {"process_setting": os.getenv("RAG_CHUNKING_MODE"),
            "env_file_setting": values[-1] if values else None, "code_default": "legacy"}


def resolve_units(document, text):
    lines = text.splitlines()
    if sha(text.encode()) != document["ocr_text_sha256"]:
        raise ValueError(f"Frozen OCR changed: {document['document_id']}")
    resolved = {}
    for field, units in document["fields"].items():
        if field not in QUERY_KEYS or not units:
            raise ValueError(f"Unknown/empty annotated field: {field}")
        if len({u["id"] for u in units}) != len(units):
            raise ValueError("Duplicate evidence unit")
        resolved[field] = []
        for unit in units:
            proofs = []
            for ids in unit["alternatives"]:
                if (not ids or len(set(ids)) != len(ids)
                        or any(type(i) is not int or not 1 <= i <= len(lines) or not lines[i - 1].strip() for i in ids)):
                    raise ValueError(f"Invalid source lines: {field} {ids}")
                if field == "line_items" and not set(unit["header_line_ids"] + unit["body_line_ids"]).issubset(ids):
                    raise ValueError("Table proof must include its header and full body")
                proofs.append({"document_id": document["document_id"], "page": document["page"],
                               "line_ids": ids, "source_text": [lines[i - 1] for i in ids]})
            if not proofs:
                raise ValueError("No gold alternative")
            resolved[field].append({**unit, "alternatives": proofs})
    return resolved


def canonical_chunk(chunk, document_id):
    """Evaluation-only adapter for legacy spans; never changes text, order or score."""
    ids = chunk.get("line_ids")
    if ids is None:
        ids = list(range(chunk["global_line_start"], chunk["global_line_end"] + 1))
    return {**chunk, "document_id": chunk.get("document_id") or document_id,
            "page": chunk.get("page", chunk.get("page_number")), "line_ids": ids}


def contains(chunk, proof, body_only=None):
    ids = proof["line_ids"] if body_only is None else body_only
    return (chunk["document_id"] == proof["document_id"] and chunk["page"] == proof["page"]
            and set(ids).issubset(chunk["line_ids"]))


def first_rank(chunks, unit, body_only=False):
    body = unit.get("body_line_ids") if body_only else None
    return next((rank for rank, chunk in enumerate(chunks, 1)
                 if any(contains(chunk, proof, body) for proof in unit["alternatives"])), None)


def evaluate(chunks, units, body_only=False):
    ranks = [first_rank(chunks[:5], u, body_only) for u in units]
    return {"ranks": ranks, "recall": {str(k): sum(r is not None and r <= k for r in ranks) / len(ranks) for k in KS}}


def diagnose(chunks, unit):
    if first_rank(chunks, unit) is not None:
        return "B"
    for proof in unit["alternatives"]:
        relevant = [c for c in chunks if c["document_id"] == proof["document_id"] and c["page"] == proof["page"]]
        present = {i for c in relevant for i in c["line_ids"]}
        if set(proof["line_ids"]).issubset(present):
            if unit.get("body_line_ids"):
                return "F" if first_rank(chunks, unit, body_only=True) is not None else "E"
            return "D"
    return "A"


def aggregate(rows):
    result = {"fields": len(rows), "units": sum(len(r["units"]) for r in rows), "modes": {}}
    for mode in MODES:
        ranks = [rank for r in rows for rank in r[mode]["ranks"]]
        result["modes"][mode] = {
            "macro_recall": {str(k): mean(r[mode]["recall"][str(k)] for r in rows) for k in KS},
            "micro_recall": {str(k): sum(rank is not None and rank <= k for rank in ranks) / len(ranks) for k in KS},
        }
    return result


def recommend(result):
    metrics = result["metrics"]["global"]["modes"]
    old, new = (metrics[m]["macro_recall"] for m in MODES)
    gains = [new[str(k)] - old[str(k)] for k in KS]
    change = result["changes"]
    measured = (f"Recall@1 : {old['1']:.2%} -> {new['1']:.2%} ; "
                f"Recall@5 : {old['5']:.2%} -> {new['5']:.2%}. "
                f"{change['improvement_units']} unites gagnent en rang, "
                f"{change['regression_units']} regressent ({change['regressed_fields']} champs). ")
    if change["regression_units"] or any(g < 0 for g in gains) or not any(g > 0 for g in gains):
        return ("A. Conserver legacy par defaut a ce stade. " + measured
                + "Ces resultats ne demontrent pas un remplacement sans regression. Les gains de coherence ne suffisent pas a eux seuls. "
                "Une strategie hybride n'a pas ete mesuree : elle reste une piste, pas une solution validee. "
                "Garder layout en option et tester un lot independant plus large avant une bascule.")
    return ("B. Les metriques de ce lot favorisent layout par defaut. " + measured
            + "Cette recommandation reste limitee a sept factures deja utilisees en Phase 2 ; "
            "la confirmer sur un lot independant, multipage et avec acomptes avant generalisation. "
            "Aucune configuration n'est modifiee par ce benchmark.")


def chunk_stats(chunks, fields, mode):
    lengths = [len(c["text"]) for c in chunks]
    tables = [u for units in fields.values() for u in units if u.get("header_line_ids")]
    groups = {}
    for unit in tables:
        proof = unit["alternatives"][0]
        key = (proof["page"], tuple(unit["header_line_ids"]))
        groups.setdefault(key, []).append(unit)
    split_count, fragment_counts = 0, []
    for (page, header), units in groups.items():
        body = {i for u in units for i in u["body_line_ids"]}
        fragments = [c for c in chunks if c["page"] == page and body.intersection(c["line_ids"])]
        fragment_counts.append(len(fragments))
        # A table is fragmented only when no chunk contains its full annotated table.
        split_count += not any(set(header).union(body).issubset(c["line_ids"]) for c in fragments)
    return {"chunks": len(chunks), "mean_chars": mean(lengths), "max_chars": max(lengths),
            "totals_chunks": sum(c.get("section") == "totals" for c in chunks) if mode == "layout" else None,
            "table_chunks": sum(c.get("kind") == "table" or c.get("section") == "line_items" for c in chunks),
            "without_section": sum(not c.get("section") for c in chunks),
            "other_section": sum(c.get("section") == "other" for c in chunks),
            "without_bbox": sum(c.get("bbox") is None for c in chunks),
            "with_parent_block_id": sum(bool(c.get("parent_block_id")) for c in chunks),
            "with_layout_label": sum(bool(c.get("layout_label")) for c in chunks),
            "gold_tables": len(groups), "gold_tables_split": split_count,
            "table_overlapping_chunk_counts": fragment_counts,
            "body_units": len(tables),
            "body_units_with_header_in_catalog": sum(first_rank(chunks, u) is not None for u in tables)}


def inventory(annotations):
    baseline = ROOT / "qa-results/groq-multisource-20260912"
    manifest = read_json(baseline / "manifest.json")["selected"]
    by_id = {Path(item["entry"]).stem: item for item in manifest}
    datasets, records = {}, []
    for document in annotations["documents"] + annotations["excluded_documents"]:
        doc_id = document["document_id"]
        item = by_id[doc_id]
        ocr_path = ROOT / f"qa-results/paddle-small-20260913/{doc_id}.ocr.json"
        layout_path = baseline / f"{doc_id}.result.json"
        frozen, layout_result = read_json(ocr_path), read_json(layout_path)
        path = Path(item["local_path"])
        image_hash = sha(path.read_bytes())
        if len({image_hash, frozen["sha256"], layout_result["sha256"], item["sha256"]}) != 1:
            raise ValueError(f"OCR/layout/image hash mismatch: {doc_id}")
        if sha(frozen["ocr_text"].encode()) != document["ocr_text_sha256"]:
            raise ValueError(f"Annotated OCR changed: {doc_id}")
        pages = {line.get("page_number", 1) for line in frozen["ocr_metadata"]["lines"]}
        if pages != {document["page"]}:
            raise ValueError("Current annotations cover single physical pages only")
        layout = {**layout_result["classification"][2], **image_layout_frame(path)}
        records.append({"document_id": doc_id, "invoice": "fields" in document,
                        "origin": item["origin"], "image_sha256": image_hash,
                        "ocr_file": str(ocr_path), "layout_file": str(layout_path),
                        "ocr_file_sha256": sha(ocr_path.read_bytes()), "layout_file_sha256": sha(layout_path.read_bytes()),
                        "ocr_text_sha256": document["ocr_text_sha256"], "page": document["page"],
                        "lines": len(frozen["ocr_text"].splitlines()), "chars": len(frozen["ocr_text"]),
                        "layout_boxes": len(layout.get("layout_boxes", [])),
                        "tags": document.get("tags", []), "notes": document.get("notes", [document.get("reason")])})
        datasets[doc_id] = (frozen, layout)
    return records, datasets


def rank_label(rank):
    return str(rank) if rank is not None else ">5/absent"


def metric_table(group):
    lines = ["| Recall | Legacy | Layout |", "|---|---:|---:|"]
    for k in KS:
        lines.append(f"| @{k} | {group['modes']['legacy']['macro_recall'][str(k)]:.2%} | {group['modes']['layout']['macro_recall'][str(k)]:.2%} |")
    return lines


def write_reports(result, output):
    rows = result["rows"]
    global_metrics = result["metrics"]["global"]
    lines = ["# Phase 2B : benchmark du chunking", "", "## Resume", "",
             f"{len(result['documents'])} factures, {len(rows)} champs document/champ, {global_metrics['units']} unites de preuve.",
             f"{len(rows) * 2} evaluations annotees ; {result['counts']['distinct_evaluated_queries']} requetes document/champ distinctes sur les deux modes.",
             f"Le code existant calcule ses {len(FIELD_QUERIES)} requetes fixes a chaque appel : {result['counts']['retrieval_calls']} appels, {result['counts']['internal_queries']} classements internes.",
             "Le controle negatif 00540 est inventorie mais exclu des metriques positives. Aucun OCR, LLM ou classement de document n'est execute.", "",
             "## Protocole", "",
             "- Memes OCR, layout, requetes, top_k=5, BM25 k1=1.2/b=0.75, all-MiniLM-L6-v2 CPU local, RRF k=60. Seul chunking_mode change.",
             "- Le RRF est inchange. Les statistiques BM25 changent naturellement avec les chunks. Les candidats fournisseur/table forces par le code restent forces.",
             "- Items retourne jusqu'a 12 chunks de tableau en ordre source, meme avec top_k=5. On mesure les premiers 1/3/5 de cette sortie, sans les retrier ni completer.",
             "- Chaque preuve exige le bon document, la bonne page et tous ses line_ids dans UN chunk. Les alternatives explicites valent une seule unite, jamais une correspondance de nombre seul.",
             "- Recall principal = moyenne des fractions de preuves retrouvees par document/champ. Les produits comptent chacun une unite (en-tete + corps). Pour un champ scalaire, Recall = hit rate.",
             "- Rangs censures a 5 : >5/absent n'est pas un rang exact au-dela de 5. Aucune union entre chunks pour valider une preuve complete.",
             "- net_to_pay utilise la requete existante total_amount, sans en inventer une autre. Les deux champs restent deux evaluations de preuves distinctes.",
             "- Annotations manuelles figees avant scoring, empreinte OCR verifiee. Pas de seconde annotation humaine independante. Bbox non requise pour juger la preuve ; elle n'est jamais inventee.",
             "- Les cases non imprimees ou ambigues sont exclues : pas de HT/TVA calcules pour 00158, pas de fournisseur reconstruit depuis une URL, pas de client suppose depuis Ship to.",
             "- Reseau bloque dans le processus de benchmark ; modules application, DB, OCR et LLM non importes. Empreintes applicatives/configuration controlees avant/apres.", "",
             "## Resultats globaux", "", *metric_table(global_metrics), "",
             "### Sensibilite de la mesure", "",
             "Le score principal favorise volontairement les chunks coherents. Le corps seul retire cette contrainte pour les produits, sans changer les sorties du retrieval.", "",
             "| Mesure | Legacy @1/@3/@5 | Layout @1/@3/@5 |", "|---|---|---|"]
    micro = {m: global_metrics["modes"][m]["micro_recall"] for m in MODES}
    body = result["metrics"]["body_only_global"]
    lines.append("| Micro, chaque unite meme poids | " + " | ".join(" / ".join(f"{micro[m][str(k)]:.2%}" for k in KS) for m in MODES) + " |")
    lines.append("| Macro, produits corps seul | " + " | ".join(" / ".join(f"{body['modes'][m]['macro_recall'][str(k)]:.2%}" for k in KS) for m in MODES) + " |")
    scalar = aggregate([r for r in rows if r["field"] != "line_items"])
    products = result["metrics"]["by_field"]["line_items"]
    lost = [c for c in result["rank_changes"] if c["legacy_rank"] is not None and c["layout_rank"] is None]
    gained = [c for c in result["rank_changes"] if c["legacy_rank"] is None and c["layout_rank"] is not None]
    lines += ["", "### Interpretation", "",
              f"- Produits avec leur en-tete : {products['modes']['legacy']['micro_recall']['5']:.2%} -> {products['modes']['layout']['micro_recall']['5']:.2%} au Top 5 ({products['units']} produits).",
              f"- Champs hors produits : {scalar['modes']['legacy']['macro_recall']['5']:.2%} -> {scalar['modes']['layout']['macro_recall']['5']:.2%} au Top 5 ({scalar['fields']} champs).",
              "- Le micro-recall global augmente parce que chaque produit compte autant qu'un champ scalaire. Le macro-recall baisse parce que chaque document/champ, dont le tableau entier, a le meme poids. Les deux vues sont necessaires.",
              f"- {len(lost)} preuves trouvees dans le Top 5 legacy disparaissent du Top 5 layout ; {len(gained)} font le trajet inverse.",
              f"- {result['changes']['regressed_fields']}/{len(rows)} champs ({result['changes']['regressed_fields'] / len(rows):.2%}) ont au moins une regression de rang. Le TTC de 00158 n'est donc pas un cas isole.",
              "- Le paiement en especes de 00158 reste au rang 1 dans les deux modes : pas de gain de retrieval mesure sur ce moyen de paiement. L'echantillon de paiement reste tres petit."]
    for title, groups in (("Par champ", result["metrics"]["by_field"]), ("Par facture", result["metrics"]["by_document"])):
        lines += ["", f"## {title}", "", "| Groupe | N champs / unites | Legacy @1/@3/@5 | Layout @1/@3/@5 |", "|---|---:|---|---|"]
        for key, group in groups.items():
            scores = [" / ".join(f"{group['modes'][m]['macro_recall'][str(k)]:.2%}" for k in KS) for m in MODES]
            lines.append(f"| {key} | {group['fields']} / {group['units']} | {' | '.join(scores)} |")
    lines += ["", "## Statistiques des chunks", "",
              "Taille en caracteres Unicode, pas en tokens. N/A = metadonnee non produite en legacy, pas zero totals.",
              "Table divisee = aucun chunk ne couvre tous ses produits annotes avec leur en-tete ; une copie par chevauchement ne suffit pas a la declarer divisee.", "",
              "| Mesure | Legacy | Layout |", "|---|---:|---:|"]
    for key in result["chunk_summary"]["legacy"]:
        values = [result["chunk_summary"][m][key] for m in MODES]
        lines.append(f"| {key} | " + " | ".join("N/A" if v is None else str(round(v, 3)) if isinstance(v, float) else str(v) for v in values) + " |")
    lines += ["", "Details par document (chunks / taille moyenne-max / tables divisees / sans bbox) :", "",
              "| Facture | Legacy | Layout |", "|---|---|---|"]
    for document in result["documents"]:
        values = []
        for mode in MODES:
            stats = document["modes"][mode]["stats"]
            values.append(f"{stats['chunks']} / {stats['mean_chars']:.1f}-{stats['max_chars']} / {stats['gold_tables_split']} / {stats['without_bbox']}")
        lines.append(f"| {document['document_id']} | {' | '.join(values)} |")
    lines += ["", "## Ameliorations et regressions", "",
              "Comparaison du premier rang de chaque unite de preuve dans les 5 premieres sorties. Chaque tableau peut produire plusieurs cas, un par produit.", "",
              f"Resume : {result['changes']['improvement_units']} unites ameliorees, {result['changes']['regression_units']} degradees, {result['changes']['tied_units']} egales/censurees.",
              f"Document/champs concernes : {result['changes']['improved_fields']} avec au moins une amelioration ; {result['changes']['regressed_fields']} avec au moins une regression (non exclusifs).", ""]
    for title, direction in (("Regressions", "regression"), ("Ameliorations", "improvement")):
        lines += [f"### {title}", "", "| Facture | Champ / unite | Rang legacy | Rang layout | Diagnostic du mode moins bon |", "|---|---|---:|---:|---|"]
        for change in result["rank_changes"]:
            if change["direction"] == direction:
                lines.append(f"| {change['document_id']} | {change['field']} / {change['unit']} | {rank_label(change['legacy_rank'])} | {rank_label(change['layout_rank'])} | {change['cause']} |")
    lines += ["", "## Diagnostic des echecs", "",
              "Diagnostic mecanique sur les line_ids, pas une expertise automatique du layout. B inclut le filtrage et les candidats forces, pas seulement le score RRF.",
              "C et H ne peuvent pas etre attribues de facon causale ici : pas d'annotation independante des blocs ni de variation de requete. G est un controle de metadonnees separe, pas la cause supposee d'un mauvais rang.", "",
              "| Cause | Legacy echecs @5 | Layout echecs @5 |", "|---|---:|---:|"]
    for code, description in CAUSES.items():
        counts = [sum(d["mode"] == m and d["cause"] == code and 5 in d["failed_at"] for d in result["diagnostics"]) for m in MODES]
        values = [str(n) for n in counts] if code not in {"C", "G", "H"} else ["Non attribue", "Non attribue"]
        lines.append(f"| {code}: {description} | {values[0]} | {values[1]} |")
    lines += ["", "Tous les echecs @1/@3/@5 (une ligne par unite/mode, plusieurs seuils possibles) :", "",
              "| Facture | Champ / unite | Mode | Seuils echoues | Cause |", "|---|---|---|---|---|"]
    for diag in result["diagnostics"]:
        lines.append(f"| {diag['document_id']} | {diag['field']} / {diag['unit']} | {diag['mode']} | {diag['failed_at']} | {diag['cause']} |")
    lines += ["", "## Limites", "",
              "Sept factures seulement, dont deux synthetiques anglaises. Pas de document multipage complet ni d'acompte exploitable. HAMMAMI est uniquement l'image marquee 2/2, pas un test multipage.",
              "Une seule facture annotee pour le paiement et une pour le net a payer. Les scans FR/AR et les tableaux alternes sont couverts, pas toutes les structures. Aucune precision des montants extraits par LLM n'est mesuree.",
              "Les lignes OCR decalees de 00791 sont conservees telles quelles. Aucune correction OCR/LLM/regle n'a ete injectee pour aider un mode. Le corpus a deja servi aux investigations Phase 2 : ce n'est pas un jeu de generalisation jamais vu.",
              "Les changements de rang ne sont pas des differences statistiquement etablies sur toutes les factures. Le gain de metadonnees ne garantit pas le bon classement.", "",
              "## Decision", "", result["recommendation"], "",
              "Configuration non modifiee : RAG_CHUNKING_MODE=legacy reste le defaut.", "",
              "## Artefacts", "",
              "- annotations.json : annotations manuelles et exclusions.",
              "- resolved_annotations.json : preuves explicites document/page/line_ids et texte source.",
              "- inventory.json : provenance, empreintes OCR/layout/image, limites par facture.",
              "- results.json : metriques, rangs, diagnostics, chunks complets, Top 5 (sortie items complete jusqu'a 12), scores disponibles et configuration.",
              "- EXAMPLES_TOP3.md : Top 3 pour chacun des champs annotes, pas seulement les exemples favorables.",
              "- INTEGRITY.json : empreintes applicatives/configuration avant/apres.", "",
              "Reproduction : `python qa_chunking_benchmark.py --output qa-results/chunking-phase2b-replay` (MiniLM doit deja etre en cache).", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    examples = ["# Top 3 exhaustifs : legacy / layout", "", "Les line_ids, pas la valeur seule, definissent les preuves attendues. Scores absents = candidat force sans score expose.", ""]
    for row in rows:
        examples += [f"## {row['document_id']} / {row['field']}", "", f"Requete : {row['query']}", "", "Preuves :"]
        for unit in row["units"]:
            examples.append(f"- {unit['id']} : " + " OU ".join(f"p{p['page']} L{p['line_ids']}" for p in unit["alternatives"]))
        for mode in MODES:
            examples += ["", f"### {mode}", ""]
            for rank, chunk in enumerate(row[mode]["retrieved"][:3], 1):
                supported = [u["id"] for u in row["units"] if first_rank([chunk], u) == 1]
                examples += [f"{rank}. {chunk.get('chunk_id', chunk['source_id'])} | page {chunk['page']} | L{chunk['line_ids']} | section {chunk.get('section', 'N/A')} | kind {chunk.get('kind')} | unites couvertes {supported}",
                             f"   BM25 {chunk.get('lexical_score', 'N/A')} ; semantic {chunk.get('semantic_score', 'N/A')} ; RRF {chunk.get('score', 'N/A')} ; bbox {chunk.get('bbox')} ; parent {chunk.get('parent_block_id')}",
                             "", "```text", chunk["text"], "```", ""]
            if not row[mode]["retrieved"]:
                examples.append("Aucun passage retourne.")
    (output / "EXAMPLES_TOP3.md").write_text("\n".join(examples), encoding="utf-8")


def run(output, annotation_path):
    before, setting = fingerprints(), configured_mode()
    annotations = read_json(annotation_path)
    if annotations["schema_version"] != 1:
        raise ValueError("Unknown annotation schema")
    inv, datasets = inventory(annotations)
    resolved = {d["document_id"]: resolve_units(d, datasets[d["document_id"]][0]["ocr_text"]) for d in annotations["documents"]}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "annotations.json").resolve() != annotation_path.resolve():
        (output / "annotations.json").write_bytes(annotation_path.read_bytes())
    # This snapshot is written before model loading and any ranking computation.
    (output / "resolved_annotations.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"schema_version": 1, "annotation_sha256": sha(annotation_path.read_bytes()),
              "parameters": {"top_k": 5, "model": "sentence-transformers/all-MiniLM-L6-v2",
                             "bm25_k1": 1.2, "bm25_b": .75, "rrf_k": 60, "device": "cpu",
                             "queries": FIELD_QUERIES, "query_mapping": QUERY_KEYS},
              "python": platform.python_version(), "setting": setting,
              "ocr_runs": 0, "llm_calls": 0, "database_writes": 0, "network_blocked": True,
              "documents": [], "rows": [], "diagnostics": [], "rank_changes": []}
    retriever = ExtractionRetriever()
    started = time.monotonic()
    with retriever.lock:
        retriever._load()
    if retriever.model is None:
        raise RuntimeError("Cached MiniLM unavailable; refusing lexical-only comparison")
    retriever.model.encode(["facture"], normalize_embeddings=True)
    result["warmup_seconds"] = round(time.monotonic() - started, 3)
    result["backend"] = retriever.backend
    for annotation in annotations["documents"]:
        doc_id = annotation["document_id"]
        frozen, layout = datasets[doc_id]
        fields = resolved[doc_id]
        doc = {"document_id": doc_id, "modes": {}}
        for mode in MODES:
            started = time.monotonic()
            evidence = retriever.retrieve(frozen["ocr_text"], top_k=5, ocr_metadata=frozen["ocr_metadata"],
                                          layout_metadata=layout, document_id=doc_id, chunking_mode=mode)
            if retriever.model is None or retriever.backend != "bm25+sentence_transformers":
                raise RuntimeError("Semantic fallback during comparison; invalid benchmark")
            raw = evidence.passages if hasattr(evidence, "passages") else source_passages(frozen["ocr_text"])
            chunks = [canonical_chunk(c, doc_id) for c in raw]
            selected = {field: [canonical_chunk(c, doc_id) for c in evidence[QUERY_KEYS[field]]] for field in fields}
            doc["modes"][mode] = {"chunks": chunks, "selected": selected,
                                   "seconds": round(time.monotonic() - started, 3),
                                   "stats": chunk_stats(chunks, fields, mode)}
            print(json.dumps({"document": doc_id, "mode": mode, "chunks": len(chunks),
                              "seconds": doc["modes"][mode]["seconds"]}), flush=True)
        result["documents"].append(doc)
        for field, units in fields.items():
            row = {"document_id": doc_id, "field": field, "query": FIELD_QUERIES[QUERY_KEYS[field]], "units": units}
            for mode in MODES:
                chunks = doc["modes"][mode]["chunks"]
                selected = doc["modes"][mode]["selected"][field]
                row[mode] = {**evaluate(selected, units), "body_only": evaluate(selected, units, body_only=True), "retrieved": selected}
                for unit, rank in zip(units, row[mode]["ranks"]):
                    failed = [k for k in KS if rank is None or rank > k]
                    if failed:
                        result["diagnostics"].append({"document_id": doc_id, "field": field, "unit": unit["id"],
                            "mode": mode, "rank": rank, "failed_at": failed, "cause": diagnose(chunks, unit)})
            for unit, old, new in zip(units, row["legacy"]["ranks"], row["layout"]["ranks"]):
                if (old or 6) != (new or 6):
                    direction = "improvement" if (new or 6) < (old or 6) else "regression"
                    worse = "legacy" if direction == "improvement" else "layout"
                    result["rank_changes"].append({"document_id": doc_id, "field": field, "unit": unit["id"],
                        "legacy_rank": old, "layout_rank": new, "direction": direction,
                        "cause": diagnose(doc["modes"][worse]["chunks"], unit)})
            result["rows"].append(row)
    rows = result["rows"]
    result["counts"] = {"retrieval_calls": len(result["documents"]) * 2,
                        "internal_queries": len(result["documents"]) * 2 * len(FIELD_QUERIES),
                        "distinct_evaluated_queries": sum(len({QUERY_KEYS[f] for f in fields}) for fields in resolved.values()) * 2}
    body_rows = [{**r, **{m: r[m]["body_only"] for m in MODES}} for r in rows]
    result["metrics"] = {"global": aggregate(rows), "body_only_global": aggregate(body_rows),
        "by_field": {key: aggregate([r for r in rows if r["field"] == key]) for key in QUERY_KEYS if any(r["field"] == key for r in rows)},
        "by_document": {d["document_id"]: aggregate([r for r in rows if r["document_id"] == d["document_id"]]) for d in result["documents"]}}
    result["chunk_summary"] = {}
    for mode in MODES:
        stats = [d["modes"][mode]["stats"] for d in result["documents"]]
        chunks = [c for d in result["documents"] for c in d["modes"][mode]["chunks"]]
        summary = {"mean_chunks_per_invoice": mean(s["chunks"] for s in stats), "total_chunks": len(chunks),
                   "mean_chunk_chars": mean(len(c["text"]) for c in chunks), "max_chunk_chars": max(len(c["text"]) for c in chunks)}
        for key in ("totals_chunks", "table_chunks", "without_section", "other_section", "without_bbox", "with_parent_block_id", "with_layout_label", "gold_tables", "gold_tables_split", "body_units", "body_units_with_header_in_catalog"):
            summary[key] = None if any(s[key] is None for s in stats) else sum(s[key] for s in stats)
        summary["table_split_frequency"] = summary["gold_tables_split"] / summary["gold_tables"]
        result["chunk_summary"][mode] = summary
    changes = result["rank_changes"]
    improved = [c for c in changes if c["direction"] == "improvement"]
    regressed = [c for c in changes if c["direction"] == "regression"]
    result["changes"] = {"improvement_units": len(improved), "regression_units": len(regressed),
                         "tied_units": result["metrics"]["global"]["units"] - len(changes),
                         "improved_fields": len({(c["document_id"], c["field"]) for c in improved}),
                         "regressed_fields": len({(c["document_id"], c["field"]) for c in regressed})}
    result["recommendation"] = recommend(result)
    after = fingerprints()
    if after != before or sha(annotation_path.read_bytes()) != result["annotation_sha256"]:
        raise RuntimeError("Application/configuration/annotations changed during benchmark; results not finalized")
    result["application_unchanged"] = True
    (output / "INTEGRITY.json").write_text(json.dumps({"before": before, "after": after, "equal": True}, indent=2), encoding="utf-8")
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(result, output)
    print(json.dumps({"global": result["metrics"]["global"], "changes": result["changes"], "output": str(output)}), flush=True)


def offline_run(output, annotations):
    def deny_network(*args, **kwargs):
        raise RuntimeError("Network disabled for Phase 2B benchmark")

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"EXTRACTION_SEMANTIC_RAG": "true", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}))
        stack.enter_context(patch.object(socket.socket, "connect", deny_network))
        stack.enter_context(patch.object(socket.socket, "connect_ex", deny_network))
        run(output, annotations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--annotations", type=Path, default=OUTPUT / "annotations.json")
    args = parser.parse_args()
    offline_run(args.output.resolve(), args.annotations.resolve())
