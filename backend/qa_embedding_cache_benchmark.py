"""Phase 3B: CPU chunk-only cache, fresh query embeddings, no production changes."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
from statistics import mean, median
import subprocess
import sys
import time
from unittest.mock import patch
from uuid import uuid4

from qa_embedding_benchmark import (ROOT, PHASE2, OUTPUT as PHASE3, MemorySampler,
    VectorCache, digest, integrity, load_dataset, metrics, model_identity, read_json, write_json)
from qa_chunk_embedding_cache import ChunkCachingEncoder


OUTPUT = ROOT / "qa-results/embedding-phase3b"
MODELS = ("minilm", "bge-m3")
WARM_REPEATS = 3


def protected_snapshot():
    current = integrity()
    approved = read_json(PHASE3 / "INTEGRITY.json")
    if current != approved["after"] or approved["before"] != approved["after"]:
        raise ValueError("Protected Phase 3 inputs/application settings changed")
    paths = [ROOT / name for name in ("extraction_retrieval.py", "extraction_embeddings.py",
        "extraction_chunks.py", "qa_embedding_benchmark.py", "qa_chunking_benchmark.py", ".env", ".env.example")]
    paths += [p for p in PHASE2.rglob("*") if p.is_file()]
    paths += [p for p in PHASE3.rglob("*") if p.is_file()]
    return {"phase3_integrity": current, "files": {str(p.relative_to(ROOT)): digest(p.read_bytes())
            for p in paths if p.exists()}}


def assert_same_evidence(actual, expected):
    """Stricter than Top 5: every returned field, passage, metadata and score."""
    if actual != expected:
        different = [field for field in set(actual) | set(expected) if actual.get(field) != expected.get(field)]
        raise ValueError(f"Phase 3 retrieval output changed: {different}")
    return digest(json.dumps(actual, sort_keys=True, ensure_ascii=False).encode())


def compare_vectors(actual, expected):
    import numpy as np

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Phase 3 vector shape/precision changed")
    if not np.allclose(actual, expected, atol=1e-7, rtol=1e-6):
        raise ValueError("Phase 3 vectors are not numerically equivalent")
    return {"bit_exact": bool(np.array_equal(actual, expected)),
            "max_absolute_difference": float(np.max(np.abs(actual - expected))),
            "atol": 1e-7, "rtol": 1e-6}


def memory_record(sample):
    return {"baseline_rss_bytes": sample.baseline, "peak_rss_bytes": sample.peak,
            "peak_increment_bytes": sample.peak - sample.baseline,
            "minimum_system_available_bytes": sample.min_available, "sampling_interval_ms": 50}


def aggregate(records):
    keys = ("retrieval_seconds", "chunk_embedding_seconds", "query_embedding_seconds",
            "cache_read_seconds", "cache_write_seconds", "other_seconds")
    return {"calls": len(records), **{key: {"mean": mean(r[key] for r in records),
            "median": median(r[key] for r in records), "min": min(r[key] for r in records),
            "max": max(r[key] for r in records)} for key in keys}}


def observed_peak_rss(run):
    """Independent 50 ms samplers can catch different peaks; retain all observations."""
    samples = [run["memory"], run["initialization"]["memory"]]
    samples += [call["memory"] for doc in run["documents"] for call in [doc["cold"], *doc["warm"]]]
    return max(sample["peak_rss_bytes"] for sample in samples)


def worker(alias, run_id, repeats):
    from extraction_retrieval import ExtractionRetriever, FIELD_QUERIES, source_passages
    from qa_chunking_benchmark import QUERY_KEYS, canonical_chunk, evaluate

    _, previous, _, data, gold = load_dataset()
    reference = read_json(PHASE3 / f"{alias}.json")
    if (reference["queries"] != FIELD_QUERIES or reference["top_k"] != 5
            or reference["chunking_mode"] != "legacy"
            or reference["annotation_sha256"] != previous["annotation_sha256"]):
        raise ValueError("Phase 3 protocol differs")
    by_doc = {d["document_id"]: d for d in reference["documents"]}
    chunks = {doc: source_passages(data[doc][0]["ocr_text"]) for doc in gold}
    if any(chunks[doc] != by_doc[doc]["chunks"] for doc in gold):
        raise ValueError("Legacy chunks changed")
    # A unique namespace makes all measured COLD calls real misses without deleting any cache.
    cache_root = OUTPUT / "runs" / run_id / "vectors" / alias
    if cache_root.exists():
        raise ValueError("Cold benchmark namespace already exists")
    result = {"model": alias, "run_id": run_id, "process_id": os.getpid(), "documents": [],
        "rows": [], "warm_rows": [], "queries": FIELD_QUERIES, "top_k": 5,
        "chunking_mode": "legacy", "annotation_sha256": reference["annotation_sha256"],
        "warm_repeats": repeats, "llm_calls": 0, "ocr_calls": 0, "database_writes": 0,
        "network_blocked": True, "query_cache_enabled": False, "torch_threads": 4,
        "started_at": datetime.now(timezone.utc).isoformat()}
    retriever = ExtractionRetriever(alias)
    with MemorySampler() as memory:
        started = time.perf_counter()
        print(json.dumps({"event": "loading", "model": alias}), flush=True)
        with MemorySampler() as load_memory:
            import torch

            torch.set_num_threads(4)
            runtime_seconds = time.perf_counter() - started
            load_start = time.perf_counter()
            retriever._load()
            load_seconds = time.perf_counter() - load_start
        if retriever.model is None:
            raise RuntimeError("Embedding model unavailable; BM25 fallback is forbidden here")
        result["initialization"] = {"runtime_import_seconds": runtime_seconds,
            "model_load_seconds_including_sentence_transformers_import": load_seconds,
            "total_seconds": runtime_seconds + load_seconds, "memory": memory_record(load_memory)}
        identity = model_identity(retriever.model, alias)
        if identity != reference["identity"]:
            raise ValueError("Model/revision/configuration differs from Phase 3")
        result["identity"] = identity
        reference_path = Path(reference["vector_cache_path"])
        if not reference_path.is_dir() or read_json(reference_path / "identity.json") != identity:
            raise ValueError("Phase 3 reference vectors missing")
        reference_cache = VectorCache(PHASE3 / "vectors", identity)
        if reference_cache.path != reference_path:
            raise ValueError("Phase 3 reference cache path mismatch")
        queries = list(FIELD_QUERIES.values())
        reference_queries = reference_cache.get(queries)
        encoder = ChunkCachingEncoder(retriever.model, cache_root, identity, queries)
        retriever.model = encoder
        print(json.dumps({"event": "loaded", "model": alias,
                          "initialization_seconds": result["initialization"]["total_seconds"]}), flush=True)

        def retrieve_document(doc_id, stage, repeat):
            texts = [c["text"] for c in chunks[doc_id]]
            with MemorySampler() as sample:
                started = time.perf_counter()
                encoder.begin_document(doc_id, texts, "legacy")
                evidence = retriever.retrieve(data[doc_id][0]["ocr_text"], top_k=5,
                    document_id=doc_id, chunking_mode="legacy")
                elapsed = time.perf_counter() - started
            if retriever.model is not encoder or encoder.call_index != 2 or len(encoder.events) != 2:
                raise RuntimeError("Unexpected fallback/encode calls during benchmark")
            q, c = encoder.events
            if q["cache_hit"] or c["cache_hit"] != (stage == "warm"):
                raise ValueError("COLD/WARM cache contract violated")
            # Validation and reference I/O occur OUTSIDE measured retrieval time.
            comparison = {"queries": compare_vectors(encoder.arrays["queries"], reference_queries),
                "chunks": compare_vectors(encoder.arrays["chunks"], reference_cache.get(texts))}
            output_hash = assert_same_evidence(evidence, by_doc[doc_id]["all_results"])
            measured = {"stage": stage, "repeat": repeat, "retrieval_seconds": elapsed,
                "query_embedding_seconds": q["embedding_seconds"],
                "chunk_embedding_seconds": c["embedding_seconds"],
                "cache_read_seconds": c["cache_read_seconds"],
                "cache_write_seconds": c["cache_write_seconds"], "memory": memory_record(sample),
                "events": encoder.events, "phase3_vectors": comparison,
                "phase3_output_exact": True, "all_results_sha256": output_hash,
                "chunk_cache_path": str(encoder.cache.path)}
            measured["other_seconds"] = elapsed - sum(measured[k] for k in (
                "query_embedding_seconds", "chunk_embedding_seconds", "cache_read_seconds", "cache_write_seconds"))
            print(json.dumps({"event": "retrieval", "model": alias, "document": doc_id,
                "stage": stage, "repeat": repeat, "total_seconds": round(elapsed, 3),
                "queries_seconds": round(q["embedding_seconds"], 3),
                "chunks_seconds": round(c["embedding_seconds"], 3), "phase3_exact": True}), flush=True)
            return measured, evidence

        for doc_id, fields in gold.items():
            cold, evidence = retrieve_document(doc_id, "cold", 0)
            doc = {"document_id": doc_id, "chunk_count": len(chunks[doc_id]), "cold": cold,
                   "warm": [], "all_results": evidence}
            result["documents"].append(doc)
            for field, units in fields.items():
                passages = [canonical_chunk(c, doc_id) for c in evidence[QUERY_KEYS[field]]]
                result["rows"].append({"document_id": doc_id, "field": field, "units": units,
                    "query": FIELD_QUERIES[QUERY_KEYS[field]], **evaluate(passages, units), "retrieved": passages})

        # Three passes, fixed original document order, queries deliberately not cached.
        for repeat in range(1, repeats + 1):
            for doc in result["documents"]:
                doc_id = doc["document_id"]
                measured, evidence = retrieve_document(doc_id, "warm", repeat)
                doc["warm"].append(measured)
                if repeat == 1:
                    for field, units in gold[doc_id].items():
                        passages = [canonical_chunk(c, doc_id) for c in evidence[QUERY_KEYS[field]]]
                        result["warm_rows"].append({"document_id": doc_id, "field": field,
                            "units": units, "query": FIELD_QUERIES[QUERY_KEYS[field]],
                            **evaluate(passages, units), "retrieved": passages})
    if result["rows"] != reference["rows"] or result["warm_rows"] != reference["rows"]:
        raise ValueError("Annotated results/ranks/recall differ from Phase 3")
    result["memory"] = memory_record(memory)
    result["cache_bytes"] = sum(p.stat().st_size for p in cache_root.rglob("*") if p.is_file())
    result["cache_matrix_files"] = len(list(cache_root.rglob("*.npy")))
    if result["cache_matrix_files"] != len(gold):
        raise ValueError("Expected exactly one matrix per document, no query cache")
    result["performance"] = {"cold": aggregate([d["cold"] for d in result["documents"]]),
        "warm": aggregate([r for d in result["documents"] for r in d["warm"]])}
    result["phase3_exact"] = True
    result["metrics"] = metrics(result["rows"])
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(OUTPUT / "runs" / run_id / f"{alias}.json", result)
    print(json.dumps({"event": "complete", "model": alias,
        "performance": result["performance"], "memory": result["memory"]}), flush=True)


def offline_worker(alias, run_id, repeats):
    def deny(*args, **kwargs):
        raise RuntimeError("Network forbidden in Phase 3B")

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"EXTRACTION_SEMANTIC_RAG": "true",
            "RAG_EMBEDDING_MODEL": alias, "RAG_CHUNKING_MODE": "legacy", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}))
        stack.enter_context(patch.object(socket.socket, "connect", deny))
        stack.enter_context(patch.object(socket.socket, "connect_ex", deny))
        worker(alias, run_id, repeats)


def make_report(run_id):
    directory = OUTPUT / "runs" / run_id
    audit = read_json(directory / "INTEGRITY.json")
    if not audit["equal"] or audit["before"] != audit["after"]:
        raise ValueError("Integrity not verified")
    runs = {alias: read_json(directory / f"{alias}.json") for alias in MODELS}
    if any(not r["phase3_exact"] or r["query_cache_enabled"] for r in runs.values()):
        raise ValueError("Invalid benchmark result")
    # Report measured tradeoffs, not an invented universal usability threshold.
    bge = runs["bge-m3"]["performance"]
    small = runs["minilm"]["performance"]
    reduction = 1 - bge["warm"]["retrieval_seconds"]["mean"] / bge["cold"]["retrieval_seconds"]["mean"]
    ratio = bge["warm"]["retrieval_seconds"]["mean"] / small["warm"]["retrieval_seconds"]["mean"]
    conclusion = (f"C. BGE-M3 seulement en mode haute qualite explicite pour le moment. "
        f"Le cache de chunks reduit le temps moyen BGE de {reduction:.1%}, mais le warm reste "
        f"{ratio:.1f} fois le warm MiniLM sur ce lot. MiniLM reste le defaut interactif ; "
        "ne pas generaliser ces temps a d'autres tailles de documents. Le routage par champ n'a pas ete teste.")
    summary = {"run_id": run_id, "dataset": {"documents": 7, "fields": 63, "units": 82},
        "models": {a: {"performance": r["performance"], "metrics": r["metrics"],
            "initialization": r["initialization"], "memory": r["memory"],
            "observed_peak_rss_bytes": observed_peak_rss(r), "cache_bytes": r["cache_bytes"]}
            for a, r in runs.items()}, "phase3_exact": True, "recommendation": conclusion,
        "llm_calls": 0, "ocr_calls": 0, "database_writes": 0}
    write_json(directory / "summary.json", summary)
    lines = ["# Phase 3B : cache experimental des chunks", "", "## Perimetre", "",
        "7 factures, 63 champs, 82 unites de preuve. Memes OCR figes, legacy, requetes, BM25, RRF, top_k=5, seuils et regles de selection que Phase 3.",
        "Aucun LLM/OCR execute, aucun endpoint appele, aucune ecriture en base applicative. Aucun changement du retriever applicatif ni du chatbot. MiniLM et legacy restent les defauts.", "",
        "## Diagnostic du recalcul", "",
        "`ExtractionRetriever.retrieve()` charge le modele via `_load()` puis appelle `model.encode(FIELD_QUERIES.values(), normalize_embeddings=True)` et `model.encode(textes_des_chunks, normalize_embeddings=True)` a CHAQUE retrieval. Les 22 requetes et tous les chunks sont recalcules, meme pour un document identique. `extraction_embeddings._models` ne cache que les objets modeles charges, pas les vecteurs.",
        "En Phase 3 les temps 'cached' reutilisaient requetes ET chunks. Ils ne repondent donc pas a la question presente. En Phase 3B seules les matrices de chunks sont cachees : les 22 requetes sont recalculees a chaque appel, y compris warm.", "",
        "## Protocole", "",
        "- Deux processus CPU separes, modeles offline deja telecharges, float32, normalisation L2, 4 threads torch, batch_size=32 et configurations natives Phase 3 inchangees.",
        "- Chaque modele charge une fois. COLD = premiere indexation de chacun des 7 documents dans un repertoire unique vide ; le chargement global est mesure separement, pas repete artificiellement pour chaque document.",
        "- WARM = meme modele deja charge et cache chunks sur disque, puis trois passes de 7 documents. Aucun prechauffage d'inference cache n'est exclu du COLD. Aucun cache des requetes.",
        "- Temps retrieval = creation/verifications du cache, encode requetes, encode/ecriture chunks (cold) ou lecture/validation chunks (warm), et code retrieval complet. Comparaison aux sorties Phase 3 et lecture de ses matrices de reference hors chronometrage.",
        "- Le cache persistant est relu depuis le disque a chaque appel ; le cache disque de Windows n'est pas vide. 'Cold' ne signifie pas disque/OS froid ni telechargement initial.",
        "- RAM = RSS echantillonnee chaque 50 ms, estimation, processus de benchmark uniquement. L'application reste ouverte. Une passe cold et trois warm ne constituent pas un benchmark de charge.", "",
        "## Temps et ressources", "", "| Mesure | MiniLM | BGE-M3 |", "|---|---:|---:|"]
    values = [
        ("Import runtime torch (s)", lambda r: r["initialization"]["runtime_import_seconds"]),
        ("Chargement modele + imports ST (s)", lambda r: r["initialization"]["model_load_seconds_including_sentence_transformers_import"]),
        ("Initialisation totale une fois (s)", lambda r: r["initialization"]["total_seconds"]),
        ("COLD retrieval / document, hors chargement (s)", lambda r: r["performance"]["cold"]["retrieval_seconds"]["mean"]),
        ("COLD chunks / document (s)", lambda r: r["performance"]["cold"]["chunk_embedding_seconds"]["mean"]),
        ("COLD requetes / document (s)", lambda r: r["performance"]["cold"]["query_embedding_seconds"]["mean"]),
        ("COLD ecriture cache / document (ms)", lambda r: 1000*r["performance"]["cold"]["cache_write_seconds"]["mean"]),
        ("WARM retrieval / document (s)", lambda r: r["performance"]["warm"]["retrieval_seconds"]["mean"]),
        ("WARM requetes / document (s)", lambda r: r["performance"]["warm"]["query_embedding_seconds"]["mean"]),
        ("WARM chunks embedding / document (s)", lambda r: r["performance"]["warm"]["chunk_embedding_seconds"]["mean"]),
        ("WARM lecture/validation cache (ms)", lambda r: 1000*r["performance"]["warm"]["cache_read_seconds"]["mean"]),
        ("Premier document + initialisation (s)", lambda r: r["initialization"]["total_seconds"] + r["documents"][0]["cold"]["retrieval_seconds"]),
        ("COLD moyen avec chargement amorti sur 7 (s)", lambda r: r["initialization"]["total_seconds"]/7 + r["performance"]["cold"]["retrieval_seconds"]["mean"]),
        ("RSS pic observe, toutes sondes (MiB)", lambda r: observed_peak_rss(r)/2**20),
        ("RSS pic COLD (MiB)", lambda r: max(d["cold"]["memory"]["peak_rss_bytes"] for d in r["documents"])/2**20),
        ("RSS pic WARM (MiB)", lambda r: max(w["memory"]["peak_rss_bytes"] for d in r["documents"] for w in d["warm"])/2**20),
        ("Cache chunks + metadata, 7 documents (KiB)", lambda r: r["cache_bytes"]/1024)]
    for title, getter in values:
        lines.append(f"| {title} | " + " | ".join(f"{getter(runs[a]):.3f}" for a in MODELS) + " |")
    lines += ["", "Les sondes RAM globales et par etape sont independantes. Le pic observe retient le maximum de toutes leurs mesures, pas uniquement la sonde globale ; les mesures brutes sont conservees dans les JSON.",
        "Les temps requetes couvrent le LOT de 22 requetes, pas une requete isolee. Les modeles passent sequentiellement : les imports, le cache OS, la charge et la temperature du CPU peuvent biaiser leurs temps de chargement. Les comparaisons cold/warm sont internes a chaque processus.",
        "", "## Temps par document", "", "WARM = moyenne de trois passages ; min/max disponibles dans les fichiers JSON.", "",
        "| Document | Chunks | MiniLM cold (s) | MiniLM warm (s) | BGE cold (s) | BGE warm (s) |", "|---|---:|---:|---:|---:|---:|"]
    for i, doc in enumerate(runs["minilm"]["documents"]):
        big = runs["bge-m3"]["documents"][i]
        lines.append(f"| {doc['document_id']} | {doc['chunk_count']} | {doc['cold']['retrieval_seconds']:.3f} | "
            f"{mean(w['retrieval_seconds'] for w in doc['warm']):.3f} | {big['cold']['retrieval_seconds']:.3f} | "
            f"{mean(w['retrieval_seconds'] for w in big['warm']):.3f} |")
    lines += ["", "## Qualite strictement preservee", "", "| Recall macro | MiniLM Phase 3 = cold = warm | BGE-M3 Phase 3 = cold = warm |", "|---|---:|---:|"]
    for k in (1, 3, 5):
        lines.append(f"| @{k} | " + " | ".join(f"{runs[a]['metrics']['macro_recall'][str(k)]:.2%}" for a in MODELS) + " |")
    lines += ["", "56 appels controles (2 modeles x 7 documents x 4 passages), chaque sortie comparee strictement a la Phase 3 : ordre, metadonnees, textes et scores exposes identiques. Controle plus fort que Top 5 : toutes les sorties, y compris les tableaux forces jusqu'a 12 passages. Les 63 lignes QA cold et warm sont egalement identiques aux lignes Phase 3 pour chaque modele.",
        "Les matrices sont comparees aux vecteurs Phase 3 (atol=1e-7, rtol=1e-6), plus verification bit-a-bit lorsqu'elle reussit. La tolerance des matrices n'autorise aucun changement du classement ni des scores exposes.", ""]
    for alias, run in runs.items():
        checks = [c for d in run["documents"] for r in [d["cold"], *d["warm"]] for c in r["phase3_vectors"].values()]
        lines.append(f"- {alias}: {sum(c['bit_exact'] for c in checks)}/{len(checks)} matrices bit-a-bit identiques ; difference absolue maximale {max(c['max_absolute_difference'] for c in checks):.9g}.")
    lines += ["", "## Isolation et invalidation", "",
        "`qa_chunk_embedding_cache.ChunkEmbeddingCache` est uniquement appele par ce benchmark. Aucun import depuis la production. Repertoires `runs/<run_id>/vectors/<modele>/...` distincts des caches Phase 3, du chatbot et de la base applicative.",
        "Cle complete = namespace SHA256 de l'identite (document_id, modele/revision, dimension, normalisation, chunking_mode, dtype, versions/configuration et schema) + SHA256 de la liste ordonnee des textes. Changement de texte/ordre/document/mode/modele/revision/configuration => autre entree.",
        "Lecture : verification de l'identite complete, nombre de lignes, hash des textes, SHA256 du fichier NPY, forme, float32, finitude et norme L2. Pickle desactive. Cache incomplet ou corrompu refuse, jamais utilise en silence. MiniLM 384D ne peut pas lire BGE 1024D. Aucune suppression des caches precedents.", "",
        "## Conclusion", "", conclusion, "",
        "Le gain porte sur les relectures du MEME document : la premiere facture importee conserve le cout des embeddings de chunks. L'encodage des requetes reste paye pour chaque document. Un cache de requetes serait une experience distincte, non activee ici.",
        "Aucun routage, reranker, quantification, nouveau backend, nouveau modele, modification des threads par rapport a Phase 3 ni activation du cache dans l'application. Ce lot deja connu ne valide pas la qualite d'extraction finale ni la generalisation a d'autres factures.", "",
        "## Reproduction", "",
        "`python qa_embedding_cache_benchmark.py --run` : deux processus offline, nouveaux caches uniques, 3 passes warm.",
        f"`python qa_embedding_cache_benchmark.py --report --run-id {run_id}` : rapport seul.",
        "Dependances deja installees : PyTorch CPU, sentence-transformers, numpy, psutil. Aucun telechargement ni installation.",
        "Artefacts : minilm.json, bge-m3.json, summary.json, INTEGRITY.json, hardware.json et vectors dans le dossier de run. Les Top 5 complets et hashes de sorties sont conserves par document.", ""]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUTPUT / "latest.json", {"run_id": run_id, "report": str(directory / "REPORT.md")})
    print(json.dumps({"event": "report", "path": str(directory / "REPORT.md"),
                      "recommendation": conclusion}), flush=True)


def run_all():
    import psutil

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    directory = OUTPUT / "runs" / run_id
    directory.mkdir(parents=True)
    before = protected_snapshot()
    write_json(directory / "before.json", before)
    write_json(directory / "hardware.json", {"platform": platform.platform(),
        "python": platform.python_version(), "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False), "memory_total_bytes": psutil.virtual_memory().total,
        "memory_available_bytes": psutil.virtual_memory().available})
    for alias in MODELS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", alias,
                        "--run-id", run_id], cwd=ROOT, check=True)
    after = protected_snapshot()
    if before != after:
        raise ValueError("Protected files changed during Phase 3B")
    write_json(directory / "INTEGRITY.json", {"before": before, "after": after, "equal": True})
    make_report(run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", choices=MODELS)
    parser.add_argument("--run-id")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.run_id and (Path(args.run_id).name != args.run_id or any(c not in "0123456789TZ-abcdef" for c in args.run_id)):
        parser.error("Invalid run ID")
    if args.run:
        run_all()
    elif args.worker and args.run_id:
        offline_worker(args.worker, args.run_id, WARM_REPEATS)
    elif args.report and args.run_id:
        make_report(args.run_id)
    else:
        parser.print_help()
