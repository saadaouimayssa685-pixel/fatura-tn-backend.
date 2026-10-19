"""Phase 3D: encode new-document chunks once, persist, then test a fresh process."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import inspect
import json
import math
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

from qa_embedding_benchmark import (ROOT, OUTPUT as PHASE3, MemorySampler, VectorCache,
    digest, load_dataset, metrics, model_identity, read_json, write_json)
from qa_embedding_cache_benchmark import assert_same_evidence, compare_vectors, memory_record
from qa_query_cache_benchmark import OUTPUT as PHASE3C, protected_snapshot as prior_snapshot, query_inventory
from qa_chunk_embedding_cache import ChunkEmbeddingCache
from qa_query_embedding_cache import QueryEncoder, QueryCachingEncoder


OUTPUT = ROOT / "qa-results/embedding-phase3d"
MODELS = ("minilm", "bge-m3")
REPEATS = 3


def phase3c_directory():
    return PHASE3C / "runs" / read_json(PHASE3C / "latest.json")["run_id"]


def protected_snapshot():
    current = prior_snapshot()
    audit = read_json(phase3c_directory() / "INTEGRITY.json")
    if current != audit["after"] or audit["before"] != audit["after"]:
        raise ValueError("Phase 3C protected inputs changed")
    paths = [ROOT / name for name in ("qa_query_embedding_cache.py", "qa_query_cache_benchmark.py",
                                    "test_query_embedding_cache.py")]
    paths += [p for p in PHASE3C.rglob("*") if p.is_file()]
    return {"phase3c_protected": current, "phase3c_files": {
        str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths}}


def cache_snapshot(root):
    return {str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()}


class EncodeMonitor:
    """Count actual encode invocations; restart worker forbids every inference."""

    def __init__(self, model, forbid=False):
        self.model = model
        self.forbid = forbid
        self.calls = []

    def encode(self, texts, normalize_embeddings=False):
        if self.forbid:
            raise RuntimeError("Restart must reuse vectors: encode is forbidden")
        texts = list(texts)
        self.calls.append({"count": len(texts), "normalize_embeddings": normalize_embeddings})
        return self.model.encode(texts, normalize_embeddings=normalize_embeddings)


def ingest_document(root, identity, document_id, texts, model, chunking_mode="legacy"):
    """Measured ingestion, separate from retrieval. Reuse exact Phase 3B cache format."""
    texts = list(texts)
    started = time.perf_counter()
    cache = ChunkEmbeddingCache(root, identity, document_id, chunking_mode)
    hit = cache.contains(texts)
    setup_seconds = time.perf_counter() - started
    encoding_seconds = write_seconds = read_seconds = 0.0
    if hit:
        read_start = time.perf_counter()
        vectors = cache.get(texts)
        read_seconds = time.perf_counter() - read_start
    else:
        encode_start = time.perf_counter()
        vectors = model.encode(texts, normalize_embeddings=True)
        encoding_seconds = time.perf_counter() - encode_start
        write_start = time.perf_counter()
        cache.put(texts, vectors)
        write_seconds = time.perf_counter() - write_start
    return {"cache_hit": hit, "chunk_count": len(texts), "chunk_embedding_seconds": encoding_seconds,
        "cache_write_seconds": write_seconds, "ingestion_cache_read_seconds": read_seconds,
        "cache_setup_seconds": setup_seconds, "ingestion_seconds": time.perf_counter() - started,
        "cache_path": str(cache.path), "text_sha256": cache.key(texts)}, vectors


def aggregate(calls):
    keys = ("total_seconds", "chunk_embedding_seconds", "cache_write_seconds", "ingestion_seconds",
            "ingestion_cache_read_seconds", "cache_setup_seconds", "retrieval_seconds")
    return {"calls": len(calls), **{key: {"mean": mean(c[key] for c in calls),
        "median": median(c[key] for c in calls), "min": min(c[key] for c in calls),
        "max": max(c[key] for c in calls)} for key in keys},
        "peak_rss_bytes": max(c["memory"]["peak_rss_bytes"] for c in calls)}


def worker(alias, phase, run_id):
    from extraction_retrieval import ExtractionRetriever, FIELD_QUERIES, source_passages
    from qa_chunking_benchmark import QUERY_KEYS, canonical_chunk, evaluate

    _, previous, _, data, gold = load_dataset()
    baseline = read_json(PHASE3 / f"{alias}.json")
    prior = read_json(phase3c_directory() / f"{alias}.json")
    query_inventory()
    if any(rows != baseline["rows"] for rows in prior["rows"].values()):
        raise ValueError("Phase 3C and Phase 3 disagree")
    ref_docs = {d["document_id"]: d for d in baseline["documents"]}
    chunks = {doc: source_passages(data[doc][0]["ocr_text"]) for doc in gold}
    if any(chunks[d] != ref_docs[d]["chunks"] for d in gold):
        raise ValueError("Legacy chunks changed")
    directory = OUTPUT / "runs" / run_id
    root = directory / "chunks" / alias
    if phase == "initial" and root.exists():
        raise ValueError("Cold documents require an empty namespace")
    if phase == "restart" and not root.is_dir():
        raise ValueError("Restart requires existing persistent chunks")
    before = cache_snapshot(root) if phase == "restart" else {}
    stages = ("cold", "warm") if phase == "initial" else ("restart",)
    result = {"model": alias, "phase": phase, "run_id": run_id, "process_id": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(), "calls": [], "rows": {s: [] for s in stages},
        "annotation_sha256": previous["annotation_sha256"], "top_k": 5, "chunking_mode": "legacy",
        "torch_threads": 4, "repeats": REPEATS, "network_blocked": True,
        "ocr_calls": 0, "llm_calls": 0, "database_writes": 0, "real_fastapi_restarted": False}
    retriever = ExtractionRetriever(alias)
    print(json.dumps({"event": "loading", "model": alias, "phase": phase}), flush=True)
    with MemorySampler() as memory:
        started = time.perf_counter()
        with MemorySampler() as load_memory:
            import torch

            torch.set_num_threads(4)
            retriever._load()
        result["initialization"] = {"seconds": time.perf_counter() - started, "memory": memory_record(load_memory)}
        if retriever.model is None:
            raise RuntimeError("Model unavailable; refusing BM25 fallback")
        model = retriever.model
        identity = model_identity(model, alias)
        if identity != baseline["identity"] or identity != prior["identity"]:
            raise ValueError("Model/configuration/revision changed")
        batch_size = inspect.signature(model.encode).parameters["batch_size"].default
        if batch_size != 32 or identity["batch_size"] != batch_size:
            raise ValueError("Existing batch_size changed")
        result["identity"] = identity
        result["batching"] = {"encode_calls_per_cold_document": 1, "default_batch_size": batch_size,
            "expected_internal_batches": {doc: math.ceil(len(cs) / batch_size) for doc, cs in chunks.items()},
            "chunk_counts": {doc: len(cs) for doc, cs in chunks.items()}}
        monitor = EncodeMonitor(model, forbid=phase == "restart")
        references = VectorCache(PHASE3 / "vectors", identity)
        queries = list(FIELD_QUERIES.values())
        query_root = phase3c_directory() / "queries" / alias
        started = time.perf_counter()
        query_encoder = QueryEncoder(monitor, query_root, identity, storage="memory")
        if not query_encoder.cache.contains(queries):
            raise ValueError("Expected pre-existing Phase 3C query cache")
        query_vectors = query_encoder.encode(queries, normalize_embeddings=True)
        result["query_preload"] = {"seconds": time.perf_counter() - started,
            "event": query_encoder.last_event, "bytes": query_encoder.memory_bytes()}
        if query_encoder.last_event["hit_source"] != "disk" or monitor.calls:
            raise ValueError("Queries must be loaded, not encoded")
        compare_vectors(query_vectors, references.get(queries))
        adapter = QueryCachingEncoder(monitor, root, identity, queries, query_encoder)
        retriever.model = adapter
        print(json.dumps({"event": "ready", "model": alias, "phase": phase,
            "load_seconds": result["initialization"]["seconds"], "query_preload_seconds": result["query_preload"]["seconds"]}), flush=True)

        def process_document(doc_id, stage, repeat):
            texts = [c["text"] for c in chunks[doc_id]]
            count_before = len(monitor.calls)
            with MemorySampler() as sample:
                started = time.perf_counter()
                ingestion, vectors = ingest_document(root, identity, doc_id, texts, monitor)
                retrieval_start = time.perf_counter()
                adapter.begin_document(doc_id, texts, "legacy")
                evidence = retriever.retrieve(data[doc_id][0]["ocr_text"], top_k=5,
                    chunking_mode="legacy", document_id=doc_id)
                retrieval_seconds = time.perf_counter() - retrieval_start
                total_seconds = time.perf_counter() - started
            if retriever.model is not adapter or adapter.call_index != 2 or len(adapter.events) != 2:
                raise RuntimeError("Unexpected fallback during retrieval")
            q, c = adapter.events
            if (q["hit_source"] != "memory" or not c["cache_hit"] or q["embedding_seconds"] != 0
                    or c["embedding_seconds"] != 0 or c["cache_write_seconds"] != 0):
                raise ValueError("Retrieval must only read pre-encoded vectors")
            expected_encodes = int(stage == "cold")
            if ingestion["cache_hit"] != (stage != "cold") or len(monitor.calls) - count_before != expected_encodes:
                raise ValueError("Ingestion COLD/WARM contract broken")
            sha = assert_same_evidence(evidence, ref_docs[doc_id]["all_results"])
            expected_vectors = references.get(texts)
            comparisons = {"ingested_chunks": compare_vectors(vectors, expected_vectors),
                "retrieved_chunks": compare_vectors(adapter.arrays["chunks"], expected_vectors),
                "queries": compare_vectors(adapter.arrays["queries"], references.get(queries))}
            call = {**ingestion, "document_id": doc_id, "stage": stage, "repeat": repeat,
                "total_seconds": total_seconds, "retrieval_seconds": retrieval_seconds,
                "encode_call_count": len(monitor.calls) - count_before,
                "retrieval_events": adapter.events, "memory": memory_record(sample),
                "phase3_vectors": comparisons, "phase3_output_exact": True, "all_results_sha256": sha}
            result["calls"].append(call)
            if repeat == 1:
                for field, units in gold[doc_id].items():
                    passages = [canonical_chunk(p, doc_id) for p in evidence[QUERY_KEYS[field]]]
                    result["rows"][stage].append({"document_id": doc_id, "field": field,
                        "units": units, "query": FIELD_QUERIES[QUERY_KEYS[field]],
                        **evaluate(passages, units), "retrieved": passages})
            print(json.dumps({"event": "document", "model": alias, "stage": stage, "document": doc_id,
                "repeat": repeat, "chunks": len(texts), "embedding_seconds": round(ingestion["chunk_embedding_seconds"], 4),
                "write_ms": round(1000*ingestion["cache_write_seconds"], 3),
                "retrieval_ms": round(1000*retrieval_seconds, 3), "total_seconds": round(total_seconds, 4),
                "encode_calls": call["encode_call_count"], "phase3_exact": True}), flush=True)
            write_json(directory / f"{alias}-{phase}.progress.json", result)

        if phase == "initial":
            for doc in gold:
                process_document(doc, "cold", 1)
                process_document(doc, "warm", 1)
            for repeat in range(2, REPEATS + 1):
                for doc in gold:
                    process_document(doc, "warm", repeat)
        else:
            for repeat in range(1, REPEATS + 1):
                for doc in gold:
                    process_document(doc, "restart", repeat)
        result["encode_calls"] = monitor.calls
        if len(monitor.calls) != (len(gold) if phase == "initial" else 0):
            raise ValueError("Unexpected total encode count")
    result["memory"] = memory_record(memory)
    samples = [result["memory"], result["initialization"]["memory"]] + [c["memory"] for c in result["calls"]]
    result["observed_peak_rss_bytes"] = max(s["peak_rss_bytes"] for s in samples)
    after = cache_snapshot(root)
    if phase == "restart" and before != after:
        raise ValueError("Restart wrote or altered persisted chunks")
    result["cache_snapshot"] = after
    result["cache_unchanged_on_restart"] = before == after if phase == "restart" else None
    result["chunk_cache_bytes"] = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if len(list(root.rglob("*.npy"))) != len(gold):
        raise ValueError("Expected seven persistent document matrices")
    if any(rows != baseline["rows"] for rows in result["rows"].values()):
        raise ValueError("Evidence ranks/recall differ from Phase 3C")
    result["metrics"] = {s: metrics(rows) for s, rows in result["rows"].items()}
    result["performance"] = {s: aggregate([c for c in result["calls"] if c["stage"] == s]) for s in stages}
    result["phase3c_exact"] = True
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(directory / f"{alias}-{phase}.json", result)
    print(json.dumps({"event": "complete", "model": alias, "phase": phase,
        "mean_total_seconds": {s: p["total_seconds"]["mean"] for s, p in result["performance"].items()}}), flush=True)


def offline_worker(alias, phase, run_id):
    def deny(*args, **kwargs):
        raise RuntimeError("Network disabled for Phase 3D")

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"EXTRACTION_SEMANTIC_RAG": "true",
            "RAG_EMBEDDING_MODEL": alias, "RAG_CHUNKING_MODE": "legacy", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}))
        stack.enter_context(patch.object(socket.socket, "connect", deny))
        stack.enter_context(patch.object(socket.socket, "connect_ex", deny))
        worker(alias, phase, run_id)


def summarize_model(initial, restart):
    if (initial["process_id"] == restart["process_id"] or not restart["cache_unchanged_on_restart"]
            or initial["cache_snapshot"] != restart["cache_snapshot"] or restart["encode_calls"]):
        raise ValueError("Fresh-process persistence not verified")
    if not initial["phase3c_exact"] or not restart["phase3c_exact"]:
        raise ValueError("Retrieval identity not verified")
    per_document = []
    for cold in [c for c in initial["calls"] if c["stage"] == "cold"]:
        doc = cold["document_id"]
        warm = [c for c in initial["calls"] if c["stage"] == "warm" and c["document_id"] == doc]
        fresh = [c for c in restart["calls"] if c["document_id"] == doc]
        per_document.append({"document_id": doc, "chunk_count": cold["chunk_count"], "cold": cold,
            "warm": aggregate(warm), "restart": aggregate(fresh), "second_treatment": warm[0],
            "first_after_restart": fresh[0]})
    return {"per_document": per_document, "performance": {**initial["performance"], **restart["performance"]},
        "metrics": {**initial["metrics"], **restart["metrics"]},
        "initialization": {"initial": initial["initialization"], "restart": restart["initialization"]},
        "query_preload": {"initial": initial["query_preload"], "restart": restart["query_preload"]},
        "process_ids": [initial["process_id"], restart["process_id"]],
        "observed_peak_rss_bytes": max(initial["observed_peak_rss_bytes"], restart["observed_peak_rss_bytes"]),
        "chunk_cache_bytes": initial["chunk_cache_bytes"], "restart_encode_calls": 0}


def make_report(run_id):
    directory = OUTPUT / "runs" / run_id
    audit = read_json(directory / "INTEGRITY.json")
    if not audit["equal"] or audit["before"] != audit["after"]:
        raise ValueError("Protected-file check failed")
    runs = {a: {p: read_json(directory / f"{a}-{p}.json") for p in ("initial", "restart")} for a in MODELS}
    combined = {a: summarize_model(r["initial"], r["restart"]) for a, r in runs.items()}
    small = combined["minilm"]["performance"]["cold"]["total_seconds"]["mean"]
    big = combined["bge-m3"]["performance"]["cold"]["total_seconds"]
    conclusion = (f"C. Conserver BGE-M3 en mode haute qualite pour le moment. Sur les nouvelles factures de ce lot, "
        f"son cout cold moyen est {big['mean']:.3f} s contre {small:.3f} s pour MiniLM "
        f"({big['mean']/small:.1f} fois), avec un maximum observe de {big['max']:.3f} s. "
        "La persistance supprime le re-encodage, y compris apres redemarrage, mais ne supprime pas ce cout initial. "
        "Un defaut BGE-M3 requerrait un budget de latence d'ingestion accepte et une validation plus large ; "
        "aucune strategie hybride par champ n'est testee ici.")
    summary = {"run_id": run_id, "models": combined, "recommendation": conclusion,
        "dataset": {"documents": 7, "fields": 63, "units": 82}, "phase3c_exact": True,
        "real_fastapi_restarted": False}
    write_json(directory / "summary.json", summary)
    lines = ["# Phase 3D : ingestion initiale et persistance des chunks", "", "## Decision", "", conclusion, "",
        "MiniLM et legacy restent les defauts. Aucun branchement automatique du cache QA a la production.", "",
        "## Protocole", "",
        "Memes 7 factures, 63 champs, 82 unites de preuve et OCR figes que Phase 3C. Legacy, BM25/RRF, 22 requetes, top_k=5, seuils, modeles et configurations inchanges. Aucun appel OCR/LLM, endpoint ou base applicative. Chatbot, prompts et regles metier inchanges.",
        "4 processus sequentiels : MiniLM initial puis restart, BGE-M3 initial puis restart. Chaque processus charge son modele CPU float32 une fois, puis charge/valide les 22 requetes Phase 3C en RAM, sans inference de requetes.",
        "COLD DOCUMENT = repertoire de chunks neuf, encodage reel a l'ingestion puis ecriture persistante, suivi du retrieval qui relit ce cache. Le premier document du processus inclut son premier forward du modele, sans prechauffage d'inference.",
        "WARM DOCUMENT = seconde ingestion immediate du meme document, verification/lecture du cache sans encode, puis retrieval. Deux repetitions supplementaires pour chaque document. Apres extinction du processus, un nouveau worker repete trois passes sur les caches disques existants.",
        "Tous les encode sont comptes. Dans le worker restart, tout appel a encode leve une erreur : aucune reconstitution cachee seulement en RAM ne peut masquer un recalcul. Les fichiers caches sont verifies strictement identiques avant/apres restart.",
        "98 traitements controles : par modele, 7 cold + 21 warm + 21 restart. Mesures d'identite/evaluation QA, lectures des references et ecriture des logs hors chronometrage.",
        "Total = ingestion mesuree + retrieval ensuite + faible surcout d'appel. Ingestion = preparation/cle + embedding (cold) + ecriture (cold), ou lecture/validation (warm). Le retrieval comprend BM25/RRF et ses propres acces aux vecteurs. On conserve donc deux lectures de chunks en warm (controle ingestion puis retrieval), sans optimiser un second composant.",
        "Model loading et prechargement des requetes sont HORS total par document et presentes separement. Cache OS non purge ; cold signifie nouveau document sans vecteurs, pas disque froid. Trois passages warm, pas un test de charge/p95 de production.", "",
        "## Resultats globaux", "", "| Mesure | MiniLM | BGE-M3 |", "|---|---:|---:|"]
    measurements = [
        ("Cold nouvelle facture, total moyen (s)", lambda r: r["performance"]["cold"]["total_seconds"]["mean"], ".3f"),
        ("Cold embedding chunks moyen (s)", lambda r: r["performance"]["cold"]["chunk_embedding_seconds"]["mean"], ".3f"),
        ("Warm meme facture, total moyen (ms)", lambda r: 1000*r["performance"]["warm"]["total_seconds"]["mean"], ".3f"),
        ("Apres redemarrage + disque, total moyen hors chargement (ms)", lambda r: 1000*r["performance"]["restart"]["total_seconds"]["mean"], ".3f"),
        ("Chargement modele + imports initial (s)", lambda r: r["initialization"]["initial"]["seconds"], ".3f"),
        ("Chargement modele + imports restart (s)", lambda r: r["initialization"]["restart"]["seconds"], ".3f"),
        ("Requetes disque -> RAM, initial (ms)", lambda r: 1000*r["query_preload"]["initial"]["seconds"], ".3f"),
        ("Requetes disque -> RAM, restart (ms)", lambda r: 1000*r["query_preload"]["restart"]["seconds"], ".3f"),
        ("Premier document, avec chargement + requetes (s)", lambda r: r["initialization"]["initial"]["seconds"] + r["query_preload"]["initial"]["seconds"] + r["per_document"][0]["cold"]["total_seconds"], ".3f"),
        ("Premier document apres restart, avec chargement + requetes (s)", lambda r: r["initialization"]["restart"]["seconds"] + r["query_preload"]["restart"]["seconds"] + r["per_document"][0]["first_after_restart"]["total_seconds"], ".3f"),
        ("Pic RAM RSS observe (MiB)", lambda r: r["observed_peak_rss_bytes"]/2**20, ".3f"),
        ("Cache chunks persistant, 7 documents (KiB)", lambda r: r["chunk_cache_bytes"]/1024, ".3f"),
        ("Encodages apres restart", lambda r: r["restart_encode_calls"], "d")]
    for k in (1, 3, 5):
        measurements.append((f"Recall@{k} macro identique a Phase 3C", lambda r, k=k: r["metrics"]["cold"]["macro_recall"][str(k)], ".2%"))
    for title, getter, fmt in measurements:
        lines.append(f"| {title} | " + " | ".join(format(getter(r), fmt) for r in combined.values()) + " |")
    lines += ["", "Les chargements initiaux incluent les imports ML et sont sensibles au cache OS et a l'ordre sequentiel des essais. Ne pas conclure a une superiorite de chargement intrinseque d'un modele. RSS echantillonnee toutes les 50 ms, maximum des sondes par etape et globales ; pas la RAM totale de l'application."]
    for alias, r in combined.items():
        lines += ["", f"## Par facture : {alias}", "",
            "Cold = nouveau document ; secondes pour embedding/total, millisecondes pour ecriture/retrieval. Warm/restart = moyenne de trois passages, hors chargement modele.", "",
            "| Facture | Chunks | Embedding cold (s) | Ecriture cache (ms) | Retrieval ensuite (ms) | Total cold (s) | Total warm (ms) | Total restart (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for d in r["per_document"]:
            c = d["cold"]
            lines.append(f"| {d['document_id']} | {d['chunk_count']} | {c['chunk_embedding_seconds']:.3f} | "
                f"{1000*c['cache_write_seconds']:.3f} | {1000*c['retrieval_seconds']:.3f} | {c['total_seconds']:.3f} | "
                f"{1000*d['warm']['total_seconds']['mean']:.3f} | {1000*d['restart']['total_seconds']['mean']:.3f} |")
        lines += ["", "Detail des traitements sans re-encodage :", "",
            "| Facture | 2e traitement total (ms) | Warm lecture ingestion (ms) | Warm retrieval (ms) | 1er apres restart total (ms) | Restart lecture ingestion (ms) | Restart retrieval (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for d in r["per_document"]:
            lines.append(f"| {d['document_id']} | {1000*d['second_treatment']['total_seconds']:.3f} | "
                f"{1000*d['warm']['ingestion_cache_read_seconds']['mean']:.3f} | {1000*d['warm']['retrieval_seconds']['mean']:.3f} | "
                f"{1000*d['first_after_restart']['total_seconds']:.3f} | {1000*d['restart']['ingestion_cache_read_seconds']['mean']:.3f} | "
                f"{1000*d['restart']['retrieval_seconds']['mean']:.3f} |")
    lines += ["", "## Batching et optimisations autorisees", "",
        "ExtractionRetriever envoie deja toute la liste des chunks a un unique model.encode(). Signature effective verifiee : batch_size=32. Les factures ont 5 a 11 chunks (51 au total), donc un batch interne attendu par facture. Un encode reel par facture cold est verifie, jamais un encode par ligne.",
        "Batching conserve. Meme normalisation L2, float32, revisions, max_seq_length et 4 threads CPU que Phase 3C. Pas de quantification, troncature additionnelle, nouveau backend, changement de precision, multi-document batching ou autre reglage melange au test. L'optimisation evaluee est la persistance/reutilisation, pas une acceleration artificielle du premier encode.", "",
        "## Cache, invalidation et redemarrage", "",
        "Cache QA reutilisant ChunkEmbeddingCache de Phase 3B. Cle : document_id + SHA256 des textes EXACTS ordonnes + chunking_mode + modele/revision/dimension/normalisation + configuration complete (dtype, max_seq_length, batch_size, prompts, versions et hashes config/pooling). Toute difference correspond a une autre entree. Aucun cache MiniLM lu comme BGE-M3.",
        "NPY sans pickle ; verification identite, ordre/hash texte, lignes, SHA256 fichier, forme/float32/finitude/norme. Cache incomplet/corrompu refuse. Aucun effacement des anciennes phases."]
    for a, r in combined.items():
        lines.append(f"- {a} : PID initial {r['process_ids'][0]}, PID restart {r['process_ids'][1]}, 7 encodages initialement, 0 au restart ; empreintes des fichiers caches identiques.")
    lines += ["", "Portee exacte : redemarrage de workers Python du benchmark, PAS du serveur FastAPI reel. Ce test prouve que les matrices survivent a un processus neuf et n'exigent aucune nouvelle inference. Le cache experimental n'est pas encore branche a FastAPI : son comportement de production n'a pas ete modifie ni valide ici. Pas de simulation de panne electrique, ecriture concurrente multi-worker ou charge multi-utilisateur.", "",
        "## Identite et limites", "",
        "Chaque traitement est compare aux sorties completes Phase 3/3C : memes chunks, ordre, metadonnees et scores exposes, y compris les sorties tableaux pouvant depasser 5. Matrices comparees avec atol=1e-7/rtol=1e-6, identite bit-a-bit comptabilisee. Toute divergence de classement fait echouer le benchmark. Recall recalcule pour cold/warm/restart."]
    for alias, phases in runs.items():
        checks = [v for r in phases.values() for c in r["calls"] for v in c["phase3_vectors"].values()]
        lines.append(f"- {alias} : {sum(len(r['calls']) for r in phases.values())} sorties exactes ; {sum(v['bit_exact'] for v in checks)}/{len(checks)} matrices bit-a-bit ; ecart maximal {max(v['max_absolute_difference'] for v in checks):.9g}.")
    lines += ["", "Sept factures deja connues, une seule mesure cold par facture, trois warm/restart ; aucune nouvelle conclusion de qualite sur d'autres factures. Le Recall mesure la recherche des preuves, pas l'extraction finale. Le cache supprime les couts repetes mais ne rend pas gratuite la premiere ingestion d'un document jamais vu.", "",
        "## Reproduction", "", "`python qa_ingestion_cache_benchmark.py --run` : quatre processus CPU offline, nouveaux caches uniques, sans telechargement.",
        f"`python qa_ingestion_cache_benchmark.py --report --run-id {run_id}` : rapport seul.",
        "Seuls qa_ingestion_cache_benchmark.py, test_ingestion_cache_benchmark.py et les artefacts de cette phase sont ajoutes. Aucune dependance supplementaire, aucun fichier de production modifie.", ""]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUTPUT / "latest.json", {"run_id": run_id, "report": str(directory / "REPORT.md")})
    print(json.dumps({"event": "report", "path": str(directory / "REPORT.md"), "recommendation": conclusion}), flush=True)


def run_all():
    import psutil

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    directory = OUTPUT / "runs" / run_id
    directory.mkdir(parents=True)
    before = protected_snapshot()
    write_json(directory / "before.json", before)
    write_json(directory / "hardware.json", {"platform": platform.platform(), "python": platform.python_version(),
        "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total, "memory_available_bytes": psutil.virtual_memory().available})
    for alias in MODELS:
        for phase in ("initial", "restart"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", alias,
                "--phase", phase, "--run-id", run_id], cwd=ROOT, check=True)
    after = protected_snapshot()
    if before != after:
        raise ValueError("Protected files changed in Phase 3D")
    write_json(directory / "INTEGRITY.json", {"before": before, "after": after, "equal": True})
    make_report(run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--worker", choices=MODELS)
    parser.add_argument("--phase", choices=("initial", "restart"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.run_id and (Path(args.run_id).name != args.run_id or any(c not in "0123456789TZ-abcdef" for c in args.run_id)):
        parser.error("Invalid run ID")
    if args.run:
        run_all()
    elif args.worker and args.phase and args.run_id:
        offline_worker(args.worker, args.phase, args.run_id)
    elif args.report and args.run_id:
        make_report(args.run_id)
    else:
        parser.print_help()
