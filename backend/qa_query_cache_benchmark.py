"""Phase 3C: off/disk/memory query cache; unchanged warm chunk-cache retrieval."""
import argparse
import ast
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

from qa_embedding_benchmark import (ROOT, OUTPUT as PHASE3, MemorySampler, VectorCache,
    digest, load_dataset, metrics, model_identity, read_json, write_json)
from qa_embedding_cache_benchmark import (OUTPUT as PHASE3B, protected_snapshot as prior_snapshot,
    assert_same_evidence, compare_vectors, memory_record)
from qa_chunk_embedding_cache import ChunkEmbeddingCache
from qa_query_embedding_cache import QueryEncoder, QueryCachingEncoder


OUTPUT = ROOT / "qa-results/embedding-phase3c"
MODELS = ("minilm", "bge-m3")
STORAGES = ("off", "disk", "memory")
REPEATS = 3


def phase3b_directory():
    return PHASE3B / "runs" / read_json(PHASE3B / "latest.json")["run_id"]


def query_inventory():
    from extraction_retrieval import FIELD_QUERIES

    code = (ROOT / "extraction_retrieval.py").read_text(encoding="utf-8")
    assignments = [node for node in ast.parse(code).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "FIELD_QUERIES" for target in node.targets)]
    if len(assignments) != 1 or ast.literal_eval(assignments[0].value) != FIELD_QUERIES:
        raise ValueError("Query dictionary is no longer a static literal")
    if FIELD_QUERIES != read_json(PHASE3 / "minilm.json")["queries"] or len(FIELD_QUERIES) != 22:
        raise ValueError("Queries differ from Phase 3")
    return {"count": len(FIELD_QUERIES), "fixed": 22, "document_dependent": 0,
            "field_specific_fixed": 22, "dynamically_generated": 0,
            "ordered_queries": [{"field": field, "text": text} for field, text in FIELD_QUERIES.items()],
            "source": "extraction_retrieval.py:FIELD_QUERIES / ExtractionRetriever.retrieve",
            "retrieval_source_sha256": digest((ROOT / "extraction_retrieval.py").read_bytes())}


def protected_snapshot():
    current = prior_snapshot()
    approved = read_json(phase3b_directory() / "INTEGRITY.json")
    if approved["before"] != approved["after"] or current != approved["after"]:
        raise ValueError("Phase 3B protected inputs changed")
    paths = [ROOT / name for name in ("qa_chunk_embedding_cache.py", "qa_embedding_cache_benchmark.py",
                                     "test_chunk_embedding_cache.py")]
    paths += [p for p in PHASE3B.rglob("*") if p.is_file()]
    return {"phase3b_protected": current, "phase3b_files": {
        str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths}, "queries": query_inventory()}


def aggregate(calls):
    keys = ("retrieval_seconds", "query_embedding_seconds", "query_cache_seconds", "query_total_seconds",
            "chunk_cache_seconds", "chunk_embedding_seconds", "other_seconds")
    return {"calls": len(calls), **{key: {"mean": mean(c[key] for c in calls),
        "median": median(c[key] for c in calls), "min": min(c[key] for c in calls),
        "max": max(c[key] for c in calls)} for key in keys},
        "peak_rss_bytes": max(c["memory"]["peak_rss_bytes"] for c in calls)}


def worker(alias, run_id):
    from extraction_retrieval import ExtractionRetriever, FIELD_QUERIES, source_passages
    from qa_chunking_benchmark import QUERY_KEYS, canonical_chunk, evaluate

    _, previous, _, data, gold = load_dataset()
    baseline = read_json(PHASE3 / f"{alias}.json")
    phase3b = read_json(phase3b_directory() / f"{alias}.json")
    if phase3b["rows"] != baseline["rows"] or phase3b["warm_rows"] != baseline["rows"]:
        raise ValueError("Phase 3/3B evidence differs")
    inventory = query_inventory()
    queries = list(FIELD_QUERIES.values())
    ref_docs = {d["document_id"]: d for d in baseline["documents"]}
    chunks = {doc: source_passages(data[doc][0]["ocr_text"]) for doc in gold}
    if any(chunks[d] != ref_docs[d]["chunks"] for d in gold):
        raise ValueError("Legacy chunks changed")
    directory = OUTPUT / "runs" / run_id
    query_root = directory / "queries" / alias
    if query_root.exists():
        raise ValueError("Query-cache creation must use a fresh namespace")
    result = {"model": alias, "run_id": run_id, "process_id": os.getpid(), "queries": inventory,
        "top_k": 5, "chunking_mode": "legacy", "repeats": REPEATS, "torch_threads": 4,
        "annotation_sha256": previous["annotation_sha256"], "calls": [],
        "rows": {storage: [] for storage in STORAGES}, "llm_calls": 0, "ocr_calls": 0,
        "database_writes": 0, "network_blocked": True, "started_at": datetime.now(timezone.utc).isoformat()}
    retriever = ExtractionRetriever(alias)
    print(json.dumps({"event": "loading", "model": alias}), flush=True)
    with MemorySampler() as memory:
        started = time.perf_counter()
        with MemorySampler() as load_memory:
            import torch

            torch.set_num_threads(4)
            retriever._load()
        result["initialization"] = {"seconds": time.perf_counter() - started,
                                    "memory": memory_record(load_memory)}
        if retriever.model is None:
            raise RuntimeError("Model unavailable; silent BM25 fallback forbidden")
        model = retriever.model
        identity = model_identity(model, alias)
        if identity != baseline["identity"] or identity != phase3b["identity"]:
            raise ValueError("Model/revision/configuration changed")
        result["identity"] = identity
        references = VectorCache(PHASE3 / "vectors", identity)
        reference_queries = references.get(queries)
        chunk_root = phase3b_directory() / "vectors" / alias
        for doc in gold:
            # Preflight read: no missing entry may trigger encoding/writing to Phase 3B.
            chunk_texts = [c["text"] for c in chunks[doc]]
            cache = ChunkEmbeddingCache(chunk_root, identity, doc, "legacy")
            compare_vectors(cache.get(chunk_texts), references.get(chunk_texts))
        print(json.dumps({"event": "loaded", "model": alias,
                          "seconds": result["initialization"]["seconds"]}), flush=True)

        encoders = {s: QueryEncoder(model, query_root, identity, storage=s) for s in STORAGES}
        with MemorySampler() as create_memory:
            started = time.perf_counter()
            initial = encoders["disk"].encode(queries, normalize_embeddings=True)
            creation_seconds = time.perf_counter() - started
        if encoders["disk"].last_event["cache_hit"]:
            raise ValueError("Initial query cache was not empty")
        result["query_creation"] = {"seconds": creation_seconds, "event": encoders["disk"].last_event,
            "memory": memory_record(create_memory), "phase3_vectors": compare_vectors(initial, reference_queries)}
        started = time.perf_counter()
        preloaded = encoders["memory"].encode(queries, normalize_embeddings=True)
        result["memory_preload"] = {"seconds": time.perf_counter() - started,
            "event": encoders["memory"].last_event, "phase3_vectors": compare_vectors(preloaded, reference_queries)}
        if result["memory_preload"]["event"]["hit_source"] != "disk":
            raise ValueError("Memory preload must validate persisted query vectors")
        print(json.dumps({"event": "query_cache_created", "model": alias,
            "seconds": creation_seconds, "memory_preload_seconds": result["memory_preload"]["seconds"]}), flush=True)

        adapters = {s: QueryCachingEncoder(model, chunk_root, identity, queries, encoders[s]) for s in STORAGES}
        for repeat in range(REPEATS):
            for doc_index, (doc_id, fields) in enumerate(gold.items()):
                # Rotate A/B orders to distribute temporal/cache-OS/thermal effects.
                shift = (repeat + doc_index) % len(STORAGES)
                for storage in STORAGES[shift:] + STORAGES[:shift]:
                    adapter = adapters[storage]
                    retriever.model = adapter
                    texts = [c["text"] for c in chunks[doc_id]]
                    with MemorySampler() as sample:
                        started = time.perf_counter()
                        adapter.begin_document(doc_id, texts, "legacy")
                        evidence = retriever.retrieve(data[doc_id][0]["ocr_text"], top_k=5,
                            chunking_mode="legacy", document_id=doc_id)
                        elapsed = time.perf_counter() - started
                    if retriever.model is not adapter or adapter.call_index != 2 or len(adapter.events) != 2:
                        raise RuntimeError("Unexpected encode path or BM25 fallback")
                    q, c = adapter.events
                    if not c["cache_hit"] or c["embedding_seconds"] != 0:
                        raise ValueError("Chunks were re-encoded")
                    if q["cache_hit"] != (storage != "off") or q["hit_source"] != (None if storage == "off" else storage):
                        raise ValueError("Query cache A/B contract violated")
                    if storage != "off" and q["embedding_seconds"] != 0:
                        raise ValueError("Query embeddings unexpectedly recalculated")
                    # Identity checks are outside timing. Strict comparison is stronger than Top 5.
                    sha = assert_same_evidence(evidence, ref_docs[doc_id]["all_results"])
                    check = {"queries": compare_vectors(adapter.arrays["queries"], reference_queries),
                             "chunks": compare_vectors(adapter.arrays["chunks"], references.get(texts))}
                    call = {"document_id": doc_id, "repeat": repeat + 1, "storage": storage,
                        "retrieval_seconds": elapsed, "query_embedding_seconds": q["embedding_seconds"],
                        "query_cache_seconds": q["cache_read_seconds"] + q["cache_write_seconds"],
                        "query_total_seconds": q["total_seconds"], "chunk_cache_seconds": c["cache_read_seconds"],
                        "chunk_embedding_seconds": c["embedding_seconds"], "events": adapter.events,
                        "memory": memory_record(sample), "phase3_vectors": check, "phase3_output_exact": True,
                        "all_results_sha256": sha}
                    call["other_seconds"] = elapsed - q["total_seconds"] - c["cache_read_seconds"]
                    result["calls"].append(call)
                    if repeat == 0:
                        for field, units in fields.items():
                            passages = [canonical_chunk(p, doc_id) for p in evidence[QUERY_KEYS[field]]]
                            result["rows"][storage].append({"document_id": doc_id, "field": field,
                                "units": units, "query": FIELD_QUERIES[QUERY_KEYS[field]],
                                **evaluate(passages, units), "retrieved": passages})
                print(json.dumps({"event": "document", "model": alias, "document": doc_id,
                    "repeat": repeat + 1, "times": {call["storage"]: round(call["retrieval_seconds"], 4)
                    for call in result["calls"][-3:]}, "phase3_exact": True}), flush=True)
                # Recover progress after an interrupted desktop turn, without redoing inference.
                write_json(directory / f"{alias}.progress.json", result)
        result["query_memory_bytes"] = encoders["memory"].memory_bytes()
        result["query_cache_bytes"] = encoders["disk"].cache.size_bytes()
        result["query_cache_path"] = str(encoders["disk"].cache.path)
        if len(list(query_root.rglob("*.npy"))) != 1:
            raise ValueError("Expected one query batch per model, not one per document")
    result["memory"] = memory_record(memory)
    samples = [result["memory"], result["initialization"]["memory"], result["query_creation"]["memory"]]
    samples += [c["memory"] for c in result["calls"]]
    result["observed_peak_rss_bytes"] = max(s["peak_rss_bytes"] for s in samples)
    result["performance"] = {s: aggregate([c for c in result["calls"] if c["storage"] == s]) for s in STORAGES}
    if any(rows != baseline["rows"] for rows in result["rows"].values()):
        raise ValueError("Recall or evidence differs from Phase 3/3B")
    result["metrics"] = {s: metrics(rows) for s, rows in result["rows"].items()}
    result["phase3_exact"] = True
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(directory / f"{alias}.json", result)
    print(json.dumps({"event": "complete", "model": alias,
        "mean_retrieval_seconds": {s: p["retrieval_seconds"]["mean"] for s, p in result["performance"].items()}}), flush=True)


def offline_worker(alias, run_id):
    def deny(*args, **kwargs):
        raise RuntimeError("Network forbidden in Phase 3C")

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"EXTRACTION_SEMANTIC_RAG": "true",
            "RAG_EMBEDDING_MODEL": alias, "RAG_CHUNKING_MODE": "legacy", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}))
        stack.enter_context(patch.object(socket.socket, "connect", deny))
        stack.enter_context(patch.object(socket.socket, "connect_ex", deny))
        worker(alias, run_id)


def make_report(run_id):
    directory = OUTPUT / "runs" / run_id
    audit = read_json(directory / "INTEGRITY.json")
    if not audit["equal"] or audit["before"] != audit["after"]:
        raise ValueError("Protected input verification failed")
    runs = {a: read_json(directory / f"{a}.json") for a in MODELS}
    old = {a: read_json(phase3b_directory() / f"{a}.json") for a in MODELS}
    if any(not r["phase3_exact"] for r in runs.values()):
        raise ValueError("Identity checks failed")
    gain = {a: 1 - r["performance"]["memory"]["retrieval_seconds"]["mean"] /
            r["performance"]["off"]["retrieval_seconds"]["mean"] for a, r in runs.items()}
    summary = {"run_id": run_id, "dataset": {"documents": 7, "fields": 63, "units": 82},
        "models": {a: {"performance": r["performance"], "metrics": r["metrics"], "gain": gain[a],
            "phase3b_warm_seconds": old[a]["performance"]["warm"]["retrieval_seconds"]["mean"],
            "query_creation_seconds": r["query_creation"]["seconds"], "memory_preload_seconds": r["memory_preload"]["seconds"],
            "observed_peak_rss_bytes": r["observed_peak_rss_bytes"], "query_memory_bytes": r["query_memory_bytes"],
            "query_cache_bytes": r["query_cache_bytes"]} for a, r in runs.items()}, "phase3_exact": True}
    bge_ms = runs["bge-m3"]["performance"]["memory"]["retrieval_seconds"]["mean"] * 1000
    summary["recommendation"] = (f"B. BGE-M3 en mode haute qualite explicite. Le retrieval warm est de {bge_ms:.2f} ms/document "
        "sur ce lot avec les deux caches : le recalcul des requetes n'est plus un frein. "
        "Ce resultat ne suffit pas a valider un defaut general : premier encodage des nouveaux documents, "
        "chargement modele, petit jeu QA et regressions Phase 3 restent a considerer. Aucun routage par champ teste.")
    write_json(directory / "summary.json", summary)
    inventory = runs["minilm"]["queries"]
    lines = ["# Phase 3C : cache des requetes fixes", "", "## Perimetre et decision", "",
        summary["recommendation"], "",
        "MiniLM et legacy restent les defauts. Aucune activation du cache en production. Aucun appel OCR/LLM ou endpoint, aucune ecriture en base applicative. Regles metier, prompts et chatbot inchanges.", "",
        "## Requetes utilisees", "",
        "22 textes fixes et deterministes, 22 formulations specifiques aux champs, 0 dependante du document, 0 variante dynamique. Ces nombres ne sont pas des categories disjointes : les 22 requetes fixes sont aussi les 22 requetes par champ.",
        "Source : FIELD_QUERIES, encode(list(FIELD_QUERIES.values())) dans ExtractionRetriever.retrieve(). L'ordre du dictionnaire est celui des lignes de la matrice. Audit du code et recherche des usages : aucune mutation du dictionnaire ni composition dynamique trouvee. Literal AST, empreinte du fichier et comparaison avec Phase 3 verifies.", "",
        "| Champ | Texte exact |", "|---|---|"]
    lines += [f"| {q['field']} | {q['text']} |" for q in inventory["ordered_queries"]]
    lines += ["", "## Protocole A/B", "",
        "7 factures / 63 champs / 82 preuves Phase 3B. OCR figes, chunks legacy, top_k=5, BM25/RRF/seuil 0.35 et formulations inchanges. Modeles CPU float32, 4 threads torch, batch32, revisions/configurations identiques a Phase 3.",
        "A = off : requetes recalculees a chaque retrieval. B1 = disk : matrice des requetes lue et validee a chaque appel. B2 = memory : meme matrice validee/chargee une fois au demarrage, puis reutilisee en RAM.",
        "Les chunks restent en cache disque Phase 3B dans TOUS les modes. Aucun recalcul des chunks. Fichiers Phase 3/3B verifies inchanges avant/apres. Aucun ancien vecteur de requete n'est copie pour creer le cache : encode reel une fois par modele.",
        "Deux processus separes, chaque modele charge une fois. Trois passes de 7 factures par mode ; ordre off/disk/memory tourne selon document et repetition pour repartir les effets temporels. 126 retrievals reels controles (2 x 3 x 7 x 3).",
        "Creation du cache et chargement RAM mesures separement. Comparaisons aux matrices/sorties Phase 3, evaluations QA et ecritures de logs hors chronometrage. Initialisation modele hors warm ; pas de telechargement.",
        "Creation initiale = encodage du lot et persistance de la matrice avec ses metadonnees. Les objets de cache et leur manifeste d'identite sont prepares avant ce chronometrage. Le prechargement RAM mesure la lecture, validation et copie immuable depuis le disque.",
        "La comparaison causale est A/B dans cette execution. Les chiffres Phase 3B sont fournis comme historique, pas comme groupe controle simultane. Temps = lot des 22 requetes, pas requete unique.", "",
        "## Comparaison principale", "", "| Mesure | MiniLM | BGE-M3 |", "|---|---:|---:|"]
    measures = [
        ("Warm avant, Phase 3B (s/doc)", lambda a, r: old[a]["performance"]["warm"]["retrieval_seconds"]["mean"], ".3f"),
        ("A : warm sans cache requetes, mesure actuelle (s/doc)", lambda a, r: r["performance"]["off"]["retrieval_seconds"]["mean"], ".3f"),
        ("B1 : warm avec cache disque (ms/doc)", lambda a, r: 1000*r["performance"]["disk"]["retrieval_seconds"]["mean"], ".3f"),
        ("B2 : warm avec cache memoire (ms/doc)", lambda a, r: 1000*r["performance"]["memory"]["retrieval_seconds"]["mean"], ".3f"),
        ("Gain B2 vs A (temps)", lambda a, r: gain[a], ".2%"),
        ("Chargement modele + imports une fois (s)", lambda a, r: r["initialization"]["seconds"], ".3f"),
        ("Creation initiale cache requetes (s)", lambda a, r: r["query_creation"]["seconds"], ".3f"),
        ("Dont embedding reel des 22 requetes (s)", lambda a, r: r["query_creation"]["event"]["embedding_seconds"], ".3f"),
        ("Dont ecriture cache (ms)", lambda a, r: 1000*r["query_creation"]["event"]["cache_write_seconds"], ".3f"),
        ("Prechargement disque -> RAM (ms)", lambda a, r: 1000*r["memory_preload"]["seconds"], ".3f"),
        ("Pic RSS observe toutes sondes (MiB)", lambda a, r: r["observed_peak_rss_bytes"]/2**20, ".3f"),
        ("Vecteurs requetes en memoire (KiB, hors objets Python)", lambda a, r: r["query_memory_bytes"]/1024, ".3f"),
        ("Cache requetes disque + metadata (KiB)", lambda a, r: r["query_cache_bytes"]/1024, ".3f"),
        ("Recall@5 macro, A = B1 = B2 = Phase 3/3B", lambda a, r: r["metrics"]["memory"]["macro_recall"]["5"], ".2%")]
    for title, getter, fmt in measures:
        lines.append(f"| {title} | " + " | ".join(format(getter(a, runs[a]), fmt) for a in MODELS) + " |")
    lines += ["", "## Detail des modes", "", "| Modele / mode | Moyenne (ms) | Mediane (ms) | Min-max (ms) | Encode requetes (ms) | Acces requetes total (ms) | RSS pic (MiB) |", "|---|---:|---:|---|---:|---:|---:|"]
    for a, r in runs.items():
        for s, p in r["performance"].items():
            t = p["retrieval_seconds"]
            lines.append(f"| {a} / {s} | {1000*t['mean']:.3f} | {1000*t['median']:.3f} | {1000*t['min']:.3f}-{1000*t['max']:.3f} | "
                f"{1000*p['query_embedding_seconds']['mean']:.3f} | {1000*p['query_total_seconds']['mean']:.3f} | {p['peak_rss_bytes']/2**20:.3f} |")
    lines += ["", "RSS mesure toutes les 50 ms, maximum des sondes globales/par etape. Les modes partagent le processus modele : RAM retenue par les allocateurs peut persister entre modes. Les 21 observations par mode ne donnent pas un p95 de production fiable. Le cache Windows n'est pas vide. Chargement modele et temperature/charge du poste peuvent biaiser les comparaisons entre modeles.",
        "", "## Qualite et identite", "", "| Recall macro | MiniLM (tous modes) | BGE-M3 (tous modes) |", "|---|---:|---:|"]
    for k in (1, 3, 5):
        lines.append(f"| @{k} | " + " | ".join(f"{r['metrics']['memory']['macro_recall'][str(k)]:.2%}" for r in runs.values()) + " |")
    for a, r in runs.items():
        checks = [v for c in r["calls"] for v in c["phase3_vectors"].values()]
        lines.append(f"\n- {a} : {len(r['calls'])} sorties completes identiques ; {sum(v['bit_exact'] for v in checks)}/{len(checks)} matrices bit-a-bit ; ecart absolu maximum {max(v['max_absolute_difference'] for v in checks):.9g}.")
    lines += ["", "Chaque retrieval compare les sorties completes (tous champs et jusqu'a 12 passages forces de tableau), ordre, texte, metadonnees et scores exposes compris. Toute divergence fait echouer le benchmark. Vecteurs compares avec atol=1e-7/rtol=1e-6, sans autoriser un seul changement de classement. Les 63 evaluations annotees de chaque mode egalent les resultats Phase 3/3B. Top 5/3 disponibles dans les lignes 'rows' des JSON ; hash des sorties conserve par appel.",
        "", "## Cle, invalidation et stockage", "",
        "Cle = SHA256 de l'identite modele/configuration complete + SHA256 de la liste ordonnee des textes EXACTS. Chaque vecteur correspond a sa position et son texte dans ce lot. Aucun nom de champ utilise comme identite. Modifier un seul texte, sa casse, ses espaces, l'ordre ou la configuration cree un miss pour le lot complet.",
        "Identite : modele, revision, dimension, normalisation, dtype, max_seq_length, batch_size, prompts/default_prompt, versions des bibliotheques, hashes configuration/pooling et schema de cache. MiniLM/BGE et caches chunks/requetes separes. Aucun effacement des caches existants.",
        "Le lot entier est conserve pour ne pas modifier les conditions de padding/calcul float32 de l'encode original. Cache NPY sans pickle, forme/dtype/finitude/normes et SHA256 verifies. Entrees corrompues/incompletes refusees. Verrou pour eviter plusieurs creations simultanees ; matrice memoire immuable ; LRU borne a 4 lots.",
        "Recommandation stockage : memoire prechargee pour ces 22 requetes, avec disque valide optionnel pour reutiliser les vecteurs apres redemarrage. Le cache disque doit etre controle a l'initialisation, puis la RAM evite les I/O repetitives. Sur ces petits tableaux, le ranking et les acces chunks peuvent dominer : ne pas promettre un gain memoire/disque au-dela des mesures.",
        "", "## Limites et suite", "",
        "Le warm mesure la relecture d'un document DEJA indexe. Le cache requetes ne supprime pas l'encodage initial des chunks d'une nouvelle facture (cout Phase 3B toujours applicable, pas remesure ici). Il ne supprime pas non plus le chargement initial du modele. Le choix d'un defaut doit prendre en compte ce parcours, la RAM et les regressions par champ observees en Phase 3.",
        "Aucun routage par champ, nouveau modele, reranker, quantification ou changement de moteur. Resultats sur 7 factures deja connues, pas une validation end-to-end de l'extraction ni un test de charge multi-utilisateur. Configuration production intacte.",
        "", "## Reproduction", "",
        "`python qa_query_cache_benchmark.py --run` : nouveaux caches experimentaux, deux workers CPU hors ligne.",
        f"`python qa_query_cache_benchmark.py --report --run-id {run_id}` : regeneration du rapport sans inference.",
        "Fichiers nouveaux : qa_query_embedding_cache.py, qa_query_cache_benchmark.py, test_query_embedding_cache.py. Aucune dependance supplementaire. Artefacts JSON, cache requetes, inventaire et empreintes conserves dans ce dossier.", ""]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUTPUT / "latest.json", {"run_id": run_id, "report": str(directory / "REPORT.md")})
    print(json.dumps({"event": "report", "report": str(directory / "REPORT.md"), "recommendation": summary["recommendation"]}), flush=True)


def run_all():
    import psutil

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    directory = OUTPUT / "runs" / run_id
    directory.mkdir(parents=True)
    before = protected_snapshot()
    write_json(directory / "before.json", before)
    write_json(directory / "query_inventory.json", query_inventory())
    write_json(directory / "hardware.json", {"platform": platform.platform(), "python": platform.python_version(),
        "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total, "memory_available_bytes": psutil.virtual_memory().available})
    for a in MODELS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", a, "--run-id", run_id], cwd=ROOT, check=True)
    after = protected_snapshot()
    if before != after:
        raise ValueError("Protected files changed during Phase 3C")
    write_json(directory / "INTEGRITY.json", {"before": before, "after": after, "equal": True})
    make_report(run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--worker", choices=MODELS)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.run_id and (Path(args.run_id).name != args.run_id or any(c not in "0123456789TZ-abcdef" for c in args.run_id)):
        parser.error("Invalid run ID")
    if args.run:
        run_all()
    elif args.worker and args.run_id:
        offline_worker(args.worker, args.run_id)
    elif args.report and args.run_id:
        make_report(args.run_id)
    else:
        parser.print_help()
