"""Phase 3: CPU-only embedding benchmark; frozen Phase 2B corpus and queries."""
import argparse
import ast
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
from statistics import mean
import subprocess
import sys
import threading
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "qa-results/embedding-phase3"
PHASE2 = ROOT / "qa-results/chunking-phase2b"
MODES = ("bm25", "minilm", "bge-m3")
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DOWNLOAD_FILES = ["1_Pooling/config.json", "config.json", "config_sentence_transformers.json",
                  "modules.json", "sentence_bert_config.json", "sentencepiece.bpe.model",
                  "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def download():
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, snapshot_download

    started = time.perf_counter()
    info = HfApi().model_info("BAAI/bge-m3", revision=BGE_REVISION, files_metadata=True, token=False)
    expected = {s.rfilename: s.size for s in info.siblings if s.rfilename in DOWNLOAD_FILES}
    print(json.dumps({"event": "download", "repository": info.id, "revision": info.sha,
                      "expected_bytes": sum(expected.values())}), flush=True)
    path = Path(snapshot_download("BAAI/bge-m3", revision=BGE_REVISION,
                                  allow_patterns=DOWNLOAD_FILES, token=False, max_workers=2))
    actual = {name: (path / name).stat().st_size for name in DOWNLOAD_FILES}
    if actual != expected:
        raise RuntimeError("Downloaded model files do not match repository sizes")
    # Set the normal cached main ref too, so explicit application selection can
    # load this verified revision offline without modifying application settings.
    ref = path.parent.parent / "refs/main"
    ref.parent.mkdir(exist_ok=True)
    if not ref.exists():
        ref.write_text(BGE_REVISION, encoding="ascii")
    record = {"repository": info.id, "revision": info.sha, "path": str(path), "files": actual,
              "download_seconds": time.perf_counter() - started, "bytes": sum(actual.values()),
              "included_in_execution_timing": False,
              "transport": "HTTPS (Xet disabled for this download process)",
              "previous_attempt": "Xet attempt interrupted after approximately 7 minutes without a usable weights file; excluded from benchmark execution timings.",
              "source": "https://huggingface.co/BAAI/bge-m3"}
    write_json(OUTPUT / "download.json", record)
    print(json.dumps(record), flush=True)


class VectorCache:
    """QA-only matrices: strict model/config/dimension/content namespaces."""
    def __init__(self, root, identity):
        self.identity = identity
        key = digest(json.dumps(identity, sort_keys=True).encode())
        self.path = root / identity["alias"] / f"{key}-d{identity['dimension']}"
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = self.path / "identity.json"
        if manifest.exists() and read_json(manifest) != identity:
            raise ValueError("Embedding cache identity mismatch")
        if not manifest.exists():
            write_json(manifest, identity)

    def key(self, texts):
        return digest(json.dumps(list(texts), ensure_ascii=False, separators=(",", ":")).encode())

    def put(self, texts, vectors):
        import numpy as np

        array = np.asarray(vectors)
        self.validate(array, len(texts))
        key = self.key(texts)
        array_path = self.path / f"{key}.npy"
        if array_path.exists():
            if not np.allclose(self.get(texts), array, atol=1e-7, rtol=1e-6):
                raise ValueError("Existing embedding differs under the same cache identity")
            return
        np.save(array_path, array, allow_pickle=False)
        write_json(self.path / f"{key}.json", {"identity": self.identity, "text_sha256": key,
                    "rows": len(texts), "array_sha256": digest(array_path.read_bytes())})

    def validate(self, array, count):
        import numpy as np

        if (array.shape != (count, self.identity["dimension"]) or array.dtype != np.float32
                or not np.isfinite(array).all()):
            raise ValueError("Invalid embedding matrix shape/dtype/values")
        if not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-4):
            raise ValueError("Expected normalized embeddings")

    def get(self, texts):
        import numpy as np

        key = self.key(texts)
        path = self.path / f"{key}.npy"
        metadata = read_json(self.path / f"{key}.json")
        if (metadata["identity"] != self.identity or metadata["rows"] != len(texts)
                or metadata["text_sha256"] != key or metadata["array_sha256"] != digest(path.read_bytes())):
            raise ValueError("Stale, foreign or corrupted embedding cache")
        array = np.load(path, allow_pickle=False)
        self.validate(array, len(texts))
        return array

    def size_bytes(self):
        return sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())


class MemorySampler:
    def __init__(self):
        import psutil

        self.process = psutil.Process()
        self.psutil = psutil
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.min_available = psutil.virtual_memory().available
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.wait(.05):
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.min_available = min(self.min_available, self.psutil.virtual_memory().available)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


class TimedEncoder:
    def __init__(self, model, cache, queries):
        self.model, self.cache, self.queries = model, cache, queries
        self.cached_only = False
        self.events = []

    def encode(self, texts, normalize_embeddings=False):
        if not normalize_embeddings:
            raise ValueError("Benchmark requires existing normalized dense embeddings")
        if self.cached_only:
            return self.cache.get(texts)
        started = time.perf_counter()
        vectors = self.model.encode(texts, normalize_embeddings=True)
        elapsed = time.perf_counter() - started
        self.events.append({"kind": "queries" if list(texts) == self.queries else "corpus",
                            "count": len(texts), "seconds": elapsed})
        self.cache.put(texts, vectors)
        return vectors


def model_identity(model, alias):
    from huggingface_hub import snapshot_download
    from extraction_embeddings import embedding_spec

    spec = embedding_spec(alias)
    snapshot = Path(snapshot_download(spec.repository, local_files_only=True))
    config = model[0].auto_model.config
    revision = getattr(config, "_commit_hash", None) or snapshot.name
    if revision != snapshot.name:
        raise ValueError("Loaded model and cached snapshot revisions differ")
    return {"alias": alias, "model": spec.repository, "revision": revision,
            "dimension": spec.dimension, "normalize_embeddings": True,
            "dtype": str(next(model.parameters()).dtype), "device": "cpu", "batch_size": 32,
            "max_seq_length": model.max_seq_length, "prompts": model.prompts,
            "default_prompt_name": model.default_prompt_name,
            "sentence_transformers_version": importlib.metadata.version("sentence-transformers"),
            "transformers_version": importlib.metadata.version("transformers"),
            "torch_version": importlib.metadata.version("torch"),
            "modules_sha256": digest((snapshot / "modules.json").read_bytes()),
            "pooling_sha256": digest((snapshot / "1_Pooling/config.json").read_bytes()),
            "model_config_sha256": digest((snapshot / "config.json").read_bytes())}


def load_dataset():
    from qa_chunking_benchmark import inventory, resolve_units

    annotations = read_json(PHASE2 / "annotations.json")
    previous = read_json(PHASE2 / "results.json")
    if digest((PHASE2 / "annotations.json").read_bytes()) != previous["annotation_sha256"]:
        raise ValueError("Phase 2B annotations changed")
    records, data = inventory(annotations)
    gold = {d["document_id"]: resolve_units(d, data[d["document_id"]][0]["ocr_text"]) for d in annotations["documents"]}
    if (len(gold), sum(len(v) for v in gold.values()), sum(len(u) for v in gold.values() for u in v.values())) != (7, 63, 82):
        raise ValueError("Expected unchanged 7/63/82 Phase 2B dataset")
    return annotations, previous, records, data, gold


def worker(mode):
    from extraction_retrieval import ExtractionRetriever, FIELD_QUERIES, source_passages
    from qa_chunking_benchmark import QUERY_KEYS, canonical_chunk, evaluate

    annotations, previous, records, data, gold = load_dataset()
    if FIELD_QUERIES != previous["parameters"]["queries"]:
        raise ValueError("Retrieval queries changed")
    previous_chunks = {d["document_id"]: d["modes"]["legacy"]["chunks"] for d in previous["documents"]}
    chunks_by_doc = {}
    for doc_id in gold:
        chunks_by_doc[doc_id] = source_passages(data[doc_id][0]["ocr_text"])
        if chunks_by_doc[doc_id] != previous_chunks[doc_id]:
            # Phase 2B adds only evaluation scope metadata to its saved catalog.
            adapted = [canonical_chunk(c, doc_id) for c in chunks_by_doc[doc_id]]
            if adapted != previous_chunks[doc_id]:
                raise ValueError("Legacy chunks differ from Phase 2B")
    result = {"mode": mode, "rows": [], "documents": [], "annotation_sha256": previous["annotation_sha256"],
              "queries": FIELD_QUERIES, "inventory": records, "identity": None, "torch_threads": None,
              "top_k": 5, "chunking_mode": "legacy", "ocr_calls": 0, "llm_calls": 0, "database_writes": 0,
              "network_blocked": True, "process_id": os.getpid(), "numpy_version": importlib.metadata.version("numpy")}
    retriever = ExtractionRetriever(embedding_model=mode if mode != "bm25" else "minilm")
    encoder, cache = None, None
    with MemorySampler() as memory:
        started = time.perf_counter()
        if mode != "bm25":
            import torch

            torch.set_num_threads(4)
            result["torch_threads"] = torch.get_num_threads()
        retriever._load()
        result["initialization_seconds"] = time.perf_counter() - started
        if mode != "bm25":
            if retriever.model is None:
                raise RuntimeError(f"{mode} unavailable; refusing silent BM25 fallback")
            identity = model_identity(retriever.model, mode)
            if identity["dtype"] != "torch.float32":
                raise ValueError("Both models must use CPU float32")
            result["identity"] = identity
            from huggingface_hub import snapshot_download
            snapshot = Path(snapshot_download(identity["model"], local_files_only=True))
            result["model_snapshot_bytes"] = sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file())
            all_texts = list(FIELD_QUERIES.values()) + [p["text"] for v in chunks_by_doc.values() for p in v]
            lengths = [len(x) for x in retriever.model.tokenizer(all_texts, truncation=False)["input_ids"]]
            result["max_input_tokens"] = max(lengths)
            result["truncated_inputs"] = sum(n > identity["max_seq_length"] for n in lengths)
            cache = VectorCache(OUTPUT / "vectors", identity)
            encoder = TimedEncoder(retriever.model, cache, list(FIELD_QUERIES.values()))
            retriever.model = encoder
        for doc_id, fields in gold.items():
            frozen, layout = data[doc_id]
            started = time.perf_counter()
            evidence = retriever.retrieve(frozen["ocr_text"], top_k=5, chunking_mode="legacy",
                ocr_metadata=frozen["ocr_metadata"], layout_metadata=layout, document_id=doc_id)
            elapsed = time.perf_counter() - started
            if mode != "bm25" and retriever.model is not encoder:
                raise RuntimeError("Embedding failed during retrieval")
            result["documents"].append({"document_id": doc_id, "uncached_batch_seconds": elapsed,
                                       "chunks": chunks_by_doc[doc_id], "all_results": evidence})
            for field, units in fields.items():
                passages = [canonical_chunk(c, doc_id) for c in evidence[QUERY_KEYS[field]]]
                measured = evaluate(passages, units)
                result["rows"].append({"document_id": doc_id, "field": field, "units": units,
                    "query": FIELD_QUERIES[QUERY_KEYS[field]], **measured, "retrieved": passages})
            print(json.dumps({"event": "retrieval", "mode": mode, "document": doc_id, "seconds": round(elapsed, 3)}), flush=True)
        # Same scoring function, now using saved matrices, to separate encoding
        # cost from ranking. These runs must reproduce every original result.
        if encoder:
            encoder.cached_only = True
        cached_times = []
        for repeat in range(3):
            for document in result["documents"]:
                doc_id = document["document_id"]
                started = time.perf_counter()
                evidence = retriever.retrieve(data[doc_id][0]["ocr_text"], top_k=5, chunking_mode="legacy", document_id=doc_id)
                cached_times.append(time.perf_counter() - started)
                if evidence != document["all_results"]:
                    raise ValueError("Cached vectors changed retrieval output")
        result["cached_batch_seconds"] = cached_times
        result["encode_events"] = encoder.events if encoder else []
        result["vector_cache_bytes"] = cache.size_bytes() if cache else 0
        result["vector_cache_path"] = str(cache.path) if cache else None
    result["memory"] = {"baseline_rss_bytes": memory.baseline, "peak_rss_bytes": memory.peak,
                         "peak_increment_bytes": memory.peak - memory.baseline,
                         "minimum_system_available_bytes": memory.min_available,
                         "sampling_interval_ms": 50}
    result["performance"] = {
        "corpus_embedding_seconds": sum(e["seconds"] for e in result["encode_events"] if e["kind"] == "corpus"),
        "query_embedding_seconds": sum(e["seconds"] for e in result["encode_events"] if e["kind"] == "queries"),
        "mean_uncached_batch_seconds": mean(d["uncached_batch_seconds"] for d in result["documents"]),
        "mean_uncached_amortized_query_seconds": mean(d["uncached_batch_seconds"] for d in result["documents"]) / len(FIELD_QUERIES),
        "mean_cached_batch_seconds": mean(cached_times),
        "mean_cached_amortized_query_seconds": mean(cached_times) / len(FIELD_QUERIES)}
    if mode == "minilm":
        old = {(r["document_id"], r["field"]): r["legacy"] for r in previous["rows"]}
        result["phase2b_rank_equivalence"] = all(r["ranks"] == old[(r["document_id"], r["field"])]["ranks"] for r in result["rows"])
        if not result["phase2b_rank_equivalence"]:
            raise ValueError("MiniLM baseline ranks do not reproduce Phase 2B")
    write_json(OUTPUT / f"{mode}.json", result)
    print(json.dumps({"event": "complete", "mode": mode, "performance": result["performance"], "memory": result["memory"]}), flush=True)


def offline_worker(mode):
    def deny(*args, **kwargs):
        raise RuntimeError("Network disabled during embedding evaluation")

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"EXTRACTION_SEMANTIC_RAG": "false" if mode == "bm25" else "true",
            "RAG_EMBEDDING_MODEL": "minilm" if mode == "bm25" else mode,
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}))
        stack.enter_context(patch.object(socket.socket, "connect", deny))
        stack.enter_context(patch.object(socket.socket, "connect_ex", deny))
        worker(mode)


def metrics(rows):
    ranks = [rank for r in rows for rank in r["ranks"]]
    found = [rank for rank in ranks if rank is not None]
    return {"fields": len(rows), "units": len(ranks), "hits_at_5": len(found),
        "macro_recall": {str(k): mean(r["recall"][str(k)] for r in rows) for k in (1, 3, 5)},
        "micro_recall": {str(k): sum(r is not None and r <= k for r in ranks) / len(ranks) for k in (1, 3, 5)},
        "mean_rank_found_only": mean(found) if found else None,
        "mean_censored_rank_6": mean(r if r is not None else 6 for r in ranks),
        "mrr_at_5": mean(1 / r if r is not None else 0 for r in ranks)}


def compare_units(minilm, bge):
    by_key = {(r["document_id"], r["field"]): r for r in bge}
    cases = []
    for old in minilm:
        new = by_key[(old["document_id"], old["field"])]
        for unit, a, b in zip(old["units"], old["ranks"], new["ranks"]):
            both = a is not None and b is not None
            cases.append({"document_id": old["document_id"], "field": old["field"], "unit": unit["id"],
                "minilm_rank": a, "bge_rank": b, "A_both_succeed": both,
                "B_bge_better": (b or 6) < (a or 6), "C_minilm_better": (a or 6) < (b or 6),
                "D_leaves_top5": "bge-m3" if a is not None and b is None else "minilm" if b is not None and a is None else None,
                "both_fail": a is None and b is None})
    return cases


def integrity():
    from extraction_retrieval import FIELD_QUERIES
    from qa_chunking_benchmark import configured_mode

    previous = read_json(PHASE2 / "INTEGRITY.json")["after"]
    allowed = {"extraction_retrieval.py", ".env.example"}
    stable = {name: digest((ROOT / name).read_bytes()) for name in previous if name not in allowed}
    if stable != {name: value for name, value in previous.items() if name not in allowed}:
        raise ValueError("An application file outside Phase 3 scope changed since Phase 2B")
    # Reconstruct only the explicitly authorized loader/constructor edits. This
    # verifies byte-for-byte that all other retrieval logic remains Phase 2B.
    code = (ROOT / "extraction_retrieval.py").read_text(encoding="utf-8")
    original = code.replace("from extraction_embeddings import embedding_spec, load_embedding_model\n", "")
    original = original.replace("def __init__(self, embedding_model=None):\n        self.embedding_model = embedding_spec(embedding_model).name", "def __init__(self):")
    original = original.replace("            self.model = load_embedding_model(self.embedding_model)",
        '            from sentence_transformers import SentenceTransformer\n            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu", local_files_only=True)')
    candidates = {digest(original.encode()), digest(original.replace("\n", "\r\n").encode())}
    if previous["extraction_retrieval.py"] not in candidates:
        raise ValueError("Retrieval changes exceed constructor/model loading scope")
    chatbot = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in (ROOT / "rag").rglob("*.py")}
    frozen = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for record in read_json(PHASE2 / "inventory.json")
              for p in (Path(record["ocr_file"]), Path(record["layout_file"]))}
    return {"stable_app_files": stable, "chatbot": chatbot, "frozen_sources": frozen,
            "annotations_sha256": digest((PHASE2 / "annotations.json").read_bytes()),
            "queries": FIELD_QUERIES, "settings": {**configured_mode(),
            "embedding_process_setting": os.getenv("RAG_EMBEDDING_MODEL"), "embedding_code_default": "minilm"},
            "retrieval_unchanged_except_loader": True}


def summarize():
    from qa_chunking_benchmark import QUERY_KEYS

    runs = {mode: read_json(OUTPUT / f"{mode}.json") for mode in MODES}
    annotation_hashes = {r["annotation_sha256"] for r in runs.values()}
    if len(annotation_hashes) != 1:
        raise ValueError("Models evaluated different annotations")
    all_metrics = {m: {"global": metrics(r["rows"]),
        "by_field": {f: metrics([row for row in r["rows"] if row["field"] == f]) for f in QUERY_KEYS},
        "by_document": {d["document_id"]: metrics([row for row in r["rows"] if row["document_id"] == d["document_id"]]) for d in r["documents"]}}
        for m, r in runs.items()}
    cases = compare_units(runs["minilm"]["rows"], runs["bge-m3"]["rows"])
    categories = {key: sum(bool(c[key]) for c in cases) for key in ("A_both_succeed", "B_bge_better", "C_minilm_better", "both_fail")}
    categories["D_bge_leaves_top5"] = sum(c["D_leaves_top5"] == "bge-m3" for c in cases)
    categories["D_minilm_leaves_top5"] = sum(c["D_leaves_top5"] == "minilm" for c in cases)
    common = [c for c in cases if c["A_both_succeed"]]
    common_means = {"count": len(common), "minilm": mean(c["minilm_rank"] for c in common), "bge-m3": mean(c["bge_rank"] for c in common)}
    conclusion = recommendation(all_metrics, cases, runs)
    summary = {"metrics": all_metrics, "categories": categories, "unit_comparisons": cases,
               "mean_rank_common_hits": common_means, "recommendation": conclusion,
               "annotation_sha256": next(iter(annotation_hashes)), "llm_calls": 0}
    write_json(OUTPUT / "summary.json", summary)
    report(runs, summary)
    print(json.dumps({"metrics": {m: all_metrics[m]["global"]["macro_recall"] for m in MODES},
                      "categories": categories, "recommendation": conclusion}), flush=True)


def recommendation(all_metrics, cases, runs):
    recalls = {m: all_metrics[m]["global"]["macro_recall"] for m in MODES}
    if all(recalls["bm25"][str(k)] >= recalls[m][str(k)] for m in ("minilm", "bge-m3") for k in (1, 3, 5)):
        return "D. Sur ce jeu QA, les embeddings n'apportent pas assez face a BM25 seul aux trois seuils. Ne pas modifier la configuration sur seulement sept factures ; confirmer sur un jeu independant."
    bge_regressions = sum(c["C_minilm_better"] for c in cases)
    if bge_regressions or recalls["bge-m3"]["5"] <= recalls["minilm"]["5"]:
        return (f"A. Garder MiniLM par defaut. BGE-M3 degrade le rang de {bge_regressions} unites ; "
                f"Recall@5 macro : MiniLM {recalls['minilm']['5']:.2%}, BGE-M3 {recalls['bge-m3']['5']:.2%}. "
                "Les gains eventuels de certains champs ne suffisent pas a valider un remplacement general ou un routage selectif non teste. "
                "Conserver BGE-M3 comme option explicite, sans changer le defaut.")
    ratio = runs["bge-m3"]["performance"]["mean_uncached_batch_seconds"] / runs["minilm"]["performance"]["mean_uncached_batch_seconds"]
    if ratio > 5:
        return (f"A. Garder MiniLM pour l'usage CPU actuel malgre le gain Recall de BGE-M3 : "
                f"son temps moyen de retrieval non cache est {ratio:.1f} fois superieur. "
                "Une utilisation selective reste une hypothese a evaluer, pas une strategie validee par ce benchmark.")
    return "B. Les metriques et le cout CPU de ce lot favorisent BGE-M3 sans regression de rang observee. Confirmer sur un jeu independant avant generalisation. Aucun changement automatique."


def report(runs, summary):
    grouped = summary["metrics"]
    labels = {"bm25": "BM25", "minilm": "MiniLM+BM25", "bge-m3": "BGE-M3+BM25"}
    lines = ["# Phase 3 : benchmark CPU des embeddings", "", "## Resume", "",
        "7 factures, 63 champs document/champ, 82 unites de preuve, annotations Phase 2B inchangees.",
        "189 evaluations annotees (63 x 3), 21 appels non caches au retriever qui calcule toujours 22 requetes par appel, soit 462 classements internes. 63 appels caches supplementaires pour chronometrage et controle d'identite des sorties.",
        "Seule la composante dense change : aucun embedding / all-MiniLM-L6-v2 / BAAI/bge-m3. Aucun appel LLM, OCR, endpoint ou base applicative.", "",
        "## Protocole", "",
        "- Legacy strictement identique, top_k=5, requetes Phase 2B exactes, BM25 k1=1.2/b=0.75, RRF k=60. Aucun reranker.",
        "- BGE-M3 utilise uniquement ses vecteurs denses SentenceTransformers. Pas de sparse/ColBERT, pas d'instruction ajoutee aux requetes, pas de changement de seuil semantique (0.35).",
        "- Normalisation L2, float32, CPU, batch_size=32 (defaut encode existant), 4 threads PyTorch pour les deux modeles. Le max_seq_length natif de chaque modele est conserve ; nombre d'entrees tronquees fourni ci-dessous.",
        "- Le filtre BM25 > 0 OU similarite >= 0.35 est inchange. Les distributions de similarite peuvent differer selon le modele ; elles ne sont pas recalibrees pour aider BGE-M3.",
        "- Fournisseur : premier chunk force. Produits : chunks table forces en ordre source, jusqu'a 12, sans nouveau tri. Mesure des premieres 1/3/5 sorties. Ne pas attribuer au modele un gain que cette politique rend impossible.",
        "- Meme definition qu'en Phase 2B : bon document/page et tous les line_ids d'une preuve dans un meme chunk. Alternatives explicites equivalentes ; produits avec en-tete obligatoire.",
        "- Macro = meme poids par document/champ ; micro = meme poids par unite de preuve. Les rangs non retrouves sont censures a 5, pas des rangs globaux exacts.",
        "- Un processus distinct par configuration. Chaque modele charge une seule fois et reutilise. Evaluation reseau bloque ; telechargement des poids separe.",
        "- MiniLM doit reproduire les rangs legacy Phase 2B. Verification stricte des chunks, annotations, requetes et du code retrieval hors constructeur/chargement.", "",
        "## Recall global (macro)", "", "| Mesure | BM25 | MiniLM+BM25 | BGE-M3+BM25 |", "|---|---:|---:|---:|"]
    for k in (1, 3, 5):
        lines.append(f"| Recall@{k} | " + " | ".join(f"{grouped[m]['global']['macro_recall'][str(k)]:.2%}" for m in MODES) + " |")
    lines += ["", "Micro et rangs (unites de preuve) :", "", "| Mesure | BM25 | MiniLM+BM25 | BGE-M3+BM25 |", "|---|---:|---:|---:|"]
    for k in (1, 3, 5):
        lines.append(f"| Micro Recall@{k} | " + " | ".join(f"{grouped[m]['global']['micro_recall'][str(k)]:.2%}" for m in MODES) + " |")
    for key in ("mean_rank_found_only", "mean_censored_rank_6", "mrr_at_5"):
        lines.append(f"| {key} | " + " | ".join(f"{grouped[m]['global'][key]:.3f}" for m in MODES) + " |")
    cm = summary["mean_rank_common_hits"]
    lines += ["", "Le rang moyen des seuls succes peut etre trompeur si les modeles ne retrouvent pas les memes preuves. Le rang penalise utilise 6 pour une absence du Top 5, sans pretendre connaitre le rang reel.",
              f"Sur les {cm['count']} preuves communes retrouvees par les deux modeles : rang moyen MiniLM {cm['minilm']:.3f}, BGE-M3 {cm['bge-m3']:.3f}."]
    for group, title in (("by_field", "Par champ"), ("by_document", "Par facture")):
        lines += ["", f"## {title}", "", "Chaque cellule : Recall@1 / @3 / @5 macro.", "", "| Groupe | N champs/unites | BM25 | MiniLM+BM25 | BGE-M3+BM25 |", "|---|---:|---|---|---|"]
        for key in grouped["bm25"][group]:
            item = grouped["bm25"][group][key]
            values = [" / ".join(f"{grouped[m][group][key]['macro_recall'][str(k)]:.2%}" for k in (1, 3, 5)) for m in MODES]
            lines.append(f"| {key} | {item['fields']}/{item['units']} | {' | '.join(values)} |")
    lines += ["", "## Comparaisons MiniLM / BGE-M3", "",
        "Les categories demandees se recouvrent : A peut aussi appartenir a B ou C ; D est un sous-ensemble des transitions B/C. Elles ne doivent pas etre additionnees comme quatre classes exclusives.", "",
        "| Categorie | Unites |", "|---|---:|"]
    for key, count in summary["categories"].items():
        lines.append(f"| {key} | {count} |")
    for title, flag in (("Ameliorations BGE-M3", "B_bge_better"), ("Regressions BGE-M3", "C_minilm_better")):
        lines += ["", f"### {title}", "", "| Facture | Champ/unite | MiniLM | BGE-M3 | Sortie Top 5 |", "|---|---|---:|---:|---|"]
        selected = [c for c in summary["unit_comparisons"] if c[flag]]
        if not selected:
            lines.append("| Aucune | | | | |")
        for case in selected:
            lines.append(f"| {case['document_id']} | {case['field']}/{case['unit']} | {case['minilm_rank'] or '>5/absent'} | {case['bge_rank'] or '>5/absent'} | {case['D_leaves_top5'] or 'non'} |")
    lines += ["", "## Cout CPU et RAM", "",
        "Temps chronometres sur ce poste pendant que l'application reste ouverte. Une seule passe non cachee sur 7 documents, puis 3 passes cachees ; pas un benchmark de charge ni un p95 fiable.",
        "Le retriever calcule 22 requetes ensemble : temps/requete = temps du lot / 22, cout AMORTI, pas la latence isolee d'une API une-requete. Temps non cache inclut l'ecriture des matrices QA ; encodage pur fourni separement.",
        "RAM = RSS du processus echantillonnee toutes les 50 ms ; pic approximatif, pas toute la RAM systeme. Initialisation inclut les imports ML et la construction du modele, pas son telechargement.", "",
        "| Mesure | BM25 | MiniLM+BM25 | BGE-M3+BM25 |", "|---|---:|---:|---:|"]
    measures = [
        ("Initialisation (s)", lambda r: r["initialization_seconds"]),
        ("Embedding corpus 51 chunks (s)", lambda r: r["performance"]["corpus_embedding_seconds"]),
        ("Embedding requetes, 22 x 7 (s)", lambda r: r["performance"]["query_embedding_seconds"]),
        ("Retrieval non cache / document (s)", lambda r: r["performance"]["mean_uncached_batch_seconds"]),
        ("Non cache / requete amortie (ms)", lambda r: 1000*r["performance"]["mean_uncached_amortized_query_seconds"]),
        ("Cache / document (ms)", lambda r: 1000*r["performance"]["mean_cached_batch_seconds"]),
        ("Cache / requete amortie (ms)", lambda r: 1000*r["performance"]["mean_cached_amortized_query_seconds"]),
        ("RAM de base (MiB)", lambda r: r["memory"]["baseline_rss_bytes"]/2**20),
        ("Pic RAM RSS (MiB)", lambda r: r["memory"]["peak_rss_bytes"]/2**20),
        ("Surcout RAM au pic (MiB)", lambda r: r["memory"]["peak_increment_bytes"]/2**20),
        ("Cache vecteurs QA + metadata (KiB)", lambda r: r["vector_cache_bytes"]/1024),
        ("Snapshot modele sur disque (MiB)", lambda r: r.get("model_snapshot_bytes",0)/2**20),
        ("Entrees tronquees", lambda r: r.get("truncated_inputs",0))]
    for name, getter in measures:
        lines.append(f"| {name} | " + " | ".join(f"{getter(runs[m]):.3f}" for m in MODES) + " |")
    if (OUTPUT / "download.json").exists():
        dl = read_json(OUTPUT / "download.json")
        lines += ["", f"Telechargement initial BGE-M3 : {dl['download_seconds']:.1f} s, {dl['bytes']/2**30:.3f} GiB. Temps totalement exclu des mesures d'execution."]
    lines += ["", "## Modeles et caches", "",
        "L'extraction applicative n'avait pas et n'ajoute pas d'index vectoriel persistant : elle recalcule ses vecteurs. Seuls les poids sont caches par Hugging Face et les modeles charges sont reutilises par processus.",
        "Le benchmark archive ses matrices dans vectors/<alias>/<empreinte-identite>-d<dimension>/. Identite = repository + revision + versions + pooling/config + dimension + normalisation + dtype + longueur maximale. Chaque matrice controle aussi le texte ordonne, sa forme et son SHA256. Lecture inter-modele refusee.",
        "Les embeddings MiniLM existants et la SQLite RAG du chatbot ne sont ni lus par le benchmark, ni modifies. Aucun cache vectoriel QA n'est branche automatiquement a l'application."]
    for mode in ("minilm", "bge-m3"):
        identity = runs[mode]["identity"]
        lines += ["", f"- {mode} : {identity['model']} @ {identity['revision']}, dimension {identity['dimension']}, max_seq_length {identity['max_seq_length']}, maximum observe {runs[mode]['max_input_tokens']} tokens."]
    lines += ["", "## Interpretation semantique et limites", "",
        "Les requetes n'ont pas ete paraphrasees : ajouter 'montant reellement a regler' aurait change une deuxieme variable. Le gain observe par champ est mesurable ; la generalisation a des paraphrases libres ne l'est pas dans cette phase.",
        "invoice_number et supplier_tax_id mesurent la recherche de preuves d'identifiants, pas une recherche par numero exact fourni par l'utilisateur. Ne pas confondre ces deux taches.",
        "Le net a payer et le moyen de paiement ne sont annotes que sur une facture chacun. Sept factures deja utilisees, dont deux synthetiques anglaises, pas de multipage complet ni d'acompte : conclusions limitees a ce lot, pas de garantie de generalisation.",
        "Les absences dues au decoupage legacy des tableaux persistent quel que soit l'embedding. Ce test ne revalide ni OCR, ni extraction LLM, ni regles metier.", "",
        "## Recommandation", "", summary["recommendation"], "",
        "MiniLM et legacy restent les defauts. Aucun redemarrage serveur ni changement automatique de configuration.", "",
        "## Reproduction et sources", "",
        "`python qa_embedding_benchmark.py --download` pour les poids manquants, puis `python qa_embedding_benchmark.py --run` pour les trois processus CPU. `--report` regenere seulement le rapport depuis les scores sauvegardes.",
        "Dependances utilisees : sentence-transformers, torch CPU, transformers, huggingface-hub, numpy et psutil deja installes. Pas de CUDA, vLLM, FlashAttention ni FlagEmbedding requis pour cette comparaison dense.",
        "[BAAI : BGE-M3, usage dense via SentenceTransformers, dimensions et absence d'instruction de requete](https://huggingface.co/BAAI/bge-m3). Les scores et temps de ce rapport proviennent exclusivement des mesures locales.",
        "Details : summary.json, bm25.json, minilm.json, bge-m3.json, EXAMPLES_TOP5.md et INTEGRITY.json. Les Top 3 sont les trois premiers passages de chaque Top 5 fourni.", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    examples = ["# Tous les Top 5 (Top 3 = rangs 1 a 3)", "", "Meme texte OCR, memes requetes et line_ids attendus. N/A = candidat force sans score expose.", ""]
    indexes = {m: {(r["document_id"], r["field"]): r for r in run["rows"]} for m, run in runs.items()}
    for key in indexes["bm25"]:
        reference = indexes["bm25"][key]
        examples += [f"## {key[0]} / {key[1]}", "", f"Requete : {reference['query']}", ""]
        for unit in reference["units"]:
            examples.append(f"- {unit['id']} : " + " OU ".join(f"page {p['page']} L{p['line_ids']}" for p in unit["alternatives"]))
        for mode in MODES:
            row = indexes[mode][key]
            examples += ["", f"### {labels[mode]}", "", f"Rangs des preuves : {row['ranks']}", ""]
            for rank, passage in enumerate(row["retrieved"][:5],1):
                examples += [f"{rank}. {passage['source_id']} | L{passage['line_ids']} | BM25 {passage.get('lexical_score','N/A')} / semantique {passage.get('semantic_score','N/A')} / RRF {passage.get('score','N/A')}",
                             "", "```text", passage["text"], "```", ""]
            if not row["retrieved"]:
                examples.append("Aucun passage.")
    (OUTPUT / "EXAMPLES_TOP5.md").write_text("\n".join(examples), encoding="utf-8")


def run_all():
    import psutil

    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = integrity()
    write_json(OUTPUT / "hardware.json", {"platform": platform.platform(), "python": platform.python_version(),
        "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total, "memory_available_bytes": psutil.virtual_memory().available})
    for mode in MODES:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", mode], cwd=ROOT, check=True)
    after = integrity()
    if before != after:
        raise ValueError("Protected files changed during benchmark")
    write_json(OUTPUT / "INTEGRITY.json", {"before": before, "after": after, "equal": True})
    summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--worker", choices=MODES)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.download:
        download()
    elif args.worker:
        offline_worker(args.worker)
    elif args.run:
        run_all()
    elif args.report:
        summarize()
    else:
        parser.print_help()
