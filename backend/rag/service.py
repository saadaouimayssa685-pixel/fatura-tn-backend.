import os
import re
import shutil
from typing import Any, Dict, List, Optional

from .chunker import DocumentChunker
from .embeddings import EmbeddingProvider, cosine_similarity
from .postgres_store import PostgresRAGStore
from .store import SQLiteRAGStore


class RAGService:
    """Indexes extracted documents and retrieves relevant chunks for questions."""

    def __init__(
        self,
        database_url: str,
        storage_dir: str = "rag_storage/documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sqlite_db_path: Optional[str] = None,
    ):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        if database_url:
            self.store = PostgresRAGStore(database_url)
        else:
            self.store = SQLiteRAGStore(sqlite_db_path or os.path.join(storage_dir, "rag.sqlite3"))
        self.chunker = DocumentChunker()
        self.embeddings = EmbeddingProvider(model_name=embedding_model)

    def index_document(
        self,
        filename: str,
        file_type: str,
        source_file_path: str,
        extracted_text: str,
        extraction_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not extracted_text.strip():
            raise ValueError("Cannot index an empty document")

        stored_file_path = self._persist_source_file(source_file_path, filename)
        document_id = self.store.add_document(
            filename=filename,
            file_type=file_type,
            file_path=stored_file_path,
            text_length=len(extracted_text),
            metadata={
                "extraction": extraction_metadata,
                "embedding_backend": self.embeddings.backend,
            },
        )

        chunks = self.chunker.split(extracted_text)
        for chunk in chunks:
            embedding = self.embeddings.embed(chunk.text)
            self.store.add_chunk(
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                chunk_text=chunk.text,
                embedding=embedding,
                metadata={"chars": len(chunk.text)},
            )

        return {
            "document_id": document_id,
            "filename": filename,
            "file_type": file_type,
            "stored_file_path": stored_file_path,
            "chunks_count": len(chunks),
            "text_length": len(extracted_text),
            "embedding_backend": self.embeddings.backend,
        }

    def query(self, question: str, document_id: Optional[int] = None, top_k: int = 5) -> Dict[str, Any]:
        if not question.strip():
            raise ValueError("Question is required")

        top_k = max(1, min(top_k, 20))
        query_embedding = self.embeddings.embed(question)
        chunks = self.store.get_chunks(document_id=document_id)

        scored_chunks: List[Dict[str, Any]] = []
        for chunk in chunks:
            semantic_score = cosine_similarity(query_embedding, chunk["embedding"])
            lexical_score = self._lexical_score(question, chunk["chunk_text"])
            score = (semantic_score * 0.75) + (lexical_score * 0.25)
            scored_chunks.append({
                "score": round(float(score), 4),
                "semantic_score": round(float(semantic_score), 4),
                "lexical_score": round(float(lexical_score), 4),
                "chunk_id": chunk["id"],
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "page_number": chunk["page_number"],
                "text": chunk["chunk_text"],
                "metadata": chunk["metadata"],
            })

        scored_chunks.sort(key=lambda item: item["score"], reverse=True)
        top_chunks = scored_chunks[:top_k]

        return {
            "question": question,
            "answer": self._build_extractive_answer(question, top_chunks),
            "sources": top_chunks,
            "retrieval": {
                "top_k": top_k,
                "chunks_searched": len(chunks),
                "embedding_backend": self.embeddings.backend,
            },
            "llm_prompt": self._build_llm_prompt(question, top_chunks),
        }

    def list_documents(self):
        return self.store.list_documents()

    def _persist_source_file(self, source_file_path: str, filename: str) -> str:
        safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in filename)
        base_name, extension = os.path.splitext(safe_name)
        candidate = os.path.join(self.storage_dir, safe_name)
        counter = 1

        while os.path.exists(candidate):
            candidate = os.path.join(self.storage_dir, f"{base_name}_{counter}{extension}")
            counter += 1

        shutil.copy2(source_file_path, candidate)
        return candidate

    def _build_extractive_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "Aucun contexte indexé ne permet de répondre à cette question."

        best = chunks[0]
        page = f" page {best['page_number']}" if best.get("page_number") else ""
        return (
            f"Passage le plus pertinent trouvé dans le document {best['document_id']}{page}. "
            "Utilise les sources retournées pour générer une réponse finale avec un LLM."
        )

    def _build_llm_prompt(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        context = "\n\n".join(
            f"[Source {index + 1} | document {chunk['document_id']} | page {chunk.get('page_number') or 'N/A'}]\n{chunk['text']}"
            for index, chunk in enumerate(chunks)
        )
        return (
            "Réponds à la question uniquement avec le contexte fourni. "
            "Si l'information n'est pas présente, dis que le document ne permet pas de répondre.\n\n"
            f"Question: {question}\n\n"
            f"Contexte:\n{context}"
        )

    def _lexical_score(self, question: str, text: str) -> float:
        question_tokens = self._tokens(question)
        text_tokens = set(self._tokens(text))
        if not question_tokens or not text_tokens:
            return 0.0

        overlap = sum(1 for token in question_tokens if token in text_tokens)
        score = overlap / len(question_tokens)

        normalized_question = " ".join(question_tokens)
        important_phrases = ["total ttc", "tva", "rib", "fournisseur", "client", "date facture"]
        for phrase in important_phrases:
            if phrase in normalized_question and phrase in text.lower():
                score += 0.35

        return min(score, 1.0)

    def _tokens(self, value: str) -> List[str]:
        return [
            token
            for token in re.findall(r"[\wÀ-ÿ]+", value.lower())
            if len(token) > 1
        ]
