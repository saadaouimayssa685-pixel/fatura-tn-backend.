import React, { useEffect, useState } from "react";
import { Search, Loader2 } from "lucide-react";
import { API_BASE_URL, apiFetch } from "./api";

type Source = { document_id: number; page_number?: number; text: string; score: number };
export function RagAssistant() {
  const [question, setQuestion] = useState("");
  const [documents, setDocuments] = useState<{ id: number; filename: string }[]>([]);
  const [documentId, setDocumentId] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [searched, setSearched] = useState(false);
  const [answer, setAnswer] = useState("");
  const [model, setModel] = useState("");
  useEffect(() => {
    let active = true;
    apiFetch(`${API_BASE_URL}/rag/documents`).then(async response => {
      if (!response.ok) throw new Error("Impossible de charger les documents indexés.");
      const data = await response.json();
      if (active) setDocuments(data.documents || []);
    }).catch(error => { if (active) setError(error.message); });
    return () => { active = false; };
  }, []);
  async function search(event: React.FormEvent) {
    event.preventDefault();
    if (!question.trim() || busy) return;
    setBusy(true); setError(""); setSources([]); setSearched(false); setAnswer(""); setModel("");
    try {
      const response = await apiFetch(`${API_BASE_URL}/rag/query`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, document_id: documentId ? Number(documentId) : null, top_k: 5 }),
      });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.detail || data.error || "Recherche indisponible.");
      setSources(data.sources || []); setSearched(true);
      if (data.generation?.status === "completed") {
        setAnswer(data.answer || "");
        setModel(data.generation?.model || "LLM local");
      } else setError(data.generation?.message || "LLM local indisponible : seuls les passages sources sont affichés.");
    } catch (error) { setError(error instanceof Error ? error.message : "Recherche indisponible."); }
    finally { setBusy(false); }
  }
  return <section className="rag-live">
    <h1>Recherche RAG</h1>
    <form onSubmit={search}>
      <label>Document<select value={documentId} onChange={event => setDocumentId(event.target.value)}>
        <option value="">Tous les documents indexés</option>
        {documents.map(document => <option key={document.id} value={document.id}>{document.filename}</option>)}
      </select></label>
      <label>Question<textarea required value={question} onChange={event => setQuestion(event.target.value)} /></label>
      <button className="btn btn-primary" disabled={busy || !question.trim()}>
        {busy ? <Loader2 size={18} /> : <Search size={18} />} {busy ? "Recherche…" : "Rechercher"}
      </button>
    </form>
    {error && <p role="alert">{error}</p>}
    {answer && <article><h2>Réponse RAG {model ? `(${model})` : "locale"}</h2><p style={{ whiteSpace: "pre-wrap" }}>{answer}</p></article>}
    {searched && sources.length === 0 && <p>Aucun passage pertinent retrouvé.</p>}
    {sources.length > 0 && <h2>Passages sources</h2>}
    {sources.map((source, index) => <article key={`${source.document_id}-${index}`}>
      <h3>[{index + 1}] {documents.find(document => Number(document.id) === source.document_id)?.filename || `Document ${source.document_id}`}</h3>
      <small>{source.page_number ? `Page ${source.page_number}` : "Page non renseignée"}</small>
      <blockquote>{source.text}</blockquote>
    </article>)}
  </section>;
}
