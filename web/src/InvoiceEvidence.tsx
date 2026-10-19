import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ChevronDown, ChevronLeft, ChevronRight, Download, LocateFixed, Minus, Plus, X } from "lucide-react";
import type { PDFDocumentProxy } from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import { equalEvidenceValue } from "./reviewData";

export function SourceHighlight({ location }: { location: any }) {
  const firstHighlight = useRef<HTMLSpanElement>(null);
  useEffect(() => { firstHighlight.current?.scrollIntoView({ block: "center", behavior: "smooth" }); }, [location]);
  const boxes = location?.boxes?.length ? location.boxes : location?.box ? [location.box] : [];
  return <>{boxes.filter((box: number[]) => box.length === 4 && box.every(Number.isFinite)
    && box[0] >= 0 && box[1] >= 0 && box[2] > 0 && box[3] > 0
    && box[0] + box[2] <= 1.002 && box[1] + box[3] <= 1.002).map((box: number[], index: number) =>
    <span ref={index === 0 ? firstHighlight : undefined} key={index} className="field-source-highlight" title={location.matched_text || location.text}
      style={{ left: `${box[0] * 100}%`, top: `${box[1] * 100}%`, width: `${box[2] * 100}%`, height: `${box[3] * 100}%` }} />)}</>;
}

export function FieldEvidence({ fieldKey, value, proof, isPdf, active, onLocate }: {
  fieldKey: string; value: string; proof: any; isPdf: boolean; active: boolean; onLocate: (location: any) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (expanded) dialog.current?.showModal();
    else dialog.current?.close();
  }, [expanded]);
  useEffect(() => setExpanded(false), [value, proof]);
  if (!value && !proof?.value) return null;
  const normalize = (v: unknown) => String(v ?? "").replace(/\s/g, "").replace(/,/g, ".").toLowerCase();
  const isDate = ["date", "due_date"].includes(fieldKey);
  const sameValue = equalEvidenceValue(value, proof?.value)
    || (isDate && normalize(value).replace(/[-.]/g, "/") === normalize(proof?.value).replace(/[-.]/g, "/"))
    || (/^\d+[.]\d+$/.test(normalize(value)) && Number(normalize(value)) === Number(proof?.value));
  if (!proof) return <small className="field-proof-note">Saisie humaine ou source non disponible</small>;
  if (proof?.source === "workflow_default_tnd" || proof?.default_applied) return <small className="field-proof-note">{sameValue ? "Devise par défaut, non prouvée par OCR" : "Valeur corrigée manuellement"}</small>;
  const hasRag = proof?.source?.type === "rag_chunk" && (proof.source.text || proof.source.preview);
  const visual = proof?.source?.type === "visual_model";
  const partyBlock = proof?.source?.type === "ocr_party_block";
  const location = hasRag && proof?.source?.page_number && proof.location?.page_number !== proof.source.page_number
    ? null : proof?.location;
  const citation = proof.source?.text || proof.source?.preview || location?.text;
  if (!citation) return <small className="field-proof-note">{isPdf ? "Preuve RAG non disponible" : "Source OCR non localisée"}</small>;
  const page = location?.page_number || proof.source?.page_number;
  const title = visual ? "Lecture visuelle locale · à confirmer" : hasRag ? "Preuves RAG" : "Source OCR · non retrouvée par RAG";
  return <span className={`field-proof ${active ? "is-selected" : ""}`}>
    {!sameValue && <small className="field-proof-note">Valeur modifiée · preuve de la valeur originale</small>}
    <span className="field-proof-heading"><small>{title}{page ? ` · page ${page}` : " · page inconnue"}</small>
      {location?.box && <button type="button" className="field-locate" aria-pressed={active}
        aria-label={`Surligner la source : ${proof.value}`} title={active ? "Effacer le surlignage" : "Surligner la source originale"}
        onClick={(event) => {
          event.preventDefault();
          onLocate(active ? null : { ...location, field_key: fieldKey });
          if (!active) document.getElementById("invoice-source-preview")?.scrollIntoView({ behavior: "smooth", block: "start" });
        }}>{active ? <X size={15} /> : <LocateFixed size={15} />}</button>}
    </span>
    <button type="button" className="field-proof-toggle" aria-expanded={expanded}
      onClick={(event) => { event.preventDefault(); setExpanded(!expanded); }}>
      Voir la preuve
      <ChevronDown size={14} style={{ transform: expanded ? "rotate(180deg)" : undefined }} />
    </button>
    {expanded && createPortal(<dialog ref={dialog} aria-label="Preuve du champ" onCancel={() => setExpanded(false)}
      style={{ width: "min(560px, calc(100vw - 32px))", maxHeight: "80vh", padding: 24, border: "1px solid #ccd8e8", borderRadius: 8, color: "#173657" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16 }}>
        <strong>{title}{page ? ` · page ${page}` : ""}</strong>
        <button type="button" aria-label="Fermer la preuve" onClick={() => setExpanded(false)}><X size={20} /></button>
      </div>
      <p><strong>{sameValue ? value : proof.value}</strong>{!sameValue && <small> · valeur originale extraite</small>}</p>
      <span className="field-proof-context" style={{ display: "block", whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
      {hasRag || visual || partyBlock ? <span dir="auto">{citation}</span> : (location?.context?.length ? location.context : [{ text: citation, line_number: location?.line_number }]).map((line: any, index: number) =>
        <span className={line.line_number === location?.line_number ? "is-source-line" : ""} key={index}>
          <small>{line.line_number ? `L${line.line_number}` : "OCR"}</small><span dir="auto">{line.text}</span>
        </span>)}
      </span>
      {location?.box ? <button type="button" className="btn" style={{ marginTop: 16 }} onClick={() => {
        onLocate({ ...location, field_key: fieldKey }); setExpanded(false);
        document.getElementById("invoice-source-preview")?.scrollIntoView({ behavior: "smooth", block: "start" });
      }}><LocateFixed size={16} /> Surligner sur la facture</button> : <p>Passage retrouvé, coordonnées indisponibles.</p>}
    </dialog>, document.body)}
  </span>;
}

export function PdfInvoicePreview({ url, name, highlight }: { url: string; name: string; highlight?: any }) {
  const [document, setDocument] = useState<PDFDocumentProxy | null>(null);
  const [page, setPage] = useState(1);
  const [zoom, setZoom] = useState(100);
  const [ratio, setRatio] = useState(0.707);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const canvasHost = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let disposed = false;
    let task: ReturnType<typeof import("pdfjs-dist")["getDocument"]> | undefined;
    setDocument(null); setPage(1); setError(""); setReady(false);
    import("pdfjs-dist").then((pdfjs) => {
      if (disposed) return;
      pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
      task = pdfjs.getDocument({ url });
      return task.promise.then((pdf) => { if (!disposed) setDocument(pdf); });
    }).catch(() => { if (!disposed) setError("Aperçu PDF indisponible."); });
    return () => { disposed = true; void task?.destroy(); };
  }, [url]);
  useEffect(() => {
    const target = Number(highlight?.page_number);
    if (document && Number.isInteger(target) && target >= 1 && target <= document.numPages) setPage(target);
  }, [highlight, document]);
  useEffect(() => {
    if (!document) return;
    let disposed = false;
    let renderTask: ReturnType<Awaited<ReturnType<PDFDocumentProxy["getPage"]>>["render"]> | undefined;
    setReady(false); setError("");
    document.getPage(page).then(async (pdfPage) => {
      if (disposed) return;
      const viewport = pdfPage.getViewport({ scale: Math.min(window.devicePixelRatio || 1, 2) * zoom / 100 * 1.5 });
      // Each render owns a canvas so cancelled page/zoom changes cannot draw over the new page.
      const canvas = window.document.createElement("canvas");
      canvas.width = Math.ceil(viewport.width); canvas.height = Math.ceil(viewport.height);
      canvas.setAttribute("aria-label", `${name}, page ${page}`);
      setRatio(viewport.width / viewport.height);
      canvasHost.current?.replaceChildren(canvas);
      renderTask = pdfPage.render({ canvas, viewport });
      await renderTask.promise;
      if (!disposed) setReady(true);
    }).catch(() => { if (!disposed) setError("Impossible d'afficher cette page."); });
    return () => { disposed = true; renderTask?.cancel(); };
  }, [document, page, zoom, name]);
  return <div className="invoice-pdf-viewer">
    <div className="invoice-pdf-toolbar">
      <button type="button" title="Page précédente" aria-label="Page précédente" disabled={!document || page <= 1} onClick={() => setPage(page - 1)}><ChevronLeft size={17} /></button>
      <span>{page} / {document?.numPages || "…"}</span>
      <button type="button" title="Page suivante" aria-label="Page suivante" disabled={!document || page >= document.numPages} onClick={() => setPage(page + 1)}><ChevronRight size={17} /></button>
      <button type="button" title="Réduire le zoom" aria-label="Réduire le zoom" disabled={zoom <= 75} onClick={() => setZoom(zoom - 25)}><Minus size={17} /></button>
      <span>{zoom}%</span>
      <button type="button" title="Agrandir" aria-label="Agrandir" disabled={zoom >= 200} onClick={() => setZoom(zoom + 25)}><Plus size={17} /></button>
      <a href={url} download={name} title="Télécharger le PDF" aria-label="Télécharger le PDF"><Download size={17} /></a>
    </div>
    {error ? <div className="pdf-preview-error" role="alert">{error} <a href={url} target="_blank" rel="noreferrer">Ouvrir le document</a></div> :
      <div className="invoice-pdf-scroll">
        <div className="invoice-pdf-page" style={{ width: `${zoom}%`, aspectRatio: ratio }} aria-busy={!ready}>
          <div ref={canvasHost} className="invoice-pdf-canvas" />
          {!ready && <span className="pdf-preview-loading" role="status">Chargement…</span>}
          {ready && page === Number(highlight?.page_number) && <SourceHighlight location={highlight} />}
        </div>
      </div>}
  </div>;
}
