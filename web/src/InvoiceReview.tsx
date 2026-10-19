import React, { useEffect, useRef, useState } from "react";
import { ArrowLeft, Check, Download, Loader2, Minus, Plus, Save, Trash2, Undo2, X } from "lucide-react";
import { API_BASE_URL, apiFetch } from "./api";
import { FieldEvidence, PdfInvoicePreview, SourceHighlight } from "./InvoiceEvidence";
import { correctionPatch, displayValue, numericFields, reviewGroups } from "./reviewData";
import { detectionCategory, detectionScore, visibleDetections } from "./visualDetections";

async function request(url: string, init?: RequestInit) {
  const response = await apiFetch(`${API_BASE_URL}${url}`, init);
  const data = await response.json();
  if (!response.ok || data.success === false) throw new Error(typeof data.detail === "string" ? data.detail : data.message || "Opération impossible.");
  return data;
}

export function InvoiceReview({ invoiceId, navigate }: { invoiceId: string; navigate: (path: string) => void }) {
  const [invoice, setInvoice] = useState<any>(null);
  const [draft, setDraft] = useState<Record<string, any>>({});
  const [events, setEvents] = useState<any[]>([]);
  const [tab, setTab] = useState("Informations");
  const [highlight, setHighlight] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [zoom, setZoom] = useState(100);
  const [imageError, setImageError] = useState(false);
  const confirmDialog = useRef<HTMLDialogElement>(null);
  const source = invoiceId.startsWith("dataset:") ? "dataset" : "processed";
  const original = invoice?.normalized_data || {};
  const dirty = JSON.stringify(draft) !== JSON.stringify(original);
  const readOnly = source === "dataset";
  const raw = invoice?.raw_data || {};
  const proofs = raw.field_proofs || {};
  const isPdf = /pdf/i.test(invoice?.file_type || "") || /\.pdf$/i.test(invoice?.filename || "");
  const fileUrl = `${API_BASE_URL}/invoices/catalog/download?${new URLSearchParams({ source, invoice_id: invoiceId, inline: "true" })}`;
  const loadEvents = () => readOnly ? Promise.resolve() : request(`/database/validation-events?invoice_id=${encodeURIComponent(invoiceId)}`)
    .then(data => setEvents(data.events || [])).catch(() => setMessage("Historique des validations indisponible."));
  useEffect(() => {
    let cancelled = false;
    setLoading(true); setError(""); setHighlight(null); setImageError(false);
    request(`/invoices/catalog/detail?${new URLSearchParams({ source, invoice_id: invoiceId })}`)
      .then(data => { if (!cancelled) { setInvoice(data.invoice); setDraft(data.invoice.normalized_data || {}); } })
      .catch(e => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    void loadEvents();
    return () => { cancelled = true; };
  }, [invoiceId]);
  useEffect(() => {
    const guard = (event: BeforeUnloadEvent) => { if (dirty && !readOnly) { event.preventDefault(); event.returnValue = ""; } };
    window.addEventListener("beforeunload", guard);
    const onNavigate = (event: Event) => { if (dirty && !readOnly && !window.confirm("Quitter sans enregistrer les corrections ?")) event.preventDefault(); };
    window.addEventListener("fatura-before-navigate", onNavigate);
    return () => { window.removeEventListener("beforeunload", guard); window.removeEventListener("fatura-before-navigate", onNavigate); };
  }, [dirty, readOnly]);
  const update = (key: string, value: any) => { setDraft(current => ({ ...current, [key]: value })); setMessage(""); setHighlight(null); };
  async function save(validate: boolean) {
    setError(""); setMessage("");
    try {
      const patch = correctionPatch(original, draft);
      setBusy(true);
      const data = await request(`/invoices/${invoiceId}${validate ? "/validate" : ""}`, {
        method: validate ? "POST" : "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ normalized_data: patch, ...(validate ? {} : { status: "draft" }) }),
      });
      setInvoice(data.invoice); setDraft(data.invoice.normalized_data || {}); setHighlight(null);
      setMessage(validate ? "Validation humaine enregistrée." : "Corrections enregistrées en brouillon.");
      await loadEvents();
    } catch (e) { setError(e instanceof Error ? e.message : "Enregistrement impossible."); }
    finally { setBusy(false); confirmDialog.current?.close(); }
  }
  function exportJson() {
    const blob = new Blob([JSON.stringify({ invoice_id: invoice.id, status: invoice.status, fields: original, evidence: proofs, validation_events: events }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = `fatura-${invoice.id}.json`; link.click(); URL.revokeObjectURL(url);
  }
  if (loading) return <p role="status"><Loader2 size={18} /> Chargement de la facture…</p>;
  if (!invoice) return <div role="alert">{error || "Facture introuvable."}<button className="btn" onClick={() => navigate("/factures?source=processed")}>Retour aux factures</button></div>;
  const issues = raw.validation_issues || [];
  const detections = visibleDetections(raw.object_detections || []);
  return <section className="invoice-review">
    <button className="back-link" onClick={() => navigate(`/factures?source=${source}`)}><ArrowLeft size={16} />Retour aux factures</button>
    <div className="review-heading"><div><h1>{invoice.filename}</h1><p>{readOnly ? "Document du dataset · lecture seule" : invoice.status === "validated" ? "Validée par un utilisateur" : "À vérifier"}{dirty ? " · Modifications non enregistrées" : ""}</p>{raw.classification && <p>{raw.classification.is_invoice === true ? "Facture" : raw.classification.is_invoice === false ? "Non-facture" : "Classification à vérifier"}{typeof raw.classification.confidence === "number" ? ` · score de classification ${Math.round(raw.classification.confidence <= 1 ? raw.classification.confidence * 100 : raw.classification.confidence)}%` : ""}</p>}</div>
      <button className="btn btn-outline" onClick={exportJson} disabled={dirty} title="Exporter les données enregistrées"><Download size={16} />Exporter JSON</button></div>
    {error && <p role="alert" className="form-error">{error}</p>}
    {message && <p role="status" className="review-notice">{message}</p>}
    <div className="review-workspace">
      <section className="review-document" aria-label="Document source" id="invoice-source-preview">
        {isPdf ? <PdfInvoicePreview url={fileUrl} name={invoice.filename} highlight={highlight} /> : <>
          <div className="invoice-pdf-toolbar"><button title="Réduire le zoom" aria-label="Réduire le zoom" disabled={zoom <= 75} onClick={() => setZoom(z => z - 25)}><Minus size={17} /></button><span>{zoom}%</span><button title="Agrandir le zoom" aria-label="Agrandir le zoom" disabled={zoom >= 200} onClick={() => setZoom(z => z + 25)}><Plus size={17} /></button><a href={fileUrl.replace("inline=true", "inline=false")} download title="Télécharger le document" aria-label="Télécharger le document"><Download size={17} /></a>{highlight && <button title="Effacer le surlignage" aria-label="Effacer le surlignage" onClick={() => setHighlight(null)}><X size={17} /></button>}</div>
          <div className="review-image-scroll">{imageError ? <p role="alert">Aperçu indisponible. <a href={fileUrl} target="_blank" rel="noreferrer">Ouvrir le document</a></p> : <div className="review-image-page" style={{ width: `${zoom}%` }}><img src={fileUrl} alt={`Facture ${invoice.filename}`} onError={() => setImageError(true)} /><SourceHighlight location={highlight} /></div>}</div>
        </>}
      </section>
      <section className="review-data" aria-label="Vérification des données">
        {readOnly ? <section className="review-field-group"><h2>Document de référence</h2><p>Image du dataset, sans extraction utilisateur ni validation humaine.</p><dl><dt>Lot</dt><dd>{raw.split || "Non renseigné"}</dd><dt>Mots annotés</dt><dd>{raw.words_count ?? "Non renseigné"}</dd><dt>Zones annotées</dt><dd>{raw.boxes_count ?? "Non renseigné"}</dd></dl>{raw.text_preview && <details><summary>Texte des annotations</summary><p dir="auto">{raw.text_preview}</p></details>}</section> : <div className="tabs" role="tablist" aria-label="Données de la facture">{["Informations", "Articles", "Contrôles", "Historique"].map(name => <button key={name} role="tab" aria-selected={tab === name} className={tab === name ? "active" : ""} onClick={() => setTab(name)}>{name}</button>)}</div>}
        {!readOnly && tab === "Informations" && reviewGroups.map(group => <section className="review-field-group" key={group.title}><h2>{group.title}</h2><div className="review-field-grid">{group.fields.map(([key, label]) => <div className={key.includes("address") || key === "notes" || key === "amount_in_words" ? "review-field wide-field" : "review-field"} key={key}>
          <label htmlFor={`review-${key}`}>{label}</label>
          <FieldEvidence fieldKey={key} value={displayValue(draft[key])} proof={proofs[key]} isPdf={isPdf} active={highlight?.field_key === key} onLocate={setHighlight} />
          <input id={`review-${key}`} value={displayValue(draft[key])} disabled={readOnly || busy} inputMode={numericFields.has(key) ? "decimal" : "text"} placeholder="Non renseigné" onChange={e => update(key, e.target.value)} />
        </div>)}</div></section>)}
        {tab === "Articles" && <section className="review-field-group"><h2>Articles ({draft.items?.length || 0})</h2>
          {!(draft.items || []).length && <p>Aucun article extrait.</p>}
          {(draft.items || []).map((item: any, index: number) => <article className="review-item" key={index}><div className="review-item-heading"><h3>Article {index + 1}</h3>{!readOnly && <button title={`Supprimer l'article ${index + 1}`} aria-label={`Supprimer l'article ${index + 1}`} disabled={busy} onClick={() => update("items", draft.items.filter((_: any, i: number) => i !== index))}><Trash2 size={17} /></button>}</div><div className="review-field-grid">
            {[["description", "Désignation"], ["reference", "Référence"], ["unit", "Unité"], ["quantity", "Quantité"], ["unit_price", "Prix unitaire"], ["total", "Montant ligne"], ["tax_rate", "Taux TVA"], ["discount_amount", "Remise"], ["price_basis", "Base du prix"]].map(([key, label]) => <label key={key}>{label}{key === "price_basis" ? <select disabled={readOnly || busy} value={item[key] || "unknown"} onChange={e => update("items", draft.items.map((row: any, i: number) => i === index ? { ...row, [key]: e.target.value } : row))}><option value="unknown">Non déterminée</option><option value="HT">HT</option><option value="TTC">TTC</option></select> : <input aria-label={`${label} article ${index + 1}`} disabled={readOnly || busy} value={displayValue(item[key])} onChange={e => update("items", draft.items.map((row: any, i: number) => i === index ? { ...row, [key]: e.target.value } : row))} />}</label>)}
          </div>{item.source && <details><summary>Source de l'extraction originale</summary><p dir="auto">{typeof item.source === "string" ? item.source : JSON.stringify(item.source)}</p></details>}</article>)}
          {!readOnly && <button className="btn btn-outline" disabled={busy} onClick={() => update("items", [...(draft.items || []), { description: "", quantity: null, unit_price: null, total: null, price_basis: "unknown" }])}><Plus size={16} />Ajouter un article</button>}
        </section>}
        {tab === "Contrôles" && <section className="review-field-group"><h2>Contrôles lors de l'extraction</h2><p>Ces résultats automatiques précèdent les éventuelles corrections humaines.</p>{issues.length ? issues.map((issue: any, index: number) => <div className="review-notice" key={index}><strong>{issue.severity === "blocking" ? "À vérifier" : "Information"}</strong><p>{issue.message || issue.code}</p>{issue.code && <small>{issue.code}</small>}</div>) : <p>Aucun avertissement enregistré. Cela ne remplace pas la relecture humaine.</p>}
          {!!original.taxes?.length && <><h2>Taxes extraites</h2><div className="review-tax-list">{original.taxes.map((tax: any, i: number) => <p key={i}><strong>{tax.label || tax.name || "Taxe"}</strong> · {displayValue(tax.rate)} · {displayValue(tax.amount)} {original.currency || ""}</p>)}</div></>}
          <h2>Éléments visuels</h2><p>Les détections ne certifient pas l'authenticité de la facture.</p>{[ ["logo", "Logo"], ["stamp", "Cachet"], ["signature", "Signature"] ].map(([key, label]) => { const found = detections.filter(d => detectionCategory(d) === key); return <p key={key}>{label} : {found.length ? `${found.length} détection(s) · score maximal ${Math.round(Math.max(...found.map(detectionScore)) * 100)}%` : "Aucune détection enregistrée"}</p>; })}
        </section>}
        {tab === "Historique" && <section className="review-field-group"><h2>Historique enregistré</h2>{!events.length && <p>Aucun événement disponible.</p>}{[...events].reverse().map(event => <article className="review-event" key={event.id}><strong>{({ validated: "Validation humaine", updated_draft: "Corrections enregistrées", created_draft: "Import enregistré" } as any)[event.action] || event.action}</strong><p>{event.created_at} · {event.user_email || "Utilisateur non renseigné"}</p>{event.action !== "created_draft" && <details><summary>Champs modifiés</summary>{Object.keys(event.after_data || {}).filter(key => JSON.stringify(event.before_data?.[key]) !== JSON.stringify(event.after_data[key])).map(key => <p key={key}><strong>{reviewGroups.flatMap(g => [...g.fields]).find(([k]) => k === key)?.[1] || key}</strong> : {JSON.stringify(event.before_data?.[key] ?? null)} → {JSON.stringify(event.after_data[key])}</p>)}</details>}</article>)}</section>}
        {!readOnly && <div className="review-savebar"><button className="btn btn-outline" disabled={busy || !dirty} onClick={() => { setDraft(original); setHighlight(null); setError(""); setMessage("Modifications annulées."); }}><Undo2 size={16} />Annuler</button><button className="btn btn-soft" disabled={busy || !dirty} onClick={() => void save(false)}>{busy ? <Loader2 size={16} /> : <Save size={16} />}Enregistrer</button><button className="btn btn-primary" disabled={busy || (invoice.status === "validated" && !dirty)} onClick={() => confirmDialog.current?.showModal()}><Check size={16} />{invoice.status === "validated" && !dirty ? "Validée" : "Valider la facture"}</button></div>}
      </section>
    </div>
    <dialog ref={confirmDialog} className="fatura-dialog" aria-labelledby="confirm-invoice-title"><h2 id="confirm-invoice-title">Confirmer la validation humaine</h2><p>Les valeurs affichées et vos corrections seront enregistrées comme validées par votre compte.</p>{issues.length > 0 && <p>{issues.length} avertissement(s) de l'extraction sont disponibles dans l'onglet Contrôles. Une validation humaine ne transforme pas une valeur corrigée en preuve OCR.</p>}<div className="form-actions"><button className="btn btn-outline" disabled={busy} onClick={() => confirmDialog.current?.close()}>Revenir à la facture</button><button className="btn btn-primary" disabled={busy} onClick={() => void save(true)}>{busy ? "Enregistrement…" : "J'ai vérifié, confirmer"}</button></div></dialog>
  </section>;
}
