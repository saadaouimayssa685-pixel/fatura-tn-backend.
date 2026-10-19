import React, { useEffect, useState } from "react";
import { Check, Clock, Inbox, RefreshCw, UploadCloud } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { API_BASE_URL, apiFetch } from "./api";

export function LiveDashboard({ navigate }: { navigate: (path: string) => void }) {
  const [stats, setStats] = useState<any>(null);
  const [invoices, setInvoices] = useState<any[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function refresh() {
    setBusy(true); setError("");
    try {
      const statsResponse = await apiFetch(`${API_BASE_URL}/database/stats`);
      if (!statsResponse.ok) throw new Error("Statistiques indisponibles.");
      setStats((await statsResponse.json()).stats);
      const invoiceResponse = await apiFetch(`${API_BASE_URL}/invoices/catalog?source=processed&page_size=5`);
      if (!invoiceResponse.ok) throw new Error("Historique indisponible.");
      setInvoices((await invoiceResponse.json()).invoices || []);
    } catch (error) { setError(error instanceof Error ? error.message : "Serveur inaccessible."); }
    finally { setBusy(false); }
  }
  useEffect(() => { void refresh(); }, []);
  const total = Number(stats?.totals?.total_invoices || 0);
  const validated = Number(stats?.totals?.validated_invoices || 0);
  const draft = Number(stats?.totals?.draft_invoices || 0);
  const statuses = [{ name: "Validées", value: validated }, { name: "À vérifier", value: draft }, { name: "Autres statuts", value: Math.max(0, total - validated - draft) }].filter(row => row.value > 0);
  return <section className="live-dashboard dashboard-page">
    <div className="page-header"><div><h1>Tableau de bord</h1><p>Factures enregistrées · base locale</p></div><div className="form-actions">
      <button className="btn btn-outline" title="Actualiser" aria-label="Actualiser le tableau de bord" disabled={busy} onClick={refresh}><RefreshCw size={18} /></button>
      <button className="btn btn-primary" onClick={() => navigate("/import")}><UploadCloud size={18} />Importer</button>
    </div></div>
    {error && <p role="alert">{error}</p>}
    {!stats && busy && <p>Chargement…</p>}
    {stats && <>
      <div className="dashboard-section-divider"><h2>Vue d'ensemble</h2><span /></div>
      <div className="live-metrics">
        {[{ label: "Importées", value: total, Icon: Inbox, tone: "blue" }, { label: "Validées", value: validated, Icon: Check, tone: "green" }, { label: "À vérifier", value: draft, Icon: Clock, tone: "amber" }].map(({ label, value, Icon, tone }) => <button className={`live-metric ${tone}`} key={label} onClick={() => navigate(label === "À vérifier" ? "/factures/a-valider" : `/factures?source=processed${label === "Validées" ? "&status=validated" : ""}`)}><Icon size={22} /><span>{label}<strong>{value}</strong></span></button>)}
      </div>
      <div className="dashboard-section-divider"><h2>Activité</h2><span /></div>
      <div className="live-chart-grid">
        <section><h3>Documents importés par mois</h3>{stats.monthly_amounts?.length ? <ResponsiveContainer width="100%" height={240}><BarChart data={stats.monthly_amounts}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="month" /><YAxis allowDecimals={false} /><Tooltip /><Bar dataKey="count" name="Factures" fill="#2563eb" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer> : <p>Aucune activité enregistrée.</p>}</section>
        <section><h3>Répartition des statuts</h3>{total ? <><ResponsiveContainer width="100%" height={200}><PieChart><Pie data={statuses} dataKey="value" nameKey="name" innerRadius={55} outerRadius={80}>{statuses.map((row, index) => <Cell key={row.name} fill={["#159666", "#d78912", "#64748b"][index]} />)}</Pie><Tooltip /></PieChart></ResponsiveContainer><div className="live-status-legend">{statuses.map(row => <span key={row.name}>{row.name} <strong>{row.value}</strong></span>)}</div></> : <p>Aucune facture enregistrée.</p>}</section>
      </div>
      <div className="dashboard-section-divider"><h2>Historique</h2><span /></div>
      {!invoices.length && !busy && <p>Aucune facture enregistrée.</p>}
      {invoices.map(invoice => <button className="live-row" key={invoice.id} onClick={() => navigate(`/factures/${invoice.id}/analyse`)}>
        <span><strong>{invoice.filename}</strong><small>{invoice.supplier_name || "Fournisseur non détecté"}</small></span>
        <span>{invoice.invoice_number || "Sans numéro"}</span>
        <span>{invoice.status === "validated" ? "Validée" : "À vérifier"}</span>
      </button>)}
      <div className="dashboard-section-divider"><h2>Fournisseurs</h2><span /></div>
      {(stats.top_suppliers || []).map((supplier: any) => <button className="live-row" key={supplier.id} onClick={() => navigate(`/factures?source=processed&supplier=${encodeURIComponent(supplier.name)}`)}>
        <strong>{supplier.name}</strong><span>{supplier.invoices_count} facture(s)</span>
      </button>)}
      {!stats.top_suppliers?.length && <p>Aucun fournisseur enregistré.</p>}
      <div className="dashboard-section-divider"><h2>Échéances renseignées</h2><span /></div>
      {(stats.due_soon || []).map((invoice: any) => <button className="live-row" key={invoice.id} onClick={() => navigate(`/factures/${invoice.id}/analyse`)}>
        <strong>{invoice.invoice_number || "Sans numéro"}</strong><span>{invoice.due_date}</span>
      </button>)}
      {!stats.due_soon?.length && <p>Aucune échéance renseignée.</p>}
    </>}
  </section>;
}
