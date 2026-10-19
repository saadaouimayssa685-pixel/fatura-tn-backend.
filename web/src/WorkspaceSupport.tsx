import React, { useEffect, useRef, useState } from "react";
import { HelpCircle, RefreshCw, X } from "lucide-react";
import { API_BASE_URL, apiFetch } from "./api";

export function HelpDialog({ account = false }: { account?: boolean }) {
  const dialog = useRef<HTMLDialogElement>(null);
  return <>
    <button type="button" className={account ? "link-btn" : "icon-btn"} title={account ? "Accès au compte" : "Aide"} aria-label={account ? "Accès au compte" : "Aide"} onClick={() => dialog.current?.showModal()}>{account ? "Accès au compte" : <HelpCircle size={18} />}</button>
    <dialog ref={dialog} className="fatura-dialog" aria-label={account ? "Accès au compte" : "Aide Fatura"}>
      <div className="review-item-heading"><h2>{account ? "Accès au compte" : "Fatura · Vérification humaine"}</h2><button type="button" title="Fermer" aria-label="Fermer l'aide" onClick={() => dialog.current?.close()}><X size={20} /></button></div>
      {account ? <p>La réinitialisation par email n'est pas disponible dans cette installation locale. En cas de mot de passe perdu, contactez l'administrateur de l'installation. Aucun email de récupération n'est envoyé.</p> : <><p>Les valeurs sans preuve suffisante restent à vérifier. Une correction humaine ne modifie pas le texte OCR original.</p><p>Une validation humaine est enregistrée avec votre compte et apparaît dans l'historique de la facture. Les scores de détection visuelle ne certifient pas son authenticité.</p></>}
    </dialog>
  </>;
}

export function AccountSettings() {
  const [user, setUser] = useState<any>(null);
  const [health, setHealth] = useState<any>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function refresh() {
    setBusy(true); setError("");
    try {
      const me = await apiFetch(`${API_BASE_URL}/auth/me`);
      if (!me.ok) throw new Error("Session indisponible. Reconnectez-vous.");
      setUser((await me.json()).user);
      const response = await apiFetch(`${API_BASE_URL}/health`);
      if (!response.ok) throw new Error("Serveur indisponible.");
      setHealth(await response.json());
    } catch (e) { setError(e instanceof Error ? e.message : "Serveur inaccessible."); }
    finally { setBusy(false); }
  }
  useEffect(() => { void refresh(); }, []);
  return <section className="account-settings"><div className="page-header"><div><h1>Compte et connexion</h1><p>Installation locale</p></div><button className="btn btn-outline" disabled={busy} onClick={refresh} title="Actualiser l'état" aria-label="Actualiser l'état"><RefreshCw size={18} /></button></div>
    {error && <p role="alert" className="form-error">{error}</p>}
    {user && <section className="review-field-group"><h2>Compte connecté</h2><dl><dt>Nom</dt><dd>{user.full_name}</dd><dt>Email</dt><dd>{user.email}</dd><dt>Entreprise</dt><dd>{user.company || "Non renseignée"}</dd><dt>Rôle</dt><dd>{user.role}</dd></dl><HelpDialog account /></section>}
    {health && <section className="review-field-group"><h2>Connexion</h2><dl><dt>Serveur</dt><dd>{API_BASE_URL}</dd><dt>État</dt><dd>{health.status === "healthy" ? "Disponible" : health.status}</dd><dt>Base de données</dt><dd>{health.services?.database || "Non renseignée"}</dd></dl></section>}
  </section>;
}
