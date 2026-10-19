export const reviewGroups = [
  { title: "Fournisseur", fields: [["vendor_name", "Raison sociale fournisseur"], ["vendor_tax_id", "Identifiant fiscal fournisseur"], ["vendor_address", "Adresse fournisseur"], ["vendor_phone", "Téléphone fournisseur"], ["vendor_email", "Email fournisseur"], ["vendor_registration_number", "Registre de commerce"]] },
  { title: "Client", fields: [["customer_name", "Raison sociale client"], ["customer_tax_id", "Identifiant fiscal client"], ["customer_address", "Adresse client"], ["customer_phone", "Téléphone client"], ["customer_email", "Email client"], ["customer_id", "Code client"]] },
  { title: "Références", fields: [["invoice_number", "Numéro de facture"], ["date", "Date de facture"], ["due_date", "Date d'échéance"], ["delivery_number", "Bon de livraison"], ["currency", "Devise"]] },
  { title: "Montants", fields: [["subtotal", "Total HT"], ["tax_amount", "Montant TVA"], ["tax_rate", "Taux TVA"], ["discount", "Montant de remise"], ["stamp_duty", "Timbre fiscal"], ["additional_tax_total", "Taxes additionnelles"], ["total_amount", "Total TTC"], ["net_payable", "Net à payer"]] },
  { title: "Paiement", fields: [["payment_method", "Mode de paiement"], ["payment_terms", "Conditions de paiement"], ["rib", "RIB / IBAN"], ["amount_in_words", "Montant en lettres"], ["notes", "Notes"]] },
] as const;

export const numericFields = new Set(["subtotal", "tax_amount", "discount", "stamp_duty", "additional_tax_total", "total_amount", "net_payable"]);

export function displayValue(value: unknown): string {
  return value === null || value === undefined ? "" : String(value);
}

export function parseAmount(value: unknown): number | null {
  const text = displayValue(value).trim().replace(/\s/g, "").replace(",", ".");
  if (!text) return null;
  if (!/^-?\d+(?:\.\d+)?$/.test(text) || !Number.isFinite(Number(text))) throw new Error("Montant invalide : " + displayValue(value));
  return Number(text);
}

export function equalEvidenceValue(value: unknown, evidence: unknown): boolean {
  const clean = (v: unknown) => displayValue(v).trim().replace(/\s/g, "").replace(/,/g, ".").toLowerCase();
  const a = clean(value), b = clean(evidence);
  if (a === b) return true;
  return /^-?\d+\.\d+$/.test(a) && /^-?\d+(\.\d+)?$/.test(b) && Number(a) === Number(b);
}

// Send only user edits. Untouched nulls, taxes, references and extraction metadata stay intact.
export function correctionPatch(original: Record<string, any>, draft: Record<string, any>): Record<string, any> {
  const patch: Record<string, any> = {};
  for (const { fields } of reviewGroups) for (const [key] of fields) {
    if (displayValue(draft[key]) === displayValue(original[key])) continue;
    patch[key] = numericFields.has(key) ? parseAmount(draft[key]) : displayValue(draft[key]).trim();
  }
  if ("vendor_tax_id" in patch) patch.tax_id = patch.vendor_tax_id;
  if (JSON.stringify(draft.items || []) !== JSON.stringify(original.items || [])) {
    patch.items = (draft.items || []).map((item: any) => {
      const result = { ...item };
      for (const key of ["quantity", "unit_price", "total", "discount_amount"]) {
        if (key in item) result[key] = parseAmount(item[key]);
      }
      return result;
    });
  }
  return patch;
}
