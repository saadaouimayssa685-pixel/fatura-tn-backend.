import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  Archive,
  Bell,
  Bot,
  Building2,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock,
  Copy,
  Download,
  Eye,
  FileCheck2,
  FileText,
  FolderSearch,
  Gauge,
  HelpCircle,
  Home,
  Inbox,
  KeyRound,
  Layers3,
  Loader2,
  Lock,
  LogIn,
  LocateFixed,
  Maximize2,
  MessageSquareText,
  Minus,
  PanelRightOpen,
  Plus,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  Stamp,
  Table2,
  UploadCloud,
  UserCog,
  Users,
  X,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./styles.css";
import "./home.css";
import { API_BASE_URL, apiFetch as fetch } from "./api";
import { LiveDashboard } from "./LiveDashboard";
import { InvoiceReview } from "./InvoiceReview";
import { AccountSettings, HelpDialog } from "./WorkspaceSupport";
import "./review.css";
import { PasswordField } from "./PasswordField";
import { FieldEvidence, PdfInvoicePreview, SourceHighlight } from "./InvoiceEvidence";
import { detectionCategory, detectionScore, visibleDetections } from "./visualDetections";


type Status = "En cours d'analyse" | "En attente de validation" | "Validée" | "Rejetée" | "Dataset";
type Classification = "Facture" | "Non-facture" | "Incertain";
type Confidence = "Très fiable" | "À vérifier" | "Faible confiance";

type Box = {
  id: string;
  type: string;
  label: string;
  confidence: number;
  page: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

type RagProof = {
  field: string;
  extractedValue: string;
  confidence: number;
  sourceText: string;
  page: number;
  boundingBox: string;
  chunkId: string;
  similarityScore: number;
};

type Field = {
  key: string;
  label: string;
  value: string;
  confidence: number;
  boxId: string;
  proof: RagProof;
};

type AgentDecision = {
  step: string;
  decision: string;
  reason: string;
  status: "completed" | "needs_attention";
  confidence?: number;
  action?: string;
};

type Invoice = {
  id: string;
  file: string;
  fileType: string;
  supplier: string;
  taxId: string;
  number: string;
  date: string;
  dueDate: string;
  amount: string;
  status: Status;
  classification: Classification;
  classificationConfidence: number;
  reason?: string[];
};

type InvoiceFormValues = {
  supplier: string;
  address: string;
  taxId: string;
  supplierPhone: string;
  customerName: string;
  customerAddress: string;
  customerTaxId: string;
  customerPhone: string;
  invoiceNumber: string;
  productDesignation: string;
  deliveryNumber: string;
  invoiceDate: string;
  dueDate: string;
  subtotal: string;
  taxAmount: string;
  taxRate: string;
  stampDuty: string;
  discount: string;
  totalAmount: string;
  netPayable: string;
  currency: string;
  paymentMethod: string;
  paymentTerms: string;
  rib: string;
  notes: string;
};

type UploadedPreview = {
  url: string;
  name: string;
  mime: string;
};

type InvoiceLine = {
  description: string;
  quantity: string;
  unitPrice: string;
  taxRate: string;
  total: string;
  confidence: number;
  priceBasis?: string;
  requiresReview?: boolean;
};

type Supplier = {
  id: number | string;
  name: string;
  tax_id?: string | null;
  address?: string | null;
  phone?: string | null;
  last_invoice_number?: string | null;
  last_invoice_date?: string | null;
  last_invoice_status?: string | null;
  average_amount?: number | null;
  invoices_count: number;
  total_amount: number;
  updated_at?: string;
};

const emptyInvoiceForm: InvoiceFormValues = {
  supplier: "",
  address: "",
  taxId: "",
  supplierPhone: "",
  customerName: "",
  customerAddress: "",
  customerTaxId: "",
  customerPhone: "",
  invoiceNumber: "",
  productDesignation: "",
  deliveryNumber: "",
  invoiceDate: "",
  dueDate: "",
  subtotal: "",
  taxAmount: "",
  taxRate: "",
  stampDuty: "",
  discount: "",
  totalAmount: "",
  netPayable: "",
  currency: "",
  paymentMethod: "",
  paymentTerms: "",
  rib: "",
  notes: "",
};

const invoices: Invoice[] = [
  {
    id: "fac-0456",
    file: "sotufab-fc-2024-0456.pdf",
    fileType: "PDF",
    supplier: "SOTUFAB",
    taxId: "1234567 A/M/000",
    number: "FC-2024-0456",
    date: "21/05/2024",
    dueDate: "20/06/2024",
    amount: "31 850,350 TND",
    status: "En attente de validation",
    classification: "Facture",
    classificationConfidence: 98,
  },
  {
    id: "fac-0789",
    file: "office-depot-0789.pdf",
    fileType: "PDF",
    supplier: "Office Depot Tunisie",
    taxId: "0987654 B/M/000",
    number: "FC-2024-0789",
    date: "16/05/2024",
    dueDate: "30/05/2024",
    amount: "458,900 TND",
    status: "En attente de validation",
    classification: "Facture",
    classificationConfidence: 96,
  },
  {
    id: "fac-0918",
    file: "telecom-mai-2024.pdf",
    fileType: "PDF",
    supplier: "Tunisie Telecom",
    taxId: "4567891 C/M/000",
    number: "TT-2024-0918",
    date: "10/05/2024",
    dueDate: "25/05/2024",
    amount: "1 284,700 TND",
    status: "Validée",
    classification: "Facture",
    classificationConfidence: 99,
  },
  {
    id: "fac-1284",
    file: "steg-energie-1284.pdf",
    fileType: "PDF",
    supplier: "STEG",
    taxId: "1112223 D/M/000",
    number: "STEG-1284",
    date: "08/05/2024",
    dueDate: "22/05/2024",
    amount: "2 942,120 TND",
    status: "En cours d'analyse",
    classification: "Incertain",
    classificationConfidence: 67,
    reason: ["structure de facture partiellement reconnue", "cachet illisible"],
  },
  {
    id: "doc-2091",
    file: "bon-livraison-carrefour.jpg",
    fileType: "JPG",
    supplier: "Carrefour Tunisie",
    taxId: "",
    number: "-",
    date: "07/05/2024",
    dueDate: "-",
    amount: "0,000 TND",
    status: "Rejetée",
    classification: "Non-facture",
    classificationConfidence: 91,
    reason: ["total TTC absent", "numéro de facture absent", "structure de facture non reconnue"],
  },
];

const detectedBoxes: Box[] = [
  { id: "logo", type: "logo", label: "Logo", confidence: 99, page: 1, x: 8, y: 6, width: 18, height: 8 },
  { id: "supplier", type: "fournisseur", label: "Fournisseur", confidence: 98, page: 1, x: 8, y: 16, width: 32, height: 8 },
  { id: "client", type: "client", label: "Client", confidence: 97, page: 1, x: 60, y: 16, width: 30, height: 10 },
  { id: "number", type: "numero", label: "Numéro facture", confidence: 99, page: 1, x: 61, y: 34, width: 26, height: 5 },
  { id: "date", type: "date", label: "Date", confidence: 99, page: 1, x: 61, y: 41, width: 24, height: 5 },
  { id: "table", type: "tableau", label: "Tableau lignes", confidence: 98, page: 1, x: 8, y: 48, width: 84, height: 22 },
  { id: "ht", type: "total_ht", label: "Total HT", confidence: 97, page: 1, x: 61, y: 72, width: 27, height: 5 },
  { id: "tva", type: "tva", label: "TVA", confidence: 96, page: 1, x: 61, y: 78, width: 27, height: 5 },
  { id: "stamp", type: "cachet", label: "Cachet", confidence: 96, page: 1, x: 10, y: 78, width: 18, height: 12 },
  { id: "signature", type: "signature", label: "Signature", confidence: 94, page: 1, x: 32, y: 81, width: 22, height: 9 },
  { id: "timbre", type: "timbre_fiscal", label: "Timbre fiscal", confidence: 92, page: 1, x: 61, y: 84, width: 27, height: 5 },
  { id: "ttc", type: "total_ttc", label: "Total TTC", confidence: 97, page: 1, x: 58, y: 90, width: 32, height: 6 },
];

const extractedFields: Field[] = [
  field("supplier", "Fournisseur", "SOTUFAB", 98, "supplier", "Société Tunisienne de Fabrication - SOTUFAB"),
  field("taxId", "Matricule fiscal", "1234567 A/M/000", 97, "supplier", "Matricule fiscal : 1234567 A/M/000"),
  field("number", "Numéro facture", "FC-2024-0456", 99, "number", "Facture N° FC-2024-0456"),
  field("date", "Date facture", "21/05/2024", 99, "date", "Date facture : 21/05/2024"),
  field("due", "Date échéance", "20/06/2024", 68, "date", "Échéance : 20/06/2024"),
  field("currency", "Devise", "TND", 100, "ttc", "Tous les montants sont exprimés en TND"),
  field("ht", "Total HT", "26 765,000 TND", 97, "ht", "Total hors taxes : 26 765,000 TND"),
  field("tva", "TVA", "5 084,350 TND", 96, "tva", "TVA 19% : 5 084,350 TND"),
  field("timbre", "Timbre fiscal", "1,000 TND", 92, "timbre", "Timbre fiscal : 1,000 TND"),
  field("ttc", "Total TTC", "31 850,350 TND", 97, "ttc", "Net à payer TTC : 31 850,350 TND"),
];

function field(key: string, label: string, value: string, confidence: number, boxId: string, sourceText: string): Field {
  return {
    key,
    label,
    value,
    confidence,
    boxId,
    proof: {
      field: label,
      extractedValue: value,
      confidence,
      sourceText,
      page: 1,
      boundingBox: boxId,
      chunkId: `chunk-${key}-p1`,
      similarityScore: confidence / 100 - 0.03,
    },
  };
}

const weekly = [
  { label: "S18", taux: 71 },
  { label: "S19", taux: 76 },
  { label: "S20", taux: 74 },
  { label: "S21", taux: 86 },
  { label: "S22", taux: 82 },
  { label: "S23", taux: 91 },
];

const statusData = [
  { name: "Validées", value: 72, color: "#16A34A" },
  { name: "À vérifier", value: 18, color: "#F59E0B" },
  { name: "Rejetées", value: 7, color: "#DC2626" },
  { name: "Doublons", value: 3, color: "#7C3AED" },
];

const uploadSteps = [
  "Upload",
  "Classification",
  "OCR",
  "RAG",
  "Extraction",
  "Contrôles",
  "Routage",
];

const agentDecisions: AgentDecision[] = [
  {
    step: "Upload",
    decision: "accept_file",
    reason: "Extension PDF et taille du fichier valides.",
    status: "completed",
    action: "continue",
  },
  {
    step: "OCR / PDF text",
    decision: "extract_text",
    reason: "Texte extrait depuis le document avec fallback OCR disponible.",
    status: "completed",
    confidence: 0.91,
    action: "continue",
  },
  {
    step: "Classification",
    decision: "invoice",
    reason: "Signaux facture détectés : numéro, TVA, Total TTC, fournisseur.",
    status: "completed",
    confidence: 0.98,
    action: "continue_to_invoice_extraction",
  },
  {
    step: "YOLO visuel",
    decision: "run_yolo",
    reason: "Zones logo, cachet, signature, tableau et totaux localisées.",
    status: "completed",
    confidence: 0.94,
    action: "continue",
  },
  {
    step: "RAG d'extraction",
    decision: "retrieve_field_evidence",
    reason: "Passages pertinents retrouvés pour Date d'échéance, TTC, TVA et fournisseur.",
    status: "completed",
    confidence: 0.94,
    action: "continue_to_field_extraction",
  },
  {
    step: "Routage modèle",
    decision: "use_local_pipeline",
    reason: "Document complexe : GLM-OCR, RAG et Qwen relisent les preuves localement.",
    status: "completed",
    confidence: 0.92,
    action: "run_structured_extraction",
  },
  {
    step: "Extraction champs",
    decision: "use_ollama",
    reason: "Qwen local + règles déterministes ont produit les champs structurés.",
    status: "completed",
    confidence: 0.93,
    action: "normalize_fields",
  },
  {
    step: "Normalisation",
    decision: "normalize_invoice_fields",
    reason: "Montants TND, dates et lignes facture normalisés.",
    status: "completed",
    action: "continue",
  },
  {
    step: "Contrôles métier",
    decision: "issues_found",
    reason: "HT + TVA cohérents, mais date d'échéance avec confiance moyenne.",
    status: "needs_attention",
    confidence: 0.68,
    action: "human_review_recommended",
  },
  {
    step: "Détection doublons",
    decision: "potential_duplicate_found",
    reason: "Une facture proche existe avec même fournisseur, montant et période.",
    status: "needs_attention",
    confidence: 0.86,
    action: "human_review_required",
  },
  {
    step: "RAG documentaire",
    decision: "index_document",
    reason: "Document découpé et vectorisé pour l'assistant IA après normalisation.",
    status: "completed",
    confidence: 0.95,
    action: "continue_to_recommendation",
  },
  {
    step: "Routage workflow",
    decision: "human_validation_required",
    reason: "La facture reste automatisée mais attend la validation humaine finale.",
    status: "needs_attention",
    action: "open_validation_queue",
  },
  {
    step: "Recommandation",
    decision: "send_to_human_review",
    reason: "Date d'échéance à vérifier et duplicata potentiel à confirmer.",
    status: "needs_attention",
    confidence: 0.87,
    action: "human_review_required",
  },
];

const agentRecommendation = {
  recommendedAction: "send_to_human_review",
  humanReviewRequired: true,
  reasons: ["date_echeance_confidence_low", "potential_duplicate_candidate"],
  workflowDestination: "human_validation_required",
};

const supplierAssets: Record<string, string> = {
  STEG: "/assets/logo-steg.svg",
  "Tunisie Telecom": "/assets/logo-telecom.svg",
  SOTUFAB: "/assets/logo-sotufab.svg",
  "Carrefour Tunisie": "/assets/logo-carrefour.svg",
  "Office Depot Tunisie": "/assets/logo-office.svg",
  SONEDE: "/assets/logo-sonede.svg",
};

const userAvatars = {
  maissa: "/assets/avatar-maissa.png",
  sarra: "/assets/avatar-sarra.svg",
  mohamed: "/assets/avatar-mohamed.svg",
  amel: "/assets/avatar-amel.svg",
  nour: "/assets/avatar-nour.svg",
};

const ragAssistantQuestions = [
  {
    question: "Résume la facture FC-2024-0456 avec fournisseur, matricule fiscal, dates, HT, TVA et TTC.",
    answer: "Facture SOTUFAB FC-2024-0456. Matricule fiscal fournisseur 1234567 A/M/000. Date facture 21/05/2024, échéance 20/06/2024. Total HT 26 765,000 TND, TVA 19% 5 084,350 TND, timbre 1,000 TND, total TTC 31 850,350 TND. Source RAG: zones fournisseur, numéro, dates et bloc totaux de la page 1.",
  },
  {
    question: "Explique l'échéance de cette facture et dis-moi si elle est en retard.",
    answer: "L'échéance est la date limite de paiement indiquée sur la facture. Ici, elle est au 20/06/2024. Si la facture n'est pas validée/payée après cette date, l'agent la classe comme en retard et l'affiche dans les alertes. Source RAG: extrait contenant 'Échéance : 20/06/2024'.",
  },
  {
    question: "Quelle est la différence entre TVA facturée, TVA déductible et montant TTC ?",
    answer: "La TVA facturée est la taxe ajoutée par le fournisseur sur la facture. La TVA déductible est la TVA que l'entreprise peut récupérer si la facture est conforme et liée à une charge professionnelle. Le montant TTC est le total à payer: HT + TVA + timbre éventuel - remise. Ici: 26 765,000 + 5 084,350 + 1,000 = 31 850,350 TND.",
  },
];

function roleLabel(role?: string) {
  const normalized = String(role || "").toLowerCase().replace(/[_-]/g, " ").trim();
  if (normalized.includes("saisie")) return "Agent de saisie";
  if (normalized.includes("admin")) return "Administrateur";
  if (normalized.includes("valid")) return "Validateur";
  if (normalized.includes("comptable")) return "Comptable";
  return "Agent de saisie";
}

function readStoredUser() {
  try {
    return JSON.parse(localStorage.getItem("fatura-user") || "{}");
  } catch {
    return {};
  }
}

function userWithLocalProfile(user: any, avatar?: string) {
  const storedUser = readStoredUser();
  return {
    ...user,
    full_name: user?.full_name || "Maissa Saadaoui",
    company: user?.company || "YASMEEN ENGINEERING SYSTEMS",
    role: roleLabel(user?.role),
    avatar: avatar || user?.avatar || storedUser.avatar || userAvatars.maissa,
  };
}

function supplierLogo(name: string) {
  return supplierAssets[name] ?? "/assets/logo-office.svg";
}

function firstValue(source: Record<string, any>, keys: string[]) {
  for (const key of keys) {
    const value = source?.[key];
    if (value !== undefined && value !== null && String(value).trim() !== "") return String(value);
  }
  return "";
}

function formatInvoiceAmount(value: any) {
  if (value === undefined || value === null || value === "") return "";
  const numeric = Number(String(value).replace(/\s/g, "").replace(",", "."));
  if (Number.isFinite(numeric)) {
    return numeric.toLocaleString("fr-FR", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
  }
  return String(value).replace(/\s*TND$/i, "").trim();
}

function formatTnd(value: any) {
  return `${formatInvoiceAmount(value || 0)} TND`;
}

function percentValue(value: any, fallback = 0) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.round(numeric <= 1 ? numeric * 100 : numeric);
}

function getClassificationSummary(result: any) {
  const raw = result?.classification || result?.document || {};
  const details = raw.details || raw.classification || {};
  const documentType = String(raw.document_type || result?.document?.document_type || "").toLowerCase();
  const isInvoice = raw.is_invoice ?? result?.document?.is_invoice ?? (documentType.includes("invoice") || documentType.includes("facture"));
  const confidence = percentValue(raw.confidence ?? result?.document?.classification_confidence, 0);
  const strongMatches = details.strong_keyword_matches ?? details.strong_matches ?? 0;
  const weakMatches = details.weak_keyword_matches ?? details.weak_matches ?? 0;
  const evidence = Array.isArray(details.evidence)
    ? details.evidence
    : [
        strongMatches ? `${strongMatches} signaux forts facture` : "",
        weakMatches ? `${weakMatches} signaux faibles` : "",
        details.reason || details.analysis_method || "",
      ].filter(Boolean);

  if (isInvoice === false) {
    return {
      label: "Non-facture",
      tone: "bad",
      confidence,
      evidence: evidence.length ? evidence.join(" · ") : "Le document ne contient pas assez de signaux facture.",
    };
  }

  if (confidence > 0 && confidence < 70) {
    return {
      label: "Document à vérifier",
      tone: "warn",
      confidence,
      evidence: evidence.length ? evidence.join(" · ") : "Classification faible, validation humaine recommandée.",
    };
  }

  return {
    label: "Facture fournisseur",
    tone: "good",
    confidence,
    evidence: evidence.length ? evidence.join(" · ") : "Signaux facture détectés dans le texte OCR/PDF.",
  };
}

function isInvoiceResult(result: any) {
  const raw = result?.classification || result?.document || {};
  const documentType = String(raw.document_type || result?.document?.document_type || "").toLowerCase();
  return raw.is_invoice ?? result?.document?.is_invoice ?? (documentType.includes("invoice") || documentType.includes("facture"));
}

function invoiceIdFromAgentResult(result: any) {
  return result?.invoice?.id ?? result?.document?.invoice_id ?? result?.document?.id ?? null;
}

function normalizedDataFromForm(values: InvoiceFormValues) {
  return {
    vendor_name: values.supplier,
    vendor_address: values.address,
    tax_id: values.taxId,
    vendor_tax_id: values.taxId,
    vendor_phone: values.supplierPhone,
    customer_name: values.customerName,
    customer_address: values.customerAddress,
    customer_tax_id: values.customerTaxId,
    customer_phone: values.customerPhone,
    invoice_number: values.invoiceNumber,
    product_designation: values.productDesignation,
    delivery_number: values.deliveryNumber,
    date: values.invoiceDate,
    due_date: values.dueDate,
    subtotal: values.subtotal,
    tax_amount: values.taxAmount,
    tax_rate: values.taxRate,
    stamp_duty: values.stampDuty,
    discount: values.discount,
    total_amount: values.totalAmount,
    net_payable: values.netPayable,
    currency: values.currency || "TND",
    payment_method: values.paymentMethod,
    payment_terms: values.paymentTerms,
    rib: values.rib,
    notes: values.notes,
  };
}

function mapBackendStatus(status?: string): Status {
  const value = String(status || "").toLowerCase();
  if (value.includes("dataset")) return "Dataset";
  if (value.includes("valid")) return "Validée";
  if (value.includes("reject") || value.includes("rejet")) return "Rejetée";
  if (value.includes("classified") || value.includes("review") || value.includes("draft")) return "En attente de validation";
  return "En cours d'analyse";
}

function mapBackendClassification(invoice: any): Classification {
  const statusValue = String(invoice?.status || "").toLowerCase();
  const sourceValue = String(invoice?.raw_data?.source || invoice?.source || "").toLowerCase();
  if (statusValue.includes("dataset") || sourceValue.includes("fatura_dataset") || String(invoice?.id || "").startsWith("dataset:")) {
    return "Facture";
  }
  const normalized = invoice?.normalized_data || {};
  const structured = invoice?.structured_data || {};
  // Normalized data is the agent's authoritative, evidence-guarded output.
  // Keep an explicit null so historical storage defaults such as 0 cannot
  // turn an unknown monetary field into a false value in the UI.
  const merged = { ...structured, ...invoice, ...normalized };
  const raw = invoice?.raw_data?.classification || invoice?.classification || {};
  const documentType = String(invoice?.document_type || raw.document_type || "").toLowerCase();
  const hasExplicitInvoiceFlag = raw.is_invoice !== undefined || invoice?.is_invoice !== undefined;
  const explicitInvoiceFlag = raw.is_invoice ?? invoice?.is_invoice;
  const hasInvoiceSignals = Boolean(
    firstValue(merged, ["invoice_number", "number", "numero_facture", "num_facture"])
    || firstValue(merged, ["total_amount", "total_ttc", "montant_ttc"])
    || firstValue(merged, ["tax_amount", "tva", "montant_tva"])
    || firstValue(merged, ["supplier_name", "vendor_name", "fournisseur", "raison_sociale"]),
  );
  const total = Number(String(firstValue(merged, ["total_amount", "total_ttc", "montant_ttc"]) || 0).replace(/\s/g, "").replace(",", "."));
  const confidence = Number(invoice?.confidence_score || raw.confidence || 0);
  const isInvoice = hasExplicitInvoiceFlag
    ? explicitInvoiceFlag
    : (documentType.includes("invoice") || documentType.includes("facture") || (hasInvoiceSignals && total > 0));
  if (isInvoice) return "Facture";
  if (confidence < 0.7 && hasInvoiceSignals) return "Incertain";
  return "Non-facture";
}

function mapBackendInvoice(invoice: any): Invoice {
  const normalized = invoice?.normalized_data || {};
  const structured = invoice?.structured_data || {};
  const merged = { ...structured, ...invoice, ...normalized };
  const status = mapBackendStatus(invoice?.status);
  const supplier = firstValue(merged, ["supplier_name", "vendor_name", "fournisseur", "raison_sociale"]) || (status === "Dataset" ? "Dataset FATURA" : "Fournisseur à confirmer");
  const invoiceNumber = firstValue(merged, ["invoice_number", "number", "numero_facture", "num_facture"]) || "Numéro non renseigné";
  const invoiceDate = firstValue(merged, ["date", "invoice_date", "date_facture"]) || "Date non renseignée";
  const total = firstValue(merged, ["total_amount", "total_ttc", "montant_ttc"]);
  const currency = firstValue(merged, ["currency", "devise"]) || "TND";
  const confidence = percentValue(invoice?.confidence_score, 0);

  return {
    id: String(invoice?.id ?? invoiceNumber),
    file: invoice?.filename || `${invoiceNumber}.pdf`,
    fileType: String(invoice?.file_type || "").toUpperCase(),
    supplier,
    taxId: firstValue(merged, ["tax_id", "vendor_tax_id", "matricule_fiscal", "mf", "identifiant_fiscal"]),
    number: invoiceNumber,
    date: invoiceDate,
    dueDate: firstValue(merged, ["due_date", "date_echeance", "echeance"]),
    amount: status === "Dataset" ? "Non analysé" : total === "" || total == null ? "Montant non renseigné" : `${formatInvoiceAmount(total)} ${currency}`.trim(),
    status: mapBackendClassification(invoice) === "Non-facture" ? "Rejetée" : status,
    classification: mapBackendClassification(invoice),
    classificationConfidence: confidence || 0,
    reason: invoice?.raw_data?.validation_issues?.map((issue: any) => issue?.message || issue?.field).filter(Boolean),
  };
}

function invoiceIdFromRoute(route: string) {
  const match = route.match(/^\/factures\/(.+)\/analyse$/);
  return match ? decodeURIComponent(match[1]) : "";
}

function invoiceCatalogSource(invoiceId: string) {
  return invoiceId.startsWith("dataset:") ? "dataset" : "processed";
}

function invoiceFileUrl(invoiceId: string, inline = false) {
  const source = invoiceCatalogSource(invoiceId);
  const params = new URLSearchParams({ source, invoice_id: invoiceId });
  if (inline) params.set("inline", "true");
  return `${API_BASE_URL}/invoices/catalog/download?${params.toString()}`;
}

function amountWithCurrency(value: any, currency = "TND") {
  const amount = formatInvoiceAmount(value);
  return amount ? `${amount} ${currency}` : "";
}

function fieldsFromInvoice(invoice: Invoice, rawInvoice: any = null): Field[] {
  const normalized = rawInvoice?.normalized_data || {};
  const structured = rawInvoice?.structured_data || {};
  const merged = { ...structured, ...rawInvoice, ...normalized };
  const rawData = rawInvoice?.raw_data || {};
  const evidence = rawData.field_evidence || rawData.evidence || {};
  const textPreview = String(rawInvoice?.extracted_text || rawData.text_preview || rawData.ocr_text || "").trim();
  const currency = firstValue(merged, ["currency", "devise"]) || invoice.amount.split(" ").at(-1) || "TND";
  const isDataset = invoice.status === "Dataset";

  const evidenceFor = (backendKey: string) => {
    const items = Array.isArray(evidence?.[backendKey]) ? evidence[backendKey] : [];
    const first = items[0] || {};
    return String(first.preview || first.text || first.source_text || textPreview || "").slice(0, 500).trim();
  };

  const confidenceFor = (backendKey: string, value: string, fallback: number) => {
    if (!value || value.startsWith("Non détect")) return 0;
    if (invoice.classification === "Non-facture" && backendKey !== "classification") return 0;
    if (isDataset) return 0;
    const items = Array.isArray(evidence?.[backendKey]) ? evidence[backendKey] : [];
    const first = items[0] || {};
    if (first.confidence !== undefined) return percentValue(first.confidence, fallback);
    if (first.score !== undefined) return Math.min(99, Math.max(55, Math.round(Number(first.score) * 10)));
    return invoice.classificationConfidence || fallback;
  };

  const makeField = (key: string, label: string, value: string, backendKey: string, boxId: string, fallbackConfidence: number): Field => {
    const cleanValue = value || (key === "currency" ? currency : key === "taxRate" ? "Non détecté" : "Non détecté");
    const confidence = confidenceFor(backendKey, cleanValue, fallbackConfidence);
    const proofText = confidence > 0
      ? evidenceFor(backendKey) || `Valeur retrouvée dans les données extraites du document : ${cleanValue}.`
      : "Aucune preuve RAG disponible pour ce champ.";
    return {
      key,
      label,
      value: cleanValue,
      confidence,
      boxId,
      proof: {
        field: label,
        extractedValue: cleanValue,
        confidence,
        sourceText: proofText,
        page: 1,
        boundingBox: boxId,
        chunkId: confidence > 0 ? `chunk-${backendKey}-p1` : "preuve-indisponible",
        similarityScore: confidence > 0 ? Math.max(0.4, confidence / 100 - 0.03) : 0,
      },
    };
  };

  if (invoice.classification === "Non-facture") {
    return [
      makeField("classification", "Classification", "Non-facture", "classification", "supplier", invoice.classificationConfidence || 0),
      makeField("supplier", "Fournisseur", "Non applicable", "supplier_name", "supplier", 0),
      makeField("number", "Numéro facture", "Non applicable", "invoice_number", "number", 0),
      makeField("ht", "Total HT", "Non applicable", "subtotal", "ht", 0),
      makeField("ttc", "Total TTC", "Non applicable", "total_amount", "ttc", 0),
    ];
  }

  const subtotal = firstValue(merged, ["subtotal", "amount_ht", "total_ht", "montant_ht"]);
  const taxAmount = firstValue(merged, ["tax_amount", "tva", "montant_tva"]);
  const stampDuty = firstValue(merged, ["stamp_duty", "timbre", "timbre_fiscal"]);
  const totalAmount = firstValue(merged, ["total_amount", "total_ttc", "montant_ttc"]) || invoice.amount.replace(/\s*TND$/i, "");

  return [
    makeField("supplier", "Fournisseur", invoice.supplier || firstValue(merged, ["supplier_name", "vendor_name", "fournisseur", "raison_sociale"]), "supplier_name", "supplier", 90),
    makeField("taxId", "Matricule fiscal", invoice.taxId || firstValue(merged, ["tax_id", "vendor_tax_id", "matricule_fiscal", "mf", "identifiant_fiscal"]), "tax_id", "supplier", 88),
    makeField("number", "Numéro facture", invoice.number || firstValue(merged, ["invoice_number", "number", "numero_facture", "num_facture"]), "invoice_number", "number", 92),
    makeField("date", "Date facture", invoice.date || firstValue(merged, ["invoice_date", "date", "date_facture"]), "invoice_date", "date", 88),
    makeField("due", "Date échéance", invoice.dueDate || firstValue(merged, ["due_date", "date_echeance", "echeance"]), "due_date", "date", 78),
    makeField("currency", "Devise", currency, "currency", "ttc", 86),
    makeField("ht", "Total HT", amountWithCurrency(subtotal, currency), "subtotal", "ht", 88),
    makeField("tva", "TVA", amountWithCurrency(taxAmount, currency), "tax_amount", "tva", 84),
    makeField("timbre", "Timbre fiscal", amountWithCurrency(stampDuty, currency), "stamp_duty", "timbre", 76),
    makeField("ttc", "Total TTC", amountWithCurrency(totalAmount, currency), "total_amount", "ttc", 90),
  ];
}

function extractInvoiceFormValues(result: any): InvoiceFormValues {
  const invoice = result?.invoice || {};
  const documentFields = result?.document?.fields || {};
  const normalized = invoice?.normalized_data || {};
  const structured = invoice?.structured_data || {};
  const storedFields = invoice?.fields || {};
  const fields = { ...documentFields, ...storedFields, ...structured, ...invoice, ...normalized };
  const lines = extractInvoiceLines(result);
  const detectedProducts = lines.map((line) => line.description).filter(Boolean).join(" | ");

  return {
    supplier: firstValue(fields, ["vendor_name", "supplier_name", "fournisseur", "raison_sociale"]),
    address: firstValue(fields, ["vendor_address", "supplier_address", "adresse_fournisseur", "address_supplier", "adresse", "address"]),
    taxId: firstValue(fields, ["tax_id", "vendor_tax_id", "matricule_fiscal", "mf", "identifiant_fiscal"]),
    supplierPhone: firstValue(fields, ["vendor_phone", "supplier_phone", "telephone_fournisseur", "tel_fournisseur", "phone"]),
    customerName: firstValue(fields, ["customer_name", "client_name", "client", "acheteur"]),
    customerAddress: firstValue(fields, ["customer_address", "client_address", "adresse_client"]),
    customerTaxId: firstValue(fields, ["customer_tax_id", "client_tax_id", "matricule_client", "mf_client"]),
    customerPhone: firstValue(fields, ["customer_phone", "client_phone", "telephone_client", "tel_client"]),
    invoiceNumber: firstValue(fields, ["invoice_number", "number", "numero_facture", "num_facture"]),
    productDesignation: detectedProducts || firstValue(fields, ["product_designation", "designation", "désignation", "product", "produit", "article"]),
    deliveryNumber: firstValue(fields, ["delivery_number", "bon_livraison", "numero_bl", "bl"]),
    invoiceDate: firstValue(fields, ["date", "invoice_date", "date_facture"]),
    dueDate: firstValue(fields, ["due_date", "date_echeance", "echeance"]),
    subtotal: formatInvoiceAmount(firstValue(fields, ["subtotal", "amount_ht", "total_ht", "montant_ht"])),
    taxAmount: formatInvoiceAmount(firstValue(fields, ["tax_amount", "tva", "montant_tva"])),
    taxRate: firstValue(fields, ["tax_rate", "taux_tva"]),
    stampDuty: formatInvoiceAmount(firstValue(fields, ["stamp_duty", "timbre", "timbre_fiscal"])),
    discount: formatInvoiceAmount(firstValue(fields, ["discount", "remise", "rabais"])),
    totalAmount: formatInvoiceAmount(firstValue(fields, ["total_amount", "total_ttc", "montant_ttc", "total"])),
    netPayable: formatInvoiceAmount(firstValue(fields, ["net_payable", "net_a_payer", "montant_a_payer"])) || formatInvoiceAmount(firstValue(fields, ["total_amount", "total_ttc", "montant_ttc", "total"])),
    currency: firstValue(fields, ["currency", "devise"]) || "TND",
    paymentMethod: firstValue(fields, ["payment_method", "mode_paiement", "payment", "reglement"]),
    paymentTerms: firstValue(fields, ["payment_terms", "conditions_paiement", "delai_paiement"]),
    rib: firstValue(fields, ["rib", "iban", "bank_account", "compte_bancaire"]),
    notes: firstValue(fields, ["notes", "amount_in_words", "amount_words", "montant_en_lettres", "commentaires"]),
  };
}

function extractInvoiceLines(result: any): InvoiceLine[] {
  const normalized = result?.invoice?.normalized_data || result?.invoice?.fields || result?.document?.fields || {};
  const candidates = [
    normalized.items,
    normalized.line_items,
    normalized.invoice_items,
    normalized.lignes,
    normalized.lignes_facture,
    result?.invoice?.items,
    result?.invoice?.line_items,
    result?.items,
    result?.line_items,
  ].find((value) => Array.isArray(value));

  if (!Array.isArray(candidates)) return [];

  return candidates
    .map((item: any): InvoiceLine => {
      const rawConfidence = Number(item.confidence || item.score || item.extraction_confidence || 0.92);
      const confidence = Math.round(rawConfidence <= 1 ? rawConfidence * 100 : rawConfidence);
      return {
        description: firstValue(item, ["description", "designation", "désignation", "label", "name", "article"]),
        quantity: firstValue(item, ["quantity", "quantite", "qté", "qty"]) || "1",
        unitPrice: formatInvoiceAmount(firstValue(item, ["unit_price", "prix_unitaire", "pu_ht", "price", "unitPrice"])),
        taxRate: firstValue(item, ["tax_rate", "taux_tva", "tva_rate", "vat_rate"]) || "",
        total: formatInvoiceAmount(firstValue(item, ["total", "total_price", "montant_ht", "total_ht", "amount"])),
        confidence,
        priceBasis: ["HT", "TTC"].includes(item.price_basis) ? item.price_basis : "unknown",
        requiresReview: Boolean(item.requires_review),
      };
    })
    .filter((line) => line.description || line.total || line.unitPrice);
}

function App() {
  const [, refreshNavigation] = useState(0);
  const currentUrl = React.useRef(window.location.pathname + window.location.search);
  const [route, setRoute] = useState(window.location.pathname || "/");
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [role, setRoleState] = useState<"guest" | "user">(() => localStorage.getItem("fatura-auth-token") ? "user" : "guest");
  const [toast, setToast] = useState("");

  const setRole = (nextRole: "guest" | "user") => {
    setRoleState(nextRole);
    if (nextRole === "guest") {
      localStorage.removeItem("fatura-auth-token");
      localStorage.removeItem("fatura-user");
    }
  };

  const navigate = (path: string) => {
    if (!window.dispatchEvent(new Event("fatura-before-navigate", { cancelable: true }))) return;
    window.history.pushState({}, "", path);
    currentUrl.current = path;
    setRoute(new URL(path, window.location.origin).pathname);
    refreshNavigation(value => value + 1);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  React.useEffect(() => {
    const onPop = () => {
      if (!window.dispatchEvent(new Event("fatura-before-navigate", { cancelable: true }))) {
        window.history.pushState({}, "", currentUrl.current);
        return;
      }
      currentUrl.current = window.location.pathname + window.location.search;
      setRoute(window.location.pathname); refreshNavigation(value => value + 1);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  React.useEffect(() => {
    const token = localStorage.getItem("fatura-auth-token");
    if (!token) return;
    fetch(`${API_BASE_URL}/auth/me?token=${encodeURIComponent(token)}`)
      .then((response) => {
        if (!response.ok) throw new Error("Session expirée");
        return response.json();
      })
      .then((data) => {
        localStorage.setItem("fatura-user", JSON.stringify(userWithLocalProfile(data.user)));
        setRoleState("user");
      })
      .catch(() => setRole("guest"));
  }, []);

  const notify = (message: string) => {
    setToast(message);
    setTimeout(() => setToast(""), 2600);
  };

  const page = renderPage(route, navigate, role, setRole, notify, () => setAssistantOpen(true));
  const showAssistant = role === "user" && route === "/dashboard";

  return (
    <>
      {page}
      {showAssistant && <AssistantButton onClick={() => setAssistantOpen(true)} />}
      <AssistantDrawer open={assistantOpen} onClose={() => setAssistantOpen(false)} route={route} navigate={navigate} />
      {toast && <div className="toast"><Check size={18} />{toast}</div>}
    </>
  );
}

function renderPage(
  route: string,
  navigate: (path: string) => void,
  role: "guest" | "user",
  setRole: (role: "guest" | "user") => void,
  notify: (message: string) => void,
  openAssistant: () => void,
) {
  if (route === "/") return <Landing navigate={navigate} />;
  if (route === "/login") return <Login navigate={navigate} setRole={setRole} />;
  if (route === "/signup") return <Signup navigate={navigate} setRole={setRole} />;
  if (route.startsWith("/admin")) return <AccessDenied navigate={navigate} />;
  if (role === "guest") return <Login navigate={navigate} setRole={setRole} />;
  return <UserLayout route={route} navigate={navigate} notify={notify} openAssistant={openAssistant} setRole={setRole} />;
}

function Logo({ dark = false }: { dark?: boolean }) {
  return (
    <button className={`logo ${dark ? "logo-dark" : ""}`} onClick={() => (window.location.href = "/")}>
      <span className="logo-mark">F</span>
      <span>
        <strong>FATURA.tn</strong>
        <small>فاتورة</small>
      </span>
    </button>
  );
}

function Landing({ navigate }: { navigate: (path: string) => void }) {
  const [stage, setStage] = useState(0);
  const stages = [
    { icon: UploadCloud, title: "Importez vos documents", detail: "PDF ou image : rassemblez vos factures dans un même espace.", label: "Importer" },
    { icon: Eye, title: "Vérifiez à la source", detail: "Relisez les informations extraites et corrigez les champs avant validation.", label: "Vérifier" },
    { icon: ShieldCheck, title: "Gardez le dernier mot", detail: "Validez les informations puis retrouvez vos factures et leurs fournisseurs.", label: "Valider" },
  ];
  const moveScene = (event: React.PointerEvent<HTMLElement>) => {
    if (event.pointerType !== "mouse" || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const rect = event.currentTarget.getBoundingClientRect();
    event.currentTarget.style.setProperty("--scene-x", `${((event.clientX - rect.left) / rect.width - .5) * 12}px`);
    event.currentTarget.style.setProperty("--scene-y", `${((event.clientY - rect.top) / rect.height - .5) * 8}px`);
  };
  return (
    <main className="fatura-home">
      <header className="fatura-home-nav">
        <Logo />
        <nav aria-label="Navigation principale"><a href="#parcours">La plateforme</a><a href="#assistant-home">Assistant</a><a href="#espace">Votre espace</a></nav>
        <button className="btn btn-primary" onClick={() => navigate("/login")}>Se connecter<LogIn size={16} /></button>
      </header>
      <section className="fatura-home-hero" onPointerMove={moveScene} onPointerLeave={(event) => {
        event.currentTarget.style.setProperty("--scene-x", "0px"); event.currentTarget.style.setProperty("--scene-y", "0px");
      }}>
        <div className="fatura-scene"><img className="fatura-home-image" src="/assets/fatura-floating-documents.png" alt="Illustration de factures et d'un dossier bleu flottants" /></div>
        <div className="fatura-home-copy">
          <span className="fatura-home-eyebrow"><span />LA CLARTÉ COMMENCE ICI</span>
          <h1>FATURA<span>.tn</span></h1>
          <h2>Vos factures.<br /><span>Une nouvelle perspective.</span></h2>
          <p>Du premier document à la dernière vérification,<br className="desktop-break" /> retrouvez le plaisir d'une gestion bien organisée.</p>
          <div className="fatura-home-actions">
            <button className="btn btn-primary" onClick={() => navigate("/login")}>Accéder à mon espace<ChevronRight size={18} /></button>
            <a className="fatura-discover" href="#parcours">Explorer la plateforme<ChevronRight size={16} /></a>
          </div>
          <div className="fatura-hero-facts"><span><FileText size={15} />PDF & images</span><span><Eye size={15} />Sources visibles</span><span><Check size={15} />Validation humaine</span></div>
        </div>
        <span className="fatura-floating-note note-left" aria-hidden="true"><FileText size={19} /><span>Tout commence<br /><b>par une facture.</b></span></span>
        <span className="fatura-floating-note note-right" aria-hidden="true"><LocateFixed size={19} /><span>Chaque détail<br /><b>compte.</b></span></span>
        <a className="fatura-scroll-link" href="#parcours">DÉCOUVRIR<ChevronRight size={14} /></a>
      </section>

      <section className="fatura-home-process" id="parcours">
        <div className="fatura-home-section-title"><span>DU DOCUMENT À LA DÉCISION</span><h2>Simple à suivre.<br />Facile à prendre en main.</h2><p>Trois étapes, avec vous aux commandes.</p></div>
        <div className="fatura-product-tour">
          <div className="fatura-tour-tabs" role="tablist" aria-label="Parcours de traitement">
            {stages.map(({ icon: Icon, title, detail, label }, index) => <button key={label} id={`tour-tab-${index}`} role="tab" aria-selected={stage === index} aria-controls="tour-panel" tabIndex={stage === index ? 0 : -1} className={stage === index ? "selected" : ""} onClick={() => setStage(index)} onKeyDown={(event) => {
              if (["ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
                event.preventDefault();
                const next = event.key === "Home" ? 0 : event.key === "End" ? 2 : (stage + (["ArrowUp", "ArrowLeft"].includes(event.key) ? 2 : 1)) % 3;
                setStage(next); document.getElementById(`tour-tab-${next}`)?.focus();
              }
            }}><span className="tour-step-number">0{index + 1}</span><span><strong>{title}</strong><small>{detail}</small></span><Icon size={21} /></button>)}
          </div>
          <div className="fatura-tour-window" id="tour-panel" role="tabpanel" aria-labelledby={`tour-tab-${stage}`} tabIndex={0}>
            <div className="fatura-tour-toolbar"><span className="mini-brand">F</span><strong>FATURA.tn</strong><small>Aperçu · données d'exemple</small></div>
            <div className="fatura-tour-content" key={stage}>
              {stage === 0 ? <div className="tour-upload"><div className="tour-paper"><FileText size={32} /><b>FACTURE</b><i /><i /><i /></div><strong>facture-exemple.pdf</strong><span>Un document. Toutes ses informations.</span><div className="tour-format-list"><span>PDF</span><span>JPG</span><span>PNG</span></div></div>
                : <div className="tour-review">
                  <div className="tour-invoice-heading"><span><FileCheck2 size={22} />Facture d'exemple</span><b className={stage === 2 ? "tour-status-complete" : ""}>{stage === 2 ? "Exemple validé" : "À vérifier"}</b></div>
                  <div className="tour-data-row"><span>Fournisseur</span><strong>Entreprise Exemple</strong></div>
                  <div className="tour-data-row"><span>Numéro</span><strong>DEMO-001</strong></div>
                  <div className="tour-data-row"><span>Montant HT</span><strong>1 200,000 TND</strong></div>
                  <div className="tour-data-row"><span>TVA · 19 %</span><strong>228,000 TND</strong></div>
                  <div className="tour-data-row"><span>Timbre</span><strong>1,000 TND</strong></div>
                  <div className="tour-data-row tour-total"><span>Total TTC</span><strong>1 429,000 TND</strong></div>
                  <div className="tour-proof">{stage === 2 ? <Check size={16} /> : <LocateFixed size={16} />}<span>{stage === 2 ? "Informations relues et validées dans cet exemple." : 'Source : « Total TTC : 1 429,000 TND »'}</span></div>
                </div>}
            </div>
            <div className="fatura-tour-pagination"><span>0{stage + 1} / 03</span><div>{stages.map((s, i) => <button key={s.label} aria-label={s.label} aria-pressed={stage === i} className={stage === i ? "active" : ""} onClick={() => setStage(i)} />)}</div><button title="Étape suivante" aria-label="Étape suivante" onClick={() => setStage((stage + 1) % 3)}><ChevronRight size={18} /></button></div>
          </div>
        </div>
      </section>

      <section className="fatura-assistant-band" id="assistant-home">
        <div><span className="fatura-section-kicker">FATURA ASSISTANT</span><h2>Une question ?<br />Revenez à vos sources.</h2><p>Recherchez dans vos factures indexées et consultez les passages retrouvés.</p><button className="btn btn-primary" onClick={() => navigate("/login")}>Retrouver mes factures<ChevronRight size={17} /></button></div>
        <div className="fatura-question-list"><span><Sparkles size={18} />Vos documents, en conversation.</span>{["Quel est le total TTC de cette facture ?", "Qui est le fournisseur ?", "Quel passage justifie ce montant ?"].map((question) => <button key={question} onClick={() => navigate("/login")}><MessageSquareText size={18} />{question}<ChevronRight size={16} /></button>)}</div>
      </section>
      <section className="fatura-home-space" id="espace"><span className="fatura-section-kicker">VOTRE PROCHAIN DOCUMENT VOUS ATTEND</span><h2>Faites place à la clarté.</h2><button className="btn btn-primary" onClick={() => navigate("/login")}>Ouvrir FATURA.tn<ChevronRight size={18} /></button></section>
      <footer className="fatura-home-footer"><Logo /><span>Pensé pour vos factures. Conçu pour vous.</span><a href="/login" onClick={(event) => { event.preventDefault(); navigate("/login"); }}>Connexion<ChevronRight size={14} /></a></footer>
    </main>
  );
}

function Login({ navigate, setRole }: { navigate: (path: string) => void; setRole: (role: "guest" | "user") => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const submitLogin = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!email.trim() || !password.trim()) {
      setError("Veuillez saisir votre email et votre mot de passe.");
      return;
    }
    setIsSubmitting(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      if (!response.ok || !data.token) throw new Error(data.detail || "Connexion impossible");
      localStorage.setItem("fatura-auth-token", data.token);
      localStorage.setItem("fatura-user", JSON.stringify(userWithLocalProfile(data.user)));
      setRole("user");
      navigate("/dashboard");
    } catch (loginError) {
      setError(loginError instanceof Error ? loginError.message : "Backend auth inaccessible.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="login-page">
      <section className="login-brand">
        <Logo dark />
        <div>
          <h1>Traitement intelligent des factures tunisiennes</h1>
          <p>Importez, analysez, validez et archivez vos factures avec l'IA — pensé pour les comptables et entreprises en Tunisie.</p>
        </div>
        <ul>
          {["OCR et extraction automatique", "Preuves disponibles par champ", "Historique des corrections", "Validation humaine des factures"].map((item) => (
            <li key={item}><Check size={17} />{item}</li>
          ))}
        </ul>
        <p className="trusted-line">FATURA.tn · Factures et validation documentaire</p>
      </section>
      <section className="login-form-wrap">
        <Card className="login-card">
          <h2>Bon retour</h2>
          <p>Connectez-vous à votre espace</p>
          <form className="auth-form" onSubmit={submitLogin}>
            <label>Email<input value={email} onChange={(event) => setEmail(event.target.value)} placeholder="vous@entreprise.tn" type="email" /></label>
            <PasswordField label="Mot de passe" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Mot de passe" autoComplete="current-password" />
            <div className="form-row">
              <HelpDialog account />
            </div>
            {error && <span className="form-error">{error}</span>}
            <button className="btn btn-primary full" type="submit" disabled={isSubmitting}>{isSubmitting ? "Connexion..." : "Se connecter"}</button>
          </form>
          <div className="demo-box">
            <span>Nouveau sur FATURA.tn ?</span>
            <button className="btn btn-outline" onClick={() => navigate("/signup")}>Créer un compte</button>
          </div>
        </Card>
      </section>
    </main>
  );
}

function Signup({ navigate, setRole }: { navigate: (path: string) => void; setRole: (role: "guest" | "user") => void }) {
  const [accepted, setAccepted] = useState(false);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [avatarPreview, setAvatarPreview] = useState(userAvatars.maissa);
  const [avatarDataUrl, setAvatarDataUrl] = useState("");
  const handleAvatarUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Veuillez choisir une image de profil valide.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      setAvatarDataUrl(result);
      setAvatarPreview(result);
      setError("");
    };
    reader.readAsDataURL(file);
  };
  const submitSignup = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const required = ["name", "company", "email", "phone", "password", "confirm"].every((key) => String(form.get(key) || "").trim());
    if (!required) {
      setError("Veuillez compléter tous les champs.");
      return;
    }
    if (form.get("password") !== form.get("confirm")) {
      setError("Les mots de passe ne correspondent pas.");
      return;
    }
    if (!accepted) {
      setError("Veuillez accepter les conditions pour créer le compte.");
      return;
    }
    setIsSubmitting(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE_URL}/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          full_name: String(form.get("name")),
          company: String(form.get("company")),
          email: String(form.get("email")),
          phone: String(form.get("phone")),
          password: String(form.get("password")),
        }),
      });
      const data = await response.json();
      if (!response.ok || !data.token) throw new Error(data.detail || "Inscription impossible");
      localStorage.setItem("fatura-auth-token", data.token);
      localStorage.setItem("fatura-user", JSON.stringify(userWithLocalProfile(data.user, avatarDataUrl || userAvatars.maissa)));
      setRole("user");
      navigate("/dashboard");
    } catch (signupError) {
      setError(signupError instanceof Error ? signupError.message : "Backend auth inaccessible.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="login-page signup-page">
      <section className="login-brand">
        <Logo dark />
        <div>
          <h1>Créez votre espace FATURA.tn</h1>
          <p>Configurez votre entreprise, invitez votre équipe comptable et commencez à traiter vos factures tunisiennes.</p>
        </div>
        <ul>
          {["Espace comptable sécurisé", "Validation humaine des formulaires", "Assistant IA RAG intégré", "Export JSON après validation"].map((item) => (
            <li key={item}><Check size={17} />{item}</li>
          ))}
        </ul>
        <p className="trusted-line">Activation démo immédiate · Données TND · Workflow facture tunisien</p>
      </section>
      <section className="login-form-wrap">
        <Card className="login-card signup-card">
          <h2>Inscription</h2>
          <p>Renseignez les informations de votre entreprise</p>
          <form className="auth-form" onSubmit={submitSignup}>
            <div className="signup-grid">
              <label className="avatar-upload">
                <img src={avatarPreview} alt="Photo de profil" />
                <span>
                  <strong>Photo de profil</strong>
                  <small>PNG ou JPG, affichée dans votre espace</small>
                </span>
                <em>Choisir</em>
                <input accept="image/*" onChange={handleAvatarUpload} type="file" />
              </label>
              <label>Nom complet<input name="name" placeholder="Maissa Saadaoui" /></label>
              <label>Entreprise<input name="company" placeholder="YASMEEN ENGINEERING SYSTEMS" /></label>
              <label>Email professionnel<input name="email" placeholder="vous@entreprise.tn" type="email" /></label>
              <label>Téléphone<input name="phone" placeholder="+216 20 000 000" /></label>
              <PasswordField label="Mot de passe" name="password" placeholder="Mot de passe" autoComplete="new-password" />
              <PasswordField label="Confirmer" name="confirm" placeholder="Confirmer" autoComplete="new-password" />
            </div>
            <label className="checkline terms-line"><input type="checkbox" checked={accepted} onChange={(event) => setAccepted(event.target.checked)} />J'accepte les conditions et la politique de confidentialité</label>
            {error && <span className="form-error">{error}</span>}
            <button className="btn btn-primary full" type="submit" disabled={isSubmitting}>{isSubmitting ? "Création..." : "Créer mon compte"}</button>
          </form>
          <div className="auth-switch">
            <span>Vous avez déjà un compte ?</span>
            <button className="link-btn" onClick={() => navigate("/login")}>Se connecter</button>
          </div>
        </Card>
      </section>
    </main>
  );
}

function UserLayout({ route, navigate, notify, openAssistant, setRole }: { route: string; navigate: (path: string) => void; notify: (message: string) => void; openAssistant: () => void; setRole: (role: "guest" | "user") => void }) {
  const [collapsed, setCollapsed] = useState(false);
  const storedUser = readStoredUser();
  const displayName = storedUser.full_name || "Maissa Saadaoui";
  const jobTitle = roleLabel(storedUser.role);
  const avatar = storedUser.avatar || userAvatars.maissa;
  const logout = async () => {
    const token = localStorage.getItem("fatura-auth-token") || "";
    if (token) {
      try {
        await fetch(`${API_BASE_URL}/auth/logout?token=${encodeURIComponent(token)}`, { method: "POST" });
      } catch {
        // La session locale est effacée même si le backend n'est pas joignable.
      }
    }
    setRole("guest");
    navigate("/login");
  };
  const nav = [
    ["/dashboard", "Tableau de bord", Home],
    ["/import", "Importer", UploadCloud],
    ["/factures", "Factures", FileText],
    ["/fournisseurs", "Fournisseurs", Users],
    ["/parametres", "Paramètres", Settings],
  ] as const;

  return (
    <div className={`app-shell ${collapsed ? "sidebar-collapsed" : ""} ${route === "/import" ? "invoice-shell" : ""}`}>
      <aside className="sidebar">
        <Logo dark />
        {route === "/import" && (
          <button className="sidebar-primary-action" onClick={() => navigate("/import")}>
            <Plus size={16} />Nouvelle facture
          </button>
        )}
        <nav className="side-nav">
          {nav.map(([path, label, Icon]) => (
            <React.Fragment key={path}>
              <button className={route === path ? "active" : ""} title={label} aria-label={label} onClick={() => navigate(path)}>
                <Icon size={18} /><span>{label}</span>
              </button>
              {route === "/import" && path === "/import" && (
                <div className="sidebar-subnav">
                  <button onClick={() => navigate("/factures")}>Toutes les factures</button>
                  <button onClick={() => navigate("/factures?source=processed")}>Historique</button>
                  <button onClick={() => navigate("/factures/a-valider")}>À valider</button>
                </div>
              )}
            </React.Fragment>
          ))}
        </nav>
        <div className="sidebar-user">
          <div><img src={avatar} alt="" /><span><strong>{displayName}</strong><small>{jobTitle}</small></span></div>
          <button onClick={logout}><LogIn size={17} />Déconnexion</button>
          <button title={collapsed ? "Déployer le menu" : "Réduire le menu"} aria-label={collapsed ? "Déployer le menu" : "Réduire le menu"} onClick={() => setCollapsed(value => !value)}>{collapsed ? <ChevronRight size={17} /> : <ChevronLeft size={17} />}<span>{collapsed ? "Déployer" : "Réduire"}</span></button>
        </div>
      </aside>
      <div className="workspace">
        {route !== "/dashboard" && <Topbar openAssistant={openAssistant} />}
        <main className="content">{renderUserRoute(route, navigate, notify)}</main>
      </div>
    </div>
  );
}

function Topbar({ openAssistant }: { openAssistant: () => void }) {
  const storedUser = readStoredUser();
  const displayName = storedUser.full_name || "Maissa Saadaoui";
  const jobTitle = roleLabel(storedUser.role);
  const avatar = storedUser.avatar || userAvatars.maissa;
  return (
    <header className="topbar">
      <HelpDialog />
      <button className="btn btn-soft" onClick={openAssistant}><Sparkles size={16} />Fatura Assistant</button>
      <div className="profile"><img src={avatar} alt="" /><div><strong>{displayName}</strong><small>{jobTitle}</small></div></div>
    </header>
  );
}

function renderUserRoute(route: string, navigate: (path: string) => void, notify: (message: string) => void) {
  if (route === "/dashboard") return <LiveDashboard navigate={navigate} />;
  if (route === "/import") return <ImportPage navigate={navigate} />;
  if (route === "/documents/a-revoir") return <ReviewDocuments />;
  if (route === "/factures/a-valider") return <InvoicesPage navigate={navigate} pending />;
  if (route === "/factures") return <InvoicesPage key={window.location.search} navigate={navigate} />;
  if (route === "/fournisseurs") return <SuppliersPage navigate={navigate} />;
  if (route.includes("/analyse")) return <InvoiceReview key={invoiceIdFromRoute(route)} invoiceId={invoiceIdFromRoute(route)} navigate={navigate} />;
  if (route === "/parametres") return <AccountSettings />;
  if (route.startsWith("/validation")) return <InvoicesPage navigate={navigate} pending />;
  if (route === "/doublons" || route === "/exports") return <InvoicesPage navigate={navigate} />;
  return <ComingSoon route={route} />;
}

function Dashboard({ navigate }: { navigate: (path: string) => void }) {
  const [activeInsight, setActiveInsight] = useState("risk");
  const insights = [
    { id: "risk", label: "À vérifier", title: "3 factures demandent une vérification", text: "Échéances incertaines et zones OCR à contrôler.", impact: "31 850,350 TND à vérifier", icon: AlertTriangle },
    { id: "duplicate", label: "Doublons", title: "Risque de doublon détecté", text: "Deux factures Office Depot présentent des informations similaires.", impact: "458,900 TND concernés", icon: Copy },
    { id: "ready", label: "Validées", title: "189 factures validées", text: "Retrouvez les documents validés et leurs sources.", impact: "72% des factures", icon: ShieldCheck },
  ];
  const currentInsight = insights.find(item => item.id === activeInsight) || insights[0];
  const InsightIcon = currentInsight.icon;
  const dashboardStats = {
    totals: { total_invoices: 263, validated_invoices: 189, draft_invoices: 47 },
    top_suppliers: [
      { name: "SOTUFAB", total_amount: 31850.350, invoices_count: 12 },
      { name: "Tunisie Telecom", total_amount: 1284.700, invoices_count: 8 },
      { name: "Office Depot", total_amount: 458.900, invoices_count: 6 },
    ],
    due_soon: [
      { supplier_name: "SOTUFAB", total_amount: 31850.350, due_date: "20/09/2026" },
      { supplier_name: "Tunisie Telecom", total_amount: 1284.700, due_date: "25/09/2026" },
    ],
  };
  const dashboardTotals = dashboardStats?.totals || {};
  const averageRate = Math.round(weekly.reduce((sum, item) => sum + item.taux, 0) / weekly.length);
  const validationDelta = weekly[weekly.length - 1].taux - weekly[0].taux;

  return (
    <div className="dashboard-page">
      <PageHeader title="Tableau de bord" subtitle="Démonstration · Données fictives" action={<button className="btn btn-primary" onClick={() => navigate("/factures")}><FileText size={17} />Ouvrir la base réelle</button>} />
      <div className="dashboard-section-divider"><h2>Vue d'ensemble</h2><span /></div>
      <div className="kpi-grid">
        <Kpi title="IMPORTÉES" value={String(dashboardTotals.total_invoices ?? 0)} trend="+18% vs mois dernier" icon={Inbox} />
        <Kpi title="VALIDÉES" value={String(dashboardTotals.validated_invoices ?? 0)} trend="Factures validées" tone="success" icon={Check} />
        <Kpi title="À VÉRIFIER" value={String(dashboardTotals.draft_invoices ?? 0)} trend="À contrôler" tone="warning" icon={Clock} />
        <Kpi title="DOUBLONS" value="6" trend="+12% vs mois dernier" tone="danger" icon={Copy} />
      </div>
      <section className="dashboard-insight">
        <div className="dashboard-insight-head"><span><Sparkles size={16} />Insight IA</span>
          <div role="tablist" aria-label="Synthèse des factures">{insights.map(item => <button type="button" role="tab" aria-selected={activeInsight === item.id} aria-controls="dashboard-insight-panel" id={`dashboard-insight-${item.id}`} key={item.id} onClick={() => setActiveInsight(item.id)}>{item.label}</button>)}</div>
        </div>
        <div className="dashboard-insight-body" id="dashboard-insight-panel" role="tabpanel" aria-labelledby={`dashboard-insight-${activeInsight}`}>
          <InsightIcon size={30} />
          <div><h2>{currentInsight.title}</h2><p>{currentInsight.text}</p></div>
          <div className="dashboard-insight-action"><strong>{currentInsight.impact}</strong><button type="button" onClick={() => navigate(activeInsight === "ready" ? "/factures" : "/factures/a-valider")}>Consulter <ChevronRight size={16} /></button></div>
        </div>
      </section>
      <div className="dashboard-section-divider"><h2>Analyse</h2><span /></div>
      <div className="grid-two">
        <Card className="chart-card-wide validation-chart-card">
          <ChartTitle title="Taux de validation" subtitle="Hebdomadaire" />
          <div className="chart-headline">
            <strong>{averageRate}%</strong>
            <span>moyenne sur 6 semaines</span>
            <em>+{validationDelta} pts</em>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={weekly} margin={{ top: 12, right: 18, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="validationBlue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#60A5FA" stopOpacity={0.45}/>
                  <stop offset="95%" stopColor="#60A5FA" stopOpacity={0.03}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="4 5" vertical={false} stroke="rgba(219,234,254,.2)" />
              <XAxis dataKey="label" tickLine={false} axisLine={false} stroke="#cbd8e7" />
              <YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} tickLine={false} axisLine={false} stroke="#cbd8e7" width={48} />
              <Tooltip content={<DashboardTooltip suffix="%" metricLabel="Taux validé" />} />
              <Area type="monotone" dataKey="taux" stroke="#93c5fd" fill="url(#validationBlue)" strokeWidth={3} dot={{ r: 4, strokeWidth: 2, fill: "#17395f", stroke: "#bfdbfe" }} activeDot={{ r: 6, strokeWidth: 2, fill: "#2563eb", stroke: "#fff" }} />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
        <Card className="status-chart-card">
          <ChartTitle title="Répartition des statuts" subtitle="Pourcentage" />
          <div className="dashboard-status-content">
            <div className="donut-wrap"><ResponsiveContainer width="100%" height={210}>
              <PieChart><Pie data={statusData} dataKey="value" innerRadius={60} outerRadius={84} paddingAngle={3} startAngle={90} endAngle={-270}>
                {statusData.map(entry => <Cell key={entry.name} fill={entry.color} />)}
              </Pie><Tooltip content={<DashboardTooltip suffix="%" metricLabel="Part" />} /></PieChart>
            </ResponsiveContainer><div className="donut-center"><strong>72%</strong><span>validées</span></div></div>
            <StatusBreakdown />
          </div>
          <button type="button" className="dashboard-review-link" onClick={() => navigate("/factures/a-valider")}>Ouvrir les factures à vérifier <ChevronRight size={16} /></button>
        </Card>
      </div>
      <div className="dashboard-section-divider"><h2>Suivi</h2><span /></div>
      <div className="dashboard-bottom-grid">
        <section><ChartTitle title="Top fournisseurs" subtitle="Montant TTC" />
          {dashboardStats.top_suppliers.map(supplier => <div className="dashboard-summary-row" key={supplier.name}>
            <span><strong>{supplier.name}</strong><small>{supplier.invoices_count} factures</small></span>
            <b>{formatTnd(supplier.total_amount)}</b>
          </div>)}
        </section>
        <section><ChartTitle title="Échéances à venir" subtitle="Date limite" />
          {dashboardStats.due_soon.map(invoice => <div className="dashboard-summary-row" key={invoice.supplier_name}>
            <span><strong>{invoice.supplier_name}</strong><small>{formatTnd(invoice.total_amount)}</small></span>
            <time>{invoice.due_date}</time>
          </div>)}
        </section>
      </div>
    </div>
  );
}

function RecentInvoicesCard({ navigate }: { navigate: (path: string) => void }) {
  const [rows, setRows] = useState<Invoice[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  React.useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    fetch(`${API_BASE_URL}/invoices/catalog?source=processed&page=1&page_size=5`)
      .then((response) => {
        if (!response.ok) throw new Error("Backend indisponible");
        return response.json();
      })
      .then((data) => {
        if (!cancelled) {
          const mappedRows: Invoice[] = Array.isArray(data?.invoices) ? data.invoices.map(mapBackendInvoice) : [];
          const uniqueRows = mappedRows.filter((invoice, index, list) => {
            const key = `${invoice.file}|${invoice.supplier}|${invoice.number}|${invoice.amount}`;
            return list.findIndex((candidate) => `${candidate.file}|${candidate.supplier}|${candidate.number}|${candidate.amount}` === key) === index;
          });
          setRows(uniqueRows);
        }
      })
      .catch((requestError) => {
        if (!cancelled) setError(requestError?.message || "Impossible de charger les factures récentes.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <Card>
      <ChartTitle title="Historique" subtitle="Voir tout" />
      {loading && <EmptyState text="Chargement des factures récentes..." />}
      {!loading && error && <div className="backend-alert"><AlertTriangle size={18} /><span>{error}</span></div>}
      {!loading && !error && rows.length === 0 && <EmptyState text="Aucune facture récente dans la base." />}
      {!loading && !error && rows.length > 0 && <InvoiceTable rows={rows} navigate={navigate} />}
    </Card>
  );
}

function ImportPage({ navigate }: { navigate: (path: string) => void }) {
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "success" | "error">("idle");
  const [uploadMessage, setUploadMessage] = useState("Backend connecté : prêt à tester un PDF ou une image.");
  const [agentResult, setAgentResult] = useState<any>(null);
  const [activeProof, setActiveProof] = useState<any>(null);
  const fieldProof = (key: string) => <InvoiceFieldProof formKey={key} value={String(formValues[key as keyof InvoiceFormValues] ?? "")} result={agentResult} onLocate={setActiveProof} isPdf={Boolean(previewFile?.mime.includes("pdf") || previewFile?.name.toLowerCase().endsWith(".pdf"))} activeKey={activeProof?.field_key} />;
  const [selectedFileName, setSelectedFileName] = useState("");
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [formValues, setFormValues] = useState<InvoiceFormValues>(emptyInvoiceForm);
  const [invoiceTab, setInvoiceTab] = useState("Informations");
  const [previewFile, setPreviewFile] = useState<UploadedPreview | null>(null);
  const [isInvoiceValidated, setIsInvoiceValidated] = useState(false);
  const [isInvoiceSaved, setIsInvoiceSaved] = useState(false);
  const [isSavingInvoice, setIsSavingInvoice] = useState(false);
  const [validationTrace, setValidationTrace] = useState<any[]>([]);
  const previousUploadStatus = React.useRef(uploadStatus);
  React.useLayoutEffect(() => {
    const justFinished = uploadStatus === "success" && previousUploadStatus.current !== "success";
    previousUploadStatus.current = uploadStatus;
    if (!justFinished) return;
    const result = document.getElementById("import-analysis-result");
    result?.focus({ preventScroll: true });
    result?.scrollIntoView({ block: "start", behavior: "auto" });
  }, [uploadStatus]);
  React.useEffect(() => {
    return () => {
      if (previewFile?.url) URL.revokeObjectURL(previewFile.url);
    };
  }, [previewFile?.url]);
  const uploadToBackend = async (file: File) => {
    if (!/\.(pdf|png|jpe?g)$/i.test(file.name) || file.size > 10 * 1024 * 1024) {
      setUploadStatus("error");
      setUploadMessage(file.size > 10 * 1024 * 1024 ? "Le fichier dépasse la limite de 10 Mo." : "Sélectionnez un fichier PDF, JPG ou PNG.");
      setSourceFile(null); setPreviewFile(null); setAgentResult(null); setSelectedFileName("");
      return;
    }
    setUploadStatus("uploading");
    setSelectedFileName(file.name);
    setUploadMessage(`Analyse en cours : ${file.name}`);
    setAgentResult(null);
    setFormValues(emptyInvoiceForm);
    setIsInvoiceValidated(false);
    setIsInvoiceSaved(false);
    setValidationTrace([]);
    setSourceFile(file);
    setPreviewFile({ url: URL.createObjectURL(file), name: file.name, mime: file.type });
    try {
      const formData = new FormData();
      formData.append("file", file);
      const token = localStorage.getItem("fatura-auth-token") || "";
      if (!token) throw new Error("Connexion requise avant l'analyse de facture.");
      setActiveProof(null);
      const response = await fetch(`${API_BASE_URL}/agent/process-document?model_choice=auto&index_for_rag=true&detect_objects=true&token=${encodeURIComponent(token)}`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok || data.success === false) {
        throw new Error(data.detail || data.error || data.message || "Traitement impossible");
      }
      setAgentResult(data);
      const isInvoice = isInvoiceResult(data);
      if (isInvoice) {
        const invoiceId = invoiceIdFromAgentResult(data);
        if (invoiceId) {
          try {
            const detailResponse = await fetch(`${API_BASE_URL}/invoices/catalog/detail?source=processed&invoice_id=${encodeURIComponent(String(invoiceId))}`);
            if (detailResponse.ok) {
              const detailData = await detailResponse.json();
              const mergedResult = { ...data, invoice: detailData?.invoice || data.invoice };
              setAgentResult(mergedResult);
              setFormValues(extractInvoiceFormValues(mergedResult));
            } else {
              setFormValues(extractInvoiceFormValues(data));
            }
          } catch {
            setFormValues(extractInvoiceFormValues(data));
          }
        } else {
          setFormValues(extractInvoiceFormValues(data));
        }
      } else {
        setFormValues(emptyInvoiceForm);
      }
      setUploadStatus("success");
      const confidence = percentValue(data.classification?.confidence, 0);
      setUploadMessage(isInvoice ? `Facture détectée · confiance ${confidence}%` : `Non-facture détectée · confiance ${confidence}%`);
      setIsInvoiceSaved(false);
      setValidationTrace([]);
      if (isInvoice && invoiceIdFromAgentResult(data)) {
        navigate(`/factures/${encodeURIComponent(String(invoiceIdFromAgentResult(data)))}/analyse`);
      }
    } catch (error) {
      setUploadStatus("error");
      setUploadMessage(error instanceof Error ? error.message : "Backend inaccessible. Vérifiez que FastAPI tourne sur le port 8002.");
    }
  };
  const reextractCurrentFile = () => {
    if (!sourceFile || uploadStatus === "uploading") return;
    void uploadToBackend(sourceFile);
  };
  const currentFileName = selectedFileName || uploadMessage.replace("Analyse en cours : ", "");
  const updateFormValue = (key: keyof InvoiceFormValues, value: string) => {
    setActiveProof(null);
    setFormValues((current) => ({ ...current, [key]: value }));
    setIsInvoiceValidated(false);
    setIsInvoiceSaved(false);
  };
  const addInvoiceToDatabase = async () => {
    if (uploadStatus !== "success" || !isInvoiceResult(agentResult)) return;
    const invoiceId = invoiceIdFromAgentResult(agentResult);
    if (!invoiceId) {
      setUploadMessage("Impossible d'ajouter à la base : identifiant facture absent dans la réponse backend.");
      return;
    }
    setIsSavingInvoice(true);
    try {
      const response = await fetch(`${API_BASE_URL}/invoices/${invoiceId}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          ...(localStorage.getItem("fatura-auth-token") ? { Authorization: `Bearer ${localStorage.getItem("fatura-auth-token")}` } : {}),
        },
        body: JSON.stringify({
          normalized_data: normalizedDataFromForm(formValues),
          status: "draft",
        }),
      });
      const data = await response.json();
      if (!response.ok || data.success === false) {
        throw new Error(data.detail || data.message || "Enregistrement impossible");
      }
      setAgentResult((current: any) => ({ ...current, invoice: data.invoice || current?.invoice }));
      setIsInvoiceSaved(true);
      setUploadMessage("Facture ajoutée à la base SQLite avec les champs corrigés.");
      window.setTimeout(() => window.location.reload(), 1800);
    } catch (error) {
      setUploadMessage(error instanceof Error ? error.message : "Erreur pendant l'ajout à la base.");
    } finally {
      setIsSavingInvoice(false);
    }
  };
  const validateInvoice = async () => {
    if (uploadStatus !== "success" || !isInvoiceResult(agentResult)) return;
    const invoiceId = invoiceIdFromAgentResult(agentResult);
    if (!invoiceId) {
      setUploadMessage("Validation impossible : identifiant facture absent.");
      return;
    }
    try {
      const token = localStorage.getItem("fatura-auth-token") || "";
      const response = await fetch(`${API_BASE_URL}/invoices/${invoiceId}/validate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ normalized_data: normalizedDataFromForm(formValues) }),
      });
      const data = await response.json();
      if (!response.ok || data.success === false) {
        throw new Error(data.detail || data.message || "Validation impossible");
      }
      const nextInvoice = data.invoice || agentResult?.invoice;
      const nextTrace = [
        ...getAgentDecisions(agentResult),
        ...getAgentDecisions({ invoice: nextInvoice }),
        {
          step: "human_validation",
          decision: "validated",
          reason: "Les champs corrigés du formulaire ont été validés et synchronisés avec la base.",
          status: "completed",
          confidence: 1,
          metadata: {
            invoice_id: invoiceId,
            rag_document_id: agentResult?.rag_document?.document_id || null,
          },
        },
      ];
      setAgentResult((current: any) => ({ ...current, invoice: nextInvoice }));
      setValidationTrace(nextTrace);
      setIsInvoiceValidated(true);
      setIsInvoiceSaved(true);
      setUploadMessage("Facture validée : traces RAG et validation backend enregistrées.");
      window.setTimeout(() => window.location.reload(), 2500);
    } catch (error) {
      setUploadMessage(error instanceof Error ? error.message : "Erreur pendant la validation.");
    }
  };
  const cancelInvoice = () => {
    setAgentResult(null);
    setFormValues(emptyInvoiceForm);
    setSelectedFileName("");
    setSourceFile(null);
    setPreviewFile(null);
    setActiveProof(null);
    setUploadStatus("idle");
    setUploadMessage("Import annulé. Vous pouvez sélectionner une nouvelle facture.");
    setValidationTrace([]);
    setIsInvoiceValidated(false);
    setIsInvoiceSaved(false);
    setInvoiceTab("Informations");
  };
  const exportValidatedJson = () => {
    if (!isInvoiceValidated) return;
    const payload = {
      status: "validated",
      validated_at: new Date().toISOString(),
      validated_by: readStoredUser().email || "demo@fatura.tn",
      fields: formValues,
      invoice_lines: extractInvoiceLines(agentResult),
      visual_authentication: agentResult?.object_detections || [],
      classification: agentResult?.classification || null,
      rag: agentResult?.rag || agentResult?.rag_evidence || null,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${formValues.invoiceNumber || selectedFileName || "facture-validee"}.json`.replace(/[^\w.-]+/g, "_");
    link.click();
    URL.revokeObjectURL(url);
  };
  return (
    <>
      <div className="invoice-create-head">
        <div>
          <h1>Nouvelle facture</h1>
          <p>Accueil <ChevronRight size={13} /> Factures <ChevronRight size={13} /> Nouvelle facture</p>
        </div>
      </div>

      <div className={`new-invoice-workspace progressive-import ${uploadStatus === "success" && isInvoiceResult(agentResult) ? "has-invoice" : "single-stage"}`}>
        {uploadStatus === "uploading" ? <section className="import-processing-stage">
          <StepLabel number="2" title="Analyse du document" />
          <UploadLoadingPanel fileName={currentFileName} />
        </section> : <>
        {uploadStatus === "success" && <ImportClassification result={agentResult} fileName={currentFileName} />}
        <div className="invoice-left-flow">
          <Card className="workflow-card upload-step-card">
            <StepLabel number="1" title={uploadStatus === "success" ? "Document source" : "Importer votre facture"} showNumber={uploadStatus !== "success"} />
            <div className="upload-drop-card" onDragOver={(event) => event.preventDefault()} onDrop={(event) => {
              event.preventDefault();
              const files = event.dataTransfer.files;
              if (files.length === 1) void uploadToBackend(files[0]);
              else setUploadMessage("Sélectionnez une seule facture à la fois.");
            }}>
              {previewFile ? (
                <div className="upload-preview-inside">
                  <div id="invoice-source-preview"><UploadedDocumentPreview file={previewFile} detections={agentResult?.object_detections || []} status={uploadStatus} highlight={activeProof} /></div>
                  <label className="btn btn-outline upload-picker">
                    <UploadCloud size={16} />Changer le fichier
                    <input type="file" accept=".pdf,.png,.jpg,.jpeg" onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) uploadToBackend(file);
                      event.currentTarget.value = "";
                    }} />
                  </label>
                </div>
              ) : (
                <>
                  <span className="upload-doc-icon"><FileText size={30} /></span>
                  <strong>Glissez-déposez votre fichier ici</strong>
                  <small>PDF, JPG, PNG · taille max : 10 Mo</small>
                  <label className="btn btn-primary upload-picker">
                    <UploadCloud size={16} />
                    Sélectionner un fichier
                    <input type="file" accept=".pdf,.png,.jpg,.jpeg" onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) uploadToBackend(file);
                      event.currentTarget.value = "";
                    }} />
                  </label>
                </>
              )}
            </div>
            {selectedFileName && <div className={`uploaded-file-card ${uploadStatus}`}>
              <FileText size={22} />
              <div>
                <strong>{currentFileName}</strong>
                <small>{uploadStatus === "success" ? "Analyse terminée" : "Traitement échoué"}</small>
              </div>
              <span>{uploadStatus === "success" ? "Fichier analysé" : "Erreur"}</span>
            </div>}
            {uploadStatus === "error" && <p className="form-error" role="alert">{uploadMessage}</p>}
            {sourceFile && <div className="import-source-actions">
              <button className="btn btn-outline" type="button" onClick={reextractCurrentFile}><RefreshCw size={15} />Ré-extraire</button>
              <button className="btn btn-outline" type="button" onClick={cancelInvoice}>Annuler</button>
            </div>}
          </Card>
        </div>

        {uploadStatus === "success" && isInvoiceResult(agentResult) && <div className="invoice-right-flow">
          <Card className="workflow-card validation-form-card">
            <StepLabel number="4" title="Vérifier et valider les informations" showNumber={false} />
            <Tabs tabs={["Informations", "Lignes de facture", "Authentification"]} active={invoiceTab} setActive={setInvoiceTab} />
            {invoiceTab === "Informations" && (
              <>
                <div className="invoice-form-grid">
                  <h3>Fournisseur</h3>
                  <label>Raison sociale fournisseur *{fieldProof("supplier")}<input value={formValues.supplier} onChange={(event) => updateFormValue("supplier", event.target.value)} placeholder="Sera rempli après extraction" /></label>
                  <label>Identifiant fiscal fournisseur{fieldProof("taxId")}<input value={formValues.taxId} onChange={(event) => updateFormValue("taxId", event.target.value)} placeholder="0000000 A/M/000" /></label>
                  <label>Téléphone fournisseur{fieldProof("supplierPhone")}<input value={formValues.supplierPhone} onChange={(event) => updateFormValue("supplierPhone", event.target.value)} placeholder="Téléphone fournisseur détecté" /></label>
                  <label className="wide-field">Adresse fournisseur{fieldProof("address")}<textarea value={formValues.address} onChange={(event) => updateFormValue("address", event.target.value)} placeholder="Adresse détectée automatiquement" /></label>

                  <details className="wide-field"><summary>Client et coordonnées</summary><div className="invoice-form-grid">
                  <label>Client / Société facturée{fieldProof("customerName")}<input value={formValues.customerName} onChange={(event) => updateFormValue("customerName", event.target.value)} placeholder="Client détecté" /></label>
                  <label>Identifiant fiscal client{fieldProof("customerTaxId")}<input value={formValues.customerTaxId} onChange={(event) => updateFormValue("customerTaxId", event.target.value)} placeholder="MF client détecté" /></label>
                  <label>Téléphone client{fieldProof("customerPhone")}<input value={formValues.customerPhone} onChange={(event) => updateFormValue("customerPhone", event.target.value)} placeholder="Téléphone client détecté" /></label>
                  <label className="wide-field">Adresse client{fieldProof("customerAddress")}<textarea value={formValues.customerAddress} onChange={(event) => updateFormValue("customerAddress", event.target.value)} placeholder="Adresse client détectée" /></label>

                  </div></details>
                  <h3>Références facture</h3>
                  <label>Numéro de facture *{fieldProof("invoiceNumber")}<input value={formValues.invoiceNumber} onChange={(event) => updateFormValue("invoiceNumber", event.target.value)} placeholder="N° facture détecté" /></label>
                  <label className="wide-field">Désignation / Produits détectés{fieldProof("productDesignation")}<input value={formValues.productDesignation} onChange={(event) => updateFormValue("productDesignation", event.target.value)} placeholder="Désignation ou produits extraits des lignes" /></label>
                  <label>Date de facture *{fieldProof("invoiceDate")}<input value={formValues.invoiceDate} onChange={(event) => updateFormValue("invoiceDate", event.target.value)} placeholder="JJ/MM/AAAA" /></label>
                  <label>Date d'échéance{fieldProof("dueDate")}<input value={formValues.dueDate} onChange={(event) => updateFormValue("dueDate", event.target.value)} placeholder="JJ/MM/AAAA" /></label>
                  <label>Bon de livraison{fieldProof("deliveryNumber")}<input value={formValues.deliveryNumber} onChange={(event) => updateFormValue("deliveryNumber", event.target.value)} placeholder="Référence BL" /></label>
                  <label>Devise{fieldProof("currency")}<input value={formValues.currency} onChange={(event) => updateFormValue("currency", event.target.value)} placeholder="Devise détectée" /></label>

                  <h3>Montants</h3>
                  <label>Montant HT *{fieldProof("subtotal")}<div className="currency-input"><input value={formValues.subtotal} onChange={(event) => updateFormValue("subtotal", event.target.value)} placeholder="Non indiqué" /><span>{formValues.currency || "TND"}</span></div></label>
                  <label>Montant TVA *{fieldProof("taxAmount")}<div className="currency-input"><input value={formValues.taxAmount} onChange={(event) => updateFormValue("taxAmount", event.target.value)} placeholder="Non indiquée" /><span>{formValues.currency || "TND"}</span></div></label>
                  <label>Taux TVA{fieldProof("taxRate")}<input value={formValues.taxRate} onChange={(event) => updateFormValue("taxRate", event.target.value)} placeholder="Non indiqué" /></label>
                  <label>Timbre fiscal{fieldProof("stampDuty")}<div className="currency-input"><input value={formValues.stampDuty} onChange={(event) => updateFormValue("stampDuty", event.target.value)} placeholder="1,000" /><span>{formValues.currency || "TND"}</span></div></label>
                  <label>Remise{fieldProof("discount")}<div className="currency-input"><input value={formValues.discount} onChange={(event) => updateFormValue("discount", event.target.value)} placeholder="0,000" /><span>{formValues.currency || "TND"}</span></div></label>
                  <label>Montant TTC *{fieldProof("totalAmount")}<div className="currency-input"><input value={formValues.totalAmount} onChange={(event) => updateFormValue("totalAmount", event.target.value)} placeholder="0,000" /><span>{formValues.currency || "TND"}</span></div></label>
                  <label>Net à payer{fieldProof("netPayable")}<div className="currency-input"><input value={formValues.netPayable} onChange={(event) => updateFormValue("netPayable", event.target.value)} placeholder="0,000" /><span>{formValues.currency || "TND"}</span></div></label>

                  <details className="wide-field"><summary>Paiement et informations complémentaires</summary><div className="invoice-form-grid">
                  <label>Mode de paiement{fieldProof("paymentMethod")}<input value={formValues.paymentMethod} onChange={(event) => updateFormValue("paymentMethod", event.target.value)} placeholder="Mode détecté" /></label>
                  <label>Conditions de paiement{fieldProof("paymentTerms")}<input value={formValues.paymentTerms} onChange={(event) => updateFormValue("paymentTerms", event.target.value)} placeholder="Ex. 30 jours fin de mois" /></label>
                  <label className="wide-field">RIB / Compte bancaire{fieldProof("rib")}<input value={formValues.rib} onChange={(event) => updateFormValue("rib", event.target.value)} placeholder="RIB détecté" /></label>
                  <label className="wide-field">Montant en lettres / Notes{fieldProof("notes")}<textarea value={formValues.notes} onChange={(event) => updateFormValue("notes", event.target.value)} placeholder="Texte complémentaire détecté" /></label>
                  </div></details>
                </div>
                {isInvoiceValidated && <div className="validation-ready-note"><Check size={15} />Facture validée. L'export JSON est maintenant disponible.</div>}
                {validationTrace.length > 0 && <ValidationTracePanel trace={validationTrace} agentResult={agentResult} />}
                <div className="form-actions">
                  <button className="btn btn-outline" onClick={cancelInvoice} type="button">Annuler</button>
                  <button
                    className={`btn ${isInvoiceSaved ? "btn-saved" : "btn-soft"}`}
                    disabled={uploadStatus !== "success" || !isInvoiceResult(agentResult) || isSavingInvoice}
                    onClick={addInvoiceToDatabase}
                    type="button"
                  >
                    {isSavingInvoice ? <Loader2 size={16} /> : isInvoiceSaved ? <Check size={16} /> : <Archive size={16} />}
                    {isSavingInvoice ? "Ajout..." : isInvoiceSaved ? "Ajoutée à la base" : "Ajouter à la base"}
                  </button>
                  {isInvoiceValidated && <button className="btn btn-soft" onClick={exportValidatedJson}><Download size={16} />Exporter JSON validé</button>}
                  <button className="btn btn-primary" disabled={uploadStatus !== "success" || !isInvoiceResult(agentResult)} onClick={validateInvoice}><Check size={16} />Valider la facture</button>
                </div>
              </>
            )}
            {invoiceTab === "Lignes de facture" && <InvoiceLinesPanel status={uploadStatus} agentResult={agentResult} currency={formValues.currency || "TND"} />}
            {invoiceTab === "Authentification" && <AuthenticationPanel ready={uploadStatus === "success"} detections={agentResult?.object_detections || []} />}
          </Card>

        </div>}
        </>}
      </div>
    </>
  );
}

const proofFieldKeys: Record<string, string> = {
  supplier: "vendor_name", taxId: "vendor_tax_id", supplierPhone: "vendor_phone", address: "vendor_address",
  customerName: "customer_name", customerTaxId: "customer_tax_id", customerPhone: "customer_phone", customerAddress: "customer_address",
  invoiceNumber: "invoice_number", productDesignation: "product_designation", invoiceDate: "date", dueDate: "due_date",
  deliveryNumber: "delivery_number", currency: "currency", subtotal: "subtotal", taxAmount: "tax_amount", taxRate: "tax_rate",
  stampDuty: "stamp_duty", discount: "discount", totalAmount: "total_amount", netPayable: "net_payable",
  paymentMethod: "payment_method", paymentTerms: "payment_terms", rib: "rib", notes: "amount_in_words",
};

function InvoiceFieldProof({ formKey, value, result, onLocate, isPdf, activeKey }: { formKey: string; value: string; result: any; onLocate: (location: any) => void; isPdf: boolean; activeKey?: string }) {
  if (!value || !result) return null;
  const key = proofFieldKeys[formKey];
  const raw = result?.invoice?.raw_data || result?.invoice?.metadata || {};
  const proofs = raw.field_proofs || result.field_proofs || {};
  return <FieldEvidence fieldKey={key} value={value} proof={proofs[key]} isPdf={isPdf} active={activeKey === key} onLocate={onLocate} />;
}

function StepLabel({ number, title, showNumber = true }: { number: string; title: string; showNumber?: boolean }) {
  return <div className={`step-label ${showNumber ? "" : "no-number"}`}>{showNumber && <span>{number}</span>}<strong>{title}</strong></div>;
}

function ExtractionSummary({ status, message, agentResult, canReextract, onReextract }: { status: "idle" | "uploading" | "success" | "error"; message: string; agentResult: any; canReextract: boolean; onReextract: () => void }) {
  const confidence = status === "success" ? percentValue(agentResult?.classification?.confidence, 0) : 0;
  const classification = status === "success" ? getClassificationSummary(agentResult) : null;
  const isInvoice = status === "success" ? isInvoiceResult(agentResult) : false;
  const cardStatus = status === "success" && !isInvoice ? "non-invoice" : status;
  const rows = [
    ["Fournisseur", Building2],
    ["Facture", FileCheck2],
    ["Montants", Gauge],
    ["TVA", ShieldCheck],
    ["Objets", Eye],
  ] as const;
  return (
    <div className="extraction-summary">
      <div className={`extraction-visual-card ${cardStatus}`}>
        <div className="extraction-hero-icon">
          {status === "error" || cardStatus === "non-invoice" ? <AlertTriangle size={29} /> : <FileCheck2 size={29} />}
          <span>{status === "error" || cardStatus === "non-invoice" ? "!" : <Check size={11} />}</span>
        </div>
        <div className="extraction-hero-copy">
          <strong>{cardStatus === "non-invoice" ? "Classification documentaire" : "Extraction automatique"}</strong>
          <small>
            {status === "idle"
              ? "En attente d'un document"
              : status === "error"
                ? message
                : isInvoice
                  ? "Facture détectée, champs prêts à valider"
                  : "Non-facture détectée, validation facture bloquée"}
          </small>
        </div>
        <div className="extraction-score">
          <strong>{confidence}%</strong>
          <small>{status === "idle" ? "Score après analyse" : isInvoice ? "Confiance globale" : "Score facture"}</small>
        </div>
        <div className="extraction-wave" aria-hidden="true">
          <svg viewBox="0 0 460 78" preserveAspectRatio="none">
            <path d="M0 52 C76 76 136 44 204 43 C292 40 309 8 383 17 C423 22 444 11 460 5" />
          </svg>
        </div>
        <div className="extraction-step-icons">
          {rows.map(([row, Icon]) => (
            <span key={row} className={status === "success" && !isInvoice ? "inactive" : ""}>
              <i><Icon size={14} />{status === "success" && isInvoice && <Check size={9} />}</i>
              <em>{row}</em>
            </span>
          ))}
        </div>
      </div>
      {classification && (
        <div className={`classification-result ${classification.tone}`}>
          <div>
            <span><FolderSearch size={15} /> Classification du document</span>
            <strong>{classification.label}</strong>
          </div>
          <p>{classification.evidence}</p>
          <em>{classification.confidence || confidence}%</em>
        </div>
      )}
      <button className="btn btn-outline full" type="button" disabled={!canReextract || status === "uploading"} onClick={onReextract}><RefreshCw size={15} />Ré-extraire</button>
    </div>
  );
}

function YoloObjectsCard({ ready, detections = [] }: { ready: boolean; detections?: any[] }) {
  if (!ready) {
    return (
      <div className="empty-detection">
        <Eye size={18} />
        <strong>Authentification en attente</strong>
        <span>YOLO vérifiera logo, cachet et signature après l'analyse.</span>
      </div>
    );
  }
  const retained = visibleDetections(detections);
  const objects = [
    { key: "logo", name: "Logo", Icon: Building2 },
    { key: "stamp", name: "Cachet", Icon: Stamp },
    { key: "signature", name: "Signature", Icon: FileCheck2 },
  ].map(category => {
    const matches = retained.filter(d => detectionCategory(d) === category.key);
    return { ...category, count: matches.length, confidence: Math.round(Math.max(0, ...matches.map(detectionScore)) * 100) };
  });
  return (
    <div className="yolo-object-list">
      <p className="yolo-auth-note">
        Détections visuelles à vérifier. Ces scores ne certifient pas l’authenticité du document.
      </p>
      {objects.length === 0 && <div className="empty-detection"><Eye size={18} /><strong>Aucune détection YOLO confirmée</strong><span>Le modèle actuel est entraîné sur logo et cachet.</span></div>}
      {objects.map((object, index) => (
        <div className="yolo-object-row" key={`${object.name}-${object.confidence}-${index}`}>
          <object.Icon size={24} aria-hidden="true" />
          <div>
            <strong>{object.name}</strong>
            <small>{object.count ? `${object.count} détection${object.count > 1 ? "s" : ""} · score maximal` : "Non détecté"}</small>
          </div>
          <span>{object.count ? `${object.confidence}%` : "—"}</span>
        </div>
      ))}
    </div>
  );
}

function AuthenticationPanel({ ready, detections = [] }: { ready: boolean; detections?: any[] }) {
  return (
    <div className="authentication-panel">
      <div className="auth-embedded-block">
        <StepLabel number="3" title="Détection des éléments visuels" />
        <YoloObjectsCard ready={ready} detections={detections} />
      </div>
      <div className="auth-panel-head">
        <ShieldCheck size={18} />
        <span>La présence d’un logo, d’un cachet ou d’une signature ne garantit pas l’authenticité de la facture.</span>
      </div>
    </div>
  );
}

function InvoiceLinesPanel({ status, agentResult, currency }: { status: "idle" | "uploading" | "success" | "error"; agentResult: any; currency: string }) {
  if (status === "uploading") {
    return (
      <div className="invoice-lines-loading">
        <Loader2 size={20} />
        <strong>Lecture des lignes en cours</strong>
        <span>OCR du tableau, séparation des colonnes, quantités, prix unitaires et totaux.</span>
        <div className="line-skeleton"><i /><i /><i /></div>
      </div>
    );
  }

  if (status !== "success") {
    return (
      <div className="invoice-lines-empty">
        <Table2 size={20} />
        <strong>Lignes en attente d'extraction</strong>
        <span>Les lignes de facture apparaîtront ici après OCR, analyse du tableau et normalisation.</span>
      </div>
    );
  }

  const rows = extractInvoiceLines(agentResult);
  if (rows.length === 0) {
    return (
      <div className="invoice-lines-empty warning">
        <AlertTriangle size={20} />
        <strong>Aucune ligne structurée détectée</strong>
        <span>Le document est analysé, mais le backend n'a pas retourné de tableau exploitable. La saisie humaine pourra compléter les lignes.</span>
      </div>
    );
  }

  return (
    <div className="invoice-lines-panel">
      <div className="invoice-lines-head">
        <div>
          <strong>Lignes détectées</strong>
          <small>{rows.length} ligne{rows.length > 1 ? "s" : ""} extraite{rows.length > 1 ? "s" : ""} depuis le document</small>
        </div>
        <span>{rows.some(row => row.requiresReview) ? "Lecture visuelle à vérifier" : "Vérification des lignes"}</span>
      </div>
      <table>
        <thead><tr><th>Désignation</th><th>Qté</th><th>Prix unitaire</th><th>Base</th><th>Total ligne</th><th>Contrôle</th></tr></thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.description}-${index}`}>
              <td><strong>{row.description || "Ligne sans désignation"}</strong></td>
              <td>{row.quantity}</td>
              <td>{row.unitPrice ? `${row.unitPrice} ${currency}` : "-"}</td>
              <td>{row.priceBasis === "unknown" ? "Non précisée" : row.priceBasis}</td>
              <td>{row.total ? `${row.total} ${currency}` : "-"}</td>
              <td><span className={row.requiresReview ? "line-confidence warn" : "line-confidence good"}>{row.requiresReview ? "À vérifier" : "Cohérente"}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="invoice-lines-actions">
        <span><Check size={14} />Prêt pour validation humaine</span>
        <button className="btn btn-outline">Ajouter une ligne</button>
      </div>
    </div>
  );
}

function ImportClassification({ result, fileName }: { result: any; fileName: string }) {
  const summary = getClassificationSummary(result);
  const method = String(result?.classification?.details?.analysis_method || result?.classification?.analysis_method || "");
  const engine = method === "paddleocr_layout_analysis" ? "PaddleOCR · analyse de structure"
    : method === "pdf_text_classification" ? "Classification du texte PDF"
    : method === "text_ocr_classification_fallback" ? "Classification OCR · règles de secours" : "Classification documentaire";
  const rawConfidence = result?.classification?.confidence ?? result?.document?.classification_confidence;
  const confidence = rawConfidence != null && Number.isFinite(Number(rawConfidence)) ? Math.min(100, Math.max(0, percentValue(rawConfidence))) : null;
  const invoice = isInvoiceResult(result);
  const extraction = result?.extraction;
  const extractionLabel = extraction?.error ? `Extraction de secours : ${extraction.error}`
    : extraction?.model_used?.startsWith("ollama:") ? "Analyse locale · champs à vérifier" : "Informations à vérifier";
  return <section id="import-analysis-result" tabIndex={-1} aria-labelledby="import-result-title" className={`import-classification result-banner ${summary.tone}`} aria-live="polite">
    <div className="result-status"><span><Check size={14} />Analyse terminée</span><small>{engine}</small></div>
    <span className="import-classification-icon">{invoice ? <FileCheck2 size={32} /> : <FileText size={32} />}</span>
    <div className="result-main"><small>RÉSULTAT DU DOCUMENT</small><h2 id="import-result-title">{invoice ? "Facture détectée" : "Document non-facture"}</h2><p className="result-file">{fileName}</p>
      <span className={`result-type ${invoice ? "is-invoice" : "is-other"}`}>{invoice ? summary.label : "Aucune facture créée"}</span>
    </div>
    <div className="import-classification-score" title="Indice de correspondance avec une facture. Ce score ne mesure pas l’exactitude des champs extraits.">
      <div className="result-score-ring" style={{ "--score": `${confidence || 0}%` } as React.CSSProperties}><strong>{confidence == null ? "—" : `${confidence}%`}</strong></div>
      <small>Score facture</small>
    </div>
    <div className="result-footer"><span>{invoice ? extractionLabel : "Document consultable ci-dessous"}</span><button type="button" onClick={() => document.getElementById("invoice-source-preview")?.scrollIntoView({ behavior: "smooth", block: "start" })}><ChevronDown size={16} />Afficher le document</button></div>
  </section>;
}

function UploadLoadingPanel({ fileName }: { fileName: string }) {
  return (
    <div className="import-loading" role="status" aria-live="polite">
      <div className="import-scan-sheet" aria-hidden="true"><FileText size={48} /><span /><i /></div>
      <h2>Analyse en cours</h2>
      <p>{fileName}</p>
      <div className="import-loading-status"><Loader2 size={16} />Lecture et vérification du document</div>
      <div className="import-indeterminate" aria-hidden="true"><span /></div>
    </div>
  );
}

function ReviewDocuments() {
  return (
    <>
      <PageHeader title="Documents à revoir" subtitle="La classification est affichée avant toute extraction." />
      <div className="cards-grid">
        {invoices.filter((i) => i.classification !== "Facture").map((doc) => (
          <Card key={doc.id} className="review-card">
            <div className="card-head"><FileText size={20} /><ClassificationBadge value={doc.classification} confidence={doc.classificationConfidence} /></div>
            <h3>{doc.file}</h3>
            <p>{doc.classification === "Incertain" ? "Revue humaine nécessaire" : "Document non-facture probable"}</p>
            <ul>{doc.reason?.map((r) => <li key={r}>{r}</li>)}</ul>
            <div className="button-row">
              <button className="btn btn-outline">Revoir manuellement</button>
              <button className="btn btn-soft">Reclasser comme facture</button>
              <button className="btn btn-danger">Rejeter</button>
            </div>
          </Card>
        ))}
      </div>
    </>
  );
}

function SuppliersPage({ navigate }: { navigate: (path: string) => void }) {
  const [suppliers, setSuppliers] = useState<Supplier[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  React.useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    fetch(`${API_BASE_URL}/database/suppliers`)
      .then((response) => {
        if (!response.ok) throw new Error("Backend indisponible");
        return response.json();
      })
      .then((data) => {
        if (!cancelled) setSuppliers(Array.isArray(data?.suppliers) ? data.suppliers : []);
      })
      .catch((requestError) => {
        if (!cancelled) setError(requestError?.message || "Impossible de charger les fournisseurs.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredSuppliers = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) return suppliers;
    return suppliers.filter((supplier) => [
      supplier.name,
      supplier.tax_id || "",
      supplier.address || "",
      supplier.phone || "",
    ].join(" ").toLowerCase().includes(normalizedQuery));
  }, [query, suppliers]);

  return (
    <>
      <PageHeader
        title="Fournisseurs"
        subtitle="Répertoire fournisseurs"
        action={<button className="btn btn-primary" onClick={() => navigate("/import")}><UploadCloud size={17} />Nouvelle facture</button>}
      />
      <section className="suppliers-card supplier-directory">
        <div className="supplier-toolbar">
          <label className="invoice-search">
            <Search size={17} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Rechercher par fournisseur, matricule fiscal, adresse ou téléphone..."
            />
          </label>
          <strong className="invoice-count">{filteredSuppliers.length} fournisseurs</strong>
        </div>
        {loading && <EmptyState text="Chargement des fournisseurs depuis SQLite..." />}
        {!loading && error && <div className="backend-alert"><AlertTriangle size={18} /><span>{error}</span></div>}
        {!loading && !error && filteredSuppliers.length === 0 && <EmptyState text="Aucun fournisseur trouvé." />}
        {!loading && !error && filteredSuppliers.length > 0 && (
          <div className="suppliers-table">
            <div className="suppliers-head">
              <span>Fournisseur</span>
              <span>Matricule fiscal</span>
              <span>Téléphone</span>
              <span>Adresse</span>
              <span>Factures</span>
              <span>Action</span>
            </div>
            {filteredSuppliers.map((supplier) => (
              <div className="supplier-row" key={supplier.id}>
                <span className="supplier-main">
                  <span className="supplier-avatar" aria-hidden="true"><Users size={20} /></span>
                  <span>
                    <strong>{supplier.name || "Fournisseur inconnu"}</strong>
                  </span>
                </span>
                <span data-label="Matricule fiscal">{supplier.tax_id || "Non renseigné"}</span>
                <span data-label="Téléphone">{supplier.phone || "Non renseigné"}</span>
                <span data-label="Adresse">{supplier.address || "Non renseignée"}</span>
                <span data-label="Factures"><b>{supplier.invoices_count || 0}</b></span>
                <a className="row-action supplier-action" href={`/factures?source=processed&supplier=${encodeURIComponent(supplier.name)}`}>
                  Voir factures
                </a>
              </div>
            ))}
          </div>
        )}
      </section>
    </>
  );
}

function InvoicesPage({ navigate, pending = false }: { navigate: (path: string) => void; pending?: boolean }) {
  const [backendRows, setBackendRows] = useState<Invoice[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [filtersOpen, setFiltersOpen] = useState(false);
  const initialParams = new URLSearchParams(window.location.search);
  const initialSupplierFilter = initialParams.get("supplier") || "";
  const initialSource = initialParams.get("source") === "dataset" ? "dataset" : "processed";
  const [statusFilter, setStatusFilter] = useState(initialParams.get("status") === "validated" ? "validated" : "");
  const [filtersDraft, setFiltersDraft] = useState({ taxId: "", invoiceName: "", company: "", supplier: initialSupplierFilter });
  const [filters, setFilters] = useState({ taxId: "", invoiceName: "", company: "", supplier: initialSupplierFilter });
  const [source, setSource] = useState<"dataset" | "processed">(pending || initialSupplierFilter ? "processed" : initialSource);
  const [page, setPage] = useState(1);
  const [totalRows, setTotalRows] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const pageSize = 10;
  const activeFilterCount = Object.values(filters).filter((value) => value.trim()).length;

  React.useEffect(() => {
    setPage(1);
  }, [filters, pending, source, statusFilter]);

  React.useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    const params = new URLSearchParams({
      page: String(page),
      page_size: String(pageSize),
    });
    if (filters.invoiceName.trim()) params.set("invoice_name", filters.invoiceName.trim());
    if (filters.supplier.trim()) params.set("supplier", filters.supplier.trim());
    if (filters.company.trim()) params.set("company", filters.company.trim());
    if (filters.taxId.trim()) params.set("tax_id", filters.taxId.trim());
    if (pending) params.set("status", "draft");
    else if (source === "processed" && statusFilter) params.set("status", statusFilter);

    fetch(`${API_BASE_URL}/invoices/catalog?source=${source}&${params.toString()}`)
      .then((response) => {
        if (!response.ok) throw new Error("Backend indisponible");
        return response.json();
      })
      .then((data) => {
        if (cancelled) return;
        const mappedRows = Array.isArray(data?.invoices) ? data.invoices.map(mapBackendInvoice) : [];
        setBackendRows(mappedRows);
        setTotalRows(Number(data?.pagination?.total || mappedRows.length));
        setTotalPages(Math.max(1, Number(data?.pagination?.total_pages || 1)));
      })
      .catch((requestError) => {
        if (cancelled) return;
        setError(requestError?.message || "Impossible de charger les factures depuis SQLite.");
        setBackendRows([]);
        setTotalRows(0);
        setTotalPages(1);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [filters, pending, page, source, statusFilter]);

  const rows = backendRows;
  const paginationStart = Math.min(Math.max(1, page - 2), Math.max(1, totalPages - 4));
  const visiblePages = Array.from(
    { length: Math.min(5, totalPages) },
    (_, index) => paginationStart + index,
  ).filter((value, index, values) => value <= totalPages && values.indexOf(value) === index);
  const applyFilters = () => {
    setFilters(filtersDraft);
    setFiltersOpen(false);
  };
  const clearFilters = () => {
    const emptyFilters = { taxId: "", invoiceName: "", company: "", supplier: "" };
    setFiltersDraft(emptyFilters);
    setFilters(emptyFilters);
  };
  return (
    <>
      <PageHeader
        title={pending ? "Factures à valider" : statusFilter && source === "processed" ? "Factures validées" : "Factures"}
        subtitle="Toutes les factures de la base, paginées et filtrées par champs métier."
      />
      <Card>
        <div className="invoice-source-tabs">
          {!pending && (
            <button type="button" className={source === "dataset" ? "active" : ""} onClick={() => setSource("dataset")}>
              Dataset FATURA
            </button>
          )}
          <button type="button" className={source === "processed" ? "active" : ""} onClick={() => setSource("processed")}>
            Historique
          </button>
        </div>
        <div className="invoice-toolbar">
          {!pending && source === "processed" && <select aria-label="Statut des factures" value={statusFilter} onChange={event => setStatusFilter(event.target.value)}><option value="">Tous les statuts</option><option value="validated">Validées</option></select>}
          <button className="add-filter-button" type="button" onClick={() => setFiltersOpen(true)}>
            <Plus size={18} /> Ajouter des filtres {activeFilterCount ? <span>{activeFilterCount}</span> : null}
          </button>
          <strong className="invoice-count">{totalRows} factures</strong>
        </div>
        {activeFilterCount > 0 && (
          <div className="active-filter-list">
            {filters.taxId && <span>Matricule fiscal: {filters.taxId}</span>}
            {filters.invoiceName && <span>Facture: {filters.invoiceName}</span>}
            {filters.company && <span>Société: {filters.company}</span>}
            {filters.supplier && <span>Fournisseur: {filters.supplier}</span>}
            <button type="button" onClick={clearFilters}>Effacer</button>
          </div>
        )}
        {loading && <EmptyState text="Chargement des factures depuis la base SQLite..." />}
        {!loading && error && (
          <div className="backend-alert">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        )}
        {!loading && !error && rows.length === 0 && <EmptyState text="Aucune facture trouvée dans la base pour le moment." />}
        {!loading && !error && rows.length > 0 && (
          <>
            <InvoiceTable rows={rows} navigate={navigate} />
            <div className="pagination-bar">
              <button type="button" disabled={page <= 1} onClick={() => setPage((current) => Math.max(1, current - 1))}>
                <ChevronLeft size={15} /> Précédente
              </button>
              <div>
                {visiblePages.map((pageNumber) => (
                  <button
                    type="button"
                    key={pageNumber}
                    className={pageNumber === page ? "active" : ""}
                    onClick={() => setPage(pageNumber)}
                  >
                    {pageNumber}
                  </button>
                ))}
              </div>
              <button type="button" disabled={page >= totalPages} onClick={() => setPage((current) => Math.min(totalPages, current + 1))}>
                Suivante <ChevronRight size={15} />
              </button>
              <span>Page {page} / {totalPages}</span>
            </div>
          </>
        )}
      </Card>
      <aside className={`filter-drawer ${filtersOpen ? "open" : ""}`} aria-hidden={!filtersOpen}>
        <div className="filter-drawer-head">
          <div><PanelRightOpen size={20} /><strong>Filtres</strong></div>
          <button type="button" onClick={() => setFiltersOpen(false)}><X size={20} /></button>
        </div>
        <div className="filter-fields">
          <h3>Recherche facture</h3>
          <label>
            <span>Numéro fiscal</span>
            <input value={filtersDraft.taxId} onChange={(event) => setFiltersDraft({ ...filtersDraft, taxId: event.target.value })} placeholder="Ex: 1234567 A/M/000" />
          </label>
          <label>
            <span>Nom de la facture</span>
            <input value={filtersDraft.invoiceName} onChange={(event) => setFiltersDraft({ ...filtersDraft, invoiceName: event.target.value })} placeholder="Ex: FC-2024-0456 ou facture_00040" />
          </label>
          <label>
            <span>Nom de la société</span>
            <input value={filtersDraft.company} onChange={(event) => setFiltersDraft({ ...filtersDraft, company: event.target.value })} placeholder="Ex: YASMEEN ENGINEERING SYSTEMS" />
          </label>
          <label>
            <span>Nom du fournisseur</span>
            <input value={filtersDraft.supplier} onChange={(event) => setFiltersDraft({ ...filtersDraft, supplier: event.target.value })} placeholder="Ex: STEG, SOTUFAB, NOVATION CITY" />
          </label>
        </div>
        <div className="filter-actions">
          <button type="button" className="clear" onClick={clearFilters}><X size={17} /> Effacer</button>
          <button type="button" className="apply" onClick={applyFilters}><Check size={17} /> Appliquer les filtres</button>
        </div>
      </aside>
      {filtersOpen && <button className="drawer-scrim" type="button" aria-label="Fermer les filtres" onClick={() => setFiltersOpen(false)} />}
    </>
  );
}

function AnalysisPage({ route, navigate }: { route: string; navigate: (path: string) => void }) {
  const invoiceId = invoiceIdFromRoute(route);
  const [detailInvoice, setDetailInvoice] = useState<Invoice | null>(null);
  const [detailRaw, setDetailRaw] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeBox, setActiveBox] = useState("signature");
  const [tab, setTab] = useState("Informations");
  const source = invoiceCatalogSource(invoiceId);
  const currentInvoice = detailInvoice || invoices[0];
  const detailFields = useMemo(() => fieldsFromInvoice(currentInvoice, detailRaw), [currentInvoice, detailRaw]);
  const detailLines = useMemo(() => extractInvoiceLines({ invoice: detailRaw }), [detailRaw]);
  const liveDecisions = useMemo<AgentDecision[]>(() => getAgentDecisions({ invoice: detailRaw })
    .map((decision: any) => ({
      step: String(decision.step || "agent"),
      decision: String(decision.decision || decision.status || "executed"),
      reason: String(decision.reason || decision.action || "Étape exécutée par le pipeline."),
      status: decision.status === "needs_attention" ? "needs_attention" : "completed",
      confidence: typeof decision.confidence === "number" ? decision.confidence : undefined,
      action: decision.action,
    })), [detailRaw]);
  const liveIssues = Array.isArray(detailRaw?.raw_data?.validation_issues) ? detailRaw.raw_data.validation_issues : [];
  const liveRecommendation = useMemo(() => ({
    recommendedAction: currentInvoice.status === "Validée" ? "archived_after_validation" : "send_to_human_validation",
    humanReviewRequired: currentInvoice.status !== "Validée" && liveIssues.some((issue: any) => issue?.severity === "blocking"),
    reasons: liveIssues.length ? liveIssues.map((issue: any) => String(issue.code || issue.message || "validation_issue")) : [currentInvoice.status === "Validée" ? "validation_humaine_enregistree" : "champs_prets_a_verifier"],
    workflowDestination: currentInvoice.status === "Validée" ? "archive_locale_validee" : "validation_humaine",
  }), [currentInvoice.status, liveIssues]);
  const [selectedField, setSelectedField] = useState<Field>(detailFields[4]);
  const isDataset = currentInvoice.status === "Dataset";
  const isNonInvoice = currentInvoice.classification === "Non-facture";
  const isUncertain = currentInvoice.classification === "Incertain";

  React.useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    fetch(`${API_BASE_URL}/invoices/catalog/detail?source=${source}&invoice_id=${encodeURIComponent(invoiceId)}`)
      .then(async (response) => {
        if (!response.ok) throw new Error(await response.text());
        return response.json();
      })
      .then((data) => {
        if (cancelled) return;
        setDetailRaw(data?.invoice || null);
        setDetailInvoice(mapBackendInvoice(data?.invoice || {}));
      })
      .catch((requestError) => {
        if (!cancelled) setError(requestError?.message || "Impossible de charger cette facture.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [invoiceId, source]);

  React.useEffect(() => {
    setSelectedField(detailFields[4] || detailFields[0]);
  }, [detailFields]);

  return (
    <div className="analysis-screen">
      <div className="analysis-head">
        <button className="back-link" onClick={() => navigate("/factures")}><ChevronLeft size={18} />Retour à la liste</button>
        <div className="analysis-title-row">
          <div>
            <h1>{currentInvoice.file}</h1>
            <p>{isDataset ? "Dataset FATURA indexé depuis SQLite · Non traité par le pipeline utilisateur" : `Importé dans la base locale · ${currentInvoice.supplier}`}</p>
          </div>
          <span className={`done-pill ${isDataset ? "dataset" : isNonInvoice ? "rejected" : ""}`}>
            {isNonInvoice ? <X size={16} /> : <Check size={16} />}
            {isDataset ? "Dataset indexé" : isNonInvoice ? "Non-facture" : "Analyse terminée"}
          </span>
          <div className="analysis-actions">
            <a className="btn btn-outline" href={invoiceFileUrl(invoiceId)} download><Download size={17} />Télécharger le fichier</a>
            <button className="btn btn-outline">Actions <ChevronRight size={16} /></button>
          </div>
        </div>
      </div>
      {error && <div className="backend-alert"><AlertTriangle size={18} /><span>{error}</span></div>}
      {loading && <EmptyState text="Chargement de la facture depuis la base..." />}

      {!loading && !error && <div className="analysis-board">
        <Card className="analysis-doc-card">
          <ChartTitle title="Aperçu du document" subtitle="" />
          <DocumentToolbar downloadUrl={invoiceFileUrl(invoiceId)} />
          <DocumentFilePreview invoice={currentInvoice} invoiceId={invoiceId} />
          <Legend />
        </Card>

        <Card className="analysis-data-card">
          <div className="analysis-card-head">
            <h3>{isDataset ? "Données du dataset" : "Données extraites"}</h3>
            <span>{isDataset ? "Source" : "Confiance classification"} <b>{isDataset ? "Index SQLite" : `${currentInvoice.classificationConfidence || 0}%`}</b></span>
          </div>
          <AnalysisFields fields={detailFields} selected={selectedField.key} onSelect={(f) => { setSelectedField(f); setActiveBox(f.boxId); }} />
          <Tabs tabs={["Informations", "Lignes facture", "Vérifications", "JSON"]} active={tab} setActive={setTab} />
          {tab === "Informations" && <RagPanel proof={selectedField.proof} onLocate={() => setActiveBox(selectedField.boxId)} />}
          {tab === "Lignes facture" && <LinesTable lines={detailLines} />}
          {tab === "Vérifications" && <Checks />}
          {tab === "JSON" && <JsonBlock data={{ invoice: detailRaw || currentInvoice, fields: detailFields, detectedObjects: isDataset ? [] : detectedBoxes }} />}
        </Card>

        <aside className="analysis-intel-stack">
          <Card className="intel-card">
            <ChartTitle title="Intelligence" />
            <IntelItem
              icon={FileCheck2}
              tone={isNonInvoice ? "danger" : isDataset || isUncertain ? "warning" : "success"}
              title="Classification du document"
              value={isDataset ? "Dataset indexé" : currentInvoice.classification}
              detail={isDataset ? "Exemple issu de FATURA.zip, pas une facture uploadée par l'utilisatrice" : isNonInvoice ? "Le document ne contient pas assez de signaux facture exploitables" : isUncertain ? "Revue humaine recommandée avant extraction" : "Type facture détecté par le pipeline"}
              score={isDataset ? "" : `${currentInvoice.classificationConfidence || 0}%`}
            />
            <IntelItem icon={FolderSearch} tone="blue" title="OCR" value={isDataset ? "Annotation disponible" : "Texte extrait"} detail={isDataset ? "Les champs viennent des annotations indexées dans SQLite" : "OCR/PDF text utilisé pour la classification et l'extraction"} score="" />
            <IntelItem icon={Sparkles} tone={isNonInvoice ? "warning" : "blue"} title="RAG d'extraction" value={isDataset || isNonInvoice ? "Non exécuté" : "Passages utiles retrouvés"} detail={isDataset ? "Lancer une vraie analyse pour créer les chunks RAG de cette facture" : isNonInvoice ? "Pas d'extraction facture après rejet de classification" : "Chunks temporaires utilisés pour justifier les champs extraits"} score="" />
            <div className="visual-detected">
              <div className="visual-title"><Sparkles size={17} />Éléments visuels détectés</div>
              {[
                ["Logo", "logo"],
                ["Cachet", "stamp"],
                ["Signature", "signature"],
                ["Tableau des lignes", "table"],
                ["Totaux", "ttc"],
              ].map(([label, box]) => (
                <button key={box} onClick={() => setActiveBox(box)} className={activeBox === box ? "active" : ""}>
                  <span>{label}</span><b>1 détecté</b>
                </button>
              ))}
            </div>
            <IntelItem icon={Archive} tone="success" title="RAG documentaire" value="Index prêt pour l'assistant" detail="Document découpé et vectorisé après normalisation" score="" />
            <IntelItem icon={Copy} tone="warning" title="Vérification des doublons" value="Aucun duplicata trouvé" detail="Comparé avec 856 factures existantes" score="" />
          </Card>

          <AgentTimeline decisions={liveDecisions.length ? liveDecisions : agentDecisions} recommendation={liveDecisions.length ? liveRecommendation : agentRecommendation} />

          <Card className="audit-card">
            <ChartTitle title="Piste d'audit" subtitle="Voir tout" />
            {["Fichier importé", "Analyse OCR terminée", "Données extraites", "Vérifications terminées"].map((item) => (
              <div className="audit-row" key={item}><span>21/05/2024 14:32</span><strong>{item}</strong><em>{item === "Fichier importé" ? "Mohamed Saidi" : "IA FATURA"}</em></div>
            ))}
          </Card>
        </aside>
      </div>}
    </div>
  );
}

function AnalysisFields({ fields, selected, onSelect }: { fields: Field[]; selected: string; onSelect: (field: Field) => void }) {
  const icons: Record<string, React.ElementType> = {
    supplier: Users,
    taxId: Building2,
    date: Clock,
    number: FileText,
    ht: Gauge,
    tva: Settings,
    ttc: Download,
    currency: Gauge,
  };
  const ordered = fields.filter((fieldItem) => ["supplier", "taxId", "date", "number", "ht", "tva", "ttc", "currency"].includes(fieldItem.key));
  return (
    <div className="analysis-fields">
      {ordered.map((fieldItem) => {
        const Icon = icons[fieldItem.key] ?? FileText;
        return (
          <button key={fieldItem.key} className={selected === fieldItem.key ? "active" : ""} onClick={() => onSelect(fieldItem)}>
            <Icon size={17} />
            <span>{fieldItem.label}</span>
            <strong>{fieldItem.value}</strong>
            <b>{fieldItem.confidence}%</b>
            <em><EditIcon /></em>
          </button>
        );
      })}
    </div>
  );
}

function EditIcon() {
  return <PanelRightOpen size={15} />;
}

function IntelItem({ icon: Icon, tone, title, value, detail, score }: { icon: React.ElementType; tone: string; title: string; value: string; detail: string; score: string }) {
  return (
    <div className={`intel-item ${tone}`}>
      <Icon size={20} />
      <div><span>{title}</span><strong>{value}</strong><small>{detail}</small></div>
      {score && <b>{score}</b>}
    </div>
  );
}

function AgentTimeline({ decisions, recommendation }: { decisions: AgentDecision[]; recommendation: typeof agentRecommendation }) {
  return (
    <Card className="agent-card">
      <ChartTitle title="Décisions de l'agent IA" subtitle="Orchestration automatique" />
      <div className={`agent-recommendation ${recommendation.humanReviewRequired ? "review" : "ready"}`}>
        <div>
          <span>Action recommandée</span>
          <strong>{recommendation.humanReviewRequired ? "Revue humaine requise" : "Prêt à valider"}</strong>
        </div>
        <ShieldCheck size={22} />
      </div>
      <div className="agent-reasons">
        {recommendation.reasons.map((reason) => <span key={reason}>{reason.replaceAll("_", " ")}</span>)}
        <span>{recommendation.workflowDestination.replaceAll("_", " ")}</span>
      </div>
      <div className="agent-timeline">
        {decisions.map((decision, index) => (
          <button key={`${decision.step}-${index}`} className={decision.status}>
            <i>{index + 1}</i>
            <div>
              <span>{decision.step}</span>
              <strong>{decision.decision}</strong>
              <small>{decision.reason}</small>
              {decision.confidence !== undefined && <em>{Math.round(decision.confidence * 100)}% confiance</em>}
            </div>
          </button>
        ))}
      </div>
    </Card>
  );
}

function ValidationPage({ notify, navigate }: { notify: (message: string) => void; navigate: (path: string) => void }) {
  const [values, setValues] = useState(Object.fromEntries(extractedFields.map((f) => [f.key, f.value])));
  const [audit, setAudit] = useState<string[]>([]);
  const [activeBox, setActiveBox] = useState("due");
  const [selectedField, setSelectedField] = useState<Field>(extractedFields[4]);
  const [tab, setTab] = useState("Informations");
  const onChange = (f: Field, value: string) => {
    setValues({ ...values, [f.key]: value });
    if (value !== f.value && !audit.some((a) => a.startsWith(f.label))) {
      setAudit([`${f.label} | IA : ${f.value} | Corrigé : ${value} | Mohamed Saidi | 21/05/2024 14:43 | confiance ${f.confidence}%`, ...audit]);
    }
  };
  return (
    <div className="validation-screen">
      <div className="validation-page-head">
        <div>
          <h1>Validation humaine</h1>
          <p>À valider <ChevronRight size={14} /> Facture FC-2024-0456</p>
        </div>
        <div className="validation-nav">
          <button className="btn btn-outline"><ChevronLeft size={16} />Précédente</button>
          <strong>3 / 18</strong>
          <button className="btn btn-outline">Suivante <ChevronRight size={16} /></button>
        </div>
      </div>

      <div className="human-validation-grid">
        <Card className="validation-document-card">
          <DocumentToolbar />
          <DocumentPreview activeId={activeBox} onBoxClick={setActiveBox} />
          <div className="thumbnail-strip"><div><FileText size={18} /><span>1</span></div></div>
        </Card>

        <Card className="validation-work-card">
          <Tabs tabs={["Informations", "Lignes facture", "Vérifications", "JSON"]} active={tab} setActive={setTab} />
          <div className="validation-work-body">
            <div className="validation-form-area">
              {tab === "Informations" && (
                <>
                  <SectionMini title="Informations générales" />
                  <EditableGrid fields={extractedFields.slice(0, 7)} values={values} onChange={(f, value) => { onChange(f, value); setSelectedField(f); }} onLocate={(id) => { setActiveBox(id); const found = extractedFields.find((f) => f.boxId === id); if (found) setSelectedField(found); }} />
                  <SectionMini title="Montants" />
                  <EditableGrid fields={extractedFields.slice(7)} values={values} onChange={(f, value) => { onChange(f, value); setSelectedField(f); }} onLocate={(id) => { setActiveBox(id); const found = extractedFields.find((f) => f.boxId === id); if (found) setSelectedField(found); }} />
                  <SectionMini title="Vérifications" />
                  <ValidationAlerts onFocus={(box) => setActiveBox(box)} />
                </>
              )}
              {tab === "Lignes facture" && <LinesTable />}
              {tab === "Vérifications" && <Checks />}
              {tab === "JSON" && <JsonBlock data={{ invoice: invoices[0], fields: extractedFields }} />}
              {audit.length > 0 && <div className="compact-audit">{audit.map((a) => <div className="audit-line" key={a}>{a}</div>)}</div>}
            </div>
            <aside className="validation-rag-side">
              <RagPanel proof={selectedField.proof} onLocate={() => setActiveBox(selectedField.boxId)} />
            </aside>
          </div>
          <div className="validation-actions">
            <button className="btn btn-outline">Enregistrer brouillon</button>
            <button className="btn btn-primary" onClick={() => { notify("Facture validée avec succès"); navigate("/factures"); }}><Check size={17} />Valider la facture</button>
          </div>
        </Card>
      </div>
    </div>
  );
}

function ValidationAlerts({ onFocus }: { onFocus: (box: string) => void }) {
  return (
    <div className="validation-alerts">
      <button className="warn" onClick={() => onFocus("number")}>
        <AlertTriangle size={19} />
        <span><strong>Doublon potentiel détecté</strong><small>Une facture similaire existe : FC-2024-0455 du 20/05/2024 (31 849,350 TND)</small></span>
        <em>Voir le détail</em>
      </button>
      <button className="danger" onClick={() => onFocus("ttc")}>
        <AlertTriangle size={19} />
        <span><strong>Montant en lettres à faible confiance</strong><small>Confiance de l'extraction : 72%</small></span>
        <em>Vérifier</em>
      </button>
    </div>
  );
}

function AssistantFull({ navigate }: { navigate: (path: string) => void }) {
  return (
    <>
      <PageHeader title="Assistant RAG FATURA" subtitle="Interrogez vos factures indexées, demandez un résumé, une explication fiscale ou une preuve par champ." />
      <Card className="chat-page">
        <div className="rag-question-head">
          <span><Sparkles size={16} />Questions RAG pertinentes</span>
          <h2>Demandez à l'agent d'expliquer la facture, pas seulement de chercher un mot.</h2>
          <p>Chaque réponse doit s'appuyer sur les champs extraits, les chunks RAG, les contrôles métier et les preuves retrouvées dans le document.</p>
          <div>
            {ragAssistantQuestions.map((item, index) => (
              <button key={item.question} onClick={() => navigate("/factures")}>
                <strong>0{index + 1}</strong>
                <span>{item.question}</span>
              </button>
            ))}
          </div>
        </div>
        {ragAssistantQuestions.map((item) => <ChatBubble key={item.question} question={item.question} answer={item.answer} />)}
        <ChatBubble question="Donne-moi les factures Tunisie Telecom supérieures à 1 000 TND et leur statut." answer="J'ai trouvé TT-2024-0918 pour 1 284,700 TND, statut Validée. L'agent peut aussi filtrer par fournisseur, période, devise TND, statut et seuil de montant à partir des champs normalisés." />
        <div className="sources">
          <h3>Sources et preuves utilisées</h3>
          {invoices.slice(0, 3).map((i) => <button key={i.id} onClick={() => navigate("/factures")}><FileText size={17} /><span>{i.number} • {i.supplier}</span><small>{i.date} • {i.amount}</small></button>)}
          <p>L'assistant doit citer la facture, la page, le champ extrait et l'extrait RAG quand il explique une valeur.</p>
        </div>
      </Card>
    </>
  );
}

function AdminLayout({ route, navigate }: { route: string; navigate: (path: string) => void }) {
  const nav = [["/admin", "Overview", Gauge], ["/admin/users", "Users", Users], ["/admin/companies", "Companies", Building2], ["/admin/roles", "Roles & Permissions", ShieldCheck], ["/admin/all-invoices", "All Invoices", FileText], ["/admin/ai-monitoring", "AI & Processing", Sparkles], ["/admin/consumption", "Consumption", Gauge], ["/admin/logs", "Logs & Errors", AlertTriangle], ["/admin/audit", "Audit Trail", ShieldCheck], ["/admin/settings", "Settings", Settings]] as const;
  return (
    <div className="app-shell admin-shell">
      <aside className="sidebar"><Logo dark /><span className="admin-badge">ADMIN</span><nav className="side-nav">{nav.map(([path, label, Icon]) => <button key={path} className={route === path ? "active" : ""} onClick={() => navigate(path)}><Icon size={18} />{label}</button>)}</nav></aside>
      <div className="workspace"><Topbar openAssistant={() => null} /><main className="content">{route === "/admin/users" ? <AdminUsers /> : route === "/admin/ai-monitoring" ? <AiMonitoring /> : <AdminDashboard />}</main></div>
    </div>
  );
}

function AdminDashboard() {
  return (
    <>
      <PageHeader title="Admin Overview" subtitle="Platform-wide metrics and health" />
      <div className="kpi-grid admin-kpis">
        {["Active users:24", "Companies:8", "Invoices processed:3,847", "Non-invoices:127", "OCR errors:18", "AI failures:6", "Storage used:4.2 GB", "Monthly volume:316"].map((raw) => {
          const [title, value] = raw.split(":");
          return <Kpi key={title} title={title} value={value} icon={Gauge} />;
        })}
      </div>
      <div className="grid-two">
        <Card><ChartTitle title="Monthly Processing Volume" subtitle="Details" /><ResponsiveContainer width="100%" height={300}><AreaChart data={weekly}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="label"/><YAxis/><Tooltip/><Area dataKey="factures" stroke="#2563EB" fill="url(#blue)" strokeWidth={2}/></AreaChart></ResponsiveContainer></Card>
        <Card><ChartTitle title="Top Users by Activity" subtitle="All users" /><MiniList items={["Sarra M. - 58", "Mohamed S. - 42", "Amel T. - 31", "Nour E. - 24"]} /></Card>
      </div>
    </>
  );
}

function AdminUsers() {
  const users = [
    [userAvatars.mohamed, "Mohamed Saidi", "mohamed@fatura.tn", "SOTUFAB", "Comptable", "Actif", "Aujourd'hui"],
    [userAvatars.amel, "Amel Trabelsi", "amel@cabinet.tn", "Cabinet Trabelsi", "Validateur", "Actif", "Hier"],
    [userAvatars.sarra, "Sarra Mansouri", "sarra@demo.tn", "Demo SARL", "Lecteur", "Suspendu", "12/05/2024"],
    [userAvatars.nour, "Nour El Houda", "admin@fatura.tn", "FATURA.tn", "Admin", "Actif", "Aujourd'hui"],
  ];
  return (
    <>
      <PageHeader title="Utilisateurs" subtitle="Gestion mockée des rôles et accès." action={<button className="btn btn-primary"><Plus size={17} />Ajouter</button>} />
      <Card><div className="data-table">{users.map((u) => <div className="data-row" key={u[2]}><span className="user-cell"><img src={u[0]} alt="" />{u[1]}</span>{u.slice(2).map((c) => <span key={c}>{c}</span>)}<div className="button-row"><button className="btn btn-soft">Modifier</button><button className="btn btn-outline">Changer rôle</button></div></div>)}</div></Card>
    </>
  );
}

function AiMonitoring() {
  const steps = ["Classification", "OCR/PDF text", "YOLO visuel", "RAG d'extraction", "Extraction champs", "Normalisation", "Contrôles & doublons", "RAG documentaire", "Validation", "Archivage"];
  return (
    <>
      <PageHeader title="IA & traitements" subtitle="Suivi simulé des étapes du pipeline." />
      <div className="cards-grid">{steps.map((s, i) => <Card key={s} className="monitor-card"><Sparkles size={22} /><strong>{s}</strong><span>Taux succès {98 - i}%</span><span>Confiance moyenne {94 - i}%</span><span>Erreurs {i + 1}</span><span>Temps moyen {(1.1 + i * 0.3).toFixed(1)}s</span></Card>)}</div>
      <Card><ChartTitle title="Traitements récents" /><MiniList items={["FC-2024-0456 - RAG d'extraction terminé", "STEG-1284 - Classification incertaine", "TT-2024-0918 - RAG documentaire indexé"]} /></Card>
    </>
  );
}

function UploadedDocumentPreview({ file, detections = [], status, highlight }: { file: UploadedPreview | null; detections?: any[]; status: "idle" | "uploading" | "success" | "error"; highlight?: any }) {
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  if (!file) {
    return (
      <div className="real-preview-empty">
        <FileText size={24} />
        <strong>Aucun document affiché</strong>
        <span>Sélectionnez une facture PDF, PNG ou JPG pour voir le vrai aperçu ici.</span>
      </div>
    );
  }

  const isPdf = file.mime.includes("pdf") || file.name.toLowerCase().endsWith(".pdf");
  const visualDetections = visibleDetections(detections);

  if (isPdf) {
    return <PdfInvoicePreview url={file.url} name={file.name} highlight={highlight} />;
  }

  return (
    <div className={`real-preview-frame image ${status === "uploading" ? "is-analyzing" : ""}`}>
      <img src={file.url} alt={`Aperçu de ${file.name}`} onLoad={(event) => {
        setImageSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight });
      }} />
      {visualDetections.map((detection, index) => {
        const boxStyle = detectionToStyle(detection, imageSize);
        if (!boxStyle) return null;
        const label = String(detection.class_name || detection.label || "Objet YOLO");
        const rawConfidence = Number(detection.confidence || detection.detection_confidence || 0);
        const confidence = Math.round(rawConfidence <= 1 ? rawConfidence * 100 : rawConfidence);
        return (
          <span className="real-yolo-box" style={boxStyle} key={`${label}-${index}`}>
            {label} {confidence ? `${confidence}%` : ""}
          </span>
        );
      })}
      {status === "uploading" && <span className="real-preview-loading"><Loader2 size={14} />Analyse en cours</span>}
      <SourceHighlight location={highlight} />
    </div>
  );
}

function detectionToStyle(detection: any, imageSize: { width: number; height: number }): React.CSSProperties | null {
  const normalized = detection.normalized_bbox;
  if (normalized && ["x1", "y1", "x2", "y2"].every((key) => Number.isFinite(Number(normalized[key])))) {
    return {
      left: `${Number(normalized.x1) * 100}%`,
      top: `${Number(normalized.y1) * 100}%`,
      width: `${(Number(normalized.x2) - Number(normalized.x1)) * 100}%`,
      height: `${(Number(normalized.y2) - Number(normalized.y1)) * 100}%`,
    };
  }

  const rawBox = detection.bbox;
  const values = Array.isArray(rawBox)
    ? rawBox
    : rawBox && ["x1", "y1", "x2", "y2"].map((key) => rawBox[key]);
  if (!values || values.length < 4 || !imageSize.width || !imageSize.height) return null;
  const [x1, y1, x2, y2] = values.map(Number);
  if (![x1, y1, x2, y2].every(Number.isFinite)) return null;
  return {
    left: `${(x1 / imageSize.width) * 100}%`,
    top: `${(y1 / imageSize.height) * 100}%`,
    width: `${((x2 - x1) / imageSize.width) * 100}%`,
    height: `${((y2 - y1) / imageSize.height) * 100}%`,
  };
}

function DocumentPreview({ activeId, onBoxClick, compact = false, yoloOnly = false }: { activeId?: string; onBoxClick?: (id: string) => void; compact?: boolean; yoloOnly?: boolean }) {
  const visibleBoxes = yoloOnly ? detectedBoxes.filter((box) => ["logo", "stamp", "signature"].includes(box.id)) : detectedBoxes;
  return (
    <div className={`invoice-doc ${compact ? "compact" : ""}`}>
      <div className="doc-pattern" />
      <div className="doc-head"><strong>SOTUFAB</strong><span>FACTURE</span></div>
      <div className="doc-address">Société Tunisienne de Fabrication<br />Matricule fiscal 1234567 A/M/000</div>
      <div className="doc-client">Client<br />Medo Consulting SARL<br />Tunis, Tunisie</div>
      <div className="doc-meta">N° FC-2024-0456<br />Date 21/05/2024<br />Échéance 20/06/2024</div>
      <table className="doc-lines"><tbody><tr><th>Désignation</th><th>Qté</th><th>PU HT</th><th>Total</th></tr><tr><td>Services administratifs</td><td>1</td><td>18 500,000</td><td>18 500,000</td></tr><tr><td>Fournitures bureau</td><td>3</td><td>2 755,000</td><td>8 265,000</td></tr></tbody></table>
      <div className="doc-stamp">CACHET</div>
      <div className="doc-sign">Signature</div>
      <div className="doc-totals">Total HT 26 765,000<br />TVA 19% 5 084,350<br />Timbre 1,000<br /><b>Total TTC 31 850,350 TND</b></div>
      {visibleBoxes.map((box) => <button key={box.id} className={`box yolo-box ${activeId === box.id ? "active" : ""}`} style={{ left: `${box.x}%`, top: `${box.y}%`, width: `${box.width}%`, height: `${box.height}%` }} title={`${box.label} ${box.confidence}%`} onClick={() => onBoxClick?.(box.id)} />)}
    </div>
  );
}

function DocumentFilePreview({ invoice, invoiceId }: { invoice: Invoice; invoiceId: string }) {
  const inlineUrl = invoiceFileUrl(invoiceId, true);
  const type = invoice.fileType.toLowerCase();
  if (["png", "jpg", "jpeg", "webp", "bmp"].includes(type)) {
    return (
      <div className="real-document-preview">
        <img src={inlineUrl} alt={`Aperçu ${invoice.file}`} />
      </div>
    );
  }
  if (type === "pdf") {
    return (
      <div className="real-document-preview pdf">
        <object data={inlineUrl} type="application/pdf" aria-label={`Aperçu ${invoice.file}`}>
          <a href={invoiceFileUrl(invoiceId)} download>Télécharger le PDF</a>
        </object>
      </div>
    );
  }
  return (
    <div className="real-document-preview empty">
      <FileText size={28} />
      <strong>Aperçu non disponible pour ce format</strong>
      <a href={invoiceFileUrl(invoiceId)} download>Télécharger le fichier</a>
    </div>
  );
}

function DocumentToolbar({ downloadUrl }: { downloadUrl?: string }) {
  return (
    <div className="doc-toolbar">
      <button><Minus size={16} /></button>
      <span>100%</span>
      <button><Plus size={16} /></button>
      <button><Maximize2 size={16} /></button>
      <span>1 / 1</span>
      {downloadUrl ? <a href={downloadUrl} download><Download size={16} /></a> : <button><Download size={16} /></button>}
    </div>
  );
}

function ExtractedFields({ fields, selected, onSelect }: { fields: Field[]; selected: string; onSelect: (field: Field) => void }) {
  return <div className="field-list">{fields.map((f) => <button key={f.key} className={selected === f.key ? "active" : ""} onClick={() => onSelect(f)}><span>{f.label}</span><strong>{f.value}</strong><ConfidenceBadge value={f.confidence} /></button>)}</div>;
}

function EditableGrid({ fields, values, onChange, onLocate }: { fields: Field[]; values: Record<string, string>; onChange: (f: Field, value: string) => void; onLocate: (id: string) => void }) {
  return <div className="editable-grid">{fields.map((f) => <label key={f.key}><span>{f.label}<button type="button" onClick={() => onLocate(f.boxId)}><Eye size={14} />Preuve</button></span><input value={values[f.key]} onChange={(e) => onChange(f, e.target.value)} /><ConfidenceBadge value={f.confidence} /></label>)}</div>;
}

function RagPanel({ proof, onLocate }: { proof: RagProof; onLocate: () => void }) {
  const hasProof = proof.confidence > 0 && proof.sourceText && proof.chunkId !== "preuve-indisponible";
  return (
    <div className="rag-panel">
      <h3>Preuve RAG</h3>
      <dl><dt>Champ</dt><dd>{proof.field}</dd><dt>Valeur extraite</dt><dd>{proof.extractedValue}</dd><dt>Confiance</dt><dd><ConfidenceBadge value={proof.confidence} /></dd><dt>Extrait</dt><dd className="source-text">{proof.sourceText}</dd><dt>Page</dt><dd>{proof.page}</dd></dl>
      <button className="btn btn-primary full" onClick={onLocate} disabled={!hasProof}>Voir dans le document</button>
      <small>{hasProof ? `${proof.chunkId} • similarité ${proof.similarityScore.toFixed(2)}` : "Preuve indisponible pour cette facture"}</small>
    </div>
  );
}

function getAgentFieldEvidence(result: any) {
  return result?.invoice?.raw_data?.field_evidence
    || result?.invoice?.metadata?.field_evidence
    || result?.document?.metadata?.field_evidence
    || result?.rag?.field_evidence
    || result?.rag_evidence
    || result?.field_evidence
    || {};
}

function getAgentDecisions(result: any): any[] {
  const candidates = [
    result?.agent?.decisions,
    result?.invoice?.raw_data?.agent_decisions,
    result?.invoice?.metadata?.agent_decisions,
    result?.document?.metadata?.agent_decisions,
    result?.rag?.decisions,
  ];
  return candidates.flatMap((items) => (Array.isArray(items) ? items : []));
}

function getEvidenceText(chunk: any) {
  return String(chunk?.text || chunk?.source_text || chunk?.chunk || chunk?.content || chunk?.excerpt || "").trim();
}

function getEvidenceConfidence(chunk: any) {
  const value = chunk?.confidence ?? chunk?.similarity_score ?? chunk?.similarity ?? chunk?.score;
  if (typeof value !== "number") return null;
  return value <= 1 ? Math.round(value * 100) : Math.round(value);
}

function getEvidenceLocation(chunk: any) {
  const parts = [
    chunk?.page ? `page ${chunk.page}` : "",
    chunk?.chunk_id || chunk?.id || chunk?.source || "",
  ].filter(Boolean);
  return parts.join(" · ");
}

function ValidationTracePanel({ trace, agentResult }: { trace: any[]; agentResult: any }) {
  const ragDocument = agentResult?.rag_document || agentResult?.document?.metadata?.rag_document || null;
  const fieldEvidence = getAgentFieldEvidence(agentResult);
  const usefulEvidence = fieldEvidence
    ? Object.entries(fieldEvidence)
        .filter(([, chunks]) => Array.isArray(chunks) && chunks.length > 0)
        .slice(0, 12)
    : [];
  return (
    <div className="validation-trace-panel">
      <div>
        <strong>Preuves RAG après validation</strong>
        <span>{ragDocument?.document_id ? `Index RAG #${ragDocument.document_id}` : "Preuves liées aux champs validés"}</span>
      </div>
      <ol>
        {trace.slice(-8).map((item, index) => (
          <li key={`${item.step || "step"}-${index}`}>
            <b>{String(item.step || "trace").replaceAll("_", " ")}</b>
            <span>{item.decision || item.status || "exécuté"}</span>
            <small>{item.reason || item.action || "Étape enregistrée par l'agent."}</small>
          </li>
        ))}
      </ol>
      {usefulEvidence.length === 0 ? (
        <div className="validation-rag-empty">
          Le backend a bien validé la facture, mais aucune preuve RAG détaillée n'a été renvoyée pour les champs.
        </div>
      ) : (
        <div className="validation-rag-evidence">
          {usefulEvidence.map(([field, chunks]) => {
            const firstChunk = Array.isArray(chunks) ? chunks[0] : null;
            const text = getEvidenceText(firstChunk);
            const confidence = getEvidenceConfidence(firstChunk);
            const location = getEvidenceLocation(firstChunk);
            return (
              <p key={field}>
                <b>{String(field).replaceAll("_", " ")}</b>
                <span>{text || "Preuve disponible dans le RAG, extrait texte non fourni."}</span>
                {(location || confidence !== null) && (
                  <small>
                    {location || "chunk RAG"}
                    {confidence !== null ? ` · confiance ${confidence}%` : ""}
                  </small>
                )}
              </p>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Checks() {
  return <div className="checks"><CheckLine good text="HT + TVA + Timbre fiscal = TTC" detail="Concordant" /><CheckLine good text="TVA 19%" detail="Concordant" /><CheckLine good text="Format date" detail="Valide" /><CheckLine warn text="Doublon" detail="Doublon potentiel" /><CheckLine warn text="Montant en lettres" detail="72% confiance" /></div>;
}

function CheckLine({ text, detail, good, warn }: { text: string; detail: string; good?: boolean; warn?: boolean }) {
  return <div className="check-row"><span className={good ? "dot success" : warn ? "dot warning" : "dot"} /> <strong>{text}</strong><em>{detail}</em></div>;
}

function LinesTable({ lines }: { lines?: InvoiceLine[] }) {
  const fallbackLines: InvoiceLine[] = [
    { description: "Services administratifs", quantity: "1", unitPrice: "18 500,000", taxRate: "", total: "18 500,000", confidence: 92 },
    { description: "Fournitures bureau", quantity: "3", unitPrice: "2 755,000", taxRate: "", total: "8 265,000", confidence: 90 },
  ];
  const rows = lines === undefined ? fallbackLines : lines;
  if (lines && lines.length === 0) {
    return <EmptyState text="Aucune ligne de facture extraite pour ce document." />;
  }
  return (
    <div className="lines-table">
      <div><b>Désignation</b><b>Qté</b><b>PU HT</b><b>Total</b></div>
      {rows.map((line, index) => (
        <div key={`${line.description}-${index}`}>
          <span>{line.description || "Non détecté"}</span>
          <span>{line.quantity || "-"}</span>
          <span>{line.unitPrice || "-"}</span>
          <span>{line.total || "-"}</span>
        </div>
      ))}
    </div>
  );
}

function JsonBlock({ data }: { data: unknown }) {
  return <div className="json-block"><button className="btn btn-soft"><Copy size={16} />Copier JSON</button><pre>{JSON.stringify(data, null, 2)}</pre></div>;
}

function AssistantButton({ onClick }: { onClick: () => void }) {
  const [position, setPosition] = useState(() => {
    const defaultX = Math.max(18, window.innerWidth - 230);
    const defaultY = Math.max(18, window.innerHeight - 154);
    try {
      const saved = JSON.parse(localStorage.getItem("fatura-assistant-position") || "null");
      if (saved && Number.isFinite(saved.x) && Number.isFinite(saved.y)) {
        const x = Math.min(Math.max(12, saved.x), Math.max(12, window.innerWidth - 216));
        const y = Math.min(Math.max(12, saved.y), Math.max(12, window.innerHeight - 68));
        const overlapsFormActions = x > window.innerWidth - 440 && y > window.innerHeight - 130;
        return overlapsFormActions ? { x, y: defaultY } : { x, y };
      }
    } catch {
      // Ignore broken browser state and use the default position.
    }
    return { x: defaultX, y: defaultY };
  });
  const [dragging, setDragging] = useState(false);
  const positionRef = React.useRef(position);
  const dragRef = React.useRef({ pointerId: 0, offsetX: 0, offsetY: 0, startX: 0, startY: 0, moved: false });

  React.useEffect(() => {
    positionRef.current = position;
  }, [position]);

  const clampPosition = (x: number, y: number) => ({
    x: Math.min(Math.max(12, x), Math.max(12, window.innerWidth - 216)),
    y: Math.min(Math.max(12, y), Math.max(12, window.innerHeight - 68)),
  });

  const moveTo = (clientX: number, clientY: number) => {
    const deltaX = Math.abs(clientX - dragRef.current.startX);
    const deltaY = Math.abs(clientY - dragRef.current.startY);
    if (deltaX > 4 || deltaY > 4) dragRef.current.moved = true;
    const next = clampPosition(clientX - dragRef.current.offsetX, clientY - dragRef.current.offsetY);
    positionRef.current = next;
    setPosition(next);
  };

  return (
    <button
      className={`assistant-float ${dragging ? "dragging" : ""}`}
      style={{ left: position.x, top: position.y }}
      onPointerDown={(event) => {
        dragRef.current = {
          pointerId: event.pointerId,
          offsetX: event.clientX - position.x,
          offsetY: event.clientY - position.y,
          startX: event.clientX,
          startY: event.clientY,
          moved: false,
        };
        setDragging(true);
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        if (!dragging || dragRef.current.pointerId !== event.pointerId) return;
        moveTo(event.clientX, event.clientY);
      }}
      onPointerUp={(event) => {
        if (dragRef.current.pointerId === event.pointerId) {
          localStorage.setItem("fatura-assistant-position", JSON.stringify(positionRef.current));
          setDragging(false);
          if (!dragRef.current.moved) onClick();
        }
      }}
      onPointerCancel={() => setDragging(false)}
      onClick={(event) => { if (event.detail === 0) onClick(); }}
      title="Déplacer ou ouvrir Fatura Assistant"
      type="button"
    >
      <Sparkles size={20} />Fatura Assistant
    </button>
  );
}

function AssistantDrawer({ open, onClose, route, navigate }: { open: boolean; onClose: () => void; route: string; navigate: (path: string) => void }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<any[]>([]);
  const [generation, setGeneration] = useState<any>(null);
  const [retrieval, setRetrieval] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const quickQuestions = [
        "Quelles factures sont en retard et quel est le montant total ?",
        "Quels fournisseurs ont le plus grand volume en TND ?",
        "Explique les champs à vérifier avant validation humaine.",
      ];
  if (!open) return null;
  return (
    <aside className={`drawer ${open ? "open" : ""}`}>
      <div className="drawer-head"><div><strong>Fatura Assistant</strong><span>Recherche dans les factures indexées</span></div><button onClick={onClose} type="button" aria-label="Fermer"><X size={18} /></button></div>
      <div className="suggestions">{quickQuestions.map((s) => <button key={s} type="button" onClick={() => setQuestion(s)}>{s}</button>)}</div>
      <div className="assistant-chat-body">
        {!answer && !sources.length && !error && <div className="assistant-empty"><b>Fatura Assistant</b><span>Pose une question sur une facture indexée. La réponse sera limitée aux preuves retrouvées.</span></div>}
        {answer && <div className="assistant-answer"><b>Fatura Assistant</b><p dir="auto" lang={/[\u0621-\u064a]/.test(answer) ? "ar" : "fr"}>{answer}</p>{generation && <small>{generation.method === "labeled_source" ? "Réponse vérifiée dans le passage source" : generation.status === "completed" ? `Réponse · ${generation.model || "LLM"}` : "Réponse basée sur les passages retrouvés"}</small>}</div>}
        {retrieval && <div className="assistant-retrieval"><span>{retrieval.chunks_searched ?? 0} passages analysés</span><span>{sources.length} preuves retenues</span></div>}
        {sources.length > 0 && <div className="assistant-sources"><b>Preuves RAG</b>{sources.slice(0, 5).map((source, index) => <blockquote key={index}><span>{`Source ${index + 1} · document ${source.document_id ?? "-"} · page ${source.page_number ?? "N/A"}`}</span><div dir="auto">{source.text || source.content}</div></blockquote>)}</div>}
        {error && <p className="form-error">{error}</p>}
      </div>
      <form onSubmit={async (event) => {
        event.preventDefault();
        if (!question.trim() || busy) return;
        setBusy(true); setError(""); setAnswer(""); setSources([]); setGeneration(null); setRetrieval(null);
        try {
          const response = await fetch(`${API_BASE_URL}/rag/query`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, top_k: 5 }) });
          const data = await response.json();
          if (!response.ok || !data.success) throw new Error(data.detail || data.error || "Assistant indisponible.");
          setAnswer(data.answer || "Réponse indisponible, mais les preuves RAG sont affichées.");
          setSources(data.sources || []);
          setGeneration(data.generation || null);
          setRetrieval(data.retrieval || null);
        } catch (requestError) { setError(requestError instanceof Error ? requestError.message : "Assistant indisponible."); }
        finally { setBusy(false); }
      }}>
        <textarea dir="auto" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Votre question / سؤالك" rows={3} />
        <button className="btn btn-primary full" disabled={busy || !question.trim()} type="submit">{busy ? "Recherche..." : "Interroger Fatura Assistant"}</button>
      </form>
    </aside>
  );
}

function InvoiceTable({ rows, navigate }: { rows: Invoice[]; navigate: (path: string) => void }) {
  return (
    <div className="invoice-table compact-invoice-list">
      {rows.map((invoice) => (
        <button className="invoice-row compact-invoice-row" key={invoice.id} onClick={() => navigate(`/factures/${encodeURIComponent(invoice.id)}/analyse`)}>
          <span className="file-cell">
            <img src={supplierLogo(invoice.supplier)} alt="" />
            <span>
              <strong>{invoice.file}</strong>
              <small>{invoice.supplier} · {invoice.date} · {invoice.fileType || "FORMAT"}</small>
            </span>
          </span>
          <span className="invoice-meta">
            <strong>{invoice.amount}</strong>
            <small>{invoice.number}</small>
          </span>
          {invoice.status !== "Dataset" && <ClassificationBadge value={invoice.classification} confidence={invoice.classificationConfidence} />}
          <WorkflowBadge status={invoice.status} />
          <span className="row-action">Ouvrir</span>
        </button>
      ))}
    </div>
  );
}

function Kpi({ title, value, trend, icon: Icon, tone = "blue" }: { title: string; value: string; trend?: string; icon: React.ElementType; tone?: string }) {
  return <Card className={`kpi ${tone}`}><Icon size={20} /><span>{title}</span><strong>{value}</strong>{trend && <small>{trend}</small>}</Card>;
}

function PageHeader({ title, subtitle, action }: { title: string; subtitle: string; action?: React.ReactNode }) {
  return <div className="page-header"><div><h1>{title}</h1><p>{subtitle}</p></div>{action}</div>;
}

function SectionTitle({ eyebrow, title }: { eyebrow: string; title: string }) {
  return <div className="section-title"><span>{eyebrow}</span><h2>{title}</h2></div>;
}

function ChartTitle({ title, subtitle }: { title: string; subtitle?: string }) {
  return <div className="chart-title"><h3>{title}</h3>{subtitle && <span>{subtitle}</span>}</div>;
}

function DashboardTooltip({ active, payload, label, suffix, metricLabel }: any) {
  if (!active || !payload?.length) return null;
  const item = payload[0];
  const name = item.payload?.name ?? label;
  return (
    <div className="chart-tooltip">
      <strong>{name}</strong>
      <span>{metricLabel}: {item.value}{suffix}</span>
    </div>
  );
}

function StatusBreakdown() {
  return (
    <div className="status-breakdown">
      {statusData.map((item) => (
        <div key={item.name} className="status-progress">
          <div>
            <span><i style={{ background: item.color }} />{item.name}</span>
            <strong>{item.value}%</strong>
          </div>
          <div className="status-track">
            <em style={{ width: `${item.value}%`, background: item.color }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function SectionMini({ title }: { title: string }) {
  return <h3 className="section-mini">{title}</h3>;
}

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <section className={`card ${className}`}>{children}</section>;
}

function ConfidenceBadge({ value }: { value: number }) {
  const label: Confidence = value >= 90 ? "Très fiable" : value >= 65 ? "À vérifier" : "Faible confiance";
  return <span className={`confidence ${label === "Très fiable" ? "good" : label === "À vérifier" ? "warn" : "bad"}`}>{value}% {label}</span>;
}

function WorkflowBadge({ status }: { status: Status }) {
  const label: Record<Status, string> = {
    "En cours d'analyse": "Analyse",
    "En attente de validation": "À valider",
    "Validée": "Validée",
    "Rejetée": "Rejetée",
    "Dataset": "Dataset",
  };
  return <span className={`workflow-badge ${status.includes("Validée") ? "good" : status.includes("validation") ? "warn" : status.includes("Rejetée") ? "bad" : status === "Dataset" ? "dataset" : "blue"}`}>{label[status] ?? status}</span>;
}

function StatusBadge({ status }: { status: string }) {
  return <span className="workflow-badge blue">{status}</span>;
}

function ClassificationBadge({ value, confidence }: { value: Classification; confidence: number }) {
  return <span className={`classification ${value === "Facture" ? "good" : value === "Incertain" ? "warn" : "bad"}`}>{value} • {confidence}%</span>;
}

function Tabs({ tabs, active, setActive }: { tabs: string[]; active: string; setActive: (tab: string) => void }) {
  return <div className="tabs">{tabs.map((t) => <button key={t} className={active === t ? "active" : ""} onClick={() => setActive(t)}>{t}</button>)}</div>;
}

function MiniList({ items, alert = false }: { items: string[]; alert?: boolean }) {
  return (
    <div className={`mini-list ${alert ? "alert-list" : ""}`}>
      {items.map((item) => {
        const [label, amount, meta] = item.split(" - ");
        return (
          <div className="mini-list-row" key={item}>
            <span className="mini-list-icon">{alert ? <AlertTriangle size={15} /> : <Clock size={15} />}</span>
            <span className="mini-list-main">
              <strong>{label}</strong>
              {meta && <small>{meta}</small>}
            </span>
            {amount && <em>{amount}</em>}
          </div>
        );
      })}
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return <div className="empty-state"><Loader2 size={18} />{text}</div>;
}

function Legend() {
  return <div className="legend boxes"><span><i className="blue-bg" />Texte</span><span><i className="purple-bg" />IA/RAG</span><span><i className="orange-bg" />À vérifier</span></div>;
}

function AccessDenied({ navigate }: { navigate: (path: string) => void }) {
  return <main className="center-page"><Card><Lock size={32} /><h1>Accès refusé</h1><p>Connectez-vous avec le profil administrateur pour accéder à cet espace.</p><button className="btn btn-primary" onClick={() => navigate("/login")}>Retour connexion</button></Card></main>;
}

function ComingSoon({ route }: { route: string }) {
  return <><PageHeader title="Module en préparation" subtitle={`Route ${route} protégée visuellement avec état empty.`} /><Card><EmptyState text="Aucune donnée à afficher pour ce module mocké." /></Card></>;
}

function ChatBubble({ question, answer }: { question: string; answer: string }) {
  return <div className="chat-bubble"><div className="q">{question}</div><div className="a">{answer}</div></div>;
}

createRoot(document.getElementById("root")!).render(<App />);
