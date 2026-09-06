from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class InvoiceItem(BaseModel):
    """Modèle pour un article de facture"""
    designation: Optional[str] = Field(None, description="Désignation de l'article")
    description: Optional[str] = Field(None, description="Description détaillée")
    quantity: Optional[float] = Field(None, description="Quantité")
    unit_price: Optional[float] = Field(None, description="Prix unitaire")
    total_price: Optional[float] = Field(None, description="Prix total de la ligne")
    unit: Optional[str] = Field(None, description="Unité de mesure")
    reference: Optional[str] = Field(None, description="Référence produit")
    category: Optional[str] = Field(None, description="Catégorie de produit")

class SupplierInfo(BaseModel):
    """Modèle pour les informations du fournisseur"""
    name: Optional[str] = Field(None, description="Nom du fournisseur")
    address: Optional[str] = Field(None, description="Adresse complète")
    city: Optional[str] = Field(None, description="Ville")
    postal_code: Optional[str] = Field(None, description="Code postal")
    country: Optional[str] = Field(None, description="Pays")
    phone: Optional[str] = Field(None, description="Téléphone")
    email: Optional[str] = Field(None, description="Email")
    website: Optional[str] = Field(None, description="Site web")
    vat_number: Optional[str] = Field(None, description="Numéro de TVA")
    registration_number: Optional[str] = Field(None, description="Numéro d'enregistrement")

class CustomerInfo(BaseModel):
    """Modèle pour les informations du client"""
    name: Optional[str] = Field(None, description="Nom du client")
    address: Optional[str] = Field(None, description="Adresse complète")
    city: Optional[str] = Field(None, description="Ville")
    postal_code: Optional[str] = Field(None, description="Code postal")
    country: Optional[str] = Field(None, description="Pays")
    phone: Optional[str] = Field(None, description="Téléphone")
    email: Optional[str] = Field(None, description="Email")
    vat_number: Optional[str] = Field(None, description="Numéro de TVA")
    customer_id: Optional[str] = Field(None, description="ID client")

class TaxInfo(BaseModel):
    """Modèle pour les informations de TVA"""
    rate: Optional[float] = Field(None, description="Taux de TVA (%)")
    amount: Optional[float] = Field(None, description="Montant de la TVA")
    base_amount: Optional[float] = Field(None, description="Montant de base")

class PaymentInfo(BaseModel):
    """Modèle pour les informations de paiement"""
    method: Optional[str] = Field(None, description="Méthode de paiement")
    due_date: Optional[str] = Field(None, description="Date d'échéance")
    terms: Optional[str] = Field(None, description="Conditions de paiement")
    bank_details: Optional[str] = Field(None, description="Détails bancaires")

class InvoiceData(BaseModel):
    """Modèle principal pour les données de facture extraites"""
    # Informations de base
    invoice_number: Optional[str] = Field(None, description="Numéro de facture")
    invoice_date: Optional[str] = Field(None, description="Date de facture")
    due_date: Optional[str] = Field(None, description="Date d'échéance")
    
    # Montants
    total_amount: Optional[float] = Field(None, description="Montant total TTC")
    subtotal: Optional[float] = Field(None, description="Sous-total HT")
    tax_amount: Optional[float] = Field(None, description="Montant total des taxes")
    currency: Optional[str] = Field(None, description="Devise")
    
    # Parties prenantes
    supplier: Optional[SupplierInfo] = Field(None, description="Informations du fournisseur")
    customer: Optional[CustomerInfo] = Field(None, description="Informations du client")
    
    # Articles
    items: List[InvoiceItem] = Field(default_factory=list, description="Liste des articles")
    
    # Taxes
    taxes: List[TaxInfo] = Field(default_factory=list, description="Détail des taxes")
    
    # Paiement
    payment_info: Optional[PaymentInfo] = Field(None, description="Informations de paiement")
    
    # Métadonnées
    language: Optional[str] = Field(None, description="Langue détectée")
    document_type: Optional[str] = Field(None, description="Type de document")
    confidence_score: Optional[float] = Field(None, description="Score de confiance global")
    
    # Données supplémentaires
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Données supplémentaires détectées")

class OCRResponse(BaseModel):
    """Modèle de réponse pour l'extraction OCR"""
    success: bool = Field(description="Succès de l'opération")
    message: str = Field(description="Message de statut")
    extracted_text: Optional[str] = Field(None, description="Texte brut extrait")
    invoice_data: Optional[InvoiceData] = Field(None, description="Données structurées de la facture")
    processing_time: Optional[float] = Field(None, description="Temps de traitement en secondes")
    errors: List[str] = Field(default_factory=list, description="Liste des erreurs rencontrées")
