import google.generativeai as genai
import json
import re
from typing import Dict, Any, Optional
from models import InvoiceData, SupplierInfo, CustomerInfo, InvoiceItem, TaxInfo, PaymentInfo
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class OptimizedAIAnalyzer:
    """Analyseur IA optimisé avec le prompt amélioré"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini AI configuré")
        else:
            print("⚠️ GEMINI_API_KEY non configuré")
            self.model = None
    
    def create_optimized_prompt(self, extracted_text: str) -> str:
        """Crée le prompt universel pour l'analyse de tous types de documents"""
        return f"""Tu es un agent intelligent expert dans l'analyse de documents administratifs et financiers. Je vais te fournir un **texte brut extrait par OCR depuis une image de document**. Ton rôle est de :
1. Lire et comprendre le contenu.
2. Identifier les éléments structurés du document (dates, montants, noms, objets, institutions, etc.).
3. Organiser ces informations de manière claire dans un objet JSON hiérarchisé.

Respecte les consignes suivantes :
- N'invente aucune donnée.
- Si une information est absente ou non détectée, mets sa valeur à `null`.
- Ne retourne **que** l'objet JSON sans explication autour.
- Adapte la structure du JSON au type de document (facture, reçu, attestation, etc.).

Voici le texte OCR :

\"\"\"
{extracted_text}
\"\"\""""
    
    def analyze_with_gemini(self, extracted_text: str) -> Optional[Dict[str, Any]]:
        """
        Analyse le texte avec Gemini AI en utilisant le prompt optimisé
        """
        if not self.model:
            print("❌ Modèle Gemini non disponible")
            return None
        
        try:
            prompt = self.create_optimized_prompt(extracted_text)
            
            print("🤖 Envoi vers Gemini AI...")
            response = self.model.generate_content(prompt)
            
            if response.text:
                # Nettoyer la réponse pour extraire uniquement le JSON
                clean_response = response.text.strip()
                
                # Supprimer les éventuels markdown ou texte supplémentaire
                # Chercher le JSON entre les accolades
                json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
                if json_match:
                    json_text = json_match.group()
                else:
                    json_text = clean_response
                
                # Parser le JSON
                try:
                    result = json.loads(json_text)
                    print("✅ Analyse Gemini réussie")
                    return result
                except json.JSONDecodeError as e:
                    print(f"❌ Erreur JSON: {e}")
                    print(f"Réponse brute: {clean_response[:500]}...")
                    return None
            
            return None
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse Gemini: {e}")
            return None
    
    def convert_to_invoice_data(self, gemini_result: Dict[str, Any], extracted_text: str) -> InvoiceData:
        """
        Convertit le résultat Gemini au format InvoiceData
        """
        try:
            # Extraire les informations générales
            infos_gen = gemini_result.get('informations_generales', {})
            vehicule = gemini_result.get('vehicule', {})
            facturation = gemini_result.get('facturation', {})
            montants = gemini_result.get('montants', {})
            coord_bancaires = gemini_result.get('coordonnees_bancaires', {})
            entreprise = gemini_result.get('entreprise', {})
            autres = gemini_result.get('autres_informations', {})
            
            # Créer les objets Pydantic
            supplier_info = None
            if entreprise:
                supplier_info = SupplierInfo(
                    name=entreprise.get('nom'),
                    address=entreprise.get('adresse'),
                    city=entreprise.get('ville'),
                    phone=entreprise.get('telephone'),
                    email=entreprise.get('email'),
                    website=entreprise.get('site_web'),
                    registration_number=entreprise.get('rc'),
                    vat_number=entreprise.get('mf')
                )
            
            # Créer un article pour le service de location
            items = []
            if facturation.get('description_service') or vehicule.get('type'):
                item = InvoiceItem(
                    designation=facturation.get('description_service') or f"Location {vehicule.get('type', 'véhicule')}",
                    description=f"Véhicule: {vehicule.get('type')}, Immat: {vehicule.get('immatriculation')}" if vehicule.get('type') else None,
                    quantity=vehicule.get('nombre_jours'),
                    unit_price=facturation.get('prix_unitaire_htva'),
                    total_price=facturation.get('prix_total_ht') or montants.get('total_ht'),
                    unit="jour" if vehicule.get('nombre_jours') else None,
                    reference=vehicule.get('numero_contrat')
                )
                items.append(item)
            
            # Créer les informations de taxe
            taxes = []
            if montants.get('tva_montant'):
                tax_info = TaxInfo(
                    rate=float(montants.get('tva_taux', '19').replace('%', '')) if montants.get('tva_taux') else 19.0,
                    amount=montants.get('tva_montant'),
                    base_amount=montants.get('total_ht')
                )
                taxes.append(tax_info)
            
            # Créer les informations de paiement
            payment_info = None
            if coord_bancaires:
                payment_info = PaymentInfo(
                    method="Virement bancaire",
                    bank_details=f"RIB: {coord_bancaires.get('rib')} - {coord_bancaires.get('nom_banque', '')}"
                )
            
            # Créer l'objet InvoiceData final
            invoice_data = InvoiceData(
                invoice_number=infos_gen.get('numero_facture'),
                invoice_date=infos_gen.get('date_facture'),
                total_amount=montants.get('total_ttc'),
                subtotal=montants.get('total_ht'),
                tax_amount=montants.get('tva_montant'),
                currency=montants.get('devise') or "TND",  # Par défaut TND pour la Tunisie
                supplier=supplier_info,
                customer=CustomerInfo(name=infos_gen.get('nom_client')) if infos_gen.get('nom_client') else None,
                items=items,
                taxes=taxes,
                payment_info=payment_info,
                language=autres.get('langue_detectee') or "fr",
                document_type=infos_gen.get('type_document') or "facture",
                confidence_score=0.9,  # Score élevé car analysé par Gemini
                additional_data={
                    'gemini_result': gemini_result,
                    'vehicule_info': vehicule,
                    'montant_lettres': montants.get('montant_lettres'),
                    'timbre_fiscal': montants.get('timbre_fiscal'),
                    'conducteur': vehicule.get('nom_conducteur')
                }
            )
            
            return invoice_data
            
        except Exception as e:
            print(f"❌ Erreur conversion Gemini vers InvoiceData: {e}")
            # Retourner un objet basique en cas d'erreur
            return InvoiceData(
                confidence_score=0.3,
                additional_data={'gemini_result': gemini_result, 'conversion_error': str(e)}
            )
    
    def convert_universal_json_to_invoice_data(self, gemini_result: Dict[str, Any], extracted_text: str) -> InvoiceData:
        """
        Convertit le JSON universel de Gemini vers InvoiceData de manière flexible
        """
        try:
            # Fonction utilitaire pour extraire des valeurs de différentes clés possibles
            def extract_value(data: dict, possible_keys: list, default=None):
                for key in possible_keys:
                    if key in data and data[key] is not None:
                        return data[key]
                    # Recherche récursive dans les sous-dictionnaires
                    for sub_key, sub_value in data.items():
                        if isinstance(sub_value, dict) and key in sub_value:
                            return sub_value[key]
                return default
            
            # Extraction flexible des informations principales
            invoice_number = extract_value(gemini_result, [
                'numero_facture', 'numero', 'number', 'invoice_number', 'facture_numero', 'n_facture'
            ])
            
            invoice_date = extract_value(gemini_result, [
                'date_facture', 'date', 'invoice_date', 'date_emission', 'date_document'
            ])
            
            total_amount = extract_value(gemini_result, [
                'total_ttc', 'total', 'montant_total', 'total_amount', 'prix_total', 'somme', 'montant'
            ])
            
            subtotal = extract_value(gemini_result, [
                'total_ht', 'sous_total', 'subtotal', 'montant_ht', 'total_htva'
            ])
            
            tax_amount = extract_value(gemini_result, [
                'tva_montant', 'tva', 'tax_amount', 'montant_tva', 'taxe'
            ])
            
            currency = extract_value(gemini_result, [
                'devise', 'currency', 'monnaie'
            ]) or "TND"
            
            # Extraction des informations du fournisseur/émetteur
            supplier_name = extract_value(gemini_result, [
                'nom_entreprise', 'entreprise', 'nom_emetteur', 'emetteur', 'fournisseur', 'supplier', 'nom_organisation', 'organisation'
            ])
            
            supplier_address = extract_value(gemini_result, [
                'adresse_entreprise', 'adresse', 'address', 'adresse_emetteur'
            ])
            
            supplier_phone = extract_value(gemini_result, [
                'telephone', 'tel', 'phone', 'telephone_entreprise'
            ])
            
            supplier_email = extract_value(gemini_result, [
                'email', 'email_entreprise', 'mail', 'courriel'
            ])
            
            # Extraction des informations du client/bénéficiaire
            customer_name = extract_value(gemini_result, [
                'nom_client', 'client', 'beneficiaire', 'destinataire', 'customer'
            ])
            
            # Type de document
            document_type = extract_value(gemini_result, [
                'type_document', 'type', 'document_type', 'categorie'
            ]) or "Document"
            
            # Langue détectée
            language = extract_value(gemini_result, [
                'langue', 'language', 'langue_detectee'
            ]) or "fr"
            
            # Créer les objets Pydantic
            supplier_info = None
            if supplier_name or supplier_address:
                supplier_info = SupplierInfo(
                    name=supplier_name,
                    address=supplier_address,
                    phone=supplier_phone,
                    email=supplier_email
                )
            
            customer_info = None
            if customer_name:
                customer_info = CustomerInfo(name=customer_name)
            
            # Conversion sécurisée des montants
            def safe_float_convert(value):
                if value is None:
                    return None
                try:
                    # Nettoyer et convertir
                    clean_value = str(value).replace(',', '.').replace(' ', '')
                    return float(clean_value) if clean_value.replace('.', '').isdigit() else None
                except:
                    return None
            
            # Créer l'objet principal
            invoice_data = InvoiceData(
                invoice_number=str(invoice_number) if invoice_number else None,
                invoice_date=str(invoice_date) if invoice_date else None,
                total_amount=safe_float_convert(total_amount),
                subtotal=safe_float_convert(subtotal),
                tax_amount=safe_float_convert(tax_amount),
                currency=currency,
                supplier=supplier_info,
                customer=customer_info,
                language=language,
                document_type=document_type,
                confidence_score=0.9,  # Score élevé car analysé par Gemini
                additional_data={
                    'gemini_result': gemini_result,
                    'document_flexible': True,
                    'universal_extraction': True
                }
            )
            
            print(f"✅ Conversion universelle réussie: {document_type}")
            return invoice_data
            
        except Exception as e:
            print(f"❌ Erreur conversion universelle: {e}")
            # Retourner un objet basique avec le JSON brut
            return InvoiceData(
                confidence_score=0.7,
                additional_data={
                    'gemini_result': gemini_result,
                    'conversion_error': str(e),
                    'raw_analysis': True
                }
            )

    def fallback_analysis(self, extracted_text: str) -> InvoiceData:
        """
        Analyse de fallback si Gemini n'est pas disponible
        """
        print("🔄 Utilisation de l'analyse de fallback...")
        
        # Analyse basique avec regex
        result = {
            'invoice_number': None,
            'total_amount': None,
            'currency': 'TND',
            'language': 'fr',
            'confidence_score': 0.4
        }
        
        text_lower = extracted_text.lower()
        
        # Chercher le numéro de facture
        patterns_invoice = [
            r'facture[^\d]*n°?\s*:?\s*(\d+)',
            r'facture[^\d]*(\d+)',
            r'n°\s*:?\s*(\d+)'
        ]
        
        for pattern in patterns_invoice:
            match = re.search(pattern, text_lower)
            if match:
                result['invoice_number'] = match.group(1)
                break
        
        # Chercher le montant total
        patterns_amount = [
            r'total\s*ttc[^\d]*(\d+[,.]?\d*)',
            r'total[^\d]*(\d+[,.]?\d*)',
            r'(\d+[,.]?\d*)\s*dinars?'
        ]
        
        for pattern in patterns_amount:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    amount_str = match.group(1).replace(',', '.')
                    result['total_amount'] = float(amount_str)
                    break
                except ValueError:
                    continue
        
        return InvoiceData(
            invoice_number=result['invoice_number'],
            total_amount=result['total_amount'],
            currency=result['currency'],
            language=result['language'],
            confidence_score=result['confidence_score'],
            additional_data={'fallback_analysis': True}
        )
    
    def analyze_invoice_text(self, extracted_text: str) -> InvoiceData:
        """
        Analyse complète du texte extrait avec prompt universel
        """
        if not extracted_text.strip():
            return InvoiceData(confidence_score=0.0)
        
        print(f"📝 Analyse universelle du texte ({len(extracted_text)} caractères)...")
        
        # Essayer avec Gemini d'abord (prompt universel)
        gemini_result = self.analyze_with_gemini(extracted_text)
        
        if gemini_result:
            try:
                # Utiliser la conversion universelle
                return self.convert_universal_json_to_invoice_data(gemini_result, extracted_text)
            except Exception as e:
                print(f"❌ Erreur conversion universelle: {e}")
                # Essayer la conversion ancienne en fallback
                try:
                    return self.convert_to_invoice_data(gemini_result, extracted_text)
                except Exception as e2:
                    print(f"❌ Erreur conversion ancienne: {e2}")
        
        # Fallback si Gemini échoue
        return self.fallback_analysis(extracted_text)
