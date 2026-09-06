import os
import json
import re
from typing import Dict, Any, Optional
from models import InvoiceData, SupplierInfo, CustomerInfo
from dotenv import load_dotenv
 
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from huggingface_hub import login
    from huggingface_hub.hf_api import HfFolder
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
 
load_dotenv()
 
class GemmaAnalyzer:
    """Analyseur utilisant Gemma-3-4b-it de Hugging Face pour l'extraction de données"""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GemmaAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self.__class__._initialized:
            return
        self.model = None
        self.tokenizer = None
        # Forcer CPU pour éviter l'erreur CUDA out of memory
        self.device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
        self._init_gemma()
        self.__class__._initialized = True

    def _init_gemma(self):
        """Initialise Gemma-3n-E4B-it depuis Hugging Face sur CPU"""
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Transformers non disponible")
            return
        
        try:
            print("🤖 Initialisation de Gemma-3-4b-it...")
            
            # Utiliser le modèle Gemma-3-4b-it (modèle réel et léger)
            model_name = "google/gemma-3-4b-it"
            hf_token = os.getenv("HF_TOKEN", "")
            
            # Se connecter à Hugging Face avec le token (méthode alternative)
            HfFolder.save_token(hf_token)
            print("✅ Token sauvegardé avec HfFolder")
            
            # Charger le tokenizer avec token
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token
            )
            
            # Charger le modèle avec token et optimisations mémoire pour CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Utiliser float32 sur CPU
                low_cpu_mem_usage=True,
                token=hf_token
            )
            
            # Forcer le modèle sur CPU
            self.model = self.model.to("cpu")
            
            # Ajouter un token de padding si nécessaire
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✅ Gemma-3-4b-it configuré sur {self.device}")
            
        except Exception as e:
            print(f"❌ Erreur Gemma: {e}")
            print("🔄 Passage en mode fallback (analyse par règles)")
            self.model = None
            self.tokenizer = None
    
    def create_analysis_prompt(self, extracted_text: str) -> str:
        """Crée le prompt d'analyse universel pour Gemma-3n-E4B-it"""
        return f"""<start_of_turn>user
Tu es un expert en analyse de documents administratifs et financiers. Analyse le texte OCR fourni et extrais les informations sous forme de JSON structuré.
 
Règles :
- N'invente aucune donnée
- Si une information est absente, mets null
- Retourne UNIQUEMENT le JSON sans explication
- Adapte la structure au type de document
 
Texte à analyser :
 
{extracted_text}
 
<end_of_turn>
<start_of_turn>model
"""
    
    def analyze_with_gemma(self, extracted_text: str) -> Optional[Dict[str, Any]]:
        """Analyse le texte avec Gemma-3-4b-it"""
        if not self.model or not self.tokenizer:
            return self._analyze_fallback(extracted_text)
        
        try:
            prompt = self.create_analysis_prompt(extracted_text)
            
            # Tokeniser avec troncature
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                max_length=2048,
                truncation=True,
                padding=True
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Générer avec Gemma
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            # Décoder la réponse
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extraire seulement la partie générée après le prompt
            if "<start_of_turn>model" in response:
                generated_part = response.split("<start_of_turn>model")[-1].strip()
            else:
                generated_part = response[len(prompt):].strip()
            
            print(f"🔍 Réponse Gemma: {generated_part[:200]}...")  # Debug
            print(f"🔍 Réponse complète: {generated_part}")  # Debug complet
            
            # Stocker la réponse brute pour l'inclure dans la réponse finale
            self.last_gemma_raw_response = generated_part
            
            # Nettoyer et extraire le JSON de la réponse
            # Essayer de reconstruire un JSON valide à partir de la réponse
            try:
                # Parser manuellement les données visibles d'abord
                manual_result = self._parse_gemma_response_manually(generated_part, extracted_text)
                if manual_result:
                    print("✅ Parsing manuel réussi")
                    # Ajouter la réponse brute dans le résultat
                    manual_result['gemma_raw_response'] = generated_part
                    return manual_result
            except Exception as manual_e:
                print(f"❌ Erreur parsing manuel: {manual_e}")
            
            # Essayer d'extraire un JSON valide en cherchant les accolades
            start_brace = generated_part.find('{')
            end_brace = generated_part.rfind('}')
            
            # if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            #     json_text = generated_part[start_brace:end_brace+1]
            #     print(f"🔍 JSON extrait brut: {json_text[:100]}...")
                
            #     # Essayer de nettoyer le JSON
            #     cleaned_json = self._clean_json_string(json_text)
            #     if cleaned_json:
            #         try:
            #             result = json.loads(cleaned_json)
            #             print("✅ Analyse Gemma réussie (JSON nettoyé)")
            #             # Ajouter la réponse brute dans le résultat
            #             result['gemma_raw_response'] = generated_part
            #             return result
            #         except json.JSONDecodeError as e:
            #             print(f"❌ Erreur JSON nettoyé: {e}")
            # else:
            #     print("❌ Pas d'accolades trouvées dans la réponse")
            
            # Fallback si pas de JSON valide
            fallback_result = self._analyze_fallback(extracted_text)
            # Ajouter la réponse brute même en fallback
            fallback_result['gemma_raw_response'] = generated_part
            return fallback_result
            
        except Exception as e:
            print(f"❌ Erreur Gemma: {e}")
            fallback_result = self._analyze_fallback(extracted_text)
            # Ajouter l'erreur dans la réponse brute
            fallback_result['gemma_raw_response'] = f"Erreur: {str(e)}"
            return fallback_result
    
    def _analyze_fallback(self, text: str) -> Dict[str, Any]:
        """Analyse de fallback avec extraction par règles"""
        print("🔄 Analyse de fallback (Gemma)...")
        
        result = {
            "type_document": "document",
            "numero": None,
            "date": None,
            "emetteur": None,
            "beneficiaire": None,
            "montant": None,
            "devise": "TND",
            "objet": None
        }
        
        text_lower = text.lower()
        
        # Type de document
        if 'reçu' in text_lower or 'recu' in text_lower:
            result["type_document"] = "reçu"
        elif 'facture' in text_lower:
            result["type_document"] = "facture"
        elif 'attestation' in text_lower:
            result["type_document"] = "attestation"
        
        # Date
        date_patterns = [
            r'(\d{1,2}\/\d{1,2}\/\d{4})',
            r'(\d{1,2}-\d{1,2}-\d{4})',
            r'(\d{1,2}\s+\w+\s+\d{4})',
            r'le\s+(\d{1,2}\s+\w+\s+\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["date"] = match.group(1)
                break
        
        # Montant
        montant_patterns = [
            r'(\d+(?:[,.]\d+)?)\s*dinars?',
            r'(\d+(?:[,.]\d+)?)\s*dt',
            r'somme\s+de\s+.*?(\d+(?:[,.]\d+)?)',
            r'total\s*:?\s*(\d+(?:[,.]\d+)?)'
        ]
        
        for pattern in montant_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    result["montant"] = float(match.group(1).replace(',', '.'))
                    break
                except:
                    continue
        
        # Émetteur/Organisation
        if 'fédération tunisienne de tennis' in text_lower:
            result["emetteur"] = "Fédération Tunisienne de Tennis"
        elif 'multicar' in text_lower:
            result["emetteur"] = "Multicar"
        elif 'société' in text_lower:
            # Chercher le nom de la société
            societe_match = re.search(r'société\s+([\w\s]+)', text, re.IGNORECASE)
            if societe_match:
                result["emetteur"] = societe_match.group(1).strip()
        
        # Bénéficiaire
        beneficiaire_patterns = [
            r'reçu\s+du\s+([\w\s]+)',
            r'avoir\s+reçu\s+du\s+([\w\s]+)',
            r'client\s*:?\s*([\w\s]+)',
            r'bénéficiaire\s*:?\s*([\w\s]+)'
        ]
        
        for pattern in beneficiaire_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["beneficiaire"] = match.group(1).strip()
                break
        
        # Objet/Description
        objet_patterns = [
            r'représentant\s*:?\s*(.*?)(?:\.|$)',
            r'frais\s+de\s+(.*?)(?:\.|$)',
            r'objet\s*:?\s*(.*?)(?:\.|$)'
        ]
        
        for pattern in objet_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["objet"] = match.group(1).strip()
                break
        
        # Ajouter une indication que c'est un fallback
        result['gemma_raw_response'] = 'Analyse de fallback (pas de réponse Gemma)'
        
        return result
    
    def convert_to_invoice_data(self, analysis_result: Dict[str, Any]) -> InvoiceData:
        """Convertit le résultat d'analyse Gemma vers InvoiceData"""
        try:
            # Fonction utilitaire pour extraire des valeurs
            def extract_value(data: dict, possible_keys: list, default=None):
                for key in possible_keys:
                    if key in data and data[key] is not None:
                        return data[key]
                    # Recherche récursive
                    for sub_key, sub_value in data.items():
                        if isinstance(sub_value, dict) and key in sub_value:
                            return sub_value[key]
                return default
            
            # Extraction flexible des informations
            invoice_number = extract_value(analysis_result, [
                'numero_facturation', 'numero_facture', 'numero', 'number', 'invoice_number'
            ])
            
            invoice_date = extract_value(analysis_result, [
                'date_vente', 'date_facture', 'ure', 'date', 'invoice_date', 'date_emission'
            ])
            
            total_amount = extract_value(analysis_result, [
                'montant_total', 'total_ttc', 'total', 'montant', 'somme'
            ])
            
            currency = extract_value(analysis_result, [
                'devise', 'currency', 'monnaie'
            ]) or "TND"
            
            # Informations fournisseur
            supplier_name = extract_value(analysis_result, [
                'fournisseur', 'nom_entreprise', 'entreprise', 'emetteur', 'recepteur'
            ])
            
            # Informations client
            customer_name = extract_value(analysis_result, [
                'client', 'nom_client', 'beneficiaire', 'destinataire', 'vendu_a', 'acteur'
            ])
            
            # Type de document
            document_type = extract_value(analysis_result, [
                'type_document', 'type', 'document_type'
            ]) or "Document"
            
            # Description/Objet
            description = extract_value(analysis_result, [
                'designation_article', 'description_des_produits', 'objet', 'article', 'nature'
            ])
            
            # Lieu
            lieu = extract_value(analysis_result, [
                'lieu', 'location', 'ville'
            ])
            
            # Source/Mode de paiement
            source = extract_value(analysis_result, [
                'source', 'origine', 'moyen_paiement'
            ])
            
            # Créer les objets Pydantic
            supplier_info = None
            if supplier_name:
                supplier_info = SupplierInfo(name=supplier_name)
            
            customer_info = None
            if customer_name:
                customer_info = CustomerInfo(name=customer_name)
            
            # Conversion sécurisée du montant
            def safe_float_convert(value):
                if value is None:
                    return None
                try:
                    clean_value = str(value).replace(',', '.').replace(' ', '')
                    return float(clean_value) if clean_value.replace('.', '').isdigit() else None
                except:
                    return None
            
            # Créer l'objet InvoiceData
            invoice_data = InvoiceData(
                invoice_number=str(invoice_number) if invoice_number else None,
                invoice_date=str(invoice_date) if invoice_date else None,
                total_amount=safe_float_convert(total_amount),
                currency=currency,
                supplier=supplier_info,
                customer=customer_info,
                language="fr",
                document_type=document_type,
                confidence_score=0.85,  # Score pour Gemma
                additional_data={
                    'gemma_result': analysis_result,
                    'gemma_raw_response': analysis_result.get('gemma_raw_response', 'Non disponible'),
                    'analysis_method': 'Gemma-3-4b-it',
                    'model_version': 'google/Gemma-3-4b-it',
                    'description': description,  # Ajouter la description extraite
                    'lieu': lieu,  # Ajouter le lieu
                    'source': source  # Ajouter la source/mode de paiement
                }
            )
            
            print(f"✅ Conversion Gemma réussie: {document_type}")
            return invoice_data
            
        except Exception as e:
            print(f"❌ Erreur conversion Gemma: {e}")
            return InvoiceData(
                confidence_score=0.65,
                additional_data={
                    'gemma_result': analysis_result,
                    'conversion_error': str(e),
                    'analysis_method': 'Gemma-3-4b-it'
                }
            )
    
    def analyze_document(self, extracted_text: str) -> InvoiceData:
        """Méthode principale d'analyse avec Gemma-3-4b-it"""
        if not extracted_text.strip():
            return InvoiceData(confidence_score=0.0)
        
        print(f"🧠 Analyse Gemma-3-4b-it du texte ({len(extracted_text)} caractères)...")
        
        # Analyser avec Gemma
        analysis_result = self.analyze_with_gemma(extracted_text)
        
        if analysis_result:
            return self.convert_to_invoice_data(analysis_result)
        else:
            return InvoiceData(
                confidence_score=0.3,
                additional_data={
                    'analysis_failed': True,
                    'analysis_method': 'Gemma-3-4b-it'
                }
            )
    
    def _clean_json_string(self, json_str: str) -> Optional[str]:
        """Nettoie une chaîne JSON malformée pour la rendre valide"""
        try:
            # Supprimer les caractères invisibles et nettoyer
            cleaned = json_str.strip()
            
            # Essayer de réparer les erreurs communes
            # 1. Supprimer les virgules en fin de ligne avant }
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            
            # 2. Supprimer les propriétés vides ou malformées
            cleaned = re.sub(r'["\']?\w+["\']?\s*:\s*["\']?\s*[,}]', '', cleaned)
            cleaned = re.sub(r'["\']?\w+["\']?\s*:\s*,', '', cleaned)
            
            # 3. Supprimer les tableaux vides mal formés
            cleaned = re.sub(r':\s*\[\s*{\s*["\']?\w+["\']?\s*:\s*["\']?[^"]*["\']?\s*,?\s*},?\s*{\s*["\']?\w+["\']?\s*:\s*["\']?[^"]*["\']?\s*,?\s*}\s*\]', '', cleaned)
            
            # 4. Nettoyer les doubles virgules
            cleaned = re.sub(r',,+', ',', cleaned)
            
            # 5. S'assurer qu'on a bien un objet JSON complet
            if not cleaned.startswith('{'):
                cleaned = '{' + cleaned
            if not cleaned.endswith('}'):
                cleaned = cleaned + '}'
            
            # 6. Dernière vérification et nettoyage
            cleaned = re.sub(r'{\s*,', '{', cleaned)
            cleaned = re.sub(r',\s*}', '}', cleaned)
            
            return cleaned
            
        except Exception as e:
            print(f"❌ Erreur nettoyage JSON: {e}")
            return None
 
    def _parse_gemma_response_manually(self, response_text: str, ocr_text: str = "") -> Optional[Dict[str, Any]]:
        """Parse manuellement la réponse de Gemma quand le JSON n'est pas valide"""
        try:
            result = {}
            
            # Patterns pour extraire les données de la réponse (améliorés)
            patterns = {
                'numero_facturation': r'(?:numero_facturation|numero_facture|numero|facture\s*n°?)["\']?\s*:?\s*["\']?([^",\n}]+)["\']?',
                'client': r'(?:client|nom_client|acteur)["\']?\s*:?\s*["\']?([^",\n}]+)["\']?',
                'fournisseur': r'(?:fournisseur|emetteur|entreprise|recepteur)["\']?\s*:?\s*["\']?([^",\n}]+)["\']?',
                'designation_article': r'(?:designation_article|article|objet|description_des_produits|nature)["\']?\s*:?\s*["\']?([^",\n}]+)["\']?',
                'montant_total': r'(?:montant_total|montant_facturé|montant|total|total_ttc)["\']?\s*:?\s*["\']?([^",\n}]+)["\']?',
                'date_vente': r'(?:date_vente|date_facture|date)["\']?\s*:?\s*["\']?([^",\n}]+)["\']?',
                'vendu_a': r'(?:vendu_a|beneficiaire)["\']?\s*:?\s*["\']?([^",\n}]+)["\']?',
                'type_document': r'(?:type_document|type)["\']?\s*:?\s*["\']?(facture|reçu|recu|attestation|document)["\']?',
                'lieu': r'(?:lieu|location|ville)["\']?\s*:?\s*["\']?([^",\n}]+)["\']?',
                'source': r'(?:source|origine|moyen_paiement)["\']?\s*:?\s*["\']?([^",\n}]+)["\']?'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip().strip('"').strip("'")
                    if value and value.lower() not in ['null', 'none', '']:
                        result[key] = value
                        print(f"🔍 Extracting {key}: {value}")  # Debug
            
            # Si pas de type de document trouvé dans la réponse Gemma, essayer de détecter depuis le texte OCR
            if 'type_document' not in result or result.get('type_document') in ['VW Polo', '{']:
                if 'facture' in (response_text + ocr_text).lower():
                    result['type_document'] = 'facture'
                elif 'reçu' in (response_text + ocr_text).lower() or 'recu' in (response_text + ocr_text).lower():
                    result['type_document'] = 'reçu'
                else:
                    result['type_document'] = 'document'
            
            # Essayer d'extraire le numéro de facture depuis le texte OCR s'il n'est pas trouvé
            if 'numero_facturation' not in result and ocr_text:
                # Rechercher dans le texte OCR original
                facture_patterns = [
                    r'facture\s*n°?\s*:?\s*(\d+)',
                    r'n°\s*:?\s*(\d+)',
                    r'numero?\s*:?\s*(\d+)'
                ]
                for pattern in facture_patterns:
                    facture_match = re.search(pattern, ocr_text, re.IGNORECASE)
                    if facture_match:
                        result['numero_facturation'] = facture_match.group(1)
                        break
            
            # Essayer d'extraire le montant total depuis le texte OCR s'il n'est pas trouvé
            if 'montant_total' not in result and ocr_text:
                # Rechercher Total TTC dans le texte OCR
                montant_patterns = [
                    r'total\s*ttc\s*:?\s*(\d+[,.]?\d*)',
                    r'total\s*:?\s*(\d+[,.]?\d*)',
                    r'arrête.*?somme.*?(\d+[,.]?\d*)',
                    r'(\d+[,.]?\d*)\s*dinars?'
                ]
                for pattern in montant_patterns:
                    total_match = re.search(pattern, ocr_text, re.IGNORECASE)
                    if total_match:
                        result['montant_total'] = total_match.group(1)
                        break
            
            # Essayer d'extraire le fournisseur depuis le texte OCR
            if 'fournisseur' not in result and ocr_text:
                if 'multicar' in ocr_text.lower():
                    result['fournisseur'] = 'Multicar'
                elif 'fédération' in ocr_text.lower() and 'tennis' in ocr_text.lower():
                    result['fournisseur'] = 'Fédération Tunisienne de Tennis'
            
            # Convertir le montant en float si possible
            if 'montant_total' in result:
                try:
                    # Nettoyer le montant : supprimer caractères étranges, garder chiffres et points/virgules
                    montant_str = result['montant_total']
                    # Extraire le premier nombre valide du string
                    montant_match = re.search(r'(\d+[,.]?\d*)', montant_str)
                    if montant_match:
                        clean_amount = montant_match.group(1).replace(',', '.')
                        result['montant_total'] = float(clean_amount)
                except:
                    pass
            
            return result if result else None
            
        except Exception as e:
            print(f"❌ Erreur parsing manuel: {e}")
            return None