# Invoice RAG OS

Application locale d'automatisation de saisie des factures a partir d'images ou de PDF.

Le projet combine OCR, extraction PDF, detection d'objets, classification de documents, extraction de champs, validation humaine, stockage PostgreSQL/JSONB et RAG pour interroger les factures indexees.

## Objectif produit

L'application sert a transformer une facture image/PDF en donnees exploitables:

1. l'utilisateur charge une image ou un PDF;
2. le backend extrait le texte avec OCR ou pdfplumber;
3. le document est classe comme facture ou non;
4. les objets visuels sont detectes avec YOLO: cachet, logo, etc.;
5. les champs facture sont extraits: numero, date, client, HT, TVA, timbre, TTC;
6. le document est indexe dans le RAG;
7. une facture brouillon est creee dans PostgreSQL;
8. un humain corrige/valide;
9. l'admin valide la facture en base;
10. le chatbot RAG repond aux questions sur les documents indexes.

## Fonctionnalites principales

- Authentification MVP avec roles `Admin` et `User`.
- Interface Next.js professionnelle avec sidebar.
- Dashboard analytics:
  - nombre de factures integrees;
  - factures validees;
  - factures a valider;
  - montant total integre;
  - fournisseur/client le plus integre;
  - factures a echeance proche.
- Upload image/PDF.
- OCR image avec Tesseract.
- Extraction PDF native avec pdfplumber et fallback OCR.
- Detection d'objets avec YOLO.
- Extraction de champs par approche hybride:
  - regles deterministes;
  - RAG-assisted extraction;
  - option LLM local;
  - option LLM cloud.
- Validation humaine avant passage en statut `validated`.
- Detection de doublons potentiels avant validation.
- Chatbot RAG connecte aux documents indexes.
- Stockage PostgreSQL unique pour factures, documents IA et chunks RAG.

## Architecture

```text
invoice-ocr-app/
  backend/
    main.py                  API FastAPI principale
    ocr_extractor.py          OCR image Tesseract optimise
    pdf_extractor.py          Extraction PDF avec pdfplumber + OCR fallback
    postgres_invoice_store.py Stockage PostgreSQL/JSONB des factures
    postgres_document_store.py Stockage PostgreSQL/JSONB des documents IA
    rag/                      Chunking, embeddings, recherche RAG PostgreSQL
    ollama_analyzer.py        Analyse LLM local via Ollama
    ai_analyzer.py            Analyse LLM cloud Gemini
    gemma_analyzer.py         Analyse Gemma si disponible
    tessdata/                 Donnees de langue Tesseract

  frontend/
    app/
      page.tsx                Accueil, auth MVP, dashboard, validation, chatbot
      extract/page.tsx        Console upload + agent IA + resultats
      api/                    Routes proxy Next.js vers le backend
    components/ui/            Composants shadcn/ui
    lib/backend.ts            URL backend

  data/
    uploads/                  Fichiers temporaires/uploads
    invoice_storage/          Fichiers factures persistants
    document_ai/              Fichiers documents IA
    rag_storage/              Documents indexes par le RAG

  models/
    extraction/best.pt        Modele YOLO local

  docs/
    DOCKER_README.md          Notes Docker
```

## Stack technique

Backend:

- Python
- FastAPI
- Tesseract OCR
- pdfplumber
- YOLO / Ultralytics
- PostgreSQL avec JSONB
- Sentence Transformers pour embeddings RAG
- Ollama pour LLM local
- Gemini pour LLM cloud

Frontend:

- Next.js 15
- React 19
- Tailwind CSS
- shadcn/ui
- Recharts
- Lucide Icons

## Prerequis

Obligatoires:

- Python 3.11+ ou version compatible installee localement.
- Node.js / pnpm.
- Tesseract OCR installe.
- Modele YOLO present dans `models/extraction/best.pt`.
- PostgreSQL local ou via Docker.

Optionnels:

- Ollama pour le mode LLM local.
- Cle Gemini pour le mode LLM cloud.

## Variables d'environnement

Backend: `backend/.env`

```env
DATABASE_URL=postgresql://invoice_user:invoice_password@localhost:5432/invoice_rag_os
GEMINI_API_KEY=your_key_here
UPLOAD_FOLDER=data/uploads
INVOICE_STORAGE_FOLDER=data/invoice_storage/documents
DOCUMENT_STORAGE_FOLDER=data/document_ai/documents
RAG_STORAGE_FOLDER=data/rag_storage/documents
YOLO_MODEL_PATH=models/extraction/best.pt
```

Frontend:

```env
NEXT_PUBLIC_BACKEND_BASE_URL=http://localhost:8002
BACKEND_BASE_URL=http://localhost:8002
```

Par defaut, le frontend utilise `http://localhost:8002`.

Le frontend ne parle pas directement au backend depuis les pages React: il passe par les routes proxy Next.js. La connexion cree une session backend et les requetes protegees transmettent son token:

```text
/api/health                  -> backend /health
/api/agent/process-document  -> backend /agent/process-document
/api/invoices                -> backend /invoices
/api/database/stats          -> backend /database/stats
/api/rag/query               -> backend /rag/query
```

## Lancement local

Backend:

```powershell
docker compose up -d postgres
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

Le backend exige `DATABASE_URL`. Sans PostgreSQL, il utilise les stores SQLite locaux en mode developpement.

Verification backend:

```text
http://127.0.0.1:8002/health
```

Frontend:

```powershell
cd frontend
pnpm install
pnpm dev
```

Interface:

```text
http://localhost:3000
```

## Comptes demo

L'authentification utilise les sessions SQLite du backend. Les comptes demo sont crees au demarrage si necessaire.

Admin:

```text
email: admin@fatura.tn
password: admin123
```

User:

```text
email: user@fatura.tn
password: user123
```

Permissions actuelles:

- `Admin`: consultation + validation humaine des factures.
- `User`: consultation et usage de l'interface sans validation finale.

Pour production, definir `ALLOWED_ORIGINS` avec les origines frontend autorisees et remplacer les identifiants demo.

## Workflow utilisateur

### 1. Connexion

Ouvrir `http://localhost:3000`, puis choisir Admin ou User.

### 2. Upload facture

Aller dans `/extract`.

Charger une facture image ou PDF.

Choisir un mode IA:

- `RAG + regles fiables`: recommande pour la saisie comptable.
- `LLM local`: mode Ollama local/offline.
- `LLM cloud`: Gemini cloud.

Cliquer sur `Lancer l'automatisation`.

### 3. Resultats

L'interface affiche:

- preview document;
- pipeline OCR/classification/objets/RAG/champs;
- objets detectes;
- champs extraits;
- graphique des montants;
- preuves RAG;
- JSON modifiable.

### 4. Validation humaine

Retourner a l'accueil.

Dans `Validation humaine`:

1. selectionner une facture;
2. corriger le JSON si necessaire;
3. cliquer sur `Valider humainement et enregistrer`.

La facture passe de `draft` a `validated`.

### 5. Chatbot RAG

Dans `Chatbot RAG`, poser une question:

```text
Quel est le montant TTC de la facture 95/2024 ?
```

Le backend recherche les chunks pertinents dans la base RAG et renvoie les sources.

## API backend principale

### Agent complet

```http
POST /agent/process-document?model_choice=fallback&index_for_rag=true&detect_objects=true
```

Parametres:

- `model_choice=fallback`: extraction fiable par regles + RAG-assisted.
- `model_choice=local`: LLM local Ollama.
- `model_choice=gemini`: LLM cloud Gemini.
- `index_for_rag=true`: indexer le document.
- `detect_objects=true`: detecter cachet/logo via YOLO.

### Documents IA

```http
GET /agent/documents
GET /agent/documents/{document_id}
```

### Factures

```http
GET /invoices
GET /invoices/{invoice_id}
PUT /invoices/{invoice_id}
POST /invoices/{invoice_id}/validate
```

### RAG

```http
POST /rag/query
```

Body:

```json
{
  "question": "Quel est le montant TTC de la facture 95/2024 ?",
  "document_id": 10,
  "top_k": 3
}
```

## Routes frontend proxy

Le frontend expose des routes Next.js qui appellent le backend:

```text
POST /api/agent/process-document
GET  /api/invoices
POST /api/invoices/[invoiceId]/validate
POST /api/rag/query
```

## Choix IA recommandes

Pour des factures, ne pas laisser le LLM decider seul des montants.

Recommande:

```text
OCR/pdfplumber -> RAG-assisted extraction -> regles -> normalisation -> validation humaine
```

Le LLM sert surtout a:

- comprendre des champs ambigus;
- enrichir les metadonnees;
- aider dans les cas non standards.

## Modes IA

### Mode fiable recommande

```text
model_choice=fallback
```

Ce mode utilise OCR/PDF + RAG-assisted extraction + regles deterministes.

### Mode local

Le mode local utilise Ollama.

Qwen a ete elimine car non fonctionnel sur ce PC.

Le modele local actuel est configure dans `backend/ollama_analyzer.py`, par exemple `phi3.5`.

### Mode cloud

Le mode cloud utilise Gemini si `GEMINI_API_KEY` est configure.

Note: le backend utilise actuellement `google.generativeai`, qui affiche un warning de deprecation. A terme, migrer vers le SDK `google.genai`.

## Test reel realise

Facture image testee:

```text
data/uploads/1753380618_facture_00008.png
```

Resultat obtenu:

```json
{
  "invoice_number": "95/2024",
  "date": "15/07/2024",
  "customer_name": "SOCIETE BRIDGE BCN IMMOBILIERE",
  "subtotal": 5045.04,
  "tax_amount": 968.143,
  "stamp_duty": 1.0,
  "total_amount": 6064.634,
  "currency": "TND"
}
```

Objets detectes:

```text
cachet: 0.7614
logo: 0.4512
```

RAG:

- document indexe;
- chunks generes;
- requete sur le montant TTC confirmee avec source contenant `MONTANT TTC 6064,634`.

## Build et verification

Frontend:

```powershell
cd frontend
pnpm build
```

Derniere verification:

```text
pnpm build: OK
```

Backend:

```powershell
cd backend
python -m py_compile main.py ocr_extractor.py postgres_invoice_store.py postgres_document_store.py rag/service.py rag/postgres_store.py
```

## Stockage des donnees

Les donnees sont stockees dans PostgreSQL uniquement:

```text
DATABASE_URL=postgresql://invoice_user:invoice_password@localhost:5432/invoice_rag_os
```

Les champs semi-structures et non structures sont stockes en JSONB.

La base combine maintenant deux niveaux:

1. donnees brutes/semi-structurees pour audit;
2. tables metier structurees pour manipulation SQL, dashboard et validation.

### Tables principales

```text
invoices
- JSON brut: normalized_json, raw_json
- colonnes metier: invoice_number, supplier_id, supplier_name, customer_name,
  invoice_date, due_date, subtotal, tax_amount, stamp_duty, total_amount,
  currency, status

suppliers
- fournisseurs/contreparties consolides
- invoices_count
- total_amount

invoice_items
- lignes de facture structurees

validation_events
- historique des creations, corrections et validations humaines

duplicate_detection
- stocke dans `raw_json`
- signale les factures similaires par numero, fournisseur + montant, ou montant identique

ai_documents
- documents non structures, OCR complet, detections, metadata, lien RAG

rag_chunks
- chunks indexes pour recherche RAG
```

Cela correspond a une base SQL avec donnees non structurees/semi-structurees:

- texte OCR complet;
- JSON normalise;
- metadata extraction;
- detections YOLO;
- preuves RAG;
- chunks RAG.

### Endpoints de manipulation SQL

```http
GET /database/stats
GET /database/suppliers
GET /database/invoice-items
GET /database/invoice-items?invoice_id=10
GET /database/validation-events
GET /database/validation-events?invoice_id=10
```

Le dashboard frontend consomme `/api/database/stats`, qui appelle `/database/stats`.

### Strategie PostgreSQL unique

La solution utilise maintenant PostgreSQL comme seule base applicative:

- meilleure concurrence;
- droits utilisateurs;
- sauvegardes robustes;
- indexes plus puissants;
- JSONB pour conserver les donnees brutes;
- possibilite pgvector pour optimiser le RAG plus tard.

Schema PostgreSQL utilise/prevu:

```text
users
suppliers
invoices
invoice_items
documents
detections
rag_chunks
validation_events
```

Les embeddings RAG sont stockes en JSONB pour rester simple au MVP. Pour une version plus grande, `pgvector` pourra remplacer ou completer `embedding_json`.

## Limites actuelles

- Authentification MVP cote frontend uniquement.
- Pas encore de vraie gestion multi-utilisateur cote backend.
- Les embeddings RAG ne sont pas encore indexes avec pgvector.
- Pas encore de table echeances dediee.
- Les line items sont encore a ameliorer selon les formats de facture.
- Les preuves RAG sont textuelles; pas encore de surlignage visuel sur l'image.
- L'interface valide le JSON; une validation champ par champ plus ergonomique serait meilleure.
- Le cloud Gemini fonctionne, mais peut etre plus lent et consomme une API externe.
- Le local LLM est utile mais doit rester controle par les regles pour les montants.

## Prochaines ameliorations recommandees

1. Auth production:
   - table users;
   - hash password;
   - JWT/session;
   - RBAC serveur Admin/User.

2. Schema SQL metier:
   - suppliers;
   - invoices;
   - invoice_items;
   - due_dates;
   - validation_events;
   - document_chunks.

3. Formulaire de validation avance:
   - edition champ par champ;
   - score de confiance par champ;
   - preuve affichee a cote du champ;
   - historique des corrections humaines.

4. RAG chatbot avance:
   - reponse finale generee par LLM avec citations;
   - filtre par facture;
   - filtre par fournisseur;
   - question sur plusieurs factures.

5. Performance:
   - mode batch;
   - traitement asynchrone;
   - queue de jobs;
   - cache OCR/RAG;
   - option desactiver YOLO pour accelerer.

6. Production:
   - Docker compose final;
   - logs;
   - monitoring;
   - sauvegardes PostgreSQL;
   - tests automatises.

## Depannage

### Backend inaccessible

Verifier:

```text
http://127.0.0.1:8002/health
```

Si le frontend ne trouve pas le backend, verifier:

```env
NEXT_PUBLIC_BACKEND_BASE_URL=http://localhost:8002
```

### Tesseract ne lit rien

Verifier:

- installation Tesseract;
- chemin `tessdata`;
- langues `fra+eng`;
- qualite de l'image.

### YOLO ne detecte rien

Verifier:

```text
models/extraction/best.pt
```

et le seuil de confiance dans le backend.

### Gemini indisponible

Verifier:

```env
GEMINI_API_KEY=...
```

Si absent, l'application retombe sur l'extraction fallback.

### LLM local lent ou incorrect

Utiliser le mode recommande:

```text
RAG + regles fiables
```

Le LLM local ne doit pas etre la seule source de verite pour les montants.

## Docker

Voir:

```text
docs/DOCKER_README.md
```

## Note projet FATURA.tn - interface, agent IA et base de donnees

### Role de l'humain dans la validation

L'humain intervient apres l'upload et apres le traitement automatique. Son role n'est pas de refaire l'OCR ou l'extraction: il valide un formulaire deja pre-rempli par les champs extraits automatiquement.

Flux attendu:

1. L'utilisateur importe un PDF ou une image.
2. Le systeme extrait le texte, classe le document et detecte les zones utiles.
3. Le RAG d'extraction retrouve les preuves par champ.
4. Le LLM et les regles remplissent le formulaire facture.
5. Le systeme normalise les montants TND, les dates, la TVA et les lignes.
6. Les controles detectent les incoherences et doublons potentiels.
7. L'humain verifie le formulaire, corrige si besoin, puis valide la facture.
8. La facture validee est archivee et devient exploitable pour dashboard, exports et assistant IA.

### Interface a garder

L'interface doit suivre les templates des captures:

- page d'accueil premium FATURA.tn avec hero, pipeline IA, assistant RAG et securite;
- page login en deux colonnes;
- dashboard utilisateur avec sidebar bleu fonce, KPIs, graphiques, factures recentes, fournisseurs, echeances et alertes;
- page analyse facture avec apercu document, champs extraits, intelligence IA, JSON, RAG et audit;
- page validation humaine avec document a gauche, formulaire au centre et preuve RAG a droite;
- admin panel avec overview, users, companies, roles, all invoices, AI processing, logs et audit.

La partie "Tous les outils dont vous avez besoin" a ete retiree de l'accueil. Les fonctionnalites doivent etre montrees a travers des blocs plus utiles et interactifs: pipeline, assistant IA, preuves RAG, controles, securite.

Direction visuelle ajoutee pour l'accueil:

- alterner certains blocs en carres blancs nets pour donner de la clarte;
- utiliser d'autres blocs en fond sombre pour creer du contraste;
- integrer des cercles avec avatars, logos fournisseurs ou photos dynamiques;
- garder des micro-interactions au survol pour que la page semble vivante;
- eviter que toutes les sections ressemblent aux memes cartes basiques.

Decision interface du 31/08/2026:

- garder uniquement l'espace utilisateur/comptable dans le prototype;
- retirer l'acces admin de la connexion et de la navigation;
- remplacer le CTA final par deux actions: `Se connecter` et `S'inscrire`;
- `S'inscrire` ouvre un formulaire avec nom, entreprise, email, telephone et mot de passe;
- la connexion et l'inscription sont reliees au backend local: endpoints `/auth/signup`, `/auth/login`, `/auth/me` et `/auth/logout`;
- l'authentification MVP utilise SQLite dans `data/auth.sqlite3`, mots de passe hashes PBKDF2, sessions serveur avec token et expiration 7 jours;
- le frontend garde seulement le token et les informations publiques utilisateur en local, puis verifie la session via `/auth/me`;
- l'analyse facture `/agent/process-document` exige maintenant un token de session valide; l'interface import transmet ce token automatiquement apres connexion;
- le compte demo `demo@fatura.tn` affiche `Maissa Saadaoui`, le metier `Agent de saisie` et l'entreprise `YASMEEN ENGINEERING SYSTEMS` dans l'interface utilisateur;
- l'inscription propose un upload de photo de profil; dans le MVP local, l'avatar est conserve cote navigateur avec le profil public, tandis que l'identite et la session restent controlees par le backend SQLite;
- pour une version SaaS, l'authentification devra ensuite migrer vers PostgreSQL, JWT ou cookies httpOnly, reset mot de passe, verification email, roles stricts et entreprise rattachee;
- garder l'administration comme idee future, mais pas dans l'interface actuelle;
- l'import frontend appelle le backend local sur `http://127.0.0.1:8002/agent/process-document`;
- le test local utilise `model_choice=local`, `index_for_rag=true` et `detect_objects=true`.
- decision actuelle: garder SQLite pour le moment afin de tester rapidement l'interface et le pipeline local;
- PostgreSQL reste la cible recommandee pour une version SaaS plus serieuse, mais il sera configure plus tard.
- la liste "Factures recentes" est affichee en format compact: fichier, fournisseur, date, montant, numero et statut court, pour eviter l'interface encombrée.
- les graphes du dashboard doivent etre plus riches que des graphes simples: carte sombre pour le taux de validation, moyenne en pourcentage, evolution en points, tooltip interactif, donut avec valeur centrale et barres de repartition par statut.
- pendant l'upload, l'interface affiche un vrai etat de chargement: spinner, bouton pulse, panneau "Analyse intelligente en cours", etapes du pipeline et barre de progression indeterminee jusqu'a la reponse backend.
- l'echelle des pages internes est reduite: sidebar plus fine, topbar moins haute, cartes plus compactes, titres et KPI moins grands pour eviter l'effet zoom.
- l'echelle des pages connexion et inscription est reduite: hero plus compact, titres et paragraphes plus petits, champs moins hauts, espacements resserres.
- les blocs "Top fournisseurs", "Echeances a venir" et "Alertes" sont affiches comme des listes structurees; les alertes utilisent un style rouge avec indicateur d'avertissement flottant.
- la page import est simplifiee: suppression des cartes `Import ZIP` et `Scanner / email`, zone d'upload compacte, un seul etat de traitement pendant l'analyse, et titre de file d'import raccourci.
- la file d'import est refaite comme une liste SaaS compacte: en-tete sobre, lignes sans cartes imbriquees, logo fournisseur, statut, progression et heure alignes proprement.
- la page import evolue vers le template "Nouvelle facture": upload a gauche, extraction automatique, bloc dedie aux objets detectes par YOLO, formulaire de verification/validation a droite, puis apercu de facture.
- la page "Nouvelle facture" utilise une variante de layout claire: sidebar blanche, bouton primaire `Nouvelle facture`, sous-navigation facture, contenu centre et workflow plus proche de la capture de reference.
- les pages/entrees separees `Doublons` et `Exports` sont retirees de l'interface utilisateur: les doublons restent un controle metier dans la validation, et l'export JSON existe uniquement comme bouton apres validation humaine des champs.
- le formulaire de verification/validation reste vide avant extraction reelle: placeholders uniquement, champs non pre-remplis et bouton `Valider la facture` desactive tant que l'analyse backend n'a pas reussi.
- avant extraction reelle, les champs du formulaire ne doivent pas proposer de listes mockees; les objets YOLO affichent aussi un etat vide jusqu'a la reponse d'analyse.
- le formulaire de validation est enrichi pour se rapprocher d'une facture reelle: fournisseur, client, identifiants fiscaux, references facture/BC/BL, dates, HT, TVA, timbre, remise, TTC, net a payer, devise, paiement, RIB et notes.
- YOLO est presente comme une brique d'authentification visuelle: detection du logo fournisseur, du cachet et de la signature pour renforcer la confiance dans la facture; les tableaux et zones totaux restent des aides a l'extraction et aux controles.
- apres analyse reussie, le formulaire de validation doit se remplir automatiquement depuis `invoice.normalized_data`; le bloc YOLO doit afficher les detections backend disponibles, avec logo/cachet/signature comme signaux prioritaires.
- dans l'interface de validation, l'onglet `Authentification` affiche les objets detectes par YOLO dans la facture; l'apercu encadre uniquement les objets utiles a l'authentification visuelle, donc logo, cachet et signature.
- dans la page `Nouvelle facture`, le bloc `Authentification visuelle YOLO` est maintenant integre directement dans l'onglet `Authentification` du formulaire de verification, et non plus comme carte separee dans la colonne gauche.
- dans la page `Nouvelle facture`, les blocs `Verifier et valider les informations` et `Apercu de la facture` sont affiches cote a cote sur desktop, sans pastilles numerotees 4/5 visibles.
- le bloc `Extraction automatique` affiche pendant l'analyse un etat loading plus vivant: emoji robot anime, icone tournante, barre de progression dynamique et etapes OCR/RAG/YOLO/controles.
- apres extraction, le bloc `Extraction automatique` utilise une carte compacte inspiree dashboard: icone document circulaire, badge de confiance globale, courbe bleue dynamique et etapes fournisseur/facture/montants/TVA/objets en ligne.
- l'apercu du document uploade s'affiche maintenant dans le meme emplacement que la zone d'upload des que le fichier est selectionne; la carte d'apercu separee a ete retiree pour eviter la duplication.
- l'onglet `Lignes de facture` lit les lignes retournees par le backend (`items`, `line_items`, `invoice_items` ou variantes), affiche quantite, PU HT, TVA, total HT et confiance; si aucune ligne structuree n'est retournee, l'interface l'indique clairement sans inventer de fausses lignes.

### Chatbot / Assistant IA RAG

Le chatbot doit repondre aux questions sur les factures indexees, expliquer les champs extraits, resumer une facture et citer les preuves RAG.

Exemples de questions:

- "Resume la facture FC-2024-0456 avec fournisseur, matricule fiscal, dates, HT, TVA et TTC."
- "Explique l'echeance de cette facture et dis-moi si elle est en retard."
- "Quelle est la difference entre TVA facturee, TVA deductible et montant TTC ?"
- "Resume cette facture."
- "Pourquoi ce champ est a verifier ?"
- "Est-ce que HT + TVA + timbre = TTC ?"
- "Quelles factures STEG depassent 1 000 TND ce mois-ci ?"
- "Existe-t-il un doublon pour cette facture ?"
- "Quelles factures sont en retard ?"

Le chatbot ne doit pas inventer. Il doit s'appuyer sur:

- les champs normalises de la facture;
- les chunks RAG indexes;
- les preuves par champ;
- l'historique d'audit;
- les statuts de workflow.

### Ouverture et interrogation SQLite

Base dataset integree:

```text
data/fatura_dataset.sqlite3
```

Commandes utiles depuis `C:\Users\maiss\OneDrive\Desktop\Back\invoice-ocr-app`:

```powershell
python -c 'import sqlite3; con=sqlite3.connect("data/fatura_dataset.sqlite3"); print([r[0] for r in con.execute("select name from sqlite_master where type=''table'' order by name")]); con.close()'
python -c 'import sqlite3; con=sqlite3.connect("data/fatura_dataset.sqlite3"); print(con.execute("select split, count(*) from dataset_splits group by split").fetchall()); con.close()'
python -c 'import sqlite3; con=sqlite3.connect("data/fatura_dataset.sqlite3"); print(con.execute("select kind, count(*) from dataset_files group by kind").fetchall()); con.close()'
```

Si `sqlite3` CLI est installe:

```powershell
sqlite3 data\fatura_dataset.sqlite3
.tables
SELECT split, COUNT(*) FROM dataset_splits GROUP BY split;
SELECT kind, COUNT(*) FROM dataset_files GROUP BY kind;
```

Connexion applicative:

- SQLite est lu par `backend/fatura_dataset_store.py`;
- FastAPI expose les donnees via `/dataset/fatura/stats`, `/dataset/fatura/samples` et `/dataset/fatura/annotation`;
- le frontend appelle le backend via `API_BASE_URL = "http://127.0.0.1:8002"`;
- l'import facture utilise `/agent/process-document` pour classification, OCR/PDF text, RAG, extraction, YOLO et validation humaine.
- la page frontend `Factures` lit maintenant les vraies factures SQLite via `GET /invoices`;
- les lignes affichees dans `Factures` viennent de `data/invoice_rag_os.sqlite3`: fichier, fournisseur, date, numero, montant, statut et classification;
- si le backend local n'est pas lance, l'interface affiche une alerte au lieu d'une liste mockee.
- la page `Factures` affiche toutes les factures retournees par la base et propose une recherche par nom de fournisseur, matricule fiscal, numero de facture, nom de fichier, date ou montant;
- des index SQLite ont ete ajoutes sur `supplier_name`, `invoice_number`, `invoice_date`, `due_date`, `status` et les lignes de facture pour garder la recherche et les vues metier rapides.
- la route `GET /invoices` accepte maintenant `search`, `invoice_name`, `supplier`, `company`, `tax_id`, `status`, `page` et `page_size`;
- la page `Factures` utilise une pagination serveur et ne charge plus toute la table d'un coup;
- les filtres de classification/format ont ete retires de la page principale car ils etaient trompeurs pour le travail metier; le panneau `Ajouter des filtres` contient maintenant uniquement les criteres facture utiles: numero fiscal, nom de facture/fichier, nom de societe/client et nom de fournisseur;
- une pagination dynamique en bas permet de naviguer entre les pages quand la base contient beaucoup de factures;
- la base `data/invoice_rag_os.sqlite3` contient les factures réellement traitees/uploadées; la base `data/fatura_dataset.sqlite3` contient le dataset FATURA indexe, avec images et annotations, mais ce ne sont pas encore toutes des factures validees dans le workflow utilisateur.
- la route `GET /invoices/catalog` expose deux sources:
  - `source=processed`: factures deja traitees par l'agent et stockees dans `invoice_rag_os.sqlite3`;
  - `source=dataset`: factures du dataset FATURA indexe depuis `fatura_dataset.sqlite3`;
- la page `Factures` affiche par defaut `Dataset FATURA`, soit 10 000 factures indexees avec pagination serveur;
- l'onglet `Factures traitees` affiche les factures réellement uploadées/analysees par l'application, actuellement 8 dans la base locale;
- les filtres metier `invoice_name`, `supplier`, `company` et `tax_id` fonctionnent sur les deux sources.

### Classification et modele local

La classification est une etape avant l'extraction des champs:

1. Upload fichier PDF/image.
2. Extraction texte OCR/PDF.
3. Classification du document: facture fournisseur, non-facture ou document a verifier.
4. RAG d'extraction: preuves par champ depuis le texte/chunks.
5. Extraction et normalisation des champs.
6. Authentification visuelle YOLO: logo, cachet, signature.
7. Validation humaine du formulaire.
8. Export JSON apres validation.

Le modele LLM local est pertinent pour expliquer, resumer, reformuler et aider quand un champ est ambigu. Pour les champs critiques comme numero, date, HT, TVA, TTC, devise et matricule fiscal, le systeme doit rester hybride: OCR/PDF text + regex + normalisation + RAG de preuve, puis LLM local en assistance. Cette architecture evite que le LLM invente des montants.

La classification est maintenant exposee dans l'interface apres analyse avec type du document, confiance et justification. Le formulaire ne doit plus se remplir avec des valeurs d'exemple si le backend ne renvoie pas de champs extraits.

Correction interface import:

- si `classification.is_invoice = false`, l'interface affiche `Non-facture detectee` au lieu de `Extraction terminee avec succes`;
- le score affiche devient le vrai score renvoye par le backend, sans fallback force a 92%;
- pour une non-facture, le libelle du score devient `Score facture` afin de montrer que les regles n'ont pas trouve assez de signaux facture;
- les coches fournisseur/facture/montants/TVA/objets sont desactivees visuellement pour eviter de faire croire que les champs facture ont ete extraits;
- le bouton `Valider la facture` est bloque si le document analyse n'est pas classe comme facture.
- apres extraction d'une vraie facture, le formulaire propose maintenant un bouton `Ajouter a la base`;
- ce bouton envoie les champs corriges vers `PUT /invoices/{invoice_id}` et met a jour la facture brouillon dans `data/invoice_rag_os.sqlite3`;
- l'ordre metier devient: upload, extraction, correction du formulaire, ajout/mise a jour en base, validation humaine, puis export JSON valide;
- l'upload cree deja un brouillon technique cote backend, mais le bouton rend l'enregistrement metier explicite pour l'utilisatrice.

### Strategie base de donnees

Pour le MVP local, SQLite peut rester pratique. Pour une version serieuse SaaS, la source principale doit etre PostgreSQL.

Integration dataset `FATURA.zip` via SQLite:

- le fichier source est `C:\Users\maiss\Downloads\FATURA.zip`;
- la base locale creee est `data/fatura_dataset.sqlite3`;
- le zip contient 10 000 images, 30 000 annotations JSON, 3 CSV de split et 1 fichier metadata;
- les images restent dans le zip pour eviter de dupliquer 363 Mo dans le projet;
- SQLite indexe les chemins, les splits `train/dev/test`, les formats d'annotation `layoutlm_HF_format`, `COCO_compatible_format`, `Original_Format`, les mots, les boites et les tags;
- routes backend ajoutees:
  - `POST /dataset/fatura/import` pour reconstruire l'index SQLite depuis le zip;
  - `GET /dataset/fatura/stats` pour verifier les compteurs;
  - `GET /dataset/fatura/samples?split=train&limit=12` pour recuperer des exemples;
  - `GET /dataset/fatura/annotation?annotation_path=...` pour lire une annotation indexee.

Schema recommande:

- `users`: comptes, roles, statut, entreprise;
- `companies`: entreprises clientes;
- `suppliers`: fournisseurs detectes;
- `invoices`: facture normalisee, statut, montants, devise, dates, fournisseur;
- `invoice_items`: lignes facture;
- `documents`: fichier source, type document, texte extrait, metadata OCR/IA;
- `document_chunks`: chunks RAG, embeddings, page, source;
- `field_evidence`: preuve RAG par champ extrait;
- `validation_events`: historique des corrections et validations humaines;
- `agent_runs`: decisions, actions, modele utilise, temps de traitement, destination workflow;
- `exports`: JSON valide genere apres validation humaine;
- `audit_logs`: actions sensibles et securite.

Usage des donnees:

- Les champs metier importants doivent etre en colonnes SQL pour filtrer, trier et calculer les KPIs.
- Le brut OCR, les reponses LLM, les decisions agentiques et les metadata peuvent rester en JSONB.
- Les embeddings RAG peuvent commencer en JSONB, puis migrer vers `pgvector` quand le volume augmente.
- Les fichiers PDF/images doivent etre stockes hors base ou dans un stockage objet; la base garde seulement le chemin et les metadata.

Nom pertinent du projet:

```text
FATURA.tn - Agent IA documentaire RAG pour factures tunisiennes
```

Ce nom est correct si le systeme garde l'orchestration automatique multi-etapes: extraction, classification, RAG, LLM, controles, routage workflow et validation humaine finale.

### Interface Assistant IA

Le bouton `Assistant IA` du frontend est maintenant flottant et deplacable:

- un clic simple ouvre le panneau assistant;
- un glisser-deposer deplace le bouton dans l'interface;
- la position est conservee dans `localStorage` avec la cle `fatura-assistant-position`;
- le bouton reste limite dans la fenetre pour ne pas sortir de l'ecran.

### Detail facture et telechargement

La page detail/analyse ne doit plus afficher une maquette fixe SOTUFAB pour toutes les factures:

- l'onglet `Factures traitees` a ete renomme `Factures recentes`;
- la page `Factures a valider` affiche directement les factures uploadées/traitees, pas le dataset;
- les anciens raccourcis vers la facture de demonstration `fac-0456` ont ete remplaces pour ne plus ouvrir une fiche inexistante;
- une facture issue du dataset affiche maintenant `Dataset indexe`, pas `Analyse terminee`;
- `Dataset indexe` signifie que le fichier et ses annotations sont presents dans `FATURA.zip`/SQLite, mais qu'il n'a pas encore ete traite par le pipeline utilisateur;
- une facture uploadée garde le statut `Analyse terminee` quand le pipeline local l'a réellement analysee;
- le frontend charge le detail via `GET /invoices/catalog/detail`;
- le bouton `Telecharger le fichier` utilise maintenant `GET /invoices/catalog/download`;
- l'icone de telechargement dans la barre d'aperçu utilise le meme lien reel;
- pour le dataset, le backend lit l'image réelle dans `FATURA.zip`;
- pour les factures uploadées, le backend sert le fichier stocke dans `data/invoice_storage/documents`;
- l'aperçu du document affiche le vrai fichier ouvert, avec rendu image ou PDF selon le format.

Correction pagination:

- `page_size` est maintenant respecte par le catalogue dataset SQLite;
- `page_size` est maintenant respecte par le catalogue des factures réellement uploadées;
- tests locaux confirmes: `page_size=3` renvoie 3 elements pour dataset et factures recentes.

Preparation Vercel:

- le frontend n'utilise plus uniquement `http://127.0.0.1:8002` en dur;
- `VITE_API_BASE_URL` peut maintenant definir l'URL publique du backend au build/deploiement;
- `Front/.env.example` documente la valeur locale;
- `Front/vercel.json` configure le build Vite et les rewrites SPA vers `index.html`.

Tests locaux du 2026-09-01:

- `GET /health`: OK;
- `POST /auth/login` avec `demo@fatura.tn`: OK;
- `GET /auth/me`: OK, societe `YASMEEN ENGINEERING SYSTEMS`;
- `GET /invoices/catalog?source=dataset&page_size=3`: OK, 3/10000;
- `GET /invoices/catalog?source=processed&page_size=3`: OK, 3/10;
- detail dataset via `/invoices/catalog/detail`: OK;
- detail facture uploadée via `/invoices/catalog/detail`: OK;
- telechargement dataset: OK, image JPEG;
- telechargement facture uploadée: OK, image PNG;
- frontend `/dashboard`: OK;
- `npm run build` frontend: OK;
- `python -m py_compile` backend: OK.

### Correctifs QA avant hebergement

Verifications realisees depuis le navigateur integre et l'API locale:

- la liste `Factures` affiche le dataset SQLite avec pagination reelle: 10 lignes par page, `Page 1 / 1000`, total `10000`;
- la liste `Factures recentes` affiche les factures uploadées presentes dans SQLite;
- les doublons visibles dans `Factures recentes` correspondent a des enregistrements réellement repetes dans la base, pas a une duplication d'interface;
- les filtres metier `matricule fiscal`, `nom facture`, `societe`, `fournisseur` affichent le compteur correct apres application;
- le detail facture ne reutilise plus la maquette SOTUFAB pour tous les documents;
- chaque page detail charge le vrai fichier via `/invoices/catalog/detail`;
- chaque aperçu charge le vrai fichier image/PDF via `/invoices/catalog/download`;
- le bouton `Telecharger le fichier` pointe vers le fichier reel stocke ou lu depuis le zip dataset;
- l'onglet `Lignes facture` n'affiche plus de lignes fictives quand le backend n'a pas extrait de lignes;
- les preuves RAG affichent `Aucune preuve RAG disponible` au lieu d'inventer une preuve;
- une image non-facture est visible comme `Non-facture` avec son score de classification et statut `Rejetee`;
- le panneau `Intelligence` du detail non-facture n'affiche plus `Facture fournisseur`.
- l'interface `Fournisseurs` est connectee a `/database/suppliers`;
- `Fournisseurs` affiche les KPI consolides depuis SQLite: fournisseurs, factures liees, volume TTC;
- la recherche fournisseur filtre par nom, matricule fiscal, adresse et telephone;
- l'action `Voir factures` ouvre la liste des factures filtrees sur le fournisseur en source `Factures recentes`;
- test confirme: `NOVATION CITY` ouvre `/factures?source=processed&supplier=NOVATION%20CITY` et affiche 3 factures.
- la classification document est durcie pour eviter les faux positifs: un CV ou document administratif contenant le mot `facture` dans un projet/experience ne suffit plus;
- une vraie facture doit presenter une structure minimale: numero facture, montants HT/TTC/TVA, date, matricule fiscal ou termes de tableau;
- les signaux CV (`profil`, `formation`, `experiences professionnelles`, `competences`, `linkedin.com`) abaissent la classification facture;
- test confirme: le CV `Sirine Labidi` est classe `Non-facture`, `document_type=other`, sans creation de facture.

Builds apres correction:

- `npm run build`: OK;
- `python -m py_compile main.py fatura_dataset_store.py invoice_store.py`: OK.
