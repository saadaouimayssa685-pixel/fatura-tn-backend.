# FATURA.tn

Application web d'aide a la saisie et a la verification des factures.
FATURA.tn associe les donnees extraites a leurs preuves documentaires et permet
de les corriger avant validation humaine.

![Facture et champs extraits](docs/screenshots/01-resultats-facture.jpg)

## Fonctionnalites

- Authentification et sessions utilisateur.
- Import de factures au format image ou PDF.
- Classification documentaire et extraction des champs fournisseur/client.
- Recuperation des references, dates, montants, taxes et articles.
- Preuves par champ et surlignage lorsque les coordonnees existent.
- Correction, enregistrement du brouillon, validation humaine et historique.
- Tableau de bord, catalogue des factures et liste des fournisseurs.
- Apercu du document, telechargement authentifie et export JSON.
- Fatura Assistant en panneau de conversation.

Une extraction automatique reste un brouillon a verifier. Les detections
visuelles ne constituent pas une certification d'authenticite.

## Demonstration

Les captures montrent le resultat enregistre de la facture de demonstration
`00158`, publie avec autorisation. Le numero imprime est **00633**, distinct
du nom du fichier.

| Champ | Resultat |
| --- | --- |
| Numero | 00633 |
| Date | 28-11-2019 |
| Fournisseur | BEN ABDELKADER |
| Articles | 2 |
| Prix unitaires TTC | 6,000 et 6,000 TND |
| Timbre | 0,600 TND |
| Total TTC imprime | 12,600 TND |
| Paiement | Especes |
| HT et TVA | Non renseignes dans le resultat |

### Articles

![Premier article](docs/screenshots/02-articles.jpg)

![Second article](docs/screenshots/03-second-article.jpg)

### Preuves documentaires

Un clic ouvre le passage utilise pour l'extraction. Le surlignage permet de
retrouver sa position sur l'image ou le PDF. Une valeur corrigee manuellement
reste distinguee de la preuve de la valeur originale.

![Passage source](docs/screenshots/04-preuve-rag.jpg)

![Total TTC surligne](docs/screenshots/05-montants-highlight.jpg)

Ces captures conservent l'etat **A verifier**, les champs vides et les erreurs
OCR eventuelles, notamment la graphie `BRIDGE IMMOBILERE` du nom client.
Elles illustrent le parcours de verification, pas une extraction parfaite.

## Architecture

```text
Image / PDF
    -> OCR et structure des tableaux
    -> Classification
    -> Recherche des preuves documentaires
    -> Extraction LLM
    -> Grounding et mapping
    -> Regles metier et validation financiere
    -> Correction et validation humaines
```

| Couche | Technologies |
| --- | --- |
| Frontend | React 19, TypeScript, Vite, Lucide, Recharts, PDF.js |
| Backend | Python, FastAPI |
| OCR | PaddleOCR, Tesseract et extraction de texte PDF selon configuration |
| Detection visuelle | Ultralytics / YOLO et poids configures |
| RAG d'extraction | Chunking legacy, BM25, all-MiniLM-L6-v2, RRF |
| Options RAG | Chunking layout, BGE-M3 en mode haute qualite |
| LLM | Groq GPT-OSS-120B, adaptateur Ollama configurable |
| Stockage local | SQLite et documents sur disque |
| Stockage heberge | PostgreSQL, JSONB et fichiers BYTEA |

Les controles financiers signalent les incoherences sans remplacer les
montants imprimes par des montants calcules. Les champs insuffisamment
prouves doivent rester vides ou etre signales pour verification.

## Contenu du depot

```text
backend/              API, OCR, extraction, preuves et stockage
web/                  Interface React / Vite
models/extraction/    Poids YOLO inclus dans le projet
assets/               Ressources backend
docs/                 Configuration, hebergement et captures
```

Le depot contient le code applicatif complet. Les bases, fichiers importes,
cles API, journaux, scripts de test et outils d'experimentation sont conserves
hors du contenu publie. Certains poids de modeles doivent etre installes
separement selon la configuration choisie.

## Installation locale

### Backend

Installer Python, les dependances systeme OCR et les poids des modeles utilises.
Verifier la compatibilite de Paddle et PyTorch avec l'environnement cible.

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Configurer le fichier `.env` :

- SQLite local : `DATABASE_URL` vide et `REQUIRE_POSTGRES=false`.
- PaddleOCR : `ENABLE_PADDLEOCR=true` et `OCR_TEXT_ENGINE=paddle`.
- Retrieval : `RAG_CHUNKING_MODE=legacy` et `RAG_EMBEDDING_MODEL=minilm`.
- Groq : `INVOICE_LLM_PROVIDER=groq` et la cle serveur `GROQ_API_KEY`.
- Adapter les chemins des modeles et du stockage a la machine.

Le fichier `.env.example` est un gabarit : il ne contient pas les secrets ni
la configuration privee du poste. Ne jamais publier le fichier `.env`.

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

### Frontend

Dans un second terminal :

```powershell
cd web
npm ci
Copy-Item .env.example .env.local
npm run dev -- --port 5176 --strictPort
```

Ouvrir [FATURA.tn en local](http://127.0.0.1:5176).
La variable `VITE_API_BASE_URL` doit pointer vers `http://127.0.0.1:8002`.
Aucune cle LLM ne doit etre exposee dans une variable `VITE_`.

Pour compiler l'interface : `npm run build` depuis `web/`.

## Hebergement et securite

- Vercel : dossier racine `web`, compilation Vite, sortie `dist`.
- Backend : service FastAPI et base PostgreSQL persistante.
- `WORKSPACE_ALLOWED_EMAILS` definit les comptes existants autorises a acceder
  a l'espace partage. Une liste vide refuse l'acces.
- En mode heberge PostgreSQL, l'inscription publique et les comptes de
  demonstration sont desactives. Les routes de donnees exigent une session.
- Les comptes autorises partagent une seule base : l'application ne propose
  pas d'isolation multi-entreprise.
- Les secrets sont configures dans les consoles d'hebergement, jamais dans Git.

Le profil `backend/Dockerfile.free` desactive PaddleOCR, YOLO et les embeddings
Sentence Transformers. Il n'est pas equivalent au pipeline local complet.
La version du depot doit etre deployee et verifiee separement : sa publication
sur GitHub ne garantit pas que le site public execute cette version.

Voir le [guide Render](docs/RENDER_FREE.md) et les
[limites du stockage heberge](backend/FILE_STORAGE.md).

## Limites

- Une verification humaine reste necessaire, notamment pour les scans peu
  lisibles, les tableaux ambigus et les identifiants incomplets.
- Les quotas et la disponibilite du fournisseur LLM peuvent interrompre
  l'analyse.
- La reinitialisation du mot de passe par email n'est pas configuree.
- Les donnees locales ne sont pas automatiquement transferees vers le service
  heberge. Une migration doit inclure les documents et l'historique.
- Les captures autorisees ne constituent pas une autorisation de publier
  d'autres factures ou des donnees privees.
