# Invoice OCR API - Docker Documentation

## 🚀 Déploiement Docker

Cette application FastAPI d'extraction et d'analyse de factures est maintenant entièrement dockerisée pour un déploiement facile et reproductible.

## 📋 Prérequis

- Docker (version 20.10+)
- Docker Compose (version 2.0+)
- Au moins 4GB de RAM libre
- 10GB d'espace disque libre

## 🔧 Configuration

### 1. Variables d'environnement

Créez ou modifiez le fichier `.env` :

```bash
# Clé API Gemini (obligatoire)
GEMINI_API_KEY=votre_cle_api_gemini_ici

# Configuration de l'upload
UPLOAD_FOLDER=uploads
MAX_FILE_SIZE=10485760
ALLOWED_EXTENSIONS=png,jpg,jpeg,tiff,bmp,webp,pdf
```

### 2. Modèles requis

Assurez-vous que les fichiers suivants sont présents :
- `ExtractionModel/best.pt` - Modèle YOLO pour la détection d'entités
- `SimSun.ttf` - Police pour le traitement de texte

## 🚀 Déploiement rapide

### Méthode 1 : Script automatique

```bash
./deploy.sh
```

### Méthode 2 : Commandes manuelles

```bash
# 1. Construire l'image
docker-compose build

# 2. Démarrer les services
docker-compose up -d

# 3. Vérifier le statut
docker-compose ps
```

## 🔍 Vérification

Une fois déployé, l'API sera accessible sur :

- **API Base** : http://localhost:8002
- **Documentation** : http://localhost:8002/docs
- **Health Check** : http://localhost:8002/health

## 📊 Endpoints principaux

| Endpoint | Description |
|----------|-------------|
| `POST /extract-invoice` | Extraction avec Gemini AI |
| `POST /extract-invoice-gemma` | Extraction avec Gemma |
| `POST /verify-invoice` | Vérification de facture |
| `POST /extract-entities` | Extraction d'entités YOLO |
| `POST /extract-entities-with-ocr` | Entités + OCR |
| `GET /health` | État de l'API |

## 🛠️ Gestion des conteneurs

### Commandes utiles

```bash
# Voir les logs en temps réel
docker-compose logs -f

# Arrêter tous les services
docker-compose down

# Redémarrer un service spécifique
docker-compose restart invoice-ocr-api

# Reconstruire et redémarrer
docker-compose up --build -d

# Nettoyer les volumes (attention : supprime les données)
docker-compose down -v
```

### Monitoring

```bash
# Voir l'utilisation des ressources
docker stats

# Voir les processus dans le conteneur
docker-compose exec invoice-ocr-api ps aux

# Accéder au shell du conteneur
docker-compose exec invoice-ocr-api bash
```

## 🔧 Dépannage

### Problèmes courants

1. **Erreur de mémoire** :
   ```bash
   # Augmenter la mémoire Docker à 6GB minimum
   ```

2. **Modèle YOLO manquant** :
   ```bash
   # Vérifier que ExtractionModel/best.pt existe
   ls -la ExtractionModel/
   ```

3. **Tesseract non trouvé** :
   ```bash
   # Vérifier dans le conteneur
   docker-compose exec invoice-ocr-api tesseract --version
   ```

4. **Permission denied** :
   ```bash
   # Corriger les permissions
   sudo chown -R $USER:$USER uploads/
   ```

### Logs détaillés

```bash
# Logs de l'application
docker-compose logs invoice-ocr-api

# Logs avec horodatage
docker-compose logs -t invoice-ocr-api

# Dernières 100 lignes
docker-compose logs --tail=100 invoice-ocr-api
```

## 🔒 Sécurité

### Bonnes pratiques implémentées

- ✅ Utilisateur non-root dans le conteneur
- ✅ Variables d'environnement pour les secrets
- ✅ Volumes montés pour la persistence
- ✅ Health checks configurés
- ✅ Réseau Docker isolé

### Configuration production

Pour la production, modifiez `docker-compose.yml` :

```yaml
environment:
  - GEMINI_API_KEY_FILE=/run/secrets/gemini_api_key
secrets:
  gemini_api_key:
    file: ./secrets/gemini_api_key.txt
```

## 📈 Performance

### Optimisations

- Image basée sur Python 3.11-slim
- Mise en cache des layers Docker
- Installation optimisée des dépendances
- Nettoyage automatique des caches

### Monitoring recommandé

- CPU : 2-4 cores recommandés
- RAM : 4-8GB selon la charge
- Stockage : SSD recommandé pour les modèles

## 🔄 Mise à jour

```bash
# 1. Arrêter les services
docker-compose down

# 2. Récupérer les dernières modifications
git pull

# 3. Reconstruire et redémarrer
docker-compose up --build -d
```

## 🆘 Support

En cas de problème :

1. Vérifiez les logs : `docker-compose logs -f`
2. Vérifiez la configuration : `docker-compose config`
3. Testez l'API : `curl http://localhost:8002/health`
4. Vérifiez les ressources : `docker stats`

## 🎯 Prochaines étapes

- [ ] Ajout de Redis pour le cache
- [ ] Base de données PostgreSQL pour l'historique
- [ ] Monitoring avec Prometheus/Grafana
- [ ] CI/CD avec GitHub Actions
- [ ] Déploiement Kubernetes
