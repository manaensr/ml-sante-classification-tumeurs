# Image Docker de l'API de classification de tumeurs.
# Le modèle est entraîné pendant la construction de l'image : le conteneur
# démarre donc prêt à répondre, sans étape supplémentaire.

FROM python:3.12-slim

# Pas de fichiers .pyc, affichage des messages sans attente, chemin du code
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DOSSIER_MODELE=/app/modeles

WORKDIR /app

# Les dépendances d'abord : cette couche est réutilisée tant que
# requirements.txt ne change pas, ce qui accélère les reconstructions
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Le code du projet
COPY src ./src
COPY scripts ./scripts

# L'application tourne sans droits administrateur
RUN useradd --create-home appli && mkdir /app/modeles && chown appli /app/modeles
USER appli

# Entraînement du modèle, enregistré dans /app/modeles
RUN python scripts/entrainer.py

EXPOSE 8000
CMD ["uvicorn", "classification_tumeurs.api_prediction:app", "--host", "0.0.0.0", "--port", "8000"]
