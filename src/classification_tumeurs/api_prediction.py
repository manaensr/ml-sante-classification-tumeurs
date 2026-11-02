"""
api_prediction.py
Rôle : expose le modèle par une petite API web (FastAPI).
Routes : GET /sante (le service répond-il ?) et POST /prediction.
Lancement : uvicorn classification_tumeurs.api_prediction:app --app-dir src
Utilise : entrainement_modele.py, schemas_api.py, chargement_donnees.py
"""

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from classification_tumeurs.chargement_donnees import noms_variables
from classification_tumeurs.entrainement_modele import charger_modele
from classification_tumeurs.schemas_api import DemandePrediction, ReponsePrediction

AVERTISSEMENT = (
    "Projet pédagogique sur données publiques. "
    "Ne remplace en aucun cas un avis médical."
)

app = FastAPI(title="Classification de tumeurs du sein", version="1.0.0")


def dossier_modele() -> str:
    """
    Donne le dossier où se trouve le modèle.

    Retourne :
        str : la valeur de la variable d'environnement DOSSIER_MODELE, ou
        "modeles" par défaut.
    """
    return os.environ.get("DOSSIER_MODELE", "modeles")


@lru_cache(maxsize=4)
def modele_en_memoire(dossier: str):
    """
    Charge le modèle une seule fois par dossier, puis le garde en mémoire.

    Sans cela, le fichier serait relu sur le disque à chaque demande.

    Paramètres :
        dossier (str) : dossier contenant modele.joblib.

    Retourne :
        tuple : (modele, seuil)

    Erreurs :
        FileNotFoundError : si le modèle n'a pas encore été entraîné.
    """
    return charger_modele(Path(dossier))


@app.get("/sante")
def verifier_sante() -> dict:
    """
    Vérifie que le service répond.

    Retourne :
        dict : {"etat": "ok"}
    """
    return {"etat": "ok"}


@app.post("/prediction", response_model=ReponsePrediction)
def predire(demande: DemandePrediction) -> ReponsePrediction:
    """
    Estime si une tumeur est maligne à partir de ses 30 mesures.

    Paramètres :
        demande (DemandePrediction) : les 30 mesures, déjà validées.

    Retourne :
        ReponsePrediction : probabilité, conclusion, seuil et avertissement.

    Erreurs :
        HTTPException 503 : si le modèle n'a pas été entraîné.
    """
    try:
        modele, seuil = modele_en_memoire(dossier_modele())
    except FileNotFoundError as erreur:
        raise HTTPException(status_code=503, detail=str(erreur)) from erreur

    # Une seule ligne, avec les colonnes dans l'ordre attendu par le modèle
    ligne = pd.DataFrame([demande.mesures], columns=noms_variables())
    probabilite = float(modele.predict_proba(ligne)[0, 1])

    return ReponsePrediction(
        probabilite_malin=round(probabilite, 4),
        diagnostic_estime="malin" if probabilite >= seuil else "bénin",
        seuil=seuil,
        avertissement=AVERTISSEMENT,
    )
