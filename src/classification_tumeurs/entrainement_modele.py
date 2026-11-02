"""
entrainement_modele.py
Rôle : construit le modèle (régression logistique), règle son seuil de
       décision pour privilégier le rappel, l'entraîne, puis l'enregistre
       et le recharge.
Utilise : scikit-learn, joblib
Utilisé par : api_prediction.py, scripts/entrainer.py, tests/
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# En dépistage, manquer une tumeur maligne est bien plus grave qu'une fausse
# alerte : on vise donc au moins 98 % des tumeurs malignes détectées.
RAPPEL_CIBLE = 0.98

# Noms des fichiers écrits dans le dossier du modèle
NOM_FICHIER_MODELE = "modele.joblib"
NOM_FICHIER_METADONNEES = "metadonnees.json"


def creer_modele() -> Pipeline:
    """
    Crée le modèle : une mise à l'échelle des mesures, puis une régression
    logistique.

    La mise à l'échelle est nécessaire car les mesures n'ont pas les mêmes
    unités (un rayon et une aire n'ont pas le même ordre de grandeur). Le
    modèle reste simple et explicable : chaque mesure reçoit un coefficient
    lisible.

    Retourne :
        Pipeline : le modèle, pas encore entraîné.
    """
    return Pipeline(
        [
            ("echelle", StandardScaler()),
            ("regression", LogisticRegression(max_iter=1000)),
        ]
    )


def regler_seuil(
    modele: Pipeline,
    mesures: pd.DataFrame,
    est_malin: pd.Series,
    rappel_cible: float = RAPPEL_CIBLE,
    graine: int = 42,
) -> float:
    """
    Cherche le seuil de décision le plus strict qui atteint le rappel visé.

    Le calcul utilise uniquement le jeu d'entraînement, par validation croisée
    (5 découpages) : chaque probabilité est prédite par un modèle qui n'a pas
    vu la tumeur concernée. Le jeu de test n'est donc jamais utilisé ici.

    Paramètres :
        modele (Pipeline) : le modèle non entraîné.
        mesures (DataFrame) : mesures du jeu d'entraînement.
        est_malin (Series) : classes du jeu d'entraînement (1 = maligne).
        rappel_cible (float) : part minimale de tumeurs malignes à détecter.
        graine (int) : graine du hasard pour la validation croisée.

    Retourne :
        float : seuil entre 0,01 et 0,50. Une tumeur est jugée maligne si sa
        probabilité atteint ce seuil.
    """
    decoupages = StratifiedKFold(n_splits=5, shuffle=True, random_state=graine)

    # Probabilité "maligne" prédite pour chaque tumeur, hors de son propre découpage
    probabilites = cross_val_predict(
        modele, mesures, est_malin, cv=decoupages, method="predict_proba"
    )[:, 1]

    # On part de 0,50 (seuil habituel) et on baisse par pas de 0,01 : on garde
    # le premier seuil qui détecte assez de tumeurs malignes.
    for seuil in np.arange(0.50, 0.0, -0.01):
        predictions = (probabilites >= seuil).astype(int)
        if recall_score(est_malin, predictions) >= rappel_cible:
            return round(float(seuil), 2)

    # Cas extrême : aucun seuil ne suffit, on prend le plus souple possible
    return 0.01


def entrainer(mesures: pd.DataFrame, est_malin: pd.Series) -> tuple[Pipeline, float]:
    """
    Règle le seuil, puis entraîne le modèle sur tout le jeu d'entraînement.

    Paramètres :
        mesures (DataFrame) : mesures du jeu d'entraînement.
        est_malin (Series) : classes du jeu d'entraînement (1 = maligne).

    Retourne :
        tuple : (modele entraîné, seuil de décision)
    """
    modele = creer_modele()
    seuil = regler_seuil(modele, mesures, est_malin)

    # L'entraînement final se fait après le réglage du seuil
    modele.fit(mesures, est_malin)
    return modele, seuil


def enregistrer_modele(
    modele: Pipeline, seuil: float, metadonnees: dict, dossier: str | Path
) -> None:
    """
    Écrit le modèle et ses métadonnées (métriques, source des données) sur disque.

    Paramètres :
        modele (Pipeline) : le modèle entraîné.
        seuil (float) : le seuil de décision réglé.
        metadonnees (dict) : informations à garder, écrites en JSON lisible.
        dossier (str ou Path) : dossier de destination, créé s'il n'existe pas.
    """
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)

    joblib.dump({"modele": modele, "seuil": seuil}, dossier / NOM_FICHIER_MODELE)
    (dossier / NOM_FICHIER_METADONNEES).write_text(
        json.dumps(metadonnees, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def charger_modele(dossier: str | Path) -> tuple[Pipeline, float]:
    """
    Recharge le modèle enregistré.

    Attention : joblib s'appuie sur pickle, qui peut exécuter du code. On ne
    charge ici que le fichier produit par ce projet, jamais un fichier
    d'origine inconnue.

    Paramètres :
        dossier (str ou Path) : dossier contenant modele.joblib.

    Retourne :
        tuple : (modele, seuil)

    Erreurs :
        FileNotFoundError : si le modèle n'a pas encore été entraîné.
    """
    chemin = Path(dossier) / NOM_FICHIER_MODELE
    if not chemin.exists():
        raise FileNotFoundError(
            f"Modèle introuvable dans {dossier} : lancer scripts/entrainer.py"
        )

    contenu = joblib.load(chemin)
    return contenu["modele"], contenu["seuil"]
