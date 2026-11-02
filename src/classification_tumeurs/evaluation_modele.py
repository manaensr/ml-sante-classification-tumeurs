"""
evaluation_modele.py
Rôle : mesure la qualité du modèle sur le jeu de test avec des indicateurs
       adaptés à la santé (rappel, précision, matrice de confusion, AUC).
Utilise : scikit-learn
Utilisé par : scripts/entrainer.py, tests/
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def predire_avec_seuil(modele: Pipeline, mesures: pd.DataFrame, seuil: float) -> np.ndarray:
    """
    Décide, pour chaque tumeur, si elle est jugée maligne.

    Paramètres :
        modele (Pipeline) : le modèle entraîné.
        mesures (DataFrame) : les 30 mesures de chaque tumeur.
        seuil (float) : probabilité à partir de laquelle on conclut "maligne".

    Retourne :
        ndarray : 1 pour une tumeur jugée maligne, 0 pour une tumeur jugée bénigne.
    """
    probabilites = modele.predict_proba(mesures)[:, 1]
    return (probabilites >= seuil).astype(int)


def evaluer(
    modele: Pipeline, mesures: pd.DataFrame, est_malin: pd.Series, seuil: float
) -> dict:
    """
    Calcule les indicateurs de qualité du modèle.

    Le rappel est l'indicateur principal : c'est la part des tumeurs malignes
    réellement détectées. Un faux négatif (tumeur maligne manquée) est plus
    grave qu'un faux positif (fausse alerte), d'où le seuil abaissé.

    Paramètres :
        modele (Pipeline) : le modèle entraîné.
        mesures (DataFrame) : mesures du jeu de test.
        est_malin (Series) : vraies classes du jeu de test (1 = maligne).
        seuil (float) : seuil de décision.

    Retourne :
        dict : seuil, rappel, precision, auc_roc et les quatre cases de la
        matrice de confusion (vrais/faux positifs et négatifs).
    """
    probabilites = modele.predict_proba(mesures)[:, 1]
    predictions = (probabilites >= seuil).astype(int)

    # Matrice de confusion : vrais négatifs, faux positifs, faux négatifs, vrais positifs
    vn, fp, fn, vp = confusion_matrix(est_malin, predictions).ravel()

    return {
        "seuil": seuil,
        "rappel": round(float(recall_score(est_malin, predictions)), 4),
        "precision": round(float(precision_score(est_malin, predictions)), 4),
        "auc_roc": round(float(roc_auc_score(est_malin, probabilites)), 4),
        "vrais_positifs": int(vp),
        "faux_negatifs": int(fn),
        "faux_positifs": int(fp),
        "vrais_negatifs": int(vn),
    }
