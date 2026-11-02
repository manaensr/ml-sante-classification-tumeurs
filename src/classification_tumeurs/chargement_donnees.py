"""
chargement_donnees.py
Rôle : charge le jeu de données Breast Cancer Wisconsin (Diagnostic) et le
       découpe en un jeu d'entraînement et un jeu de test.
Source : Wolberg, Mangasarian, Street et Street (1993), UCI Machine Learning
         Repository, doi.org/10.24432/C5DW2B. Licence CC BY 4.0.
         Le jeu est fourni avec scikit-learn : aucun téléchargement.
Utilise : scikit-learn, pandas
Utilisé par : entrainement_modele.py, schemas_api.py, scripts/entrainer.py, tests/
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Part des tumeurs gardée de côté pour le test final (20 %)
PART_TEST = 0.2

# Graine du hasard : elle rend le découpage identique à chaque exécution
GRAINE = 42


def charger_jeu_donnees() -> tuple[pd.DataFrame, pd.Series]:
    """
    Charge les 569 tumeurs et leurs 30 mesures.

    Retourne :
        tuple : (mesures, est_malin)
            mesures (DataFrame) : une ligne par tumeur, 30 colonnes.
            est_malin (Series) : 1 si la tumeur est maligne, 0 si elle est bénigne.
    """
    brut = load_breast_cancer(as_frame=True)

    # Dans scikit-learn, 0 veut dire "malin" et 1 "bénin". On inverse pour que
    # 1 désigne la classe qu'on cherche à détecter (la tumeur maligne).
    est_malin = (1 - brut.target).rename("malin")

    return brut.data, est_malin


def noms_variables() -> list[str]:
    """
    Donne les noms des 30 mesures, dans l'ordre attendu par le modèle.

    Retourne :
        list[str] : par exemple ["mean radius", "mean texture", ...].
    """
    # scikit-learn renvoie des chaînes NumPy : on les convertit en chaînes Python
    # classiques, sinon le modèle ne reconnaît pas les noms de colonnes.
    return [str(nom) for nom in load_breast_cancer().feature_names]


def decouper_donnees(
    mesures: pd.DataFrame,
    est_malin: pd.Series,
    part_test: float = PART_TEST,
    graine: int = GRAINE,
):
    """
    Sépare les données en un jeu d'entraînement et un jeu de test.

    Le découpage est stratifié : la proportion de tumeurs malignes est la même
    dans les deux jeux. Sans cela, le jeu de test pourrait en contenir trop
    peu pour que l'évaluation ait un sens.

    Paramètres :
        mesures (DataFrame) : les 30 mesures de chaque tumeur.
        est_malin (Series) : la classe de chaque tumeur (1 = maligne).
        part_test (float) : part du jeu réservée au test, entre 0 et 1.
        graine (int) : graine du hasard pour obtenir un résultat reproductible.

    Retourne :
        tuple : (mesures_entrainement, mesures_test, malin_entrainement, malin_test)
    """
    return train_test_split(
        mesures,
        est_malin,
        test_size=part_test,
        stratify=est_malin,
        random_state=graine,
    )
