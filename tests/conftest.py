"""
conftest.py
Rôle : prépare les éléments partagés par les tests (données, modèle entraîné).
Utilise : le paquet classification_tumeurs
Utilisé par : pytest, automatiquement
"""

import pytest

from classification_tumeurs.chargement_donnees import charger_jeu_donnees, decouper_donnees
from classification_tumeurs.entrainement_modele import entrainer


@pytest.fixture(scope="session")
def donnees_decoupees():
    """
    Charge et découpe les données une seule fois pour toute la session de tests.

    Retourne :
        tuple : (mesures_entrainement, mesures_test, malin_entrainement, malin_test)
    """
    mesures, est_malin = charger_jeu_donnees()
    return decouper_donnees(mesures, est_malin)


@pytest.fixture(scope="session")
def modele_entraine(donnees_decoupees):
    """
    Entraîne le modèle une seule fois (l'entraînement prend quelques secondes).

    Retourne :
        tuple : (modele, seuil)
    """
    entr_x, _, entr_y, _ = donnees_decoupees
    return entrainer(entr_x, entr_y)
