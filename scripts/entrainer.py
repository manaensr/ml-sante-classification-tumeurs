"""
entrainer.py
Rôle : script de bout en bout. Charge les données, règle le seuil, entraîne le
       modèle, l'évalue sur le jeu de test, puis l'enregistre dans modeles/.
Lancement : python scripts/entrainer.py
Utilise : le paquet classification_tumeurs (dossier src/)
"""

import os
import sys
from pathlib import Path

# Le code du projet est dans src/ : on l'ajoute au chemin de recherche de Python
RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from classification_tumeurs.chargement_donnees import (  # noqa: E402
    charger_jeu_donnees,
    decouper_donnees,
    noms_variables,
)
from classification_tumeurs.entrainement_modele import (  # noqa: E402
    RAPPEL_CIBLE,
    enregistrer_modele,
    entrainer,
)
from classification_tumeurs.evaluation_modele import evaluer  # noqa: E402
from classification_tumeurs.explication_variables import variables_influentes  # noqa: E402

# Source citée dans les métadonnées, pour que le modèle garde la trace de ses données
SOURCE_DONNEES = {
    "jeu": "Breast Cancer Wisconsin (Diagnostic)",
    "reference": "Wolberg, Mangasarian, Street et Street (1993), UCI, doi.org/10.24432/C5DW2B",
    "licence": "CC BY 4.0",
}


def afficher_resume(metriques: dict, influentes: dict) -> None:
    """
    Affiche les résultats dans le terminal, en français.

    Paramètres :
        metriques (dict) : résultat de evaluer().
        influentes (dict) : résultat de variables_influentes().
    """
    total_malignes = metriques["vrais_positifs"] + metriques["faux_negatifs"]
    print(f"Seuil de décision      : {metriques['seuil']}")
    print(
        f"Rappel (malignes)      : {metriques['rappel']:.1%} "
        f"({metriques['vrais_positifs']} sur {total_malignes})"
    )
    print(f"Précision              : {metriques['precision']:.1%}")
    print(f"AUC ROC                : {metriques['auc_roc']}")
    print(
        f"Faux négatifs / positifs : "
        f"{metriques['faux_negatifs']} / {metriques['faux_positifs']}"
    )
    print("Mesures qui poussent vers « malin » :")
    for nom, coefficient in influentes["vers_malin"]:
        print(f"  {nom} ({coefficient})")


def main() -> None:
    """
    Enchaîne toutes les étapes : données, entraînement, évaluation, sauvegarde.
    """
    # 1. Données : découpage reproductible, test gardé de côté
    mesures, est_malin = charger_jeu_donnees()
    entr_x, test_x, entr_y, test_y = decouper_donnees(mesures, est_malin)

    # 2. Entraînement : le seuil est réglé sans jamais regarder le jeu de test
    modele, seuil = entrainer(entr_x, entr_y)

    # 3. Évaluation unique, sur le jeu de test
    metriques = evaluer(modele, test_x, test_y, seuil)
    influentes = variables_influentes(modele, noms_variables())
    afficher_resume(metriques, influentes)

    # 4. Sauvegarde du modèle et de ses métadonnées
    dossier = os.environ.get("DOSSIER_MODELE", str(RACINE / "modeles"))
    metadonnees = {
        "donnees": SOURCE_DONNEES,
        "rappel_cible": RAPPEL_CIBLE,
        "metriques_test": metriques,
        "variables_influentes": influentes,
        "avertissement": "Projet pédagogique. Ne remplace pas un avis médical.",
    }
    enregistrer_modele(modele, seuil, metadonnees, dossier)
    print(f"Modèle enregistré dans {dossier}")


if __name__ == "__main__":
    main()
