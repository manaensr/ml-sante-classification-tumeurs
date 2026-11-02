"""
explication_variables.py
Rôle : indique quelles mesures pèsent le plus dans la décision du modèle.
Utilise : le modèle entraîné (régression logistique)
Utilisé par : scripts/entrainer.py, tests/
"""

from sklearn.pipeline import Pipeline


def variables_influentes(modele: Pipeline, noms: list[str], nombre: int = 5) -> dict:
    """
    Classe les mesures selon leur coefficient dans la régression logistique.

    Comme les mesures ont été mises à l'échelle avant l'entraînement, les
    coefficients sont comparables entre eux : un grand coefficient positif
    pousse vers "malin", un grand coefficient négatif pousse vers "bénin".

    Paramètres :
        modele (Pipeline) : le modèle entraîné.
        noms (list[str]) : noms des 30 mesures, dans l'ordre des colonnes.
        nombre (int) : combien de mesures garder dans chaque sens.

    Retourne :
        dict : deux listes de couples (nom, coefficient), "vers_malin" et
        "vers_benin", triées de la plus influente à la moins influente.
    """
    coefficients = modele.named_steps["regression"].coef_[0]

    # Tri décroissant : les plus grands coefficients (vers "malin") en premier
    couples = sorted(zip(noms, coefficients), key=lambda c: c[1], reverse=True)

    vers_malin = [(nom, round(float(coef), 3)) for nom, coef in couples[:nombre]]

    # Les plus négatifs sont en fin de liste : on les remet du plus fort au plus faible
    vers_benin = [(nom, round(float(coef), 3)) for nom, coef in reversed(couples[-nombre:])]

    return {"vers_malin": vers_malin, "vers_benin": vers_benin}
