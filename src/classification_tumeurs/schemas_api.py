"""
schemas_api.py
Rôle : décrit la forme des demandes et des réponses de l'API, et vérifie
       que la demande contient bien les 30 mesures attendues.
Utilise : pydantic
Utilisé par : api_prediction.py, tests/
"""

from pydantic import BaseModel, field_validator

from classification_tumeurs.chargement_donnees import noms_variables


class DemandePrediction(BaseModel):
    """
    Demande de prédiction : les 30 mesures d'une tumeur, par leur nom.

    Exemple : {"mesures": {"mean radius": 17.99, "mean texture": 10.38, ...}}
    """

    mesures: dict[str, float]

    @field_validator("mesures")
    @classmethod
    def verifier_mesures(cls, valeur: dict[str, float]) -> dict[str, float]:
        """
        Refuse la demande s'il manque une mesure ou s'il y en a une inconnue.

        Paramètres :
            valeur (dict) : les mesures reçues.

        Retourne :
            dict : les mêmes mesures, si elles sont valides.

        Erreurs :
            ValueError : si des mesures manquent ou sont inconnues. Pydantic
            la transforme en réponse HTTP 422 avec le détail de l'erreur.
        """
        attendues = set(noms_variables())
        manquantes = attendues - set(valeur)
        inconnues = set(valeur) - attendues

        if manquantes or inconnues:
            raise ValueError(
                f"Mesures manquantes : {sorted(manquantes)} ; "
                f"mesures inconnues : {sorted(inconnues)}"
            )
        return valeur


class ReponsePrediction(BaseModel):
    """
    Réponse de l'API : la probabilité, la conclusion et un avertissement.
    """

    probabilite_malin: float
    diagnostic_estime: str
    seuil: float
    avertissement: str
