"""
test_api.py
Rôle : vérifie les routes de l'API de prédiction.
Teste : api_prediction.py, schemas_api.py
"""

import pytest
from fastapi.testclient import TestClient

from classification_tumeurs.api_prediction import app, modele_en_memoire
from classification_tumeurs.entrainement_modele import enregistrer_modele


@pytest.fixture
def client(modele_entraine, tmp_path, monkeypatch):
    """
    Crée un client de test dont l'API lit un modèle enregistré dans un
    dossier temporaire (le vrai dossier modeles/ n'est pas touché).

    Retourne :
        TestClient : client prêt à interroger l'API.
    """
    modele, seuil = modele_entraine
    enregistrer_modele(modele, seuil, {"essai": True}, tmp_path)
    monkeypatch.setenv("DOSSIER_MODELE", str(tmp_path))
    modele_en_memoire.cache_clear()
    return TestClient(app)


def mesures_d_une_tumeur(donnees_decoupees) -> dict:
    """
    Prend la première tumeur du jeu de test comme exemple de demande.

    Paramètres :
        donnees_decoupees (tuple) : sortie de la fixture du même nom.

    Retourne :
        dict : les 30 mesures, par nom.
    """
    _, test_x, _, _ = donnees_decoupees
    return {nom: float(valeur) for nom, valeur in test_x.iloc[0].items()}


def test_sante_repond_ok(client):
    """La route /sante confirme que le service répond."""
    reponse = client.get("/sante")
    assert reponse.status_code == 200
    assert reponse.json() == {"etat": "ok"}


def test_prediction_renvoie_une_probabilite_et_l_avertissement(client, donnees_decoupees):
    """Une demande valide renvoie une probabilité entre 0 et 1 et l'avertissement."""
    reponse = client.post(
        "/prediction", json={"mesures": mesures_d_une_tumeur(donnees_decoupees)}
    )
    corps = reponse.json()
    assert reponse.status_code == 200
    assert 0.0 <= corps["probabilite_malin"] <= 1.0
    assert corps["diagnostic_estime"] in ("malin", "bénin")
    assert "avis médical" in corps["avertissement"]


def test_prediction_refuse_une_mesure_manquante(client, donnees_decoupees):
    """Une demande incomplète est refusée avec le code 422."""
    mesures = mesures_d_une_tumeur(donnees_decoupees)
    mesures.pop("mean radius")
    reponse = client.post("/prediction", json={"mesures": mesures})
    assert reponse.status_code == 422


def test_prediction_sans_modele_renvoie_503(tmp_path, monkeypatch, donnees_decoupees):
    """Si le modèle n'est pas entraîné, l'API répond 503 avec un message clair."""
    monkeypatch.setenv("DOSSIER_MODELE", str(tmp_path / "vide"))
    modele_en_memoire.cache_clear()
    reponse = TestClient(app).post(
        "/prediction", json={"mesures": mesures_d_une_tumeur(donnees_decoupees)}
    )
    assert reponse.status_code == 503
    assert "entrainer.py" in reponse.json()["detail"]
