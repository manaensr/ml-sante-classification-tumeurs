"""
test_modele.py
Rôle : vérifie l'entraînement, l'évaluation, l'explication et la sauvegarde.
Teste : entrainement_modele.py, evaluation_modele.py, explication_variables.py
"""

from classification_tumeurs.chargement_donnees import noms_variables
from classification_tumeurs.entrainement_modele import charger_modele, enregistrer_modele
from classification_tumeurs.evaluation_modele import evaluer, predire_avec_seuil
from classification_tumeurs.explication_variables import variables_influentes


def test_le_seuil_est_abaisse_pour_privilegier_le_rappel(modele_entraine):
    """Le seuil est plus bas que 0,5, pour manquer le moins de tumeurs malignes."""
    _, seuil = modele_entraine
    assert 0.01 <= seuil <= 0.5


def test_le_modele_detecte_presque_toutes_les_tumeurs_malignes(
    modele_entraine, donnees_decoupees
):
    """Sur le jeu de test, le rappel et l'AUC restent élevés."""
    modele, seuil = modele_entraine
    _, test_x, _, test_y = donnees_decoupees

    metriques = evaluer(modele, test_x, test_y, seuil)

    assert metriques["rappel"] >= 0.95
    assert metriques["auc_roc"] >= 0.98
    # Les quatre cases de la matrice de confusion couvrent bien les 114 tumeurs de test
    total = sum(metriques[c] for c in (
        "vrais_positifs", "faux_negatifs", "faux_positifs", "vrais_negatifs"))
    assert total == len(test_y)


def test_les_variables_influentes_sont_connues(modele_entraine):
    """L'explication renvoie 5 mesures par sens, toutes issues du jeu de données."""
    modele, _ = modele_entraine
    influentes = variables_influentes(modele, noms_variables())
    for sens in ("vers_malin", "vers_benin"):
        assert len(influentes[sens]) == 5
        assert all(nom in noms_variables() for nom, _ in influentes[sens])


def test_le_modele_recharge_donne_les_memes_predictions(
    modele_entraine, donnees_decoupees, tmp_path
):
    """Enregistrer puis recharger le modèle ne change aucune prédiction."""
    modele, seuil = modele_entraine
    _, test_x, _, _ = donnees_decoupees

    enregistrer_modele(modele, seuil, {"essai": True}, tmp_path)
    modele_relu, seuil_relu = charger_modele(tmp_path)

    assert seuil_relu == seuil
    avant = predire_avec_seuil(modele, test_x, seuil)
    apres = predire_avec_seuil(modele_relu, test_x, seuil_relu)
    assert (avant == apres).all()
