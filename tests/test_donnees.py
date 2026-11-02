"""
test_donnees.py
Rôle : vérifie le chargement et le découpage des données.
Teste : chargement_donnees.py
"""

from classification_tumeurs.chargement_donnees import (
    charger_jeu_donnees,
    decouper_donnees,
    noms_variables,
)


def test_jeu_a_569_tumeurs_et_30_mesures():
    """Le jeu contient 569 tumeurs décrites par 30 mesures."""
    mesures, est_malin = charger_jeu_donnees()
    assert mesures.shape == (569, 30)
    assert len(est_malin) == 569
    assert list(mesures.columns) == noms_variables()


def test_la_classe_1_designe_les_tumeurs_malignes():
    """Le jeu compte 212 tumeurs malignes (classe 1) et 357 bénignes (classe 0)."""
    _, est_malin = charger_jeu_donnees()
    assert int(est_malin.sum()) == 212
    assert int((est_malin == 0).sum()) == 357


def test_decoupage_garde_la_meme_proportion_de_malignes(donnees_decoupees):
    """Le découpage stratifié garde environ 37 % de malignes dans chaque jeu."""
    _, _, entr_y, test_y = donnees_decoupees
    assert abs(entr_y.mean() - test_y.mean()) < 0.01
    assert len(test_y) == 114
