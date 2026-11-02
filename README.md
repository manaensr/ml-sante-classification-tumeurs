# Classification de tumeurs du sein

Projet pédagogique de machine learning appliqué à la santé : un modèle qui classe des tumeurs du sein en malignes ou bénignes à partir de mesures, avec une API, un conteneur Docker, des tests et une intégration continue.

> Projet pédagogique sur données publiques. Il ne remplace en aucun cas un avis médical.

## Objectif

Construire une chaîne complète, de la donnée à un service utilisable : chargement des données, entraînement, évaluation avec des métriques adaptées à la santé, explication des décisions, API, conteneur et intégration continue.

## Données

- **Jeu** : Breast Cancer Wisconsin (Diagnostic), 569 tumeurs, 30 mesures, 212 malignes et 357 bénignes.
- **Source** : Wolberg, W., Mangasarian, O., Street, N. et Street, W. (1993), UCI Machine Learning Repository, [doi.org/10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).
- **Licence** : CC BY 4.0. Le jeu est fourni avec scikit-learn, aucun téléchargement n'est nécessaire.
- Données publiques et anonymes : aucune donnée de patient identifiable.

## Choix de modélisation

- **Modèle** : régression logistique précédée d'une mise à l'échelle des mesures. Un modèle simple et **explicable** : on peut lire l'influence de chaque mesure.
- **Métrique principale : le rappel sur les tumeurs malignes.** En dépistage, manquer une tumeur maligne (faux négatif) est bien plus grave qu'une fausse alerte. On accepte donc quelques fausses alertes pour manquer le moins de tumeurs possible.
- **Seuil de décision** : réglé par validation croisée sur le seul jeu d'entraînement, pour viser un rappel de 98 %. Le jeu de test n'est utilisé qu'une fois, pour l'évaluation finale (20 % des données, découpage stratifié, graine fixe).

## Résultats (jeu de test, 114 tumeurs)

| Indicateur | Valeur |
|---|---|
| Seuil de décision | 0,20 |
| Rappel (tumeurs malignes détectées) | 97,6 % (41 sur 42) |
| Précision (alertes justifiées) | 91,1 % |
| AUC ROC | 0,996 |
| Faux négatifs / faux positifs | 1 / 4 |

Mesures qui poussent le plus vers « malin » : `worst texture`, `radius error`, `worst symmetry`, `mean concave points`, `worst concavity`. Le détail complet est écrit dans `modeles/metadonnees.json` à chaque entraînement.

Ces chiffres viennent d'un seul découpage des données : ils donnent un ordre de grandeur, pas une garantie.

## Installation et lancement

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt

python scripts/entrainer.py        # entraîne le modèle et affiche les métriques
pytest                             # lance les 11 tests
uvicorn classification_tumeurs.api_prediction:app --app-dir src   # lance l'API
```

La documentation interactive de l'API est ensuite disponible sur `http://localhost:8000/docs`.

### Avec Docker

```bash
docker build -t classification-tumeurs .
docker run -p 8000:8000 classification-tumeurs
```

Le modèle est entraîné pendant la construction de l'image, et le conteneur tourne sans droits administrateur.

## API

| Route | Rôle |
|---|---|
| `GET /sante` | Vérifie que le service répond |
| `POST /prediction` | Renvoie la probabilité de malignité d'une tumeur |

Corps de `POST /prediction` : `{"mesures": {"mean radius": 17.99, ...}}` avec les 30 mesures. Une demande incomplète est refusée (code 422). Si le modèle n'est pas entraîné, l'API répond 503.

Exemple de réponse :

```json
{
  "probabilite_malin": 1.0,
  "diagnostic_estime": "malin",
  "seuil": 0.2,
  "avertissement": "Projet pédagogique sur données publiques. Ne remplace en aucun cas un avis médical."
}
```

## Organisation du dépôt

```
src/classification_tumeurs/
    chargement_donnees.py       chargement et découpage des données
    entrainement_modele.py      modèle, réglage du seuil, sauvegarde
    evaluation_modele.py        rappel, précision, AUC, matrice de confusion
    explication_variables.py    mesures les plus influentes
    schemas_api.py              forme des demandes et des réponses
    api_prediction.py           API FastAPI
scripts/entrainer.py            entraînement de bout en bout
tests/                          tests des données, du modèle et de l'API
Dockerfile                      image de l'API
.github/workflows/ci.yml        intégration continue
```

## Fait et à faire

**Fait**
- Chargement, découpage stratifié et reproductible.
- Modèle explicable, seuil réglé sans toucher au jeu de test.
- Évaluation orientée santé et explication des mesures influentes.
- API avec validation des entrées, tests (11), Dockerfile, intégration continue.

**À faire**
- Comparer avec d'autres modèles (arbres, forêts) et mesurer la variabilité des résultats sur plusieurs découpages.
- Calibrer les probabilités.
- Ajouter une interface simple pour saisir les mesures.

## Technologies

Python, scikit-learn, pandas, FastAPI, pydantic, pytest, Docker, GitHub Actions.

## Licence

MIT. Voir le fichier `LICENSE`.
