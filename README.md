# Optimisation de portefeuille avec frais de transaction

Le projet reproduit la relaxation convexe et l’heuristique de Lobo, Fazel et Boyd pour le rééquilibrage d’un portefeuille avec frais fixes et proportionnels. Il ajoute un filtre qui élimine des directions de transaction avant la recherche exacte. La version resserrée du filtre conserve une partie de la courbure de la variance sans résoudre de problème conique supplémentaire.

## Installation et exécution

Depuis la racine du projet, dans un environnement Python où les dépendances peuvent être installées :

```powershell
python -m pip install -r requirements.txt
python main.py --suite echelle
```

`--suite echelle` compare les bornes affine et quadratique jusqu’à 80 actifs. Pour reproduire les chronométrages avec énumération exhaustive sur 10 et 12 actifs, utiliser `python main.py --suite temps`. Pour lancer toutes les expériences et enregistrer leur sortie, utiliser `python main.py --sortie resultats.txt`. Les chronométrages dépendent de la machine.

## Organisation

- `main.py` lance les expériences.
- `portfolio.py` contient le modèle, la méthode de l’article et le filtre.
- `benchmark_cash.py` définit les portefeuilles synthétiques et les comparaisons.
- `docs/presentations/` contient les présentations [française](docs/presentations/Presentation-fr.pdf) et [anglaise](docs/presentations/Presentation-en.pdf).
- `docs/reference/` contient [l’article anglais de référence](docs/reference/lobo-fazel-boyd-2002.pdf).

## Résultat présenté

Sur le portefeuille synthétique à 12 actifs risqués, le filtre affine laisse 4 096 configurations. Avec la courbure, deux actifs sont figés et il reste 1 024 configurations. Lors de l’essai présenté, le temps passe de 44,61 à 13,02 secondes, préparation comprise. Les valeurs optimales concordent à la précision numérique. Le gain varie selon le portefeuille et les frais.

## Référence

Miguel Sousa Lobo, Maryam Fazel et Stephen Boyd. *Portfolio Optimization with Linear and Fixed Transaction Costs*. La copie incluse correspond à la [version anglaise de septembre 2002 hébergée par Stanford](https://web.stanford.edu/~boyd/papers/pdf/portfolio_submitted.pdf).
