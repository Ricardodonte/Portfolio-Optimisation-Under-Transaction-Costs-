"""Point d'entrée pour reproduire les expériences du projet."""

import argparse
from contextlib import redirect_stdout
from pathlib import Path

from benchmark_cash import (
    print_large,
    print_screening,
    print_screening_scale,
    print_screening_timing,
    print_small,
)


EXPERIENCES = {
    "correction": ("Correction par l'actif sans risque", print_small),
    "correction-grande": ("Correction sur 20, 50 et 100 actifs", print_large),
    "filtre": ("Filtrage et optimum exhaustif", print_screening),
    "temps": ("Temps de recherche exacte et structures de frais", print_screening_timing),
    "echelle": ("Comparaison des bornes jusqu'à 80 actifs", print_screening_scale),
}


def run(suite):
    selected = EXPERIENCES.values() if suite == "tout" else (EXPERIENCES[suite],)
    for title, experiment in selected:
        print(f"\n=== {title} ===", flush=True)
        experiment()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("tout", *EXPERIENCES), default="tout",
        help="choisir une expérience ou les lancer toutes (par défaut)",
    )
    parser.add_argument("--sortie", type=Path, help="enregistrer les résultats en UTF-8")
    args = parser.parse_args()
    if args.sortie is None:
        run(args.suite)
    else:
        with args.sortie.open("w", encoding="utf-8") as output, redirect_stdout(output):
            run(args.suite)
        print(f"Résultats enregistrés dans {args.sortie}")


if __name__ == "__main__":
    main()
