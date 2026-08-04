"""
Скрининг: расширенная библиотека → RDKit-фичи → суррогат предсказывает
DFT-дескрипторы → target.py считает recovery/selectivity/desirability
→ ранжирование.

Объединяет расширенную (surrogate-predicted) библиотеку с исходными 15
кандидатами (настоящий DFT) в одну таблицу с явной пометкой источника
gap (real_dft / surrogate_predicted) -- surrogate LOO R²=0.78 на gap
заметно ниже точности настоящего DFT, и итоговый рейтинг не должен
маскировать разницу в надёжности данных под одной колонкой.

Топ-K разветвлённых кандидатов C6+ из surrogate-части отдельно
выводятся как рекомендованные на реальную DFT-валидацию -- дешёвый
скрининг сам по себе не основание для выводов, только для отбора
кандидатов на дорогую проверку (см. PLAN.md, неделя 3).

Запуск из корня проекта:
    python3 -m src.screening.rank_candidates
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.qspr.dft_surrogate import FEATURES as SURROGATE_FEATURES
from src.qspr.dft_surrogate import fit_surrogates
from src.qspr.featurize import rdkit_descriptors
from src.qspr.target import desirability, recovery, selectivity

ORIGINAL_DATASET_CSV = Path("data/03_processed/qspr_dataset.csv")
EXPANDED_CANDIDATES_CSV = Path("data/02_interim/expanded_candidates.csv")
OUTPUT_CSV = Path("data/03_processed/screening_ranked.csv")

VALIDATION_TOP_K = 5   # сколько лучших разветвлённых C6+ кандидатов
                        # рекомендовать на реальный DFT
VALIDATION_MIN_N = 6

FIELDNAMES = ["name", "smiles", "n_carbons", "branched", "gap_ev",
              "gap_source", "recovery", "selectivity", "desirability"]


def _surrogate_feature_vector(n_carbons: int, branched: int, branch_distance: int,
                               mol_weight: float, logp: float, tpsa: float) -> list[float]:
    feat_values = {
        "n_carbons": n_carbons, "branched": branched,
        "branched_x_n": branched * n_carbons,
        "branch_distance": branch_distance,
        "mol_weight": mol_weight, "logp": logp, "tpsa": tpsa,
    }
    return [feat_values[f] for f in SURROGATE_FEATURES]


def load_original_candidates() -> list[dict]:
    if not ORIGINAL_DATASET_CSV.exists():
        raise RuntimeError(
            f"нет {ORIGINAL_DATASET_CSV} -- сначала python3 -m src.qspr.featurize"
        )
    rows = []
    with ORIGINAL_DATASET_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "name": r["name"], "smiles": r["smiles"],
                "n_carbons": int(r["n_carbons"]), "branched": int(r["branched"]),
                "gap_ev": round(float(r["gap_ev"]), 4), "gap_source": "real_dft",
                "recovery": round(float(r["recovery"]), 4),
                "selectivity": round(float(r["selectivity"]), 4),
                "desirability": round(float(r["desirability"]), 4),
            })
    return rows


def score_expanded_candidates(surrogates: dict) -> list[dict]:
    if not EXPANDED_CANDIDATES_CSV.exists():
        raise RuntimeError(
            f"нет {EXPANDED_CANDIDATES_CSV} -- сначала "
            f"python3 -m src.molecules.expanded_library"
        )
    rows = []
    with EXPANDED_CANDIDATES_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rd = rdkit_descriptors(r["smiles"])
            n, branched = rd["n_carbons"], int(rd["branched"])
            x = [_surrogate_feature_vector(
                n, branched, rd["branch_distance"], rd["mol_weight"], rd["logp"], rd["tpsa"]
            )]
            gap_pred = float(surrogates["gap_ev"].predict(x)[0])

            rows.append({
                "name": r["name"], "smiles": r["smiles"],
                "n_carbons": n, "branched": branched,
                "gap_ev": round(gap_pred, 4), "gap_source": "surrogate_predicted",
                "recovery": round(recovery(n, bool(branched), gap_pred), 4),
                "selectivity": round(selectivity(n, gap_pred), 4),
                "desirability": round(desirability(n, bool(branched), gap_pred), 4),
            })
    return rows


def main() -> None:
    surrogates = fit_surrogates()
    original = load_original_candidates()
    expanded = score_expanded_candidates(surrogates)

    all_rows = original + expanded
    all_rows.sort(key=lambda r: -r["desirability"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"{len(original)} исходных (real_dft) + {len(expanded)} расширенных "
          f"(surrogate_predicted) = {len(all_rows)} кандидатов ранжировано\n")

    print(f"{'name':<18}{'n':<4}{'br':<4}{'gap':<8}{'источник':<20}"
          f"{'recovery':<10}{'select.':<10}{'desirability'}")
    for r in all_rows[:15]:
        print(f"{r['name']:<18}{r['n_carbons']:<4}{r['branched']:<4}"
              f"{r['gap_ev']:<8.3f}{r['gap_source']:<20}"
              f"{r['recovery']:<10.3f}{r['selectivity']:<10.3f}{r['desirability']:.3f}")

    candidates_for_validation = sorted(
        (r for r in expanded
         if r["branched"] and r["n_carbons"] >= VALIDATION_MIN_N),
        key=lambda r: -r["desirability"],
    )[:VALIDATION_TOP_K]

    print(f"\nРекомендовано на реальный DFT (разветвлённые, C{VALIDATION_MIN_N}+, "
          f"топ-{VALIDATION_TOP_K} по surrogate-desirability):")
    for r in candidates_for_validation:
        print(f"  {r['name']:<18} {r['smiles']:<28} "
              f"desirability={r['desirability']:.3f} (gap предсказан суррогатом)")

    print(f"\nРезультаты: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
