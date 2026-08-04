"""
Расширенная виртуальная библиотека для скрининга Недели 3.

Систематически генерирует: прямые цепи C1-C12, плюс одноветочные
разветвлённые изомеры C4-C12 (метильная и этильная ветка на каждой
допустимой позиции главной цепи). Это НЕ исчерпывающий перебор
абсолютно всех структурных изомеров (для C10-C12 их сотни, задача
комбинаторно тяжелее) -- систематическая, но частичная выборка,
покрывающая основные паттерны разветвления на каждой длине цепи.
Прозрачно об этом ограничении, а не притворяется полным перебором.

Кандидаты этой библиотеки НЕ считаются через настоящий Psi4 (дорого,
см. PLAN.md неделя 3) -- их DFT-дескрипторы предсказываются через
src.qspr.dft_surrogate, обученный на 15 реальных DFT-точках.

Дедупликация -- через RDKit-канонизацию (одна и та же структура,
сгенерированная разными путями, схлопывается в одну запись).

Запуск из корня проекта:
    python3 -m src.molecules.expanded_library
"""
from __future__ import annotations

import csv
from pathlib import Path

from rdkit import Chem

from src.molecules.candidate_library import build_candidate_library

OUTPUT_CSV = Path("data/02_interim/expanded_candidates.csv")

XANTHATE_SUFFIX = "OC(=S)[S-]"  # та же ксантогенатная голова, что и в
                                 # src.molecules.candidate_library

MIN_TOTAL_N = 4    # разветвление имеет смысл начиная примерно с C4
MAX_TOTAL_N = 12   # дальше вручную не считали DFT ни на одном примере
                    # исходной библиотеки -- суррогат экстраполирует
                    # только за счёт RDKit-фич, но осторожно за C10-C12
                    # не заходим, пока не появится реальных точек в этом
                    # диапазоне
BRANCH_LENGTHS = (1, 2)  # метил, этил


def _straight_chain(n: int) -> str:
    return "C" * n


def _insert_branch(main_chain: str, position: int, branch: str) -> str:
    """position: 1-индексация с конца, присоединяемого к O (root=1)."""
    idx = position - 1
    return main_chain[:idx + 1] + f"({branch})" + main_chain[idx + 1:]


def _single_branch_variants(total_n: int, branch_len: int) -> list[str]:
    """
    Одна ветка длины branch_len на каждой позиции 1..(M-1) главной
    цепи длины M = total_n - branch_len. Позиция M (последний атом
    цепи) не используется -- ветка там эквивалентна просто более
    длинной прямой цепи, не новой структуре.
    """
    main_len = total_n - branch_len
    if main_len < 2:
        return []
    branch = "C" * branch_len
    main = _straight_chain(main_len)
    return [_insert_branch(main, pos, branch) for pos in range(1, main_len)]


def generate_alkyl_fragments() -> set[str]:
    """Все SMILES-фрагменты алкильной цепи (без ксантогенатной головы),
    возможны дубликаты между разными (total_n, branch_len) -- убираются
    на уровне множества строк здесь, но НЕ на уровне химической
    эквивалентности (для этого нужна RDKit-канонизация, см. main())."""
    fragments = set()
    for n in range(1, MAX_TOTAL_N + 1):
        fragments.add(_straight_chain(n))
    for total_n in range(MIN_TOTAL_N, MAX_TOTAL_N + 1):
        for branch_len in BRANCH_LENGTHS:
            fragments.update(_single_branch_variants(total_n, branch_len))
    return fragments


def build_xanthate_smiles(alkyl_fragment: str) -> str:
    return alkyl_fragment + XANTHATE_SUFFIX


def _original_canonical_smiles() -> set[str]:
    """Канонические SMILES исходных 15 -- исключить их из расширенной
    библиотеки: не имеет смысла дублировать структуру с предсказанным
    gap, если для неё уже есть настоящий DFT."""
    canon = set()
    for c in build_candidate_library():
        mol = Chem.MolFromSmiles(c.smiles)
        if mol is not None:
            canon.add(Chem.MolToSmiles(mol))
    return canon


def main() -> None:
    fragments = generate_alkyl_fragments()
    print(f"Сгенерировано алкильных фрагментов (до RDKit-дедупликации): {len(fragments)}")

    original = _original_canonical_smiles()
    seen_canonical: dict[str, str] = {}  # canonical_smiles -> первый исходный SMILES
    invalid = 0
    already_original = 0
    for frag in fragments:
        smiles = build_xanthate_smiles(frag)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid += 1
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in original:
            already_original += 1
            continue
        seen_canonical.setdefault(canonical, smiles)

    if invalid:
        print(f"Невалидных SMILES (отброшены): {invalid}")
    if already_original:
        print(f"Уже есть в исходных 15 с настоящим DFT (исключены отсюда): {already_original}")

    rows = []
    for i, canonical in enumerate(sorted(seen_canonical), start=1):
        n_carbons = sum(1 for atom in Chem.MolFromSmiles(canonical).GetAtoms()
                         if atom.GetSymbol() == "C") - 1
        rows.append({
            "name": f"gen_{i:03d}_C{n_carbons}",
            "smiles": canonical,
            "n_carbons": n_carbons,
        })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "smiles", "n_carbons"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"После RDKit-дедупликации: {len(rows)} уникальных структур")
    print(f"Записано в {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
