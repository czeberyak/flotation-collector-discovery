"""
Featurization: RDKit структурные дескрипторы + объединение с уже
посчитанными DFT-дескрипторами (data/03_processed/dft_descriptors.csv)
+ применение синтетического таргета (src.qspr.target).

Длина алкильной цепи и признак "разветвлена" считаются из САМОЙ
структуры молекулы через RDKit (не из имени-строки) -- корректно
работает и на расширенной библиотеке Недели 3, которую этот код
никогда раньше не видел.

Запуск из корня проекта:
    python3 -m src.qspr.featurize
"""
from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors

from src.molecules.candidate_library import build_candidate_library
from src.qspr.target import recovery, selectivity, desirability

DFT_CSV = Path("data/03_processed/dft_descriptors.csv")
OUTPUT_CSV = Path("data/03_processed/qspr_dataset.csv")


def _find_alpha_carbon(mol):
    """
    Найти алкильный углерод, присоединённый к эфирному O ксантогенатной
    головы -- точку, с которой физически начинается алкильная цепь.
    """
    thio_carbon = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE:
                thio_carbon = atom
                break
        if thio_carbon is not None:
            break
    if thio_carbon is None:
        raise ValueError("не нашли тиокарбонатный C(=S) -- не ксантогенат?")

    ether_o = None
    for bond in thio_carbon.GetBonds():
        other = bond.GetOtherAtom(thio_carbon)
        if other.GetSymbol() == "O":
            ether_o = other
            break
    if ether_o is None:
        raise ValueError("не нашли эфирный O у тиокарбонатного углерода")

    for bond in ether_o.GetBonds():
        other = bond.GetOtherAtom(ether_o)
        if other.GetIdx() != thio_carbon.GetIdx():
            return other  # alpha-углерод алкильной цепи
    raise ValueError("эфирный O не присоединён к алкильной цепи")


def branch_distance_from_root(mol) -> int | None:
    """
    Расстояние (число связей) от корня (alpha-углерод, у O) до
    БЛИЖАЙШЕЙ точки ветвления. None, если разветвлений нет.

    0 -- ветка прямо в корне (isopropyl-тип). 1 -- ветка на первом шаге
    вглубь (isobutyl-тип). И т.д. BFS, не DFS -- гарантирует, что при
    нескольких развилках находится САМАЯ БЛИЖНЯЯ, а не первая попавшаяся
    в порядке обхода.

    Отдельная фича, не просто is_branched_alkyl: молекулярная масса,
    LogP (Crippen) и TPSA -- все локально-аддитивные по типам атомов и
    НЕ различают, где именно на цепи сидит развилка (см. README,
    известное ограничение скрининга Недели 3). Эта фича даёт то самое
    разрешение, которого не хватало.
    """
    alpha = _find_alpha_carbon(mol)

    def carbon_neighbors(atom):
        return [n for n in atom.GetNeighbors() if n.GetSymbol() == "C"]

    if len(carbon_neighbors(alpha)) >= 2:
        return 0

    visited = {alpha.GetIdx()}
    queue = deque((n, alpha, 1) for n in carbon_neighbors(alpha))
    while queue:
        atom, parent, dist = queue.popleft()
        if atom.GetIdx() in visited:
            continue
        visited.add(atom.GetIdx())
        children = [n for n in carbon_neighbors(atom) if n.GetIdx() != parent.GetIdx()]
        if len(children) >= 2:
            return dist
        queue.extend((c, atom, dist + 1) for c in children)
    return None


def is_branched_alkyl(mol) -> bool:
    """Разветвлена ли алкильная цепь -- см. branch_distance_from_root
    для деталей алгоритма и обоснования BFS."""
    return branch_distance_from_root(mol) is not None


def alkyl_chain_length(mol) -> int:
    """Число атомов C в алкильной группе (без тиокарбонатного C)."""
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C") - 1


def rdkit_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"невалидный SMILES: {smiles}"
    distance = branch_distance_from_root(mol)
    return {
        "n_carbons": alkyl_chain_length(mol),
        "branched": distance is not None,
        "branch_distance": distance if distance is not None else -1,  # -1 = нет ветки;
                                                                         # вне диапазона 0..n,
                                                                         # не путается с "ветка в корне"=0
        "mol_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
    }


def load_dft_descriptors(csv_path: Path) -> dict:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def main() -> None:
    library = build_candidate_library()
    dft = load_dft_descriptors(DFT_CSV)

    missing = [c.name for c in library if c.name not in dft]
    if missing:
        raise RuntimeError(
            f"нет DFT-дескрипторов для: {missing} -- сначала прогнать "
            f"python3 -m src.dft.batch_compute"
        )

    rows = []
    for c in library:
        rd = rdkit_descriptors(c.smiles)
        d = dft[c.name]
        gap = float(d["gap_ev"])
        n = rd["n_carbons"]
        branched = rd["branched"]

        rows.append({
            "name": c.name,
            "smiles": c.smiles,
            "n_carbons": n,
            "branched": int(branched),
            "branch_distance": rd["branch_distance"],
            "mol_weight": round(rd["mol_weight"], 3),
            "logp": round(rd["logp"], 3),
            "tpsa": round(rd["tpsa"], 3),
            "homo_ev": d["homo_ev"],
            "lumo_ev": d["lumo_ev"],
            "gap_ev": d["gap_ev"],
            "dipole_debye": d["dipole_debye"],  # оставлен в CSV для справки,
                                                  # но НЕ используется как фича
                                                  # в model.py -- см. README
            "recovery": round(recovery(n, branched, gap), 4),
            "selectivity": round(selectivity(n, gap), 4),
            "desirability": round(desirability(n, branched, gap), 4),
        })

    fieldnames = list(rows[0].keys())
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Записано {len(rows)} строк в {OUTPUT_CSV}\n")
    print(f"{'name':<14}{'n':<4}{'branch':<8}{'recovery':<10}"
          f"{'selectivity':<12}{'desirability'}")
    for r in rows:
        print(f"{r['name']:<14}{r['n_carbons']:<4}"
              f"{'yes' if r['branched'] else '':<8}{r['recovery']:<10.3f}"
              f"{r['selectivity']:<12.3f}{r['desirability']:.3f}")


if __name__ == "__main__":
    main()
