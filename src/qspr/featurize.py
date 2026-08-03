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


def is_branched_alkyl(mol) -> bool:
    """
    Разветвлена ли алкильная цепь (не считая тиокарбонатного C).

    Правило: у корня (alpha-углерод, присоединённый к O) любая
    углерод-углеродная связь уже означает развилку -- в отличие от
    всех остальных углеродов цепи, у корня нет связи "назад". У
    любого не-корневого узла развилка -- это 2+ дочерних углерода
    (то есть carbon-degree >= 3, если считать и связь к родителю).
    Даёт корректный ответ и для isopropyl (развилка в корне), и для
    isobutyl/isoamyl/2-ethylhexyl (развилка на 1-2 шага вглубь).
    """
    alpha = _find_alpha_carbon(mol)

    def carbon_neighbors(atom):
        return [n for n in atom.GetNeighbors() if n.GetSymbol() == "C"]

    if len(carbon_neighbors(alpha)) >= 2:
        return True

    visited = {alpha.GetIdx()}
    stack = [(n, alpha) for n in carbon_neighbors(alpha)]
    while stack:
        atom, parent = stack.pop()
        if atom.GetIdx() in visited:
            continue
        visited.add(atom.GetIdx())
        children = [n for n in carbon_neighbors(atom) if n.GetIdx() != parent.GetIdx()]
        if len(children) >= 2:
            return True
        stack.extend((c, atom) for c in children)
    return False


def alkyl_chain_length(mol) -> int:
    """Число атомов C в алкильной группе (без тиокарбонатного C)."""
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C") - 1


def rdkit_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"невалидный SMILES: {smiles}"
    return {
        "n_carbons": alkyl_chain_length(mol),
        "branched": is_branched_alkyl(mol),
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
