"""
DFT-дескрипторы молекул-кандидатов через Psi4.

Уровень теории по умолчанию изменен на B3LYP/6-31+G*, так как добавление 
диффузных функций (+) критически важно для корректной оптимизации 
геометрии и расчета орбиталей заряженных частиц (анионов ксантогенатов).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import psi4
from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class DFTDescriptors:
    name: str
    homo_ev: float
    lumo_ev: float
    gap_ev: float
    dipole_debye: float
    scf_energy_hartree: float


def smiles_to_xyz_block(smiles: str, seed: int = 42, num_confs: int = 10) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"невалидный SMILES: {smiles}"
    mol = Chem.AddHs(mol)

    conf_ids = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed,
        useExpTorsionAnglePrefs=True, useBasicKnowledge=True,
    )
    if not conf_ids:
        raise RuntimeError(f"RDKit не смог сгенерировать конформер: {smiles}")

    best_cid, best_energy = None, float("inf")
    for cid in conf_ids:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        ff.Minimize()
        e = ff.CalcEnergy()
        if e < best_energy:
            best_cid, best_energy = cid, e

    conf = mol.GetConformer(best_cid)
    lines = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "-1 1\n" + "\n".join(lines)


def compute_dft_descriptors(name: str, smiles: str,
                             functional: str = "b3lyp",
                             basis: str = "6-31+G*") -> DFTDescriptors:
    """
    Выполняет квантово-химическую оптимизацию геометрии и рассчитывает 
    дескрипторы граничных орбиталей (HOMO/LUMO) и дипольный момент.
    """
    import os
    os.makedirs("reports/psi4_logs", exist_ok=True)
    
    psi4.core.set_output_file(f"reports/psi4_logs/{name}.log", False)

    geom_block = smiles_to_xyz_block(smiles)
    mol = psi4.geometry(geom_block)

    psi4.set_options({"reference": "rhf"})  
    level_of_theory = f"{functional}/{basis}"

    # Оптимизация геометрии + расчет энергии
    scf_energy, wfn = psi4.optimize(
        level_of_theory, molecule=mol, return_wfn=True
    )

    # Извлечение энергий орбиталей
    eps_a = wfn.epsilon_a_subset("AO", "ALL").np
    n_occ = wfn.nalpha()
    
    homo = eps_a[n_occ - 1]
    lumo = eps_a[n_occ]
    HARTREE_TO_EV = 27.2114

    # Извлечение дипольного момента для Psi4 >= 1.6
    # Возвращает массив [x, y, z] в атомных единицах (a.u. / e-bohr)
    dipole_au = psi4.variable("SCF DIPOLE")
    # Считаем длину вектора и переводим в Дебаи (1 a.u. = 2.541746 Debye)
    dipole_debye = float(sum(d**2 for d in dipole_au) ** 0.5) * 2.541746

    return DFTDescriptors(
        name=name,
        homo_ev=homo * HARTREE_TO_EV,
        lumo_ev=lumo * HARTREE_TO_EV,
        gap_ev=(lumo - homo) * HARTREE_TO_EV,
        dipole_debye=dipole_debye,
        scf_energy_hartree=scf_energy,
    )


if __name__ == "__main__":
    # Локальный тест модуля на этилксантогенате
    psi4.set_memory("2 GB")
    psi4.set_num_threads(4)
    
    test_name = "ethyl"
    test_smiles = "CCOC(=S)[S-]"
    
    print(f"Запуск тестового расчета для {test_name} ({test_smiles})...")
    result = compute_dft_descriptors(test_name, test_smiles)
    
    print("\n--- Результаты ---")
    print(f"HOMO:   {result.homo_ev:.4f} eV")
    print(f"LUMO:   {result.lumo_ev:.4f} eV")
    print(f"Gap:    {result.gap_ev:.4f} eV")
    print(f"Dipole: {result.dipole_debye:.4f} Debye")
    print(f"Energy: {result.scf_energy_hartree:.4f} Hartree")