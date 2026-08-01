import os
import sys
import time
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import psi4

# Добавляем корень проекта в PYTHONPATH для импорта модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.molecules.candidate_library import build_candidate_library
from src.dft.psi4_descriptors import compute_dft_descriptors


def run_batch():
    # Настройки ресурсов для батча
    psi4.set_memory("3 GB")
    psi4.set_num_threads(4)
    
    library = build_candidate_library()
    results = []
    
    print(f"Запуск батч-расчета для {len(library)} молекул...")
    print("-" * 60)
    
    for idx, cand in enumerate(library, 1):
        t0 = time.time()
        print(f"[{idx}/{len(library)}] {cand.name:<15}...", end=" ", flush=True)
        
        # 1. RDKit Дескрипторы (Стерика и Гидрофобность)
        mol = Chem.MolFromSmiles(cand.smiles)
        molwt = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        
        # 2. DFT Дескрипторы (Электроника)
        try:
            dft_res = compute_dft_descriptors(cand.name, cand.smiles)
            homo = dft_res.homo_ev
            lumo = dft_res.lumo_ev
            gap = dft_res.gap_ev
            dipole = dft_res.dipole_debye
            energy = dft_res.scf_energy_hartree
            status = "OK"
        except Exception as e:
            print(f"\n  -> ОШИБКА DFT: {e}")
            homo = lumo = gap = dipole = energy = None
            status = "FAIL"
            
        calc_time = time.time() - t0
        print(f"{status} ({calc_time:.1f} сек)")
        
        results.append({
            "name": cand.name,
            "smiles": cand.smiles,
            "family": cand.family,              # ИСПРАВЛЕНО: было mol_class
            "chain_length": cand.chain_length,  # ДОБАВЛЕНО: из вашей библиотеки
            "branched": cand.branched,          # ДОБАВЛЕНО: из вашей библиотеки
            "molwt": molwt,
            "logp": logp,
            "tpsa": tpsa,
            "rot_bonds": rot_bonds,
            "homo_ev": homo,
            "lumo_ev": lumo,
            "gap_ev": gap,
            "dipole_debye": dipole,
            "scf_energy_hartree": energy
        })
        
    # Сохраняем в папку interim согласно архитектуре проекта
    df = pd.DataFrame(results)
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/02_interim'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "molecular_descriptors.csv")
    
    df.to_csv(out_path, index=False)
    print("-" * 60)
    print(f"Батч-расчет завершен! Результаты сохранены в:\n{out_path}")


if __name__ == "__main__":
    run_batch()