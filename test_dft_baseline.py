import time
from rdkit import Chem
from rdkit.Chem import AllChem
import psi4

# 1. Настройка окружения Psi4
# Выделяем ресурсы. Если у вас мощный процессор, можете увеличить threads до 8
psi4.set_num_threads(4)
psi4.set_memory('4 GB')
psi4.core.set_output_file('psi4_baseline.log', False)

# Контрольная группа из PLAN.md
test_mols = {
    'ethyl': 'CCOC(=S)[S-]',
    'isobutyl': 'CC(C)COC(=S)[S-]',
    'n-amyl': 'CCCCCOC(=S)[S-]'
}

hartree_to_ev = 27.2114
level_of_theory = 'b3lyp/6-31+g*'

print(f"Запуск DFT скрининга ({level_of_theory})...")
print(f"{'Name':<12} {'HOMO (eV)':<12} {'LUMO (eV)':<12} {'Gap (eV)':<10} {'Time (s)'}")
print("-" * 60)

results = {}

for name, smiles in test_mols.items():
    t0 = time.time()
    
    # 2. RDKit: Генерация 3D-структуры
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    
    # 3. Подготовка XYZ для Psi4
    conf = mol.GetConformer()
    # КРИТИЧНО: Заряд -1, Мультиплетность 1 (анион)
    xyz_string = "-1 1\n" 
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        xyz_string += f"{atom.GetSymbol()} {pos.x:.5f} {pos.y:.5f} {pos.z:.5f}\n"
    
    psi4_mol = psi4.geometry(xyz_string)
    
    # 4. DFT Оптимизация и расчет энергии
    # В отличие от прошлого примера, мы добавляем шаг оптимизации геометрии, 
    # так как UFF из RDKit недостаточно точен для квантовой химии
    try:
        # Сначала оптимизируем геометрию
        psi4.optimize(level_of_theory, molecule=psi4_mol)
        # Затем считаем свойства
        energy, wfn = psi4.energy(level_of_theory, molecule=psi4_mol, return_wfn=True)
        
        # 5. Извлечение дескрипторов
        lumo_idx = wfn.nalpha()
        homo_idx = lumo_idx - 1
        
        homo_ev = wfn.epsilon_a_subset("AO", "ALL").np[homo_idx] * hartree_to_ev
        lumo_ev = wfn.epsilon_a_subset("AO", "ALL").np[lumo_idx] * hartree_to_ev
        gap = lumo_ev - homo_ev
        
        calc_time = time.time() - t0
        print(f"{name:<12} {homo_ev:<12.4f} {lumo_ev:<12.4f} {gap:<10.4f} {calc_time:.1f}")
        
        results[name] = gap
        
    except Exception as e:
        print(f"{name:<12} ОШИБКА РАСЧЕТА: {e}")

print("-" * 60)
print("Проверка ранжирования (Amyl > Isobutyl > Ethyl):")
# Сортируем по убыванию Gap (или HOMO, в зависимости от того, что сильнее коррелирует с литературой)
sorted_mols = sorted(results.items(), key=lambda item: item[1])  # меньше gap — эффективнее
for rank, (name, gap_val) in enumerate(sorted_mols, 1):
    print(f"{rank}. {name} (Gap: {gap_val:.4f} eV)")