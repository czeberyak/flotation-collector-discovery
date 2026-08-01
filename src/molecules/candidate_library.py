"""
Генератор виртуальной библиотеки алкилксантогенатов (xanthate collectors) --
кандидатов-собирателей для сульфидной флотации.

Обоснование выбора этого класса молекул (не произвольное):
  - Ксантогенаты -- самый распространённый и наиболее изученный класс
    собирателей для сульфидной флотации (галенит, халькопирит, пирит и т.д.).
  - Структурно-функциональная зависимость хорошо задокументирована в
    литературе: собирающая способность растёт с длиной алкильной цепи,
    но селективность падает; существует оптимальная длина цепи (баланс
    силы адсорбции и растворимости); разветвлённые изомеры короче ~C5
    эффективнее прямоцепочечных аналогов, длиннее -- наоборот.
  - Есть прямой литературный прецедент использования DFT-дескрипторов
    (энергии граничных орбиталей HOMO/LUMO) для предсказания относительной
    эффективности флотации у этила/изобутила/амила ксантогената -- то есть
    сам подход "DFT-дескрипторы -> QSPR" для ЭТОГО класса молекул уже
    валидирован в открытой литературе, это не хайпотеза с нуля.

СМІLES строятся строковыми шаблонами (без RDKit) -- в этой песочнице RDKit
недоступен офлайн для тестирования. Первый шаг реального пайплайна (после
установки RDKit в рабочем окружении) -- ОБЯЗАТЕЛЬНО валидировать каждую
строку через Chem.MolFromSmiles(s) is not None, прежде чем считать её
валидной структурой. Ниже это отмечено TODO явно.

Форма аниона: R-O-C(=S)-[S-] (депротонированная ксантогеновая кислота,
физиологически релевантная форма в щелочной пульпе флотации).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candidate:
    name: str
    smiles: str
    chain_length: int  # число атомов углерода в алкильном радикале R
    branched: bool
    family: str = "xanthate"


# Прямоцепочечные алкильные радикалы C1..C10 (SMILES-фрагмент перед O)
_STRAIGHT_CHAIN_NAMES = {
    1: "methyl", 2: "ethyl", 3: "n-propyl", 4: "n-butyl", 5: "n-amyl",
    6: "n-hexyl", 7: "n-heptyl", 8: "n-octyl", 9: "n-nonyl", 10: "n-decyl",
}

# Классические промышленно значимые разветвлённые ксантогенаты
# (реальные, именованные продукты, а не произвольные комбинаторные изомеры)
_BRANCHED_ISOMERS = {
    "isopropyl": ("CC(C)", 3),
    "isobutyl": ("CC(C)C", 4),
    "sec-butyl": ("CCC(C)", 4),
    "isoamyl": ("CC(C)CC", 5),
    "2-ethylhexyl": ("CCCCC(CC)C", 8),
}


def _straight_chain_smiles(n: int) -> str:
    """SMILES алкильного радикала прямой цепи из n атомов углерода."""
    return "C" * n


def build_candidate_library(max_chain: int = 10) -> list[Candidate]:
    """
    Собирает виртуальную библиотеку: прямоцепочечные C1..max_chain +
    классические разветвлённые изомеры.
    """
    candidates: list[Candidate] = []

    for n in range(1, max_chain + 1):
        alkyl = _straight_chain_smiles(n)
        smiles = f"{alkyl}OC(=S)[S-]"
        name = _STRAIGHT_CHAIN_NAMES.get(n, f"C{n}-straight")
        candidates.append(Candidate(name=name, smiles=smiles, chain_length=n, branched=False))

    for name, (alkyl_smiles, n) in _BRANCHED_ISOMERS.items():
        smiles = f"{alkyl_smiles}OC(=S)[S-]"
        candidates.append(Candidate(name=name, smiles=smiles, chain_length=n, branched=True))

    return candidates


def to_dataframe_records(candidates: list[Candidate]) -> list[dict]:
    return [
        {
            "name": c.name,
            "smiles": c.smiles,
            "chain_length": c.chain_length,
            "branched": c.branched,
            "family": c.family,
        }
        for c in candidates
    ]


# --------------------------------------------------------------------- #
# TODO (первый шаг реального пайплайна, требует RDKit -- недоступен в
# этой песочнице для тестирования, проверить в рабочем окружении):
#
#   from rdkit import Chem
#   for c in candidates:
#       mol = Chem.MolFromSmiles(c.smiles)
#       assert mol is not None, f"невалидный SMILES: {c.name} -> {c.smiles}"
#       c.smiles = Chem.MolToSmiles(mol)  # канонизация
#
# Ниже -- self-check БЕЗ RDKit: проверяет только структуру данных
# (уникальность имён/SMILES, разумный диапазон длины цепи), не химическую
# валидность SMILES как таковую.
# --------------------------------------------------------------------- #

def self_check(candidates: list[Candidate]) -> None:
    names = [c.name for c in candidates]
    smiles_list = [c.smiles for c in candidates]
    assert len(names) == len(set(names)), "дублирующиеся имена кандидатов"
    assert len(smiles_list) == len(set(smiles_list)), "дублирующиеся SMILES"
    assert all(1 <= c.chain_length <= 20 for c in candidates), "аномальная длина цепи"
    assert any(c.branched for c in candidates), "нет ни одного разветвлённого изомера"
    assert any(not c.branched for c in candidates), "нет ни одного прямоцепочечного изомера"


if __name__ == "__main__":
    lib = build_candidate_library(max_chain=10)
    self_check(lib)
    print(f"Библиотека кандидатов: {len(lib)} структур\n")
    for c in lib:
        tag = "разветв." if c.branched else "прямая "
        print(f"  {c.name:14s} [{tag}, C{c.chain_length}]  {c.smiles}")
    print("\nSelf-check (без RDKit) пройден.")
    print("TODO перед следующим шагом: провалидировать все SMILES через")
    print("rdkit.Chem.MolFromSmiles в рабочем окружении с установленным RDKit.")
