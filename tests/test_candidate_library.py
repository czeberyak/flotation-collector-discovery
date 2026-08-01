"""
Unit-тесты для src.molecules.candidate_library.build_candidate_library().

Запуск из корня проекта:
    python3 -m unittest tests.test_candidate_library -v

Тесты, помеченные RDKit, автоматически пропускаются, если пакет rdkit
не установлен в текущем окружении (пропуск, а не падение — так тесты
можно гонять и вне conda-окружения flotation-collectors).
"""
import re
import unittest

from src.molecules.candidate_library import build_candidate_library

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# Ожидаемая длина алкильной цепи (число атомов C в самой цепи,
# без учёта тиокарбонатного углерода) — взято из зафиксированного
# вывода build_candidate_library() при составлении плана проекта.
EXPECTED_CHAIN_LENGTH = {
    "methyl": 1, "ethyl": 2, "n-propyl": 3, "n-butyl": 4, "n-amyl": 5,
    "n-hexyl": 6, "n-heptyl": 7, "n-octyl": 8, "n-nonyl": 9, "n-decyl": 10,
    "isopropyl": 3, "isobutyl": 4, "sec-butyl": 4, "isoamyl": 5,
    "2-ethylhexyl": 8,
}
STRAIGHT_CHAIN_NAMES = {
    "methyl", "ethyl", "n-propyl", "n-butyl", "n-amyl",
    "n-hexyl", "n-heptyl", "n-octyl", "n-nonyl", "n-decyl",
}
BRANCHED_NAMES = {"isopropyl", "isobutyl", "sec-butyl", "isoamyl", "2-ethylhexyl"}

# Ксантогенатная функциональная группа: O-C(=S)-S(-)
XANTHATE_SUFFIX = "OC(=S)[S-]"


class TestCandidateLibrary(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lib = build_candidate_library()

    def test_01_library_has_15_candidates(self):
        self.assertEqual(len(self.lib), 15)

    def test_02_names_match_expected_set(self):
        names = {c.name for c in self.lib}
        self.assertEqual(names, STRAIGHT_CHAIN_NAMES | BRANCHED_NAMES)

    def test_03_names_are_unique(self):
        names = [c.name for c in self.lib]
        self.assertEqual(len(names), len(set(names)), "есть повторяющиеся имена")

    def test_04_smiles_are_unique(self):
        smiles = [c.smiles for c in self.lib]
        self.assertEqual(len(smiles), len(set(smiles)), "есть повторяющиеся SMILES")

    def test_05_all_smiles_end_in_xanthate_group(self):
        for c in self.lib:
            self.assertTrue(
                c.smiles.endswith(XANTHATE_SUFFIX),
                f"{c.name}: SMILES не заканчивается ксантогенатной группой: {c.smiles}",
            )

    def test_06_alkyl_chain_carbon_count_matches_name(self):
        # Число атомов C в SMILES минус 1 (углерод самой группы C(=S))
        # должно совпадать с длиной алкильной цепи по имени.
        # Валидно, пока в библиотеке нет ароматики/Cl/др. C-содержащих
        # групп кроме простой алифатической цепи + ксантогенат.
        for c in self.lib:
            expected = EXPECTED_CHAIN_LENGTH[c.name]
            n_carbons_total = len(re.findall(r"C", c.smiles))
            n_alkyl_carbons = n_carbons_total - 1
            self.assertEqual(
                n_alkyl_carbons, expected,
                f"{c.name}: ожидалось {expected} атомов C в цепи, "
                f"по SMILES получилось {n_alkyl_carbons} ({c.smiles})",
            )

    def test_07_ten_straight_chain_and_five_branched(self):
        straight = [c for c in self.lib if c.name in STRAIGHT_CHAIN_NAMES]
        branched = [c for c in self.lib if c.name in BRANCHED_NAMES]
        self.assertEqual(len(straight), 10)
        self.assertEqual(len(branched), 5)

    def test_08_smiles_are_nonempty_strings(self):
        for c in self.lib:
            self.assertIsInstance(c.smiles, str)
            self.assertGreater(len(c.smiles), 0)

    @unittest.skipUnless(HAS_RDKIT, "RDKit не установлен в этом окружении")
    def test_09_all_smiles_valid_in_rdkit(self):
        for c in self.lib:
            mol = Chem.MolFromSmiles(c.smiles)
            self.assertIsNotNone(
                mol, f"{c.name}: RDKit не смог распарсить SMILES {c.smiles}"
            )

    @unittest.skipUnless(HAS_RDKIT, "RDKit не установлен в этом окружении")
    def test_10_canonical_smiles_are_unique_in_rdkit(self):
        canonical = set()
        for c in self.lib:
            mol = Chem.MolFromSmiles(c.smiles)
            canonical.add(Chem.MolToSmiles(mol))
        self.assertEqual(
            len(canonical), len(self.lib),
            "после канонизации RDKit нашлись дублирующиеся структуры",
        )


if __name__ == "__main__":
    unittest.main()
