"""
Синтетический литературно-обоснованный таргет: recovery-прокси,
selectivity-прокси, desirability (геометрическое среднее, тот же
принцип Харрингтона, что в flotation_reagent_opt).

recovery: растёт с длиной алкильной цепи с насыщением (см. PLAN.md,
раздел 2); разветвлённые изомеры короче ~C5 получают небольшой
бонус, длиннее -- штраф; мягкая положительная связь с электронной
плотностью через HOMO-LUMO gap (уже gap -> чуть выше recovery).

selectivity: падает с длиной цепи, растёт с шириной HOMO-LUMO gap
(уже gap -> реакционноспособнее -> менее селективная хемосорбция,
см. PLAN.md, неделя 2). Сигмоида -- гарантированно в (0, 1), в отличие
от первой черновой версии на линейной формуле (которая на methyl
вылезала за 1.0).

Коэффициенты подобраны эмпирически на реальных 15 DFT-точках, не
выведены теоретически -- см. data/03_processed/dft_descriptors.csv.
"""
from __future__ import annotations

import math

GAP_REF = 4.0   # верхняя граница gap в текущей библиотеке (methyl)
GAP_MIN = 2.4   # нижняя граница (n-nonyl)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def recovery(n_carbons: int, branched: bool, gap_ev: float) -> float:
    base = 1.0 - math.exp(-n_carbons / 4.0)
    branch_adj = 1.0 + 0.15 * (5 - n_carbons) / 5.0 if branched else 1.0
    gap_nudge = 1.0 + 0.05 * (GAP_REF - gap_ev) / GAP_REF
    return min(max(base * branch_adj * gap_nudge, 0.0), 1.0)


def selectivity(n_carbons: int, gap_ev: float) -> float:
    x = 1.1 - 0.35 * (n_carbons - 1) + 0.9 * (gap_ev - GAP_MIN)
    return _sigmoid(x)


def desirability(n_carbons: int, branched: bool, gap_ev: float) -> float:
    r = recovery(n_carbons, branched, gap_ev)
    s = selectivity(n_carbons, gap_ev)
    return (r * s) ** 0.5


if __name__ == "__main__":
    _demo = [
        ("methyl", 1, False, 4.0140), ("n-amyl", 5, False, 2.9272),
        ("n-decyl", 10, False, 2.4635), ("isobutyl", 4, True, 3.2870),
        ("2-ethylhexyl", 8, True, 2.9724),
    ]
    print(f"{'name':<14}{'recovery':<10}{'selectivity':<12}{'desirability'}")
    for name, n, br, gap in _demo:
        print(f"{name:<14}{recovery(n, br, gap):<10.3f}"
              f"{selectivity(n, gap):<12.3f}{desirability(n, br, gap):.3f}")
