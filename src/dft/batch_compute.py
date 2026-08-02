"""
Batch-прогон DFT-дескрипторов для всей библиотеки кандидатов.

Читает src.molecules.candidate_library.build_candidate_library(), для
каждого кандидата считает src.dft.psi4_descriptors.compute_dft_descriptors
и пишет результат в data/03_processed/dft_descriptors.csv -- построчно,
сразу после каждой молекулы, а не в конце.

Резюмируемость: при перезапуске имена, уже присутствующие в CSV,
пропускаются. Молекулы, упавшие с ошибкой, в CSV не попадают и поэтому
будут пересчитаны при следующем запуске автоматически -- специально
следить за "недоделанными" не нужно.

Запуск из корня проекта:
    python3 -m src.dft.batch_compute
"""
from __future__ import annotations

import csv
import time
import traceback
from pathlib import Path

import psi4

from src.molecules.candidate_library import build_candidate_library
from src.dft.psi4_descriptors import compute_dft_descriptors, DFTDescriptors

OUTPUT_CSV = Path("data/03_processed/dft_descriptors.csv")
FIELDNAMES = [
    "name", "smiles", "homo_ev", "lumo_ev", "gap_ev",
    "dipole_debye", "scf_energy_hartree", "elapsed_sec",
]


def already_done(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {row["name"] for row in csv.DictReader(f)}


def append_result(csv_path: Path, row: dict) -> None:
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def main() -> None:
    psi4.set_memory("2 GB")
    psi4.set_num_threads(4)

    library = build_candidate_library()
    done = already_done(OUTPUT_CSV)
    todo = [c for c in library if c.name not in done]

    print(f"Библиотека: {len(library)}, уже посчитано: {len(done)}, "
          f"осталось: {len(todo)}")

    failed = []
    batch_start = time.time()

    for i, candidate in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {candidate.name} ({candidate.smiles})...")
        t0 = time.time()
        try:
            result: DFTDescriptors = compute_dft_descriptors(
                candidate.name, candidate.smiles
            )
            elapsed = time.time() - t0
            append_result(OUTPUT_CSV, {
                "name": result.name,
                "smiles": candidate.smiles,
                "homo_ev": f"{result.homo_ev:.4f}",
                "lumo_ev": f"{result.lumo_ev:.4f}",
                "gap_ev": f"{result.gap_ev:.4f}",
                "dipole_debye": f"{result.dipole_debye:.4f}",
                "scf_energy_hartree": f"{result.scf_energy_hartree:.6f}",
                "elapsed_sec": f"{elapsed:.1f}",
            })
            print(f"  OK   HOMO={result.homo_ev:.4f}  LUMO={result.lumo_ev:.4f}  "
                  f"Gap={result.gap_ev:.4f} eV   ({elapsed:.1f} сек)")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  ОШИБКА после {elapsed:.1f} сек: {exc}")
            traceback.print_exc()
            failed.append(candidate.name)
        finally:
            # сброс scratch-состояния Psi4 между молекулами
            psi4.core.clean()

    total = time.time() - batch_start
    print(f"\n{'=' * 60}")
    print(f"Готово за {total / 60:.1f} мин. Успешно: {len(todo) - len(failed)}/{len(todo)}")
    if failed:
        print(f"Не сошлись: {failed}")
        print("Их нет в CSV -- повторный запуск пересчитает только их.")
    print(f"Результаты: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
