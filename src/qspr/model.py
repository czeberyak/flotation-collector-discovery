"""
QSPR-модель: ridge-регрессия recovery/selectivity от дескрипторов.

15 точек, до 8 признаков -- полагаться на in-sample R² было бы
обманчиво (переобучение почти гарантировано), поэтому основная
метрика -- leave-one-out CV, а не просто model.fit().score().

dipole_debye сознательно исключён из признаков: по итогам батч-расчёта
Недели 1 он выглядит как origin-dependent артефакт для заряженной
частицы (растёт с длиной цепи в 5 раз, что физически неправдоподобно
для дипольного момента) -- см. README, раздел "Известное ограничение".
HOMO/LUMO/gap этой проблемы не имеют, остаются в фичах.

Запуск из корня проекта:
    python3 -m src.qspr.model
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASET_CSV = Path("data/03_processed/qspr_dataset.csv")

FEATURES = [
    "n_carbons", "branched", "branched_x_n", "mol_weight", "logp", "tpsa",
    "homo_ev", "gap_ev",
]
TARGETS = ["recovery", "selectivity"]

# lumo_ev сознательно исключён вместе с dipole_debye: gap = lumo - homo
# точное тождество, а не эмпирическая корреляция -- втроём homo/lumo/gap
# несут только 2 степени свободы и произвольно делят между собой вес.
# branched_x_n -- interaction term: без него линейная модель не может
# выразить branching-эффект, меняющий знак в зависимости от длины цепи
# (короткие разветвлённые лучше прямых аналогов, длинные -- хуже).


def load_dataset(csv_path: Path):
    names, X, y = [], [], {t: [] for t in TARGETS}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["name"])
            n = float(row["n_carbons"])
            branched = float(row["branched"])
            feat_values = {
                "n_carbons": n,
                "branched": branched,
                "branched_x_n": branched * n,
                "mol_weight": float(row["mol_weight"]),
                "logp": float(row["logp"]),
                "tpsa": float(row["tpsa"]),
                "homo_ev": float(row["homo_ev"]),
                "gap_ev": float(row["gap_ev"]),
            }
            names_ok = set(feat_values) == set(FEATURES)
            assert names_ok, f"FEATURES и feat_values разошлись: {set(FEATURES) ^ set(feat_values)}"
            X.append([feat_values[f] for f in FEATURES])
            for t in TARGETS:
                y[t].append(float(row[t]))
    return names, np.array(X), {t: np.array(v) for t, v in y.items()}


def fit_and_evaluate(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    loo = LeaveOneOut()
    y_pred_cv = cross_val_predict(model, X, y, cv=loo)
    r2_cv = r2_score(y, y_pred_cv)

    model.fit(X, y)
    r2_fit = r2_score(y, model.predict(X))

    ridge = model.named_steps["ridge"]
    coefs = dict(zip(FEATURES, ridge.coef_))

    return {
        "r2_in_sample": r2_fit,
        "r2_loo_cv": r2_cv,
        "coefficients": coefs,
        "intercept": ridge.intercept_,
        "model": model,
        "y_pred_cv": y_pred_cv,
    }


def main() -> None:
    names, X, y = load_dataset(DATASET_CSV)

    print(f"{len(names)} кандидатов, {len(FEATURES)} признаков "
          f"(dipole_debye исключён -- origin-dependent артефакт)\n")

    results = {}
    for target in TARGETS:
        result = fit_and_evaluate(X, y[target])
        results[target] = result

        print(f"=== {target} ===")
        print(f"R² (in-sample, оптимистично):  {result['r2_in_sample']:.3f}")
        print(f"R² (leave-one-out CV, честно): {result['r2_loo_cv']:.3f}")
        print("Коэффициенты (после стандартизации, сравнимы по величине):")
        for feat, coef in sorted(result["coefficients"].items(),
                                  key=lambda kv: -abs(kv[1])):
            print(f"  {feat:<12} {coef:+.3f}")

        print("\nХудшие по |ошибка| в LOO CV (кандидаты на разбор):")
        errors = np.abs(y[target] - result["y_pred_cv"])
        worst = np.argsort(-errors)[:3]
        for i in worst:
            print(f"  {names[i]:<14} факт={y[target][i]:.3f}  "
                  f"LOO-предсказание={result['y_pred_cv'][i]:.3f}  "
                  f"|ошибка|={errors[i]:.3f}")
        print()


if __name__ == "__main__":
    main()
