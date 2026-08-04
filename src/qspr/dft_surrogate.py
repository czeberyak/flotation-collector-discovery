"""
Суррогат DFT: предсказывает HOMO/gap из ТОЛЬКО RDKit-структурных
дескрипторов, без единого значения реального DFT на входе.

Нужен для расширенной библиотеки Недели 3: PLAN.md прямо говорит, что
новые кандидаты не считаются через настоящий Psi4 (дорого) -- только
через RDKit-дескрипторы + "QSPR-предсказание DFT-дескрипторов". Это
предсказание и есть данный модуль.

target.py.recovery()/selectivity() используют только gap_ev (не homo
напрямую) -- так что для скрининга критичен именно gap-суррогат;
homo_ev предсказывается тоже, для отчётности/дашборда, не потому что
нужен ниже по пайплайну.

Обучается и валидируется на тех же 15 реальных DFT-точках, тем же
leave-one-out CV, что и recovery/selectivity-модель в model.py.

Запуск из корня проекта:
    python3 -m src.qspr.dft_surrogate
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

# ТОЛЬКО RDKit-фичи -- ни одного DFT-значения на входе, иначе смысл
# суррогата (для кандидатов БЕЗ реального DFT) теряется
FEATURES = ["n_carbons", "branched", "branched_x_n", "mol_weight", "logp", "tpsa"]
TARGETS = ["homo_ev", "gap_ev"]


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
            }
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
        "model": model, "r2_in_sample": r2_fit, "r2_loo_cv": r2_cv,
        "coefficients": coefs, "intercept": ridge.intercept_,
        "y_pred_cv": y_pred_cv,
    }


def fit_surrogates(csv_path: Path = DATASET_CSV) -> dict:
    """
    Обучить оба суррогата (homo_ev, gap_ev) на реальных 15 точках,
    вернуть готовые к .predict() sklearn-пайплайны -- используется
    screening-скриптом для расширенной библиотеки.
    """
    names, X, y = load_dataset(csv_path)
    return {t: fit_and_evaluate(X, y[t])["model"] for t in TARGETS}


def main() -> None:
    names, X, y = load_dataset(DATASET_CSV)
    print(f"{len(names)} точек, {len(FEATURES)} RDKit-фич (без DFT на входе)\n")

    for target in TARGETS:
        result = fit_and_evaluate(X, y[target])
        print(f"=== {target} ===")
        print(f"R² (in-sample, оптимистично):  {result['r2_in_sample']:.3f}")
        print(f"R² (leave-one-out CV, честно): {result['r2_loo_cv']:.3f}")
        print("Коэффициенты:")
        for feat, coef in sorted(result["coefficients"].items(),
                                  key=lambda kv: -abs(kv[1])):
            print(f"  {feat:<14} {coef:+.3f}")

        errors = np.abs(y[target] - result["y_pred_cv"])
        worst = np.argsort(-errors)[:3]
        print("Худшие по |ошибка| в LOO CV:")
        for i in worst:
            print(f"  {names[i]:<14} факт={y[target][i]:.4f}  "
                  f"LOO={result['y_pred_cv'][i]:.4f}  |ошибка|={errors[i]:.4f}")
        print()


if __name__ == "__main__":
    main()
