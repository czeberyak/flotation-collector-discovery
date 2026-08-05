"""
Streamlit-дашборд: библиотека кандидатов, DFT-дескрипторы, QSPR-
диагностика (R², коэффициенты), ранжированный скрининг.

Тонкий слой поверх уже готового пайплайна -- не пересчитывает и не
дублирует логику src/qspr/, src/dft/, src/screening/, только читает
их CSV-результаты и по требованию переобучает модели для живой
диагностики (дёшево: ridge на 15-90 точках, доли секунды даже на
каждый rerun Streamlit).

Запуск из корня проекта:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.qspr.dft_surrogate import FEATURES as SURROGATE_FEATURES
from src.qspr.dft_surrogate import TARGETS as SURROGATE_TARGETS
from src.qspr.dft_surrogate import fit_and_evaluate as fit_surrogate
from src.qspr.dft_surrogate import load_dataset as load_surrogate_dataset
from src.qspr.model import TARGETS as MODEL_TARGETS
from src.qspr.model import fit_and_evaluate as fit_model
from src.qspr.model import load_dataset as load_model_dataset

QSPR_DATASET_CSV = Path("data/03_processed/qspr_dataset.csv")
SCREENING_CSV = Path("data/03_processed/screening_ranked.csv")

st.set_page_config(page_title="Flotation Collector Discovery", layout="wide")
st.title("🧬 Flotation Collector Discovery Pipeline")
st.caption(
    "Виртуальный скрининг молекул-собирателей для сульфидной флотации — "
    "RDKit + DFT (Psi4) + QSPR, на литературно обоснованном синтетическом "
    "таргете (не реальные экспериментальные данные флотации — см. PLAN.md)."
)


@st.cache_data
def load_qspr_dataset() -> pd.DataFrame:
    if not QSPR_DATASET_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(QSPR_DATASET_CSV)


@st.cache_data
def load_screening() -> pd.DataFrame:
    if not SCREENING_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SCREENING_CSV)


def _missing_file_warning(path: Path, how_to_generate: str) -> None:
    st.warning(f"Нет `{path}` — сначала `{how_to_generate}`.")


tab_candidates, tab_dft, tab_qspr, tab_screening = st.tabs(
    ["Библиотека кандидатов", "DFT-дескрипторы", "QSPR-диагностика", "Скрининг"]
)

# --------------------------------------------------------- Библиотека
with tab_candidates:
    df = load_qspr_dataset()
    if df.empty:
        _missing_file_warning(QSPR_DATASET_CSV, "python3 -m src.qspr.featurize")
    else:
        st.subheader(f"Исходная библиотека — {len(df)} кандидатов, настоящий DFT")
        st.dataframe(
            df[["name", "smiles", "n_carbons", "branched", "branch_distance",
                "mol_weight", "logp", "tpsa"]],
            use_container_width=True, hide_index=True,
        )

# --------------------------------------------------------- DFT
with tab_dft:
    df = load_qspr_dataset()
    if df.empty:
        _missing_file_warning(QSPR_DATASET_CSV, "python3 -m src.qspr.featurize")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("HOMO диапазон, эВ",
                    f"{df['homo_ev'].min():.2f} … {df['homo_ev'].max():.2f}")
        col2.metric("Gap диапазон, эВ",
                    f"{df['gap_ev'].min():.2f} … {df['gap_ev'].max():.2f}")
        col3.metric("Батч-прогон", "55.3 мин, 15/15")

        straight = df[df["branched"] == 0].sort_values("n_carbons")
        branched = df[df["branched"] == 1].sort_values("n_carbons")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=straight["n_carbons"], y=straight["gap_ev"],
            mode="lines+markers", name="прямая цепь",
            text=straight["name"],
            hovertemplate="%{text}<br>n=%{x}<br>gap=%{y:.3f} эВ<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=branched["n_carbons"], y=branched["gap_ev"],
            mode="markers", name="разветвлённые",
            marker=dict(symbol="triangle-up", size=13),
            text=branched["name"],
            hovertemplate="%{text}<br>n=%{x}<br>gap=%{y:.3f} эВ<extra></extra>",
        ))
        fig.update_layout(
            title="HOMO-LUMO gap vs длина цепи",
            xaxis_title="Число атомов C в алкильной цепи",
            yaxis_title="Gap, эВ",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Все DFT-дескрипторы (15 исходных)")
        st.dataframe(
            df[["name", "n_carbons", "branched", "homo_ev", "lumo_ev",
                "gap_ev", "dipole_debye"]],
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "`dipole_debye`: вероятно, origin-dependent артефакт для "
            "заряженной частицы — не использовать как надёжный сигнал "
            "(см. README, «Известное ограничение»)."
        )

# --------------------------------------------------------- QSPR
with tab_qspr:
    if not QSPR_DATASET_CSV.exists():
        _missing_file_warning(QSPR_DATASET_CSV, "python3 -m src.qspr.featurize")
    else:
        st.subheader("recovery / selectivity — ridge, leave-one-out CV")
        st.caption(
            "In-sample R² почти всегда оптимистичен на 15 точках — "
            "ориентир это LOO, не in-sample."
        )
        names, X, y = load_model_dataset(QSPR_DATASET_CSV)
        for target in MODEL_TARGETS:
            result = fit_model(X, y[target])
            c1, c2 = st.columns(2)
            c1.metric(f"{target}: R² in-sample", f"{result['r2_in_sample']:.3f}")
            c2.metric(f"{target}: R² leave-one-out CV", f"{result['r2_loo_cv']:.3f}")

            coef_items = sorted(result["coefficients"].items(),
                                 key=lambda kv: abs(kv[1]))
            coef_fig = go.Figure(go.Bar(
                x=[v for _, v in coef_items],
                y=[k for k, _ in coef_items],
                orientation="h",
            ))
            coef_fig.update_layout(
                title=f"Коэффициенты — {target} (после стандартизации)",
                xaxis_title="коэффициент", height=320,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(coef_fig, use_container_width=True)

        st.divider()
        st.subheader("Суррогат HOMO/gap — только RDKit-фичи, без DFT на входе")
        st.caption(
            "Нужен для скрининга расширенной библиотеки (см. вкладку "
            "«Скрининг»): дешёвая оценка HOMO/gap там, где реальный Psi4 "
            "не считался."
        )
        s_names, s_X, s_y = load_surrogate_dataset(QSPR_DATASET_CSV)
        for target in SURROGATE_TARGETS:
            result = fit_surrogate(s_X, s_y[target])
            c1, c2 = st.columns(2)
            c1.metric(f"{target}: R² in-sample", f"{result['r2_in_sample']:.3f}")
            c2.metric(f"{target}: R² leave-one-out CV", f"{result['r2_loo_cv']:.3f}")

# --------------------------------------------------------- Скрининг
with tab_screening:
    df = load_screening()
    if df.empty:
        _missing_file_warning(SCREENING_CSV, "python3 -m src.screening.rank_candidates")
    else:
        st.subheader(f"Ранжированный скрининг — {len(df)} кандидатов")
        st.caption(
            "`real_dft` — настоящий Psi4-расчёт, надёжно. "
            "`surrogate_predicted` — предсказание из RDKit-фич "
            "(LOO R² ≈ 0.87 на gap), годится для отбора, не для выводов."
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            source_filter = st.multiselect(
                "Источник gap", options=sorted(df["gap_source"].unique().tolist()),
                default=sorted(df["gap_source"].unique().tolist()),
            )
        with col2:
            branched_only = st.checkbox("Только разветвлённые")

        filtered = df[df["gap_source"].isin(source_filter)]
        if branched_only:
            filtered = filtered[filtered["branched"] == 1]
        filtered = filtered.sort_values("desirability", ascending=False)

        st.dataframe(
            filtered,
            use_container_width=True, hide_index=True,
            column_config={
                "desirability": st.column_config.ProgressColumn(
                    "desirability", min_value=0.0, max_value=1.0, format="%.3f",
                ),
            },
        )

        st.subheader("Рекомендовано на реальную DFT-валидацию")
        st.caption(
            "Разветвлённые кандидаты C6+ из surrogate-части, лучшие по "
            "desirability — дешёвый скрининг сам по себе не основание для "
            "выводов, только для отбора на дорогую проверку настоящим Psi4."
        )
        validation_candidates = (
            df[(df["gap_source"] == "surrogate_predicted")
               & (df["branched"] == 1) & (df["n_carbons"] >= 6)]
            .sort_values("desirability", ascending=False)
            .head(5)
        )
        st.dataframe(
            validation_candidates[["name", "smiles", "n_carbons", "gap_ev", "desirability"]],
            use_container_width=True, hide_index=True,
        )
