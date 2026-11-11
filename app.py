"""Streamlit front-end for the salary prediction model.

Run:  streamlit run app.py
Expects next to this file (or paths set via env vars):
  best_salary_model.joblib      – saved by the notebook (section 9.10)
  data-job-postings.csv         – original dataset (used for dropdown options and market context)
  salary_model_comparison.csv   – optional, used to show the model's typical error
"""
import os
from datetime import date
from pathlib import Path

import altair as alt
import streamlit as st

from salary_core import (
    SENIORITY_ORDER,
    build_features,
    load_model,
    load_reference,
    make_title,
    model_mape,
    predict,
    predict_variants,
)

HERE = Path(__file__).parent
MODEL_PATH = Path(os.getenv("SALARY_MODEL_PATH", HERE / "best_salary_model.joblib"))
DATA_PATH = Path(os.getenv("SALARY_DATA_PATH", HERE / "data-job-postings.csv"))
COMPARISON_PATH = Path(os.getenv("SALARY_COMPARISON_PATH", HERE / "salary_model_comparison.csv"))

SENIORITY_LABELS = {"junior": "Junior", "mid": "Middle", "senior": "Senior", "staff": "Staff", "lead": "Lead"}
REMOTE_LABELS = {"remote": "Удалённо", "hybrid": "Гибрид", "onsite": "В офисе"}
SIZE_LABELS = {"startup": "Стартап", "scaleup": "Scale-up", "midmarket": "Средняя", "enterprise": "Корпорация"}
COUNTRY_NAMES = {"US": "США", "GB": "Великобритания", "DE": "Германия", "NL": "Нидерланды",
                 "CA": "Канада", "IN": "Индия", "BR": "Бразилия"}

st.set_page_config(page_title="Прогноз зарплаты", page_icon="💼", layout="wide")


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data
def get_reference():
    return load_reference(DATA_PATH)


def bar_chart(data, cat_col, order, cat_title, horizontal=False):
    """Bars in a fixed order; the currently selected option is highlighted."""
    cat = alt.X if not horizontal else alt.Y
    val = alt.Y if not horizontal else alt.X
    return (
        alt.Chart(data)
        .mark_bar(cornerRadius=3)
        .encode(
            cat(f"{cat_col}:N", sort=order, title=cat_title),
            val("predicted_salary_usd:Q", title="USD в год", axis=alt.Axis(format="~s")),
            color=alt.condition(alt.datum.current, alt.value("#2563eb"), alt.value("#94a3b8")),
            tooltip=[alt.Tooltip(f"{cat_col}:N", title=cat_title),
                     alt.Tooltip("predicted_salary_usd:Q", title="Прогноз, USD", format=",.0f")],
        )
        .properties(height=320)
    )


def usd(x: float) -> str:
    return f"${x:,.0f}".replace(",", " ")


# ---------------------------------------------------------------- checks
missing = [p.name for p in (MODEL_PATH, DATA_PATH) if not p.exists()]
if missing:
    st.error(
        "Не найдены файлы: " + ", ".join(missing) +
        ". Запустите ноутбук до конца (он сохраняет `best_salary_model.joblib`) "
        "и положите файлы рядом с `app.py`."
    )
    st.stop()

model = get_model()
ref = get_reference()
model_name, mape = model_mape(COMPARISON_PATH)

# ---------------------------------------------------------------- header
st.title("💼 Прогноз зарплаты в data-профессиях")
st.caption(
    "Модель обучена на вакансиях 2023–2025 гг. (7 стран, 10 городов, 8 ролей). "
    "Прогноз — середина вилки в USD в год."
)

# ---------------------------------------------------------------- inputs
with st.sidebar:
    st.header("Параметры вакансии")

    role = st.selectbox("Роль", ref.roles)
    seniority = st.select_slider(
        "Уровень", options=SENIORITY_ORDER, value="mid", format_func=SENIORITY_LABELS.get
    )

    cities = sorted(ref.city_country, key=lambda c: (ref.city_country[c], c))
    city = st.selectbox(
        "Город",
        cities,
        index=cities.index("London") if "London" in cities else 0,
        format_func=lambda c: f"{c} · {COUNTRY_NAMES.get(ref.city_country[c], ref.city_country[c])}",
    )
    country = ref.city_country[city]

    remote = st.radio(
        "Формат работы", ref.remote_options, horizontal=True,
        format_func=lambda v: REMOTE_LABELS.get(v, v),
    )
    company_size = st.selectbox(
        "Размер компании", ref.company_sizes, format_func=lambda v: SIZE_LABELS.get(v, v)
    )

    role_top_skills = ref.role_skills.get(role, [])
    skill_options = role_top_skills + [s for s in ref.skills if s not in role_top_skills]
    skills = st.multiselect(
        "Навыки",
        skill_options,
        default=role_top_skills[:4],
        key=f"skills_{role}",  # reset defaults when the role changes
        help="Сначала идут навыки, типичные для выбранной роли.",
    )
    posted_on = st.date_input("Дата публикации", value=date.today())

base = dict(
    role=role, seniority=seniority, city=city, country=country, remote=remote,
    company_size=company_size, skills=skills, posted_on=posted_on,
)

# ---------------------------------------------------------------- prediction
features = build_features(**base, all_skills=ref.skills)
pred = float(predict(model, features)[0])
low, high = pred * (1 - mape / 100), pred * (1 + mape / 100)

title = make_title(role, seniority)
st.subheader(f"{title} · {city}, {COUNTRY_NAMES.get(country, country)}")

c1, c2, c3 = st.columns(3)
c1.metric("Прогноз (год)", usd(pred))
c2.metric("Прогноз (месяц)", usd(pred / 12))
c3.metric("Типичный разброс", f"{usd(low)} – {usd(high)}", help=f"± средняя ошибка модели на тесте (MAPE {mape:.1f}%)")

peers = ref.salaries[
    (ref.salaries["role"] == role)
    & (ref.salaries["seniority"] == seniority)
    & (ref.salaries["city"] == city)
]["salary_mid_usd"]
if len(peers) >= 3:
    diff = (pred / peers.median() - 1) * 100
    st.info(
        f"В данных **{len(peers)}** похожих вакансий ({title}, {city}) с медианой **{usd(peers.median())}**. "
        f"Прогноз {'выше' if diff >= 0 else 'ниже'} медианы на {abs(diff):.1f}%."
    )
else:
    st.warning(
        f"В данных почти нет вакансий «{title}» в {city} ({len(peers)} шт.) — "
        "прогноз опирается на похожие роли и города, точность ниже."
    )

# ---------------------------------------------------------------- what-if charts
st.divider()
left, right = st.columns(2)

with left:
    st.markdown("**Как меняется прогноз с уровнем**")
    by_level = predict_variants(model, base, "seniority", SENIORITY_ORDER, ref.skills)
    by_level["level"] = by_level["seniority"].map(SENIORITY_LABELS)
    by_level["current"] = by_level["seniority"] == seniority
    st.altair_chart(
        bar_chart(by_level, "level", [SENIORITY_LABELS[s] for s in SENIORITY_ORDER], "Уровень"),
        use_container_width=True,
    )

with right:
    st.markdown("**Тот же профиль в других городах**")
    by_city = predict_variants(model, base, "city", cities, ref.skills, ref.city_country)
    by_city = by_city.sort_values("predicted_salary_usd", ascending=False)
    by_city["current"] = by_city["city"] == city
    st.altair_chart(
        bar_chart(by_city, "city", by_city["city"].tolist(), "Город", horizontal=True),
        use_container_width=True,
    )

# ---------------------------------------------------------------- details
with st.expander("Что именно ушло в модель"):
    st.dataframe(features.T.rename(columns={0: "значение"}), use_container_width=True)
    if model_name:
        st.caption(f"Модель: {model_name} · MAPE на тесте {mape:.2f}%")

st.caption(
    "⚠️ Данные, по всей видимости, синтетические: прогноз отражает закономерности этого датасета, "
    "а не реальный рынок. Навыки внутри одной роли на прогноз почти не влияют."
)
