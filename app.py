"""
EMI Risk Assessment Platform  —  Streamlit app
Run:  streamlit run app.py
Needs (produced by the notebook): models/best_classifier.joblib, models/best_regressor.joblib,
models/metadata.json, data/sample_data.csv        (mlflow.db optional)
"""

import json, os, sqlite3, datetime as dt
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import emi_utils as eu  # must be importable: the saved pipelines reference emi_utils.add_features

st.set_page_config(page_title="EMI Risk Platform", page_icon="💳", layout="wide")

# ------------------------------------------------------------------ bigger fonts (change this one number to scale everything)
BASE_FONT_PX = 20  # Streamlit default is 16
st.markdown(
    f"""
<style>
html {{ font-size: {BASE_FONT_PX}px !important; }}                         /* scales every rem-based size: nav, text, widgets */
h1 {{ font-size: 3rem !important; }}  h2 {{ font-size: 2.3rem !important; }}  h3 {{ font-size: 1.8rem !important; }}
h4, h5 {{ font-size: 1.35rem !important; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ font-size: 1.05rem; line-height: 1.6; }}
[data-testid="stCaptionContainer"] p {{ font-size: 0.95rem !important; }}
[data-testid="stMetricLabel"] p {{ font-size: 1.05rem !important; }}
[data-testid="stMetricValue"] {{ font-size: 2.6rem !important; }}
[data-testid="stMetricDelta"] {{ font-size: 1rem !important; }}
button[data-baseweb="tab"] p {{ font-size: 1.1rem !important; font-weight: 600; }}
[data-testid="stWidgetLabel"] p {{ font-size: 1rem !important; }}
input, textarea, [data-baseweb="select"] * {{ font-size: 1rem !important; }}
.stButton button p, .stDownloadButton button p {{ font-size: 1.1rem !important; font-weight: 600; }}
header a, header button, [data-testid="stHeader"] a {{ font-size: 1.1rem !important; font-weight: 600; }}   /* top navigation bar */
table td, table th {{ font-size: 1rem !important; }}
[data-testid="stAlert"] p {{ font-size: 1.05rem !important; }}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ constants
MODEL_DIR, DATA_DIR = "models", "data"
DB_PATH = os.path.join(DATA_DIR, "applications.db")
COLORS = {"Eligible": "#2e7d32", "High_Risk": "#f9a825", "Not_Eligible": "#c62828"}
ICONS = {"Eligible": "✅", "High_Risk": "⚠️", "Not_Eligible": "⛔"}
SCENARIOS = {  # amount range (INR), tenure range (months) — from the project brief
    "E-commerce Shopping EMI": ((10_000, 200_000), (3, 24)),
    "Home Appliances EMI": ((20_000, 300_000), (6, 36)),
    "Vehicle EMI": ((80_000, 1_500_000), (12, 84)),
    "Personal Loan EMI": ((50_000, 1_000_000), (12, 60)),
    "Education EMI": ((50_000, 500_000), (6, 48)),
}
OPTIONS = {
    "gender": ["Male", "Female"],
    "marital_status": ["Single", "Married"],
    "education": ["High School", "Graduate", "Post Graduate", "Professional"],
    "employment_type": ["Private", "Government", "Self-employed"],
    "company_type": ["MNC", "Large Indian", "Mid-size", "Small", "Startup"],
    "house_type": ["Rented", "Own", "Family"],
    "existing_loans": ["No", "Yes"],
}
DEFAULTS = dict(
    age=34,
    gender="Male",
    marital_status="Married",
    education="Graduate",
    monthly_salary=60_000,
    employment_type="Private",
    years_of_employment=6.0,
    company_type="Mid-size",
    house_type="Rented",
    monthly_rent=12_000,
    family_size=3,
    dependents=1,
    school_fees=0,
    college_fees=0,
    travel_expenses=5_000,
    groceries_utilities=12_000,
    other_monthly_expenses=6_000,
    existing_loans="No",
    current_emi_amount=0,
    credit_score=720,
    bank_balance=250_000,
    emergency_fund=90_000,
    emi_scenario="Personal Loan EMI",
    requested_amount=300_000,
    requested_tenure=36,
)


# ------------------------------------------------------------------ loaders
@st.cache_resource(show_spinner="Loading models …")
def load_models():
    try:
        clf = joblib.load(f"{MODEL_DIR}/best_classifier.joblib")
        reg = joblib.load(f"{MODEL_DIR}/best_regressor.joblib")
        meta = json.load(open(f"{MODEL_DIR}/metadata.json"))
        return clf, reg, meta
    except Exception as e:  # noqa
        return None, None, {"error": str(e)}


@st.cache_data(show_spinner=False)
def load_sample():
    p = f"{DATA_DIR}/sample_data.csv"
    return pd.read_csv(p) if os.path.exists(p) else None


def predict(records: pd.DataFrame):
    """records: DataFrame with eu.RAW_FEATURES columns -> (labels, probas, max_emi)"""
    clf, reg, meta = load_models()
    X = records[eu.RAW_FEATURES]
    proba = clf.predict_proba(X)
    labels = [meta["class_order"][i] for i in proba.argmax(1)]
    emi = np.maximum(reg.predict(X), meta.get("emi_min", 500))
    return labels, proba, emi


def money(x):
    return f"₹{x:,.0f}"


# ------------------------------------------------------------------ input form (shared by Predict + Admin)
def applicant_form(prefix: str, vals: dict) -> dict:
    """Renders all inputs. Scenario is chosen first so amount/tenure limits adapt."""
    v = {**DEFAULTS, **{k: x for k, x in vals.items() if k in DEFAULTS and pd.notna(x)}}
    k = lambda n: f"{prefix}_{n}"
    out = {}

    st.markdown("##### 🎯 Loan request")
    c1, c2, c3 = st.columns(3)
    out["emi_scenario"] = c1.selectbox(
        "EMI scenario",
        list(SCENARIOS),
        index=list(SCENARIOS).index(v["emi_scenario"]),
        key=k("scn"),
    )
    (amin, amax), (tmin, tmax) = SCENARIOS[out["emi_scenario"]]
    out["requested_amount"] = c2.number_input(
        f"Requested amount (₹{amin:,}–{amax:,})",
        amin,
        amax,
        int(np.clip(v["requested_amount"], amin, amax)),
        5_000,
        key=k(f"amt_{out['emi_scenario']}"),
    )
    out["requested_tenure"] = c3.number_input(
        f"Tenure in months ({tmin}–{tmax})",
        tmin,
        tmax,
        int(np.clip(v["requested_tenure"], tmin, tmax)),
        1,
        key=k(f"ten_{out['emi_scenario']}"),
    )

    st.markdown("##### 👤 Personal")
    c = st.columns(4)
    out["age"] = c[0].number_input(
        "Age", 25, 60, int(np.clip(v["age"], 25, 60)), key=k("age")
    )
    out["gender"] = c[1].selectbox(
        "Gender", OPTIONS["gender"], OPTIONS["gender"].index(v["gender"]), key=k("gen")
    )
    out["marital_status"] = c[2].selectbox(
        "Marital status",
        OPTIONS["marital_status"],
        OPTIONS["marital_status"].index(v["marital_status"]),
        key=k("mar"),
    )
    edu = v["education"] if v["education"] in OPTIONS["education"] else "Graduate"
    out["education"] = c[3].selectbox(
        "Education", OPTIONS["education"], OPTIONS["education"].index(edu), key=k("edu")
    )

    st.markdown("##### 💼 Employment & income")
    c = st.columns(4)
    out["monthly_salary"] = c[0].number_input(
        "Monthly salary (₹)",
        15_000,
        200_000,
        int(np.clip(v["monthly_salary"], 15_000, 200_000)),
        1_000,
        key=k("sal"),
    )
    out["employment_type"] = c[1].selectbox(
        "Employment type",
        OPTIONS["employment_type"],
        OPTIONS["employment_type"].index(v["employment_type"]),
        key=k("emp"),
    )
    out["years_of_employment"] = c[2].number_input(
        "Years of employment",
        0.0,
        40.0,
        float(v["years_of_employment"]),
        0.5,
        key=k("yoe"),
    )
    out["company_type"] = c[3].selectbox(
        "Company type",
        OPTIONS["company_type"],
        OPTIONS["company_type"].index(v["company_type"]),
        key=k("cmp"),
    )

    st.markdown("##### 🏠 Household & monthly expenses")
    c = st.columns(4)
    out["house_type"] = c[0].selectbox(
        "House type",
        OPTIONS["house_type"],
        OPTIONS["house_type"].index(v["house_type"]),
        key=k("house"),
    )
    rent_default = float(v["monthly_rent"]) if out["house_type"] == "Rented" else 0.0
    out["monthly_rent"] = c[1].number_input(
        "Monthly rent (₹)",
        0,
        100_000,
        int(rent_default),
        500,
        key=k(f"rent_{out['house_type']}"),
        disabled=out["house_type"] != "Rented",
    )
    if out["house_type"] != "Rented":
        out["monthly_rent"] = 0
    out["family_size"] = c[2].number_input(
        "Family size", 1, 10, int(v["family_size"]), key=k("fam")
    )
    out["dependents"] = c[3].number_input(
        "Dependents", 0, 9, int(min(v["dependents"], 9)), key=k("dep")
    )
    c = st.columns(5)
    out["school_fees"] = c[0].number_input(
        "School fees", 0, 100_000, int(v["school_fees"]), 500, key=k("sch")
    )
    out["college_fees"] = c[1].number_input(
        "College fees", 0, 100_000, int(v["college_fees"]), 500, key=k("col")
    )
    out["travel_expenses"] = c[2].number_input(
        "Travel", 0, 100_000, int(v["travel_expenses"]), 500, key=k("trv")
    )
    out["groceries_utilities"] = c[3].number_input(
        "Groceries & utilities",
        0,
        200_000,
        int(v["groceries_utilities"]),
        500,
        key=k("gro"),
    )
    out["other_monthly_expenses"] = c[4].number_input(
        "Other expenses",
        0,
        200_000,
        int(v["other_monthly_expenses"]),
        500,
        key=k("oth"),
    )

    st.markdown("##### 🏦 Credit & savings")
    c = st.columns(5)
    out["existing_loans"] = c[0].selectbox(
        "Existing loans?",
        OPTIONS["existing_loans"],
        OPTIONS["existing_loans"].index(v["existing_loans"]),
        key=k("loans"),
    )
    emi_default = (
        float(v["current_emi_amount"]) if out["existing_loans"] == "Yes" else 0.0
    )
    out["current_emi_amount"] = c[1].number_input(
        "Current EMI (₹/mo)",
        0,
        150_000,
        int(emi_default),
        500,
        key=k(f"cemi_{out['existing_loans']}"),
        disabled=out["existing_loans"] == "No",
    )
    if out["existing_loans"] == "No":
        out["current_emi_amount"] = 0
    out["credit_score"] = c[2].number_input(
        "Credit score (300–850)",
        300,
        850,
        int(np.clip(v["credit_score"], 300, 850)),
        key=k("cs"),
    )
    out["bank_balance"] = c[3].number_input(
        "Bank balance (₹)", 0, 5_000_000, int(v["bank_balance"]), 5_000, key=k("bb")
    )
    out["emergency_fund"] = c[4].number_input(
        "Emergency fund (₹)", 0, 3_000_000, int(v["emergency_fund"]), 5_000, key=k("ef")
    )
    return out


# ------------------------------------------------------------------ result renderer
def show_result(rec: dict, label, proba, max_emi):
    meta = load_models()[2]
    order = meta["class_order"]
    req_emi = rec["requested_amount"] / rec["requested_tenure"]
    color = COLORS[label]

    st.markdown(
        f"<div style='padding:18px 24px;border-radius:12px;background:{color}18;border-left:8px solid {color}'>"
        f"<span style='font-size:1.9rem;font-weight:700;color:{color}'>{ICONS[label]} {label.replace('_', ' ')}</span><br>"
        f"<span style='color:#555'>Model confidence: {proba.max():.1%}</span></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    m = st.columns(4)
    m[0].metric("Max safe monthly EMI", money(max_emi))
    m[1].metric(
        "Requested EMI (amount ÷ tenure)",
        money(req_emi),
        delta=f"{(req_emi - max_emi):+,.0f} vs max",
        delta_color="inverse",
    )
    m[2].metric(
        "Affordable loan at this tenure", money(max_emi * rec["requested_tenure"])
    )
    sal = rec["monthly_salary"]
    m[3].metric("Requested EMI / salary", f"{req_emi / sal:.0%}")

    left, right = st.columns([1, 1])
    with left:
        pdf = pd.DataFrame({"Class": order, "Probability": proba})
        ch = (
            alt.Chart(pdf)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Probability:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y("Class:N", sort=order, title=None),
                color=alt.Color(
                    "Class:N",
                    scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                    legend=None,
                ),
                tooltip=["Class", alt.Tooltip("Probability:Q", format=".1%")],
            )
            .properties(height=150, title="Class probabilities")
        )
        st.altair_chart(ch, width="stretch")
    with right:
        exp = sum(
            rec[c]
            for c in [
                "monthly_rent",
                "school_fees",
                "college_fees",
                "travel_expenses",
                "groceries_utilities",
                "other_monthly_expenses",
            ]
        )
        free = sal - exp - rec["current_emi_amount"]
        st.markdown("**Key financial ratios**")
        st.table(
            pd.DataFrame(
                {
                    "Metric": [
                        "Expenses / income",
                        "Existing EMI / income",
                        "Disposable income (₹)",
                        "Loan / annual income",
                        "Emergency fund (months of expenses)",
                    ],
                    "Value": [
                        f"{exp / sal:.0%}",
                        f"{rec['current_emi_amount'] / sal:.0%}",
                        f"{free:,.0f}",
                        f"{rec['requested_amount'] / (sal * 12):.0%}",
                        f"{rec['emergency_fund'] / max(exp, 1):.1f}",
                    ],
                }
            ).set_index("Metric")
        )

    # plain-language recommendation
    if label == "Eligible" and req_emi <= max_emi:
        st.success(
            "**Recommendation: Approve.** Requested EMI is within the applicant's safe capacity."
        )
    elif label == "Eligible":
        tenure_needed = int(np.ceil(rec["requested_amount"] / max_emi))
        st.info(
            f"**Approve with changes.** Profile is healthy but the requested EMI exceeds the safe limit — "
            f"extend the tenure to ~{tenure_needed} months or reduce the amount to ≈ {money(max_emi * rec['requested_tenure'])}."
        )
    elif label == "High_Risk":
        st.warning(
            f"**Marginal case.** Consider a lower amount (≈ {money(max_emi * rec['requested_tenure'])}), a co-applicant, or risk-based pricing (higher interest rate)."
        )
    else:
        st.error(
            "**Not recommended.** Capacity is insufficient. Improving credit score, clearing existing EMIs or lowering the request would help."
        )


# ================================================================== PAGES
def page_home():
    clf, reg, meta = load_models()
    st.title("💳 EMI Risk Assessment Platform")
    st.caption(
        "Machine-learning eligibility check + maximum safe EMI for walk-in & digital lending"
    )
    if clf is None:
        st.error(
            "Models not found. Run the notebook first so it creates `models/best_classifier.joblib`, "
            "`models/best_regressor.joblib` and `models/metadata.json`."
        )
        st.code(meta.get("error", ""))
        return
    cres = next(
        r
        for r in meta["classification_results"]
        if r["model"] == meta["best_classifier"]
    )
    rres = next(
        r for r in meta["regression_results"] if r["model"] == meta["best_regressor"]
    )
    c = st.columns(4)
    c[0].metric("Classifier in production", meta["best_classifier"])
    c[1].metric("Test accuracy", f"{cres['test_accuracy']:.1%}")
    c[2].metric("Regressor in production", meta["best_regressor"])
    c[3].metric("Test RMSE", money(rres["test_rmse"]))
    st.markdown("""
**What you can do here**
- 🔮 **Predict** – enter an applicant's profile and get eligibility, risk probabilities, the maximum safe EMI and a recommendation instantly.
- 📊 **Data Explorer** – slice the portfolio by scenario, income, credit score and more.
- 📈 **Model Performance** – compare every trained model and browse MLflow experiments.
- 🛠 **Admin** – create, read, update and delete applicant records (SQLite).

**How it works:** 22 financial & demographic inputs → feature engineering (debt-to-income, affordability, stress ratio …) →
best-of-4 classifier (Eligible / High-Risk / Not-Eligible) and best-of-4 regressor (max monthly EMI).
""")
    st.caption(
        f"Models trained {meta.get('trained_at', '?')} on {meta.get('n_train', 0):,} training rows."
    )


def page_predict():
    st.title("🔮 Real-time EMI Prediction")
    if load_models()[0] is None:
        st.error("Models not loaded — run the notebook first.")
        return
    rec = applicant_form("pred", st.session_state.get("pred_defaults", {}))
    if st.button("Assess applicant", type="primary", width="stretch"):
        st.session_state["last_pred"] = rec
    if "last_pred" in st.session_state:
        rec = st.session_state["last_pred"]
        try:
            labels, proba, emi = predict(pd.DataFrame([rec]))
            st.divider()
            show_result(rec, labels[0], proba[0], float(emi[0]))
            if st.button("💾 Save this applicant to the database"):
                db_insert(rec, labels[0], float(emi[0]), float(proba[0].max()))
                st.success("Saved — see the Admin page.")
        except Exception as e:  # noqa
            st.error(f"Prediction failed: {e}")


def page_explorer():
    st.title("📊 Data Explorer")
    df = load_sample()
    if df is None:
        st.warning("`data/sample_data.csv` not found (created by the notebook).")
        return
    st.caption(f"Showing a clean random sample of {len(df):,} applications")

    with st.expander("🔎 Filters", expanded=True):
        f1, f2 = st.columns(2)
        scn = f1.multiselect(
            "EMI scenario",
            sorted(df.emi_scenario.unique()),
            default=sorted(df.emi_scenario.unique()),
        )
        emp = f2.multiselect(
            "Employment type",
            sorted(df.employment_type.unique()),
            default=sorted(df.employment_type.unique()),
        )
        f3, f4 = st.columns(2)
        sal = f3.slider("Monthly salary", 15_000, 200_000, (15_000, 200_000), 5_000)
        cs = f4.slider("Credit score", 300, 850, (300, 850))
    f = df[
        df.emi_scenario.isin(scn)
        & df.employment_type.isin(emp)
        & df.monthly_salary.between(*sal)
        & (df.credit_score.isna() | df.credit_score.between(*cs))
    ]

    c = st.columns(4)
    c[0].metric("Applications", f"{len(f):,}")
    c[1].metric("% Eligible", f"{(f.emi_eligibility == 'Eligible').mean():.1%}")
    c[2].metric("Median salary", money(f.monthly_salary.median()))
    c[3].metric("Median max EMI", money(f.max_monthly_emi.median()))

    t1, t2, t3, t4 = st.tabs(
        ["Eligibility", "Distributions", "Relationships", "Raw data"]
    )
    with t1:
        a, b = st.columns(2)
        d1 = f.emi_eligibility.value_counts().reset_index()
        d1.columns = ["Eligibility", "Count"]
        a.altair_chart(
            alt.Chart(d1)
            .mark_arc(innerRadius=60)
            .encode(
                theta="Count:Q",
                color=alt.Color(
                    "Eligibility:N",
                    scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                ),
                tooltip=["Eligibility", "Count"],
            )
            .properties(title="Eligibility split", height=300),
            width="stretch",
        )
        d2 = (
            pd.crosstab(f.emi_scenario, f.emi_eligibility, normalize="index")
            .mul(100)
            .reset_index()
            .melt("emi_scenario", var_name="Eligibility", value_name="%")
        )
        b.altair_chart(
            alt.Chart(d2)
            .mark_bar()
            .encode(
                y=alt.Y("emi_scenario:N", title=None),
                x=alt.X("%:Q", stack="normalize", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "Eligibility:N",
                    scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                ),
                tooltip=[
                    "emi_scenario",
                    "Eligibility",
                    alt.Tooltip("%:Q", format=".1f"),
                ],
            )
            .properties(title="Mix by scenario", height=300),
            width="stretch",
        )
    with t2:
        col = st.selectbox(
            "Numeric column", [c for c in eu.NUM_COLS + [eu.REG_TARGET]], index=1
        )
        st.altair_chart(
            alt.Chart(f)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=40)),
                y="count()",
                color=alt.Color(
                    "emi_eligibility:N",
                    scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                ),
            )
            .properties(height=350),
            width="stretch",
        )
        cat = st.selectbox("Categorical column", eu.CAT_COLS, index=3)
        d = (
            f.assign(Eligible=f.emi_eligibility == "Eligible")
            .groupby(cat)["Eligible"]
            .mean()
            .mul(100)
            .reset_index()
        )
        st.altair_chart(
            alt.Chart(d)
            .mark_bar(color="#2e7d32")
            .encode(
                x=alt.X(f"{cat}:N", title=None),
                y=alt.Y("Eligible:Q", title="% eligible"),
            )
            .properties(height=250),
            width="stretch",
        )
    with t3:
        a, b = st.columns(2)
        x = a.selectbox("X", eu.NUM_COLS, index=1)
        y = b.selectbox("Y", [eu.REG_TARGET] + eu.NUM_COLS, index=0)
        s = f.sample(min(4000, len(f)), random_state=1)
        st.altair_chart(
            alt.Chart(s)
            .mark_circle(size=18, opacity=0.5)
            .encode(
                x=f"{x}:Q",
                y=f"{y}:Q",
                color=alt.Color(
                    "emi_eligibility:N",
                    scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                ),
                tooltip=[x, y, "emi_eligibility"],
            )
            .properties(height=420)
            .interactive(),
            width="stretch",
        )
        corr = (
            f[eu.NUM_COLS + [eu.REG_TARGET]]
            .corr()[eu.REG_TARGET]
            .drop(eu.REG_TARGET)
            .sort_values()
            .reset_index()
        )
        corr.columns = ["feature", "corr"]
        st.altair_chart(
            alt.Chart(corr)
            .mark_bar()
            .encode(
                x="corr:Q",
                y=alt.Y("feature:N", sort="x"),
                color=alt.condition(
                    alt.datum["corr"] > 0, alt.value("#1565c0"), alt.value("#c62828")
                ),
            )
            .properties(title="Correlation with max_monthly_emi", height=420),
            width="stretch",
        )
    with t4:
        st.dataframe(f.head(1000), width="stretch")
        st.download_button(
            "⬇ Download filtered data (CSV)",
            f.to_csv(index=False),
            "filtered_emi_data.csv",
        )


def page_performance():
    st.title("📈 Model Performance & MLflow")
    meta = load_models()[2]
    if "classification_results" not in meta:
        st.warning("metadata.json not found — run the notebook first.")
        return
    st.success(
        f"**Production models** → Classifier: **{meta['best_classifier']}** · Regressor: **{meta['best_regressor']}** "
        f"(registry aliases: `{meta['registry']['classifier']}`, `{meta['registry']['regressor']}`)"
    )

    t1, t2, t3 = st.tabs(["Classification", "Regression", "MLflow runs"])
    with t1:
        cdf = pd.DataFrame(meta["classification_results"]).set_index("model")
        cols = [
            "test_accuracy",
            "test_precision_macro",
            "test_recall_macro",
            "test_f1_macro",
            "test_roc_auc_ovr",
            "fit_s",
        ]
        st.dataframe(
            cdf[cols]
            .style.format("{:.4f}", subset=cols[:-1])
            .highlight_max(subset=cols[:-1], color="#c8e6c9"),
            width="stretch",
        )
        m = (
            cdf[cols[:-1]]
            .reset_index()
            .melt("model", var_name="metric", value_name="score")
        )
        st.altair_chart(
            alt.Chart(m)
            .mark_bar()
            .encode(
                x="metric:N",
                y=alt.Y("score:Q", scale=alt.Scale(domain=[0.5, 1])),
                color="model:N",
                xOffset="model:N",
                tooltip=["model", "metric", alt.Tooltip("score:Q", format=".4f")],
            )
            .properties(height=350),
            width="stretch",
        )
        a, b = st.columns(2)
        cm = pd.DataFrame(
            meta["confusion_matrix_test"],
            index=meta["class_order"],
            columns=meta["class_order"],
        )
        a.markdown("**Confusion matrix (test) — selected model**")
        a.dataframe(cm, width="stretch")
        fi = pd.Series(meta["clf_feature_importance"]).sort_values().reset_index()
        fi.columns = ["feature", "importance"]
        b.altair_chart(
            alt.Chart(fi)
            .mark_bar(color="#1565c0")
            .encode(x="importance:Q", y=alt.Y("feature:N", sort="x"))
            .properties(title="Top features", height=330),
            width="stretch",
        )
    with t2:
        rdf = pd.DataFrame(meta["regression_results"]).set_index("model")
        cols = ["test_rmse", "test_mae", "test_r2", "test_mape", "fit_s"]
        st.dataframe(
            rdf[cols]
            .style.format("{:.3f}", subset=cols[:-1])
            .highlight_min(
                subset=["test_rmse", "test_mae", "test_mape"], color="#c8e6c9"
            ),
            width="stretch",
        )
        m = (
            rdf[["test_rmse", "test_mae"]]
            .reset_index()
            .melt("model", var_name="metric", value_name="INR")
        )
        st.altair_chart(
            alt.Chart(m)
            .mark_bar()
            .encode(x="model:N", y="INR:Q", color="metric:N", xOffset="metric:N")
            .properties(height=320),
            width="stretch",
        )
        fi = pd.Series(meta["reg_feature_importance"]).sort_values().reset_index()
        fi.columns = ["feature", "importance"]
        st.altair_chart(
            alt.Chart(fi)
            .mark_bar(color="#1565c0")
            .encode(x="importance:Q", y=alt.Y("feature:N", sort="x"))
            .properties(title="Top features (regressor)", height=330),
            width="stretch",
        )
    with t3:
        if not os.path.exists("mlflow.db"):
            st.info(
                "`mlflow.db` not found next to the app. Locally, run the notebook then `mlflow ui --backend-store-uri sqlite:///mlflow.db`. "
                "On Streamlit Cloud this tab needs `mlflow.db` + `mlruns/` committed to the repo, or point `MLFLOW_TRACKING_URI` at a remote server."
            )
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(
                os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
            )
            exps = [e for e in mlflow.search_experiments() if e.name.startswith("EMI_")]
            exp = st.selectbox("Experiment", exps, format_func=lambda e: e.name)
            runs = mlflow.search_runs([exp.experiment_id])
            keep = (
                ["tags.mlflow.runName"]
                + [c for c in runs.columns if c.startswith("metrics.test_")]
                + ["params.fit_seconds", "start_time"]
            )
            st.dataframe(
                runs[[c for c in keep if c in runs.columns]].rename(
                    columns=lambda c: c.split(".", 1)[-1]
                ),
                width="stretch",
            )
            st.markdown("**Model registry**")
            from mlflow.tracking import MlflowClient

            cl = MlflowClient()
            rows = []
            for rm in cl.search_registered_models():
                for mv in cl.search_model_versions(f"name='{rm.name}'"):
                    rows.append(
                        {
                            "model": rm.name,
                            "version": mv.version,
                            "type": mv.tags.get("model_type", ""),
                            "aliases": ", ".join(getattr(mv, "aliases", []) or []),
                        }
                    )
            st.dataframe(pd.DataFrame(rows), width="stretch")
        except Exception as e:  # noqa
            st.error(f"Could not read MLflow store: {e}")


# ------------------------------------------------------------------ SQLite CRUD
DB_COLS = eu.RAW_FEATURES + ["pred_eligibility", "pred_max_emi", "pred_confidence"]


def db_conn():
    os.makedirs(DATA_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    types = {c: "REAL" for c in eu.NUM_COLS}
    cols = ", ".join(f"{c} {types.get(c, 'TEXT')}" for c in eu.RAW_FEATURES)
    con.execute(
        f"CREATE TABLE IF NOT EXISTS applicants (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, {cols}, "
        f"pred_eligibility TEXT, pred_max_emi REAL, pred_confidence REAL)"
    )
    return con


def db_insert(rec, label, emi, conf):
    with db_conn() as con:
        cols = eu.RAW_FEATURES + [
            "pred_eligibility",
            "pred_max_emi",
            "pred_confidence",
            "created_at",
        ]
        vals = [rec[c] for c in eu.RAW_FEATURES] + [
            label,
            emi,
            conf,
            dt.datetime.now().isoformat(timespec="seconds"),
        ]
        con.execute(
            f"INSERT INTO applicants ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            vals,
        )


def db_update(rid, rec, label, emi, conf):
    with db_conn() as con:
        cols = eu.RAW_FEATURES + ["pred_eligibility", "pred_max_emi", "pred_confidence"]
        vals = [rec[c] for c in eu.RAW_FEATURES] + [label, emi, conf, rid]
        con.execute(
            f"UPDATE applicants SET {','.join(c + '=?' for c in cols)} WHERE id=?", vals
        )


def db_delete(ids):
    with db_conn() as con:
        con.executemany("DELETE FROM applicants WHERE id=?", [(int(i),) for i in ids])


def db_read():
    with db_conn() as con:
        return pd.read_sql("SELECT * FROM applicants ORDER BY id DESC", con)


def page_admin():
    st.title("🛠 Admin — Applicant Records (CRUD)")
    st.caption(
        "Records live in a local SQLite file (`data/applications.db`). On Streamlit Cloud the file system is ephemeral — "
        "use an external database (Postgres, Supabase …) for permanent storage."
    )
    if load_models()[0] is None:
        st.error("Models not loaded — predictions unavailable.")
        return

    tab_r, tab_c, tab_u, tab_d = st.tabs(
        ["📖 Read", "➕ Create", "✏️ Update", "🗑 Delete"]
    )

    with tab_r:
        df = db_read()
        a, b, c = st.columns([2, 1, 1])
        q = a.selectbox("Filter by prediction", ["All"] + eu.CLASS_ORDER)
        if q != "All":
            df = df[df.pred_eligibility == q]
        b.metric("Records", len(df))
        c.metric(
            "Avg predicted max EMI", money(df.pred_max_emi.mean()) if len(df) else "—"
        )
        st.dataframe(df, width="stretch", height=380)
        d1, d2 = st.columns(2)
        if len(df):
            d1.download_button("⬇ Export CSV", df.to_csv(index=False), "applicants.csv")
        samp = load_sample()
        if samp is not None and d2.button(
            "Load 200 sample applicants (scored by the model)"
        ):
            s = samp.sample(
                200, random_state=int(dt.datetime.now().timestamp()) % 10_000
            )
            labels, proba, emi = predict(s)
            for (_, row), l, p, e in zip(s.iterrows(), labels, proba, emi):
                db_insert(
                    {c: (None if pd.isna(row[c]) else row[c]) for c in eu.RAW_FEATURES},
                    l,
                    float(e),
                    float(p.max()),
                )
            st.success("200 records added.")
            st.rerun()

    with tab_c:
        rec = applicant_form("new", {})
        if st.button("Score & save new applicant", type="primary"):
            labels, proba, emi = predict(pd.DataFrame([rec]))
            db_insert(rec, labels[0], float(emi[0]), float(proba[0].max()))
            st.success(f"Saved — predicted **{labels[0]}**, max EMI {money(emi[0])}")

    with tab_u:
        df = db_read()
        if df.empty:
            st.info("No records yet.")
        else:
            rid = st.selectbox("Record ID", df.id.tolist(), key="upd_id")
            row = df[df.id == rid].iloc[0].to_dict()
            rec = applicant_form(f"upd{rid}", row)
            if st.button("Update & re-score", type="primary"):
                labels, proba, emi = predict(pd.DataFrame([rec]))
                db_update(
                    int(rid), rec, labels[0], float(emi[0]), float(proba[0].max())
                )
                st.success(
                    f"Record {rid} updated — now **{labels[0]}**, max EMI {money(emi[0])}"
                )
                st.rerun()

    with tab_d:
        df = db_read()
        if df.empty:
            st.info("No records yet.")
        else:
            ids = st.multiselect("Select record IDs to delete", df.id.tolist())
            sure = st.checkbox("I understand this cannot be undone")
            if st.button("Delete selected", disabled=not (ids and sure)):
                db_delete(ids)
                st.success(f"Deleted {len(ids)} record(s).")
                st.rerun()
            if st.button("Delete ALL records", disabled=not sure):
                db_delete(df.id.tolist())
                st.rerun()


# ================================================================== ROUTER (top navigation bar)
pages = [
    st.Page(page_home, title="Home", icon="🏠", url_path="home", default=True),
    st.Page(page_predict, title="Predict", icon="🔮", url_path="predict"),
    st.Page(page_explorer, title="Data Explorer", icon="📊", url_path="explorer"),
    st.Page(
        page_performance, title="Model Performance", icon="📈", url_path="performance"
    ),
    st.Page(page_admin, title="Admin", icon="🛠", url_path="admin"),
]
st.navigation(pages, position="top").run()
