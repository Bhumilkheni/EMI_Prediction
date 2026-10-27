"""
Shared cleaning + feature engineering for the EMI platform.
Imported by BOTH the notebook and the Streamlit app so training and
inference use exactly the same logic (no train/serve skew).
"""
import re
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ schema
CAT_COLS = ["gender", "marital_status", "education", "employment_type",
            "company_type", "house_type", "existing_loans", "emi_scenario"]
NUM_COLS = ["age", "monthly_salary", "years_of_employment", "monthly_rent",
            "family_size", "dependents", "school_fees", "college_fees",
            "travel_expenses", "groceries_utilities", "other_monthly_expenses",
            "current_emi_amount", "credit_score", "bank_balance",
            "emergency_fund", "requested_amount", "requested_tenure"]
RAW_FEATURES = CAT_COLS + NUM_COLS            # the 25 raw inputs (22 + scenario cols)
CLF_TARGET, REG_TARGET = "emi_eligibility", "max_monthly_emi"
CLASS_ORDER = ["Eligible", "High_Risk", "Not_Eligible"]

# valid business ranges (from the project brief)
VALID = {"age": (18, 70), "monthly_salary": (15_000, 200_000),
         "credit_score": (300, 850)}

ENGINEERED = [
    "total_expenses", "expense_to_income", "debt_to_income", "disposable_income",
    "requested_emi_est", "emi_to_income", "affordability_ratio", "loan_to_income",
    "savings_ratio", "emergency_months", "employment_stability",
    "dependents_per_member", "credit_risk_score", "stress_ratio",
    "credit_x_dti", "income_per_member",
]


# ------------------------------------------------------------------ cleaning
def _fix_numeric_string(s: pd.Series) -> pd.Series:
    """'58.0.0' / '23400.0.0.0' -> 58.0 / 23400.0 ; then to float."""
    s = s.astype("string").str.strip()
    s = s.str.replace(r"^(-?\d+\.\d+)(\.\d+)+$", r"\1", regex=True)
    return pd.to_numeric(s, errors="coerce")


def clean_data(df: pd.DataFrame, drop_salary_outliers: bool = True,
               verbose: bool = True) -> pd.DataFrame:
    """Deterministic cleaning only (imputation happens inside the sklearn
    pipeline, fitted on train data, so there is no leakage)."""
    df = df.copy()
    n0 = len(df)
    log = {}

    # 1. numeric columns that arrived as text / with malformed values
    for c in NUM_COLS + [REG_TARGET]:
        if c in df.columns:
            df[c] = _fix_numeric_string(df[c])

    # 2. categorical standardisation
    g = df["gender"].astype("string").str.strip().str.lower()
    df["gender"] = g.map({"male": "Male", "m": "Male",
                          "female": "Female", "f": "Female"})
    for c in ["marital_status", "education", "employment_type", "company_type",
              "house_type", "existing_loans", "emi_scenario"]:
        df[c] = df[c].astype("string").str.strip()

    # 3. duplicates
    before = len(df); df = df.drop_duplicates(); log["duplicates_removed"] = before - len(df)

    # 4. rule-based invalid values -> NaN  (imputed later) / rows dropped
    for c, (lo, hi) in VALID.items():
        bad = (df[c] < lo) | (df[c] > hi)
        log[f"{c}_out_of_range"] = int(bad.sum())
        if c == "monthly_salary" and drop_salary_outliers:
            df = df.loc[~bad]          # salary drives the label -> unsafe to impute
        else:
            df.loc[bad, c] = np.nan

    # 5. logical consistency
    df.loc[df["existing_loans"] == "No", "current_emi_amount"] = \
        df.loc[df["existing_loans"] == "No", "current_emi_amount"].fillna(0)
    df.loc[df["house_type"] != "Rented", "monthly_rent"] = \
        df.loc[df["house_type"] != "Rented", "monthly_rent"].fillna(0)
    df = df.reset_index(drop=True)

    # 6. plain float64 / object dtypes (NaN, not pd.NA) -> safe for sklearn & MLflow
    for c in NUM_COLS + [REG_TARGET]:
        df[c] = df[c].astype("float64")
    for c in CAT_COLS + [CLF_TARGET]:
        col = df[c].astype(object)
        df[c] = col.where(col.notna(), np.nan)

    if verbose:
        print(f"clean_data: {n0:,} -> {len(df):,} rows | {log}")
    return df


# --------------------------------------------------------- feature engineering
def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Derived financial ratios / risk scores / interactions.
    Works on raw (possibly NaN) input; uses safe division."""
    X = X.copy()
    num = lambda c: pd.to_numeric(X[c], errors="coerce")
    sal = num("monthly_salary").clip(lower=1)

    exp_cols = ["monthly_rent", "school_fees", "college_fees", "travel_expenses",
                "groceries_utilities", "other_monthly_expenses"]
    X["total_expenses"] = sum(num(c).fillna(0) for c in exp_cols)
    X["expense_to_income"] = X["total_expenses"] / sal
    X["debt_to_income"] = num("current_emi_amount").fillna(0) / sal
    X["disposable_income"] = sal - X["total_expenses"] - num("current_emi_amount").fillna(0)
    X["requested_emi_est"] = num("requested_amount") / num("requested_tenure").clip(lower=1)
    X["emi_to_income"] = X["requested_emi_est"] / sal
    X["affordability_ratio"] = (X["requested_emi_est"] /
                                X["disposable_income"].clip(lower=1000)).clip(upper=50)
    X["loan_to_income"] = num("requested_amount") / (sal * 12)
    X["savings_ratio"] = (num("bank_balance") + num("emergency_fund")) / (sal * 12)
    X["emergency_months"] = num("emergency_fund") / (X["total_expenses"] + 1)
    X["employment_stability"] = num("years_of_employment") / (num("age") - 17).clip(lower=1)
    X["dependents_per_member"] = num("dependents") / num("family_size").clip(lower=1)
    X["income_per_member"] = sal / num("family_size").clip(lower=1)
    # simple composite credit-risk score (0 = safest, 1 = riskiest)
    cs = ((850 - num("credit_score")) / 550).clip(0, 1)
    X["credit_risk_score"] = (0.5 * cs + 0.3 * X["debt_to_income"].clip(0, 1)
                              + 0.2 * (1 - X["employment_stability"].clip(0, 1)))
    X["stress_ratio"] = (X["total_expenses"] + num("current_emi_amount").fillna(0)
                         + X["requested_emi_est"]) / sal
    X["credit_x_dti"] = num("credit_score") * (1 - X["debt_to_income"].clip(0, 1))

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X
