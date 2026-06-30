"""
app.py — Cardiovascular Disease Dashboard (MSBA382 Healthcare Analytics)
=======================================================================
A Streamlit dashboard analysing cardiovascular disease (CVD) risk in 68,350
patients, with filters, subtypes, and a prediction model.

Data:
  - Patient data: Kaggle "Cardiovascular Disease dataset" (Ulianova, 2019),
    cleaned -> cardio_clean.csv  (bundled in this repo)

RUN LOCALLY:  pip install -r requirements.txt   then   streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Cardiovascular Disease Dashboard",
                   layout="wide")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
DATA_FILE = os.path.join(BASE_DIR, "cardio_clean.csv")

# ====== EDIT THESE ======================================================
APP_TITLE = "Cardiovascular Disease — Risk Analytics Dashboard"
AUTHOR = "Antonio Chakhtoura"          # <-- put your name
MAJOR = "MSBA — Healthcare Analytics (MSBA382)"
# Password: set APP_PASSWORD in Streamlit "Secrets" when you publish.
# Until then it falls back to the value below — CHANGE IT.
DEFAULT_PASSWORD = "cardio2026"
# ========================================================================

POS = "#C0392B"   # disease / high
NEG = "#2E86C1"   # healthy / low
SEX_COLORS = {"Male": "#2E86C1", "Female": "#C0392B"}


# ----------------------------------------------------------------------
# Password landing page
# ----------------------------------------------------------------------
def check_password() -> bool:
    def _entered():
        pw = st.session_state.get("pw_input", "")
        try:
            target = st.secrets.get("APP_PASSWORD", DEFAULT_PASSWORD)
        except Exception:
            target = DEFAULT_PASSWORD
        st.session_state["auth"] = (pw == target)

    if st.session_state.get("auth"):
        return True

    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        st.markdown(f"###  {APP_TITLE}")
        st.caption("Protected dashboard — please enter the password to continue.")
        st.text_input("Password", type="password", key="pw_input",
                      on_change=_entered)
        if st.session_state.get("auth") is False:
            st.error("Incorrect password. Try again.")
        st.caption(f"{AUTHOR} · {MAJOR}")
    return False


if not check_password():
    st.stop()


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
@st.cache_data
def load_patients():
    df = pd.read_csv(DATA_FILE)
    df["age_group"] = pd.Categorical(
        df["age_group"], ["30-39", "40-49", "50-59", "60-65"], ordered=True)
    df["bp_category"] = pd.Categorical(
        df["bp_category"], ["Normal", "Elevated", "Hypertension St.1",
                            "Hypertension St.2"], ordered=True)
    df["bmi_category"] = pd.Categorical(
        df["bmi_category"], ["Underweight", "Normal", "Overweight", "Obese"],
        ordered=True)
    for c in ["cholesterol", "glucose"]:
        df[c] = pd.Categorical(df[c], ["Normal", "Abnormal", "High"], ordered=True)
    return df


try:
    df = load_patients()
except FileNotFoundError:
    st.error(
        "Could not find **cardio_clean.csv**. Make sure it is in the **same "
        "folder** as app.py (and uploaded to your GitHub repo when publishing)."
    )
    st.stop()


def cvd_rate(frame):
    return (frame["cardio"] == "Yes").mean() * 100


def rate_by(frame, col):
    g = (frame.assign(has=(frame["cardio"] == "Yes"))
         .groupby(col, observed=True)["has"].mean().mul(100).reset_index())
    g.columns = [col, "cvd_rate"]
    return g


# ----------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------
st.sidebar.title("Filters")
g_sel = st.sidebar.multiselect("Gender", ["Male", "Female"], ["Male", "Female"])
ages = ["30-39", "40-49", "50-59", "60-65"]
a_sel = st.sidebar.multiselect("Age group", ages, ages)
smoke_sel = st.sidebar.radio("Smoker", ["All", "Yes", "No"], horizontal=True)
active_sel = st.sidebar.radio("Physically active", ["All", "Yes", "No"], horizontal=True)
st.sidebar.markdown("---")
if st.sidebar.button("Log out"):
    st.session_state["auth"] = False
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

f = df[df["gender"].isin(g_sel) & df["age_group"].isin(a_sel)]
if smoke_sel != "All":
    f = f[f["smoke"] == smoke_sel]
if active_sel != "All":
    f = f[f["active"] == active_sel]

st.title(APP_TITLE)
st.caption("Patient data: Kaggle (Ulianova, 2019), n≈68,350")

if len(f) == 0:
    st.warning("No patients match the current filters. Widen them in the sidebar.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Gender & Age", "Risk Factors", "Predict Risk",
])

# ======================================================================
# TAB 1 — OVERVIEW
# ======================================================================
with tab1:
    st.subheader("At a glance (current filters applied)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{len(f):,}")
    c2.metric("Have CVD", f"{cvd_rate(f):.1f}%")
    c3.metric("Avg age", f"{f['age'].mean():.0f}")
    hyp = f["bp_category"].isin(["Hypertension St.1", "Hypertension St.2"]).mean() * 100
    c4.metric("Hypertensive", f"{hyp:.0f}%")

    cc1, cc2 = st.columns(2)
    with cc1:
        counts = f["cardio"].value_counts().reindex(["No", "Yes"]).reset_index()
        counts.columns = ["cardio", "n"]
        fig = px.pie(counts, names="cardio", values="n", hole=0.5,
                     color="cardio", color_discrete_map={"No": NEG, "Yes": POS},
                     title="Patients with vs without CVD")
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        st.markdown("**About this dashboard**")
        st.write(
            "This tool analyses cardiovascular disease (CVD) risk from routine "
            "medical-examination data — age, blood pressure, cholesterol, glucose, "
            "BMI and lifestyle. Use the **sidebar filters** to focus on a group, "
            "explore **gender & age** patterns and **risk factors**, and "
            "**predict** an individual's risk."
        )
        st.info("CVD = diseases of the heart and blood vessels. Target: `cardio` "
                "(Yes/No). The dataset is roughly balanced (~50% have CVD).")

# ======================================================================
# TAB 2 — GENDER & AGE  (required by the brief)
# ======================================================================
with tab2:
    st.subheader("Distribution across gender and age")

    by_age_sex = (f.assign(has=(f["cardio"] == "Yes"))
                  .groupby(["age_group", "gender"], observed=True)["has"]
                  .mean().mul(100).reset_index(name="cvd_rate"))
    fig = px.bar(by_age_sex, x="age_group", y="cvd_rate", color="gender",
                 barmode="group", color_discrete_map=SEX_COLORS,
                 labels={"cvd_rate": "CVD rate (%)", "age_group": "Age group",
                         "gender": ""},
                 title="CVD rate by age group and gender")
    st.plotly_chart(fig, use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        cnt = f.groupby(["age_group", "gender"], observed=True).size().reset_index(name="n")
        fig2 = px.bar(cnt, x="age_group", y="n", color="gender", barmode="group",
                      color_discrete_map=SEX_COLORS,
                      labels={"n": "Patients", "age_group": "Age group", "gender": ""},
                      title="Number of patients")
        st.plotly_chart(fig2, use_container_width=True)
    with cc2:
        fig3 = px.histogram(f, x="age", color="cardio", nbins=30, barmode="overlay",
                            color_discrete_map={"No": NEG, "Yes": POS},
                            labels={"age": "Age (years)", "cardio": "CVD"},
                            title="Age distribution by CVD status")
        fig3.update_traces(opacity=0.7)
        st.plotly_chart(fig3, use_container_width=True)

    st.caption("CVD rate rises clearly with age — the strongest demographic "
               "signal — and differs between men and women.")

# ======================================================================
# TAB 3 — RISK FACTORS
# ======================================================================
with tab3:
    st.subheader("Which factors carry the most risk?")
    factor = st.selectbox(
        "Risk factor",
        ["bp_category", "cholesterol", "glucose", "bmi_category",
         "smoke", "alcohol", "active"],
        format_func=lambda c: {
            "bp_category": "Blood-pressure category", "cholesterol": "Cholesterol",
            "glucose": "Glucose", "bmi_category": "BMI category",
            "smoke": "Smoking", "alcohol": "Alcohol", "active": "Physically active",
        }[c])

    g = rate_by(f, factor)
    fig = px.bar(g, x=factor, y="cvd_rate", color="cvd_rate",
                 color_continuous_scale="Reds",
                 labels={"cvd_rate": "CVD rate (%)", factor: ""},
                 title=f"CVD rate by {factor.replace('_', ' ')}")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    pivot = (f.assign(has=(f["cardio"] == "Yes"))
             .groupby(["bmi_category", "bp_category"], observed=True)["has"]
             .mean().mul(100).reset_index(name="cvd_rate"))
    fig2 = px.density_heatmap(
        pivot, x="bp_category", y="bmi_category", z="cvd_rate",
        color_continuous_scale="Reds", text_auto=".0f",
        labels={"cvd_rate": "CVD %", "bp_category": "Blood pressure",
                "bmi_category": "BMI"},
        title="CVD rate (%) by BMI and blood-pressure category")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("High blood pressure and higher BMI stack up: the top-right cells "
               "carry the greatest risk.")

# ======================================================================
# TAB 4 — PREDICT (model)
# ======================================================================
@st.cache_resource(show_spinner=False)
def train():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score,
                                 confusion_matrix)

    data = pd.read_csv(DATA_FILE)
    num = ["age", "bmi", "ap_hi", "ap_lo"]
    cat = ["gender", "cholesterol", "glucose", "smoke", "alcohol", "active"]
    X = data[num + cat]
    y = (data["cardio"] == "Yes").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                             ("num", "passthrough", num)])
    pipe = Pipeline([("pre", pre),
                     ("rf", RandomForestClassifier(n_estimators=200, max_depth=12,
                                                   random_state=42))])
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(yte, pred),
        "recall": recall_score(yte, pred),
        "auc": roc_auc_score(yte, proba),
        "cm": confusion_matrix(yte, pred),
    }
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pipe.named_steps["rf"].feature_importances_
    grouped = {}
    for nm, v in zip(names, imp):
        body = nm.split("__", 1)[1]
        base = body
        for fcat in cat:
            if body.startswith(fcat + "_"):
                base = fcat
                break
        for fnum in num:
            if body == fnum:
                base = fnum
        grouped[base] = grouped.get(base, 0) + v
    fi = (pd.DataFrame({"feature": list(grouped), "importance": list(grouped.values())})
          .sort_values("importance", ascending=False))
    return pipe, metrics, fi


with tab4:
    st.subheader("Predict cardiovascular disease risk")
    pipe, metrics, fi = train()

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    m2.metric("Recall (catches sick)", f"{metrics['recall']*100:.1f}%")
    m3.metric("ROC-AUC", f"{metrics['auc']:.2f}")

    cc1, cc2 = st.columns(2)
    with cc1:
        fig = px.bar(fi.sort_values("importance"), x="importance", y="feature",
                     orientation="h", color_discrete_sequence=[POS],
                     labels={"importance": "Importance", "feature": ""},
                     title="What drives the prediction?")
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        cm = metrics["cm"]
        fig2 = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         x=["Pred No", "Pred Yes"], y=["True No", "True Yes"],
                         title="Confusion matrix")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Enter a patient's details")
    c = st.columns(4)
    in_age = c[0].number_input("Age", 30, 65, 50)
    in_gender = c[1].selectbox("Gender", ["Male", "Female"])
    in_height = c[2].number_input("Height (cm)", 120, 220, 165)
    in_weight = c[3].number_input("Weight (kg)", 30, 200, 70)
    c = st.columns(4)
    in_aphi = c[0].number_input("Systolic (ap_hi)", 90, 200, 120)
    in_aplo = c[1].number_input("Diastolic (ap_lo)", 60, 130, 80)
    in_chol = c[2].selectbox("Cholesterol", ["Normal", "Abnormal", "High"])
    in_gluc = c[3].selectbox("Glucose", ["Normal", "Abnormal", "High"])
    c = st.columns(3)
    in_smoke = c[0].selectbox("Smoker", ["No", "Yes"])
    in_alco = c[1].selectbox("Alcohol", ["No", "Yes"])
    in_active = c[2].selectbox("Physically active", ["No", "Yes"])

    if st.button("Predict risk", type="primary"):
        bmi = round(in_weight / (in_height / 100) ** 2, 1)
        row = pd.DataFrame([{
            "age": in_age, "bmi": bmi, "ap_hi": in_aphi, "ap_lo": in_aplo,
            "gender": in_gender, "cholesterol": in_chol, "glucose": in_gluc,
            "smoke": in_smoke, "alcohol": in_alco, "active": in_active}])
        risk = pipe.predict_proba(row)[0, 1] * 100
        st.metric("Estimated CVD risk", f"{risk:.0f}%")
        if risk >= 60:
            st.error("High estimated risk — clinical follow-up advised.")
        elif risk >= 40:
            st.warning("Moderate estimated risk.")
        else:
            st.success("Lower estimated risk.")
        st.caption("Educational estimate from a model on examination data — not a "
                   "medical diagnosis.")

st.markdown("---")
st.caption(f"{AUTHOR} · {MAJOR} · Patient data: Ulianova (2019), Kaggle")
