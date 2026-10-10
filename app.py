import html as html_lib
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------ settings
BROWN = "#2A1A0E"      # primary
CREAM = "#FEF3E2"      # secondary
SAND = "#F6E2C0"       # tint of the cream, used for panels
LINE = "#E3CBA3"       # tint, used for borders and gridlines
MUTED = "#7A6049"      # tint of the brown, used for secondary text
TAN = "#B9A184"
OLIVE = "#5E7F3B"
GOLD = "#C9A227"
RUST = "#B5523B"

TARGET = 0.80          # the product manager's goal: right 80% of the time

MODEL_PATH = Path("model/recipe_model.joblib")
METRICS_PATH = Path("model/metrics.json")
CLEAN_PATH = Path("data/recipes_clean.csv")

CATEGORIES = ["Beverages", "Breakfast", "Chicken", "Dessert", "Lunch/Snacks",
              "Meat", "One Dish Meal", "Pork", "Potato", "Vegetable"]
FEATURES = ["calories", "carbohydrate", "sugar", "protein", "category", "servings"]
TABLE_COLS = ["name", "category", "servings", "calories", "carbohydrate",
              "sugar", "protein", "known_result"]

st.set_page_config(page_title="Tasty Bytes | Homepage recipe picker",
                   page_icon="🍲", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=DM+Sans:wght@400;500;700&display=swap');

html, body, .stApp, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: @@CREAM@@; color: @@BROWN@@; }
[data-testid="stHeader"] { background: transparent; }
footer, #MainMenu { visibility: hidden; }
.block-container { padding-top: 1.6rem; max-width: 1180px; }

h1, h2, h3, h4 { font-family: 'Fraunces', serif; color: @@BROWN@@; letter-spacing: -0.01em; }
h3 { font-size: 1.35rem; margin-top: .4rem; }

[data-testid="stSidebar"] { background: @@SAND@@; border-right: 1px solid @@LINE@@; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid @@LINE@@; }
.stTabs [data-baseweb="tab"] { font-weight: 600; color: @@MUTED@@; padding: 10px 16px; }
.stTabs [aria-selected="true"] { color: @@BROWN@@; }
.stTabs [data-baseweb="tab-highlight"] { background-color: @@BROWN@@; }

.stButton > button, .stDownloadButton > button {
  background: @@BROWN@@; color: @@CREAM@@; border: 0; border-radius: 8px; font-weight: 600; padding: .5rem 1rem;
}
.stButton > button p, .stDownloadButton > button p { color: @@CREAM@@; }
.stButton > button:hover, .stDownloadButton > button:hover { background: #46301b; color: @@CREAM@@; }

.hero { background: @@BROWN@@; color: @@CREAM@@; border-radius: 16px; padding: 30px 34px; margin-bottom: 18px; }
.hero-title { font-family: 'Fraunces', serif; font-weight: 700; font-size: 2.3rem; line-height: 1.15; max-width: 760px; }
.hero-sub { margin-top: 10px; font-size: 1.02rem; opacity: .86; max-width: 700px; line-height: 1.5; }

.panel { background: @@SAND@@; border: 1px solid @@LINE@@; border-radius: 12px; padding: 16px 18px; }
.panel p { margin: 0; line-height: 1.5; }

.verdict { background: @@BROWN@@; color: @@CREAM@@; border-radius: 18px; padding: 26px 28px; }
.pill { display: inline-block; padding: 4px 13px; border-radius: 999px; font-weight: 700; font-size: .88rem; }
.pill.go { background: #7fa152; color: @@BROWN@@; }
.pill.maybe { background: @@GOLD@@; color: @@BROWN@@; }
.pill.skip { background: @@RUST@@; color: @@CREAM@@; }
.vnum { font-family: 'Fraunces', serif; font-weight: 700; font-size: 4.8rem; line-height: 1; margin: 16px 0 4px 0; }
.vsub { opacity: .85; font-size: 1rem; }
.meter { position: relative; height: 12px; border-radius: 999px; background: rgba(254,243,226,.2); margin: 22px 0 6px 0; }
.meter-fill { height: 12px; border-radius: 999px; background: @@CREAM@@; }
.meter-thr { position: absolute; top: -5px; width: 3px; height: 22px; background: @@GOLD@@; border-radius: 2px; }
.meter-cap { display: flex; justify-content: space-between; font-size: .82rem; opacity: .8; }

.stat { background: @@SAND@@; border: 1px solid @@LINE@@; border-radius: 12px; padding: 14px 16px; height: 100%; }
.stat-label { font-size: .86rem; color: @@MUTED@@; }
.stat-value { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.9rem; line-height: 1.15; color: @@BROWN@@; }
.stat-note { font-size: .82rem; color: @@MUTED@@; margin-top: 2px; }

.side-stat { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid @@LINE@@; font-size: .92rem; }
.side-stat b { font-family: 'Fraunces', serif; font-size: 1.05rem; }
</style>
""".replace("@@BROWN@@", BROWN).replace("@@CREAM@@", CREAM).replace("@@SAND@@", SAND) \
   .replace("@@LINE@@", LINE).replace("@@MUTED@@", MUTED).replace("@@GOLD@@", GOLD) \
   .replace("@@RUST@@", RUST)

st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def flat(s: str) -> str:
    """Remove indentation and line breaks so Markdown never treats HTML as code."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def esc(x) -> str:
    return html_lib.escape(str(x))


def stat(label, value, note=""):
    note_html = f'<div class="stat-note">{esc(note)}</div>' if note else ""
    return flat(f"""<div class="stat"><div class="stat-label">{esc(label)}</div>
        <div class="stat-value">{esc(value)}</div>{note_html}</div>""")


def style_chart(chart, height=320):
    return (chart.properties(height=height, background="transparent")
            .configure_axis(labelColor=BROWN, titleColor=BROWN, gridColor=LINE,
                            domainColor=LINE, tickColor=LINE,
                            labelFont="DM Sans", titleFont="DM Sans")
            .configure_legend(labelColor=BROWN, titleColor=BROWN, labelFont="DM Sans")
            .configure_view(strokeWidth=0))


def clean_category(s: pd.Series) -> pd.Series:
    return s.replace({"Chicken Breast": "Chicken"})


def parse_servings(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


# ------------------------------------------------------------------ loading
if not MODEL_PATH.exists() or not CLEAN_PATH.exists():
    st.error("The model or cleaned data is missing. Run `python train_model.py` in the "
             "project folder first, then restart the app.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_clean():
    return pd.read_csv(CLEAN_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


model = load_model()
data = load_clean()
metrics = load_metrics()

base_rate = float(data["high_traffic"].mean())
cat_rates = data.groupby("category")["high_traffic"].mean().sort_values(ascending=False)
curve = pd.DataFrame(metrics["threshold_curve"]) if metrics else None


def score(frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[FEATURES])[:, 1]


def curve_at(t: float):
    if curve is None:
        return None
    i = (curve["threshold"] - t).abs().idxmin()
    return curve.loc[i]


def default_threshold() -> float:
    """Lowest bar that reached the 80% target in cross-validation."""
    if curve is None:
        return 0.50
    ok = curve[curve["precision"].fillna(0) >= TARGET]
    if len(ok):
        return float(ok["threshold"].min())
    return float(curve.loc[curve["precision"].fillna(0).idxmax(), "threshold"])


def verdict(p: float, thr: float):
    if p >= thr:
        return "go", "Feature it"
    if p >= thr - 0.15:
        return "maybe", "Borderline"
    return "skip", "Skip for the homepage"


def meter(p: float, thr: float) -> str:
    return flat(f"""<div class="meter"><div class="meter-fill" style="width:{p*100:.1f}%"></div>
        <div class="meter-thr" style="left:calc({thr*100:.1f}% - 1px)"></div></div>
        <div class="meter-cap"><span>0%</span><span>Bar to clear: {thr:.0%}</span><span>100%</span></div>""")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Tasty Bytes")
    st.caption("Homepage recipe picker")
    thr = st.slider(
        "How sure should we be before calling a recipe popular?",
        min_value=0.30, max_value=0.95, value=round(default_threshold(), 2), step=0.05,
        help="Raise it to be pickier. You will flag fewer recipes, but more of the ones "
             "you flag will turn out to be high traffic.")
    thr = round(float(thr), 2)

    row = curve_at(thr)
    if row is not None and pd.notna(row["precision"]):
        st.markdown("**At this setting, in testing**")
        st.markdown(flat(f"""
            <div class="side-stat"><span>Flagged recipes that were right</span><b>{row['precision']:.0%}</b></div>
            <div class="side-stat"><span>High traffic recipes we caught</span><b>{row['recall']:.0%}</b></div>
            <div class="side-stat"><span>Recipes we would flag</span><b>{row['share_flagged']:.0%}</b></div>
            """), unsafe_allow_html=True)
        if row["precision"] >= TARGET:
            st.success("Meets the 80% goal.")
        else:
            st.warning("Below the 80% goal. Try raising the bar.")
    st.caption("Numbers come from cross-validation, where every recipe is scored by a "
               "model that never saw it.")

# ------------------------------------------------------------------ hero
n_recipes = metrics["n_recipes"] if metrics else len(data)
st.markdown(flat(f"""
    <div class="hero">
      <div class="hero-title">Which recipe should go on the homepage today?</div>
      <div class="hero-sub">Score a recipe before you feature it. The model learned from
      {n_recipes:,} recipes and what happened to site traffic when each one was shown.</div>
    </div>"""), unsafe_allow_html=True)

tab_one, tab_pick, tab_data, tab_model, tab_track = st.tabs(
    ["Check a recipe", "Pick today's recipe", "What the data shows",
     "How reliable is it", "Track it and next steps"])

# ================================================================ tab 1
with tab_one:
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.markdown("### Recipe details")
        category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index("Lunch/Snacks"))
        servings = st.select_slider("Servings", options=[1, 2, 3, 4, 5, 6], value=4)
        c1, c2 = st.columns(2)
        calories = c1.number_input("Calories", 0.0, 4000.0, float(round(data["calories"].median())), step=10.0)
        protein = c2.number_input("Protein (g)", 0.0, 600.0, float(round(data["protein"].median())), step=1.0)
        carbs = c1.number_input("Carbohydrate (g)", 0.0, 600.0, float(round(data["carbohydrate"].median())), step=1.0)
        sugar = c2.number_input("Sugar (g)", 0.0, 600.0, float(round(data["sugar"].median())), step=1.0)

    one = pd.DataFrame([{"calories": calories, "carbohydrate": carbs, "sugar": sugar,
                         "protein": protein, "category": category, "servings": int(servings)}])
    p = float(score(one)[0])
    state, label = verdict(p, thr)

    with right:
        st.markdown(flat(f"""
            <div class="verdict"><span class="pill {state}">{label}</span>
            <div class="vnum">{p:.0%}</div>
            <div class="vsub">chance this recipe drives high traffic</div>{meter(p, thr)}</div>"""),
                    unsafe_allow_html=True)
        st.markdown("")
        st.markdown(flat(f"""<div class="panel"><p>In the data, <b>{esc(category)}</b> recipes
            were high traffic <b>{cat_rates[category]:.0%}</b> of the time. The site average
            is {base_rate:.0%}.</p></div>"""), unsafe_allow_html=True)

    st.markdown("### Same nutrition, different category")
    st.caption("Category moves the prediction more than anything else. This shows how the "
               "same recipe would score if it were filed somewhere else.")
    what_if = pd.DataFrame({"category": CATEGORIES})
    for col, val in [("calories", calories), ("carbohydrate", carbs), ("sugar", sugar),
                     ("protein", protein), ("servings", int(servings))]:
        what_if[col] = val
    what_if["p"] = score(what_if)
    what_if["pick"] = np.where(what_if["category"] == category, "Selected", "Other")

    bars = alt.Chart(what_if).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("p:Q", title="Chance of high traffic", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("category:N", sort="-x", title=None),
        color=alt.Color("pick:N", legend=None,
                        scale=alt.Scale(domain=["Selected", "Other"], range=[OLIVE, BROWN])),
        tooltip=[alt.Tooltip("category:N"), alt.Tooltip("p:Q", format=".0%", title="Chance")])
    rule = alt.Chart(pd.DataFrame({"t": [thr]})).mark_rule(
        color=GOLD, strokeWidth=3, strokeDash=[5, 4]).encode(x="t:Q")
    st.altair_chart(style_chart(bars + rule, 300), width="stretch", theme=None)
    st.caption("The gold line is your bar to clear. Bars past it would be flagged.")

# ================================================================ tab 2
with tab_pick:
    st.markdown("### Choose between several recipes")
    st.write("Add the recipes you are deciding between. The table scores them live and ranks "
             "them, so the best one for today is always at the top. Edit any cell, add rows "
             "with the + at the bottom, or load real recipes to try it out.")


    def sample_candidates(seed=None) -> pd.DataFrame:
        s = data.sample(5, random_state=seed).copy()
        out = pd.DataFrame({
            "name": "Recipe #" + s["recipe"].astype(str),
            "category": s["category"], "servings": s["servings"],
            "calories": s["calories"].round(1), "carbohydrate": s["carbohydrate"].round(1),
            "sugar": s["sugar"].round(1), "protein": s["protein"].round(1),
            "known_result": np.where(s["high_traffic"] == 1, "High traffic", "Not high"),
        })
        return out.reset_index(drop=True)


    def reload_sample():
        st.session_state.cands = sample_candidates()
        st.session_state.ver += 1


    if "cands" not in st.session_state:
        st.session_state.cands = sample_candidates(seed=7)
        st.session_state.ver = 0
        st.session_state.last_upload = None

    b1, b2 = st.columns([1, 2.4])
    b1.button("Load 5 recipes from the library", on_click=reload_sample)
    b2.caption("The last column shows what actually happened in the data, so you can see "
               "whether the ranking would have been right.")

    with st.expander("Upload a CSV instead"):
        st.caption("Columns needed: category, servings, calories, carbohydrate, sugar, "
                   "protein. A name column is optional.")
        up = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
        if up is not None and st.session_state.last_upload != up.file_id:
            try:
                raw = pd.read_csv(up)
                raw.columns = [c.strip().lower() for c in raw.columns]
                raw = raw.rename(columns={"carbs": "carbohydrate"})
                missing = [c for c in FEATURES if c not in raw.columns]
                if missing:
                    st.error("These columns are missing: " + ", ".join(missing))
                else:
                    if "name" not in raw.columns:
                        raw.insert(0, "name", [f"Recipe {i + 1}" for i in range(len(raw))])
                    raw["category"] = clean_category(raw["category"].astype(str).str.strip())
                    raw["servings"] = parse_servings(raw["servings"])
                    raw["known_result"] = None
                    st.session_state.cands = raw[TABLE_COLS].reset_index(drop=True)
                    st.session_state.ver += 1
                    st.session_state.last_upload = up.file_id
                    st.rerun()
            except Exception as e:
                st.error(f"Could not read that file: {e}")

    edited = st.data_editor(
        st.session_state.cands, key=f"editor_{st.session_state.ver}",
        num_rows="dynamic", hide_index=True, width="stretch",
        disabled=["known_result"],
        column_config={
            "name": st.column_config.TextColumn("Recipe"),
            "category": st.column_config.SelectboxColumn("Category", options=CATEGORIES, required=True),
            "servings": st.column_config.NumberColumn("Servings", min_value=1, max_value=12, step=1),
            "calories": st.column_config.NumberColumn("Calories", min_value=0),
            "carbohydrate": st.column_config.NumberColumn("Carbs (g)", min_value=0),
            "sugar": st.column_config.NumberColumn("Sugar (g)", min_value=0),
            "protein": st.column_config.NumberColumn("Protein (g)", min_value=0),
            "known_result": st.column_config.TextColumn("Result in the data"),
        })

    work = edited.copy()
    work["servings"] = pd.to_numeric(work["servings"], errors="coerce")
    ready = work.dropna(subset=FEATURES)
    ready = ready[ready["category"].isin(CATEGORIES)].copy()
    skipped = len(work) - len(ready)

    if ready.empty:
        st.info("Add at least one complete recipe to see a ranking.")
    else:
        ready["p"] = score(ready)
        ready = ready.sort_values("p", ascending=False).reset_index(drop=True)
        ready["call"] = [verdict(v, thr)[1] for v in ready["p"]]
        top = ready.iloc[0]
        top_state, top_label = verdict(float(top["p"]), thr)
        top_name = esc(top["name"]) if pd.notna(top["name"]) and str(top["name"]).strip() else "The top recipe"

        if top_state == "go":
            msg = (f"Feature <b>{top_name}</b>. It has a {top['p']:.0%} chance of high "
                   f"traffic and clears your {thr:.0%} bar.")
        else:
            msg = (f"None of these clear your {thr:.0%} bar. <b>{top_name}</b> is the best of the "
                   f"group at {top['p']:.0%}. Add more candidates, or lower the bar if you must pick today.")
        st.markdown(flat(f"""<div class="verdict"><span class="pill {top_state}">{top_label}</span>
            <div style="margin-top:14px;font-size:1.15rem;line-height:1.5">{msg}</div></div>"""),
                    unsafe_allow_html=True)
        st.markdown("")

        show = pd.DataFrame({
            "Rank": np.arange(1, len(ready) + 1),
            "Recipe": ready["name"], "Category": ready["category"],
            "Chance": ready["p"] * 100, "Call": ready["call"],
            "Result in the data": ready["known_result"],
        })
        if show["Result in the data"].isna().all():
            show = show.drop(columns=["Result in the data"])
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "Chance": st.column_config.ProgressColumn(
                "Chance of high traffic", min_value=0, max_value=100, format="%.0f%%")})
        if skipped:
            st.caption(f"{skipped} row(s) left out because a value is missing or the category is not recognised.")
        st.download_button("Download this ranking", show.to_csv(index=False).encode("utf-8"),
                           file_name="recipe_ranking.csv", mime="text/csv")

# ================================================================ tab 3
with tab_data:
    st.markdown("### What drives traffic")
    t1, t2, t3, t4 = st.columns(4)
    t1.markdown(stat("Recipes studied", f"{len(data):,}"), unsafe_allow_html=True)
    t2.markdown(stat("High traffic overall", f"{base_rate:.0%}", "of recipes shown"), unsafe_allow_html=True)
    t3.markdown(stat("Best category", cat_rates.index[0], f"{cat_rates.iloc[0]:.0%} high traffic"),
                unsafe_allow_html=True)
    t4.markdown(stat("Weakest category", cat_rates.index[-1], f"{cat_rates.iloc[-1]:.0%} high traffic"),
                unsafe_allow_html=True)
    st.markdown("")

    top3 = list(cat_rates.index[:3])
    st.markdown(flat(f"""<div class="panel"><p><b>{esc(top3[0])}</b>, <b>{esc(top3[1])}</b> and
        <b>{esc(top3[2])}</b> recipes lead, each high traffic at least {cat_rates.iloc[2]:.0%} of the
        time. <b>{esc(cat_rates.index[-1])}</b> sits at the bottom at {cat_rates.iloc[-1]:.0%}.
        Category is the clearest signal in the data, so it is what the model leans on most.</p></div>"""),
                unsafe_allow_html=True)
    st.markdown("")

    rate_df = cat_rates.reset_index()
    rate_df.columns = ["category", "rate"]
    rbars = alt.Chart(rate_df).mark_bar(cornerRadiusEnd=3, color=BROWN).encode(
        x=alt.X("rate:Q", title="Share of recipes that were high traffic",
                axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("category:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("category:N"), alt.Tooltip("rate:Q", format=".0%")])
    avg = alt.Chart(pd.DataFrame({"t": [base_rate]})).mark_rule(
        color=GOLD, strokeWidth=3, strokeDash=[5, 4]).encode(x="t:Q")
    st.markdown("#### Traffic by category")
    st.altair_chart(style_chart(rbars + avg, 330), width="stretch", theme=None)
    st.caption(f"The gold line is the site average, {base_rate:.0%}.")

    st.markdown("#### Nutrition and traffic")
    nutr = st.radio("Look at", ["Calories", "Carbohydrate (g)", "Sugar (g)", "Protein (g)"],
                    horizontal=True)
    col = {"Calories": "calories", "Carbohydrate (g)": "carbohydrate",
           "Sugar (g)": "sugar", "Protein (g)": "protein"}[nutr]
    box_df = data[[col, "high_traffic"]].copy()
    box_df["outcome"] = np.where(box_df["high_traffic"] == 1, "High traffic", "Not high")
    box = alt.Chart(box_df).mark_boxplot(extent=1.5, size=70, outliers=False,
                                         median={"color": CREAM}).encode(
        x=alt.X("outcome:N", title=None, sort=["Not high", "High traffic"]),
        y=alt.Y(f"{col}:Q", title=nutr),
        color=alt.Color("outcome:N", legend=None,
                        scale=alt.Scale(domain=["Not high", "High traffic"], range=[TAN, BROWN])))
    cA, cB = st.columns([1.4, 1], gap="large")
    cA.altair_chart(style_chart(box, 300), width="stretch", theme=None)
    meds = box_df.groupby("outcome")[col].median()
    cB.markdown(stat("Median when not high", f"{meds.get('Not high', float('nan')):.1f}"), unsafe_allow_html=True)
    cB.markdown("")
    cB.markdown(stat("Median when high traffic", f"{meds.get('High traffic', float('nan')):.1f}"), unsafe_allow_html=True)
    st.caption("Extreme values are hidden so the boxes stay readable. Where the two boxes "
               "overlap heavily, that nutrient tells you little on its own.")

# ================================================================ tab 4
with tab_model:
    st.markdown("### How reliable is it")
    if not metrics:
        st.info("Run `python train_model.py` to create the model metrics.")
    else:
        st.write("The question is a yes or no: will this recipe bring high traffic? That makes "
                 "it a binary classification problem. Two models were tried. The baseline is "
                 "Logistic Regression. The comparison model is a Random Forest.")

        names = ["Logistic Regression", "Random Forest"]
        lr_cv = metrics["cv"]["Logistic Regression"]["precision"]["mean"]
        rf_cv = metrics["cv"]["Random Forest"]["precision"]["mean"]
        gap = lr_cv - TARGET
        if gap >= 0.03:
            read = f"above the 80% goal by {gap*100:.0f} points"
        elif gap >= -0.005:
            read = "right at the 80% goal, with very little room to spare"
        else:
            read = f"below the 80% goal by {abs(gap)*100:.0f} points"
        st.markdown(flat(f"""<div class="panel"><p>Across five rounds of testing, when the baseline
            model called a recipe popular it was right <b>{lr_cv:.0%}</b> of the time. That is {read}.
            Treat the app as a strong shortlist tool, not a guarantee.</p></div>"""),
                    unsafe_allow_html=True)
        st.markdown("")

        rows = []
        for n in names:
            t, c = metrics["test"][n], metrics["cv"][n]
            rows.append({"Model": n,
                         "Accuracy": f"{t['accuracy']:.1%}", "Precision": f"{t['precision']:.1%}",
                         "Recall": f"{t['recall']:.1%}", "F1": f"{t['f1']:.1%}",
                         "Precision, 5-fold": f"{c['precision']['mean']:.1%} +/- {c['precision']['std']:.1%}"})
        st.markdown("#### The two models side by side")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("The first four columns come from a held-out 20% test set. The last column "
                   "averages five rounds of cross-validation, which is steadier on a small dataset.")

        cv_rows = [{"Model": n, "Measure": m.capitalize() if m != "f1" else "F1",
                    "Score": metrics["cv"][n][m]["mean"]}
                   for n in names for m in ["accuracy", "precision", "recall", "f1"]]
        cv_df = pd.DataFrame(cv_rows)
        gbars = alt.Chart(cv_df).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("Measure:N", title=None, sort=["Accuracy", "Precision", "Recall", "F1"],
                    axis=alt.Axis(labelAngle=0)),
            xOffset="Model:N",
            y=alt.Y("Score:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", scale=alt.Scale(domain=names, range=[BROWN, TAN]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Model", "Measure", alt.Tooltip("Score:Q", format=".1%")])
        goal = alt.Chart(pd.DataFrame({"t": [TARGET]})).mark_rule(
            color=GOLD, strokeWidth=3, strokeDash=[5, 4]).encode(y="t:Q")
        st.altair_chart(style_chart(gbars + goal, 320), width="stretch", theme=None)
        st.caption("Cross-validated scores. The gold line is the 80% goal.")

        if lr_cv >= rf_cv:
            why = ("Logistic Regression is the model inside this app. It scored higher on "
                   "precision than the Random Forest, and it is simpler to explain to the "
                   "product team, which matters when people need to trust a pick.")
        else:
            why = ("Logistic Regression is the model inside this app because it is simpler to "
                   "explain, even though the Random Forest scored slightly higher on precision.")
        st.markdown(f"**Why this model.** {why}")

        st.markdown("#### The trade-off behind the slider")
        st.write("Precision is how often a recipe we call popular really is. Recall is how many "
                 "of the truly popular recipes we manage to catch. Raising the bar buys precision "
                 "and costs recall.")
        cd = curve.dropna(subset=["precision"]).melt(
            id_vars="threshold", value_vars=["precision", "recall"],
            var_name="Measure", value_name="Score")
        cd["Measure"] = cd["Measure"].str.capitalize()
        lines = alt.Chart(cd).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("threshold:Q", title="Bar to clear", axis=alt.Axis(format="%")),
            y=alt.Y("Score:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0.3, 1])),
            color=alt.Color("Measure:N", scale=alt.Scale(domain=["Precision", "Recall"],
                                                          range=[BROWN, OLIVE]),
                            legend=alt.Legend(orient="bottom", title=None)))
        goal2 = alt.Chart(pd.DataFrame({"t": [TARGET]})).mark_rule(
            color=GOLD, strokeWidth=3, strokeDash=[5, 4]).encode(y="t:Q")
        now = alt.Chart(pd.DataFrame({"t": [thr]})).mark_rule(
            color=MUTED, strokeWidth=2).encode(x="t:Q")
        st.altair_chart(style_chart(lines + goal2 + now, 320), width="stretch", theme=None)
        st.caption("Gold line: the 80% goal. Grey line: where your slider is now.")

        st.markdown("#### What pushes a prediction up or down")
        co = pd.DataFrame(metrics["coefficients"]).sort_values("coef")
        co = pd.concat([co.head(6), co.tail(6)]).drop_duplicates()
        co["Effect"] = np.where(co["coef"] >= 0, "Raises the chance", "Lowers the chance")
        cbar = alt.Chart(co).mark_bar(cornerRadius=3).encode(
            x=alt.X("coef:Q", title="Push on the prediction"),
            y=alt.Y("feature:N", sort="x", title=None),
            color=alt.Color("Effect:N", scale=alt.Scale(
                domain=["Raises the chance", "Lowers the chance"], range=[OLIVE, RUST]),
                legend=alt.Legend(orient="bottom", title=None)))
        st.altair_chart(style_chart(cbar, 340), width="stretch", theme=None)
        st.caption("Longer bars mean a stronger push. Category dominates, as the earlier charts suggested.")

# ================================================================ tab 5
with tab_track:
    st.markdown("### The number to watch")
    st.markdown(flat(f"""<div class="panel"><p><b>Hit rate of homepage picks.</b> Of the recipes we
        feature, what share bring high traffic? The goal is <b>{TARGET:.0%} or more</b>. Check it every
        week, and look closely if it falls below the goal two weeks running.</p></div>"""),
                unsafe_allow_html=True)
    st.markdown("")

    row = curve_at(thr)
    s1, s2, s3 = st.columns(3)
    s1.markdown(stat("Today, without the model", f"{base_rate:.0%}",
                     "Share of all recipes that were high traffic"), unsafe_allow_html=True)
    if row is not None and pd.notna(row["precision"]):
        s2.markdown(stat("With the model", f"{row['precision']:.0%}",
                         f"Estimated at a {thr:.0%} bar"), unsafe_allow_html=True)
        lift = row["precision"] - base_rate
        s3.markdown(stat("Expected gain", f"{lift*100:+.0f} points",
                         "Versus picking without the model"), unsafe_allow_html=True)
    st.caption("Starting values come from the existing data. A recipe picked with no model is "
               "a high traffic recipe about as often as the overall share.")

    st.markdown("#### Try your own numbers")
    k1, k2 = st.columns(2)
    featured = k1.number_input("Days a recipe was featured", 1, 365, 30)
    won = k2.number_input("Days traffic was high", 0, 365, 18)
    if won > featured:
        st.error("Days with high traffic cannot be more than days featured.")
    else:
        hit = won / featured
        d1, d2 = st.columns(2)
        d1.metric("Your hit rate", f"{hit:.0%}", f"{(hit - TARGET)*100:+.0f} points vs the 80% goal")
        if row is not None and pd.notna(row["precision"]):
            d2.metric(f"Expected high traffic days in {featured}",
                      f"{row['precision'] * featured:.0f} with the model",
                      f"{(row['precision'] - base_rate) * featured:+.0f} vs without")
        st.caption("Each high traffic day can lift traffic to the rest of the site by as much "
                   "as 40%, so every extra hit counts.")

    st.markdown("### What to do next")
    st.markdown("""
- **Start using it as a shortlist.** Each morning, score the candidate recipes in the second tab and feature the top one that clears the bar.
- **Run a four to eight week pilot.** Alternate between model picks and your own picks, and compare hit rates before committing.
- **Keep a daily log.** Record the recipe, its details, and whether traffic was high. More rows means a better model, and 947 is small.
- **Share more about each recipe.** Time to make, cost per serving, ingredients and photos are on the site but not in this data. They may add real signal.
- **Retrain every month.** Run `python train_model.py` with the new log, and compare the new hit rate with the old one before swapping the model.
- **Be careful with the claim.** The result sits near the 80% goal, not far above it. Hold a review if the weekly hit rate drops below it.
""")

st.markdown("---")
st.caption("Built from the Tasty Bytes recipe traffic data. Logistic Regression, "
           "cross-validated, for decision support only.")
