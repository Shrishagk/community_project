# =====================================================================
# Predicting Course Failure Early  |  SDG 4: Quality Education
# UCI Student Performance (id=320) -> Decision Tree + Random Forest
# HEADLINE METRIC: Recall on the FAILING class
# Paste into Google Colab (or split at the "# %%" markers into cells).
# =====================================================================

# %% 1. Imports & settings
import io, os, zipfile, urllib.request
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Safe plotting backend for headless / script execution
if os.environ.get("MPLBACKEND") is None and not hasattr(plt, "isinteractive"):
    matplotlib.use("Agg")

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (recall_score, precision_score, f1_score, accuracy_score,
                             confusion_matrix, classification_report, ConfusionMatrixDisplay)

RANDOM_STATE = 42
PASS_MARK = 10          # Portuguese 0-20 scale: final grade G3 < 10 => FAIL
TARGET_RECALL = 0.90    # we tune the decision threshold to reach at least this recall

# --- THE KEY DESIGN CHOICE: how "early" are we predicting? ---------
# G1 = 1st-period grade, G2 = 2nd-period grade, G3 = final grade.
# "mid-term" watchlist  -> USE_G1 = True  (only G1 exists by then)
# "pre-term" watchlist  -> USE_G1 = False (purely background/behaviour data)
# G2 and G3 are ALWAYS dropped: G3 IS the answer, G2 is too late.
#
# IMPORTANT FIX FOR G1 DOMINANCE / GRADE COLLINEARITY:
# When USE_G1 = True, G1 strongly correlates with G3 (r = -0.56 with fail),
# absorbing ~37% of model importance and dominating predictions.
# Setting USE_G1 = False trains a true pre-term early warning model that
# evaluates students based on actionable behavioral drivers (past class failures,
# absences, weekly study time, alcohol use, and parental support).
USE_G1 = False

# %% 2. Fetch the dataset (checks local cache first, then UCI)
URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

def fetch_uci_student(url=URL):
    """Check local file first, otherwise download UCI zip and return Portuguese-course file."""
    local_candidates = [
        "student-por.csv",
        os.path.join(os.path.dirname(__file__), "student-por.csv"),
        os.path.join(os.getcwd(), "community_project", "student-por.csv"),
    ]
    for candidate in local_candidates:
        if os.path.exists(candidate):
            print(f"Loading cached dataset from: {candidate}")
            return pd.read_csv(candidate, sep=";")

    raw = urllib.request.urlopen(url).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = z.namelist()
    # Some mirrors nest the data inside another zip (student.zip)
    if not any(n.endswith("student-por.csv") for n in names):
        inner = [n for n in names if n.endswith(".zip")][0]
        z = zipfile.ZipFile(io.BytesIO(z.read(inner)))
        names = z.namelist()
    por = [n for n in names if n.endswith("student-por.csv")][0]
    return pd.read_csv(z.open(por), sep=";")

df = fetch_uci_student()
print("Shape:", df.shape)          # expect (649, 33)
df.head()

# %% 3. Target creation & quick sanity checks
df["fail"] = (df["G3"] < PASS_MARK).astype(int)
print(f"\nFailing students: {df['fail'].sum()} of {len(df)} ({df['fail'].mean():.1%})")
print("Missing values:", int(df.isna().sum().sum()))

# %% 4. Feature selection (leakage control) & preprocessing
drop_cols = ["G3", "G2", "fail"] + ([] if USE_G1 else ["G1"])
X = df.drop(columns=drop_cols)
y = df["fail"]

cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("string")]
num_cols = [c for c in X.columns if c not in cat_cols]
print(f"\n{len(cat_cols)} categorical + {len(num_cols)} numeric features used")

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
    ("num", "passthrough", num_cols),
])

# Stratified split keeps the (small) failing class represented in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

# %% 5. Build models, tuned for RECALL (not accuracy)
# class_weight="balanced" makes missing a failing student costlier than a false alarm.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

dt_pipe = Pipeline([("prep", preprocess),
                    ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE))])
rf_pipe = Pipeline([("prep", preprocess),
                    ("clf", RandomForestClassifier(class_weight="balanced_subsample",
                                                   n_jobs=1, random_state=RANDOM_STATE))])

dt_grid = {"clf__max_depth": [3, 4, 5, 6, None],
           "clf__min_samples_leaf": [1, 5, 10, 20]}
rf_grid = {"clf__n_estimators": [200, 400],
           "clf__max_depth": [4, 6, 8, None],
           "clf__min_samples_leaf": [1, 3, 5]}

def tune(pipe, grid, name):
    gs = GridSearchCV(pipe, grid, scoring="recall", cv=cv, n_jobs=1)  # <-- optimise RECALL
    gs.fit(X_train, y_train)
    print(f"{name}: best CV recall = {gs.best_score_:.3f} | params = {gs.best_params_}")
    return gs.best_estimator_

dt_best = tune(dt_pipe, dt_grid, "Decision Tree")
rf_best = tune(rf_pipe, rf_grid, "Random Forest")

# %% 6. Threshold tuning: push recall to the target without any peeking at the test set
def pick_threshold(model, X_tr, y_tr, target=TARGET_RECALL):
    """Use out-of-fold probabilities on TRAIN data; return the highest threshold
    that still reaches the target recall (highest = fewest unnecessary flags)."""
    oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    for t in np.arange(0.60, 0.04, -0.01):
        if recall_score(y_tr, oof >= t) >= target:
            return round(float(t), 2)
    return 0.05

thresholds = {"Decision Tree": pick_threshold(dt_best, X_train, y_train),
              "Random Forest": pick_threshold(rf_best, X_train, y_train)}
print("\nChosen decision thresholds:", thresholds)

# %% 7. Evaluation on the held-out test set - RECALL FIRST
models = {"Decision Tree": dt_best, "Random Forest": rf_best}
rows, fitted_probs = [], {}
for name, m in models.items():
    p = m.predict_proba(X_test)[:, 1]
    fitted_probs[name] = p
    for label, thr in [("default 0.50", 0.50), ("recall-tuned", thresholds[name])]:
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        rows.append({"Model": name, "Threshold": f"{label} ({thr})",
                     "RECALL (fail)": recall_score(y_test, pred),
                     "Missed fails (FN)": fn, "Caught fails (TP)": tp,
                     "False alarms (FP)": fp,
                     "Precision (fail)": precision_score(y_test, pred, zero_division=0),
                     "F1 (fail)": f1_score(y_test, pred),
                     "Accuracy": accuracy_score(y_test, pred)})

results = pd.DataFrame(rows).round(3)
print("\n=== TEST RESULTS (sorted: recall is the headline) ===")
display(results.sort_values("RECALL (fail)", ascending=False)) if "display" in globals() else print(results)

# Confusion matrices for the recall-tuned versions
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, (name, m) in zip(axes, models.items()):
    pred = (fitted_probs[name] >= thresholds[name]).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Pass", "Fail"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{name}\nrecall = {recall_score(y_test, pred):.2f}")
plt.tight_layout()
if matplotlib.get_backend().lower() != "agg":
    plt.show()
else:
    plt.savefig("confusion_matrices.png", bbox_inches="tight")
    plt.close()

print("\nFull report - Random Forest (recall-tuned):")
print(classification_report(y_test, (fitted_probs['Random Forest'] >= thresholds['Random Forest']).astype(int),
                            target_names=["Pass", "Fail"], zero_division=0))

# %% 8. What drives failure? (helps teachers and your report)
feat_names = rf_best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=feat_names)
imp.index = [i.split("__")[1] for i in imp.index]
imp.sort_values().tail(12).plot.barh(figsize=(7, 4), title="Top 12 risk drivers (Random Forest)")
plt.tight_layout()
if matplotlib.get_backend().lower() != "agg":
    plt.show()
else:
    plt.savefig("confusion_matrices.png", bbox_inches="tight")
    plt.close()

# %% 9. THE WATCHLIST - teacher-friendly output
# Uses the Random Forest (usually the more stable ranker). Swap to dt_best if preferred.
CHOSEN = "Random Forest"
thr = thresholds[CHOSEN]

def reasons(r):
    """Plain-language flags a teacher can act on, built from raw columns."""
    out = []
    if USE_G1 and r.get("G1", 20) < PASS_MARK:         out.append(f"Low 1st-term grade ({r['G1']}/20)")
    if r["failures"] >= 1:                              out.append(f"{r['failures']} past class failure(s)")
    if r["absences"] >= 6:                              out.append(f"{r['absences']} absences")
    if r["studytime"] == 1:                             out.append("Studies <2 hrs/week")
    if r["Medu"] <= 1 and r["Fedu"] <= 1:               out.append("Low parental education")
    if r["higher"] == "no":                             out.append("No plan for higher education")
    if r["Dalc"] >= 3 or r["Walc"] >= 4:                out.append("High alcohol use")
    if r["goout"] >= 4 and r["studytime"] <= 2:         out.append("High social time, low study time")
    return "; ".join(out) if out else "No single obvious flag - review holistically"

def tier(p):
    if p >= 0.60:   return "HIGH - meet this week"
    if p >= thr:    return "MEDIUM - check in / monitor"
    return "LOW"

def build_watchlist(model, data, probs, threshold):
    wl = data.copy()
    wl["Student ID"] = wl.index
    wl["Fail risk %"] = (probs * 100).round(0).astype(int)
    wl["Priority"] = [tier(p) for p in probs]
    wl["Why flagged"] = wl.apply(reasons, axis=1)
    wl = wl[probs >= threshold]
    cols = ["Student ID", "Priority", "Fail risk %", "Why flagged"] + (["G1"] if USE_G1 else []) + ["absences", "failures", "studytime"]
    return wl.sort_values("Fail risk %", ascending=False)[cols].reset_index(drop=True)

watchlist = build_watchlist(rf_best, X_test, fitted_probs[CHOSEN], thr)

# Reality check: how many REAL failing students did the list catch?
real_fail_ids = set(y_test[y_test == 1].index)
caught = real_fail_ids & set(watchlist["Student ID"])
print(f"\nWATCHLIST: {len(watchlist)} of {len(X_test)} students flagged "
      f"({len(watchlist)/len(X_test):.0%} of the class)")
print(f"Of {len(real_fail_ids)} students who actually failed, the list caught {len(caught)} "
      f"(recall = {len(caught)/len(real_fail_ids):.0%})")

pd.set_option("display.max_colwidth", 90)
display(watchlist.head(20)) if "display" in globals() else print(watchlist.head(20))

out_name = "mid_term_watchlist.csv" if USE_G1 else "pre_term_watchlist.csv"
out_csv = os.path.join(os.path.dirname(__file__), out_name)
watchlist.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}  (Colab: Files panel -> download)")

# %% 10. Honest-limits reminder (put this in your report too)
print("""
NOTES FOR YOUR REPORT
- Data is from 2 Portuguese schools; validate locally before real use.
- Flag count > true failures by design: a false alarm costs a conversation,
  a missed failure costs a student's year.
- The list ranks students for human judgement; it must not be used to label or penalise.
- Compare USE_G1=True vs False to show how much earlier prediction costs in recall.
""")
