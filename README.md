# Predicting Course Failure Early

**SDG 4: Quality Education** | Decision Tree and Random Forest | Recall-first evaluation

A machine learning project that flags students at risk of failing a course **at mid-term**, so a teacher can step in while support can still change the outcome. The output is a ranked, plain-language **watchlist**, not just a table of scores.

---

## Problem

Teachers with large classes cannot track every student. Warning signs such as low early grades, past failures and absences are usually noticed only after the final exam, when it is too late to help.

## Approach

| Item | Choice |
|---|---|
| Dataset | [UCI Student Performance (id 320)](https://archive.ics.uci.edu/dataset/320/student+performance), Portuguese-course file: 649 students, 33 attributes |
| Target | `fail = 1` if final grade `G3 < 10` (0-20 scale), else `0` |
| Models | Decision Tree (interpretable), Random Forest (more stable) |
| **Headline metric** | **Recall on the failing class** |
| Deliverable | `mid_term_watchlist.csv`, a prioritized list for teachers |

### Why recall?

A **missed failing student (false negative)** costs a student their year. A **false alarm** costs a short conversation. The pipeline is built around that asymmetry:

1. `class_weight="balanced"` makes errors on the failing class costlier.
2. Grid search optimizes `scoring="recall"`, not accuracy.
3. The decision threshold is tuned on out-of-fold **training** predictions to reach a target recall (default 90%). The test set is never used for tuning.
4. Results always show **missed failures (FN)** next to recall.

### Leakage control (how "early" is early?)

| Column | Treatment |
|---|---|
| `G3` | Dropped. It is the answer. |
| `G2` | Dropped. It arrives too late for an early warning. |
| `G1` | Kept when `USE_G1 = True` (mid-term watchlist). Dropped when `False` (pre-term watchlist). |

Run both settings and compare recall. The gap shows what predicting earlier costs.

---

## Quick start (Google Colab)

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook.
2. Paste the contents of `predicting_course_failure_early.py`. Split at the `# %%` markers if you want separate cells.
3. Run all cells. The dataset downloads automatically from UCI, so internet access is required.
4. Download `mid_term_watchlist.csv` from the Files panel.

### Local setup

```bash
pip install pandas numpy scikit-learn matplotlib
python predicting_course_failure_early.py
```

---

## Pipeline overview

1. **Fetch** the UCI zip and load `student-por.csv`.
2. **Create target** (`G3 < 10`) and check class balance and missing values.
3. **Preprocess**: one-hot encode categorical columns, pass numeric columns through. A stratified 75/25 split keeps the failing class in both sets.
4. **Tune** both models with 5-fold stratified CV, optimizing recall.
5. **Tune threshold** on training out-of-fold probabilities.
6. **Evaluate** on the held-out test set at the default 0.50 and the recall-tuned threshold: recall, FN, TP, FP, precision, F1, accuracy and confusion matrices.
7. **Explain** with Random Forest feature importances (top 12 risk drivers).
8. **Generate watchlist** with risk %, priority tier and reasons.

## Watchlist format

| Column | Meaning |
|---|---|
| Student ID | Row identifier from the dataset |
| Priority | `HIGH - meet this week`, `MEDIUM - check in / monitor` |
| Fail risk % | Model probability of failing |
| Why flagged | Plain-language reasons (e.g. "2 past class failure(s); 9 absences") |
| G1, absences, failures, studytime | Key context columns |

Students are sorted by risk. Only students at or above the tuned threshold appear.

---

## Reading the results

Look at these first, in order:

1. **Recall (fail)** on the test set.
2. **Missed fails (FN)**, the students the model failed to flag.
3. **Watchlist size** as a share of the class. If it flags most of the class, it is not practical for a teacher.
4. Precision and accuracy, which are secondary.

> Add your own numbers here after running the notebook (recall, FN, watchlist size for both models, with and without G1).

---

## Limitations and ethics

- Data comes from two Portuguese secondary schools. **Validate on local data** before any real use.
- The test set is small, so recall estimates are noisy. Consider repeated stratified CV for the final report.
- The watchlist **supports teacher judgement**. It must not be used to label, rank publicly, or penalize students.
- Some features (parental education, alcohol use) are sensitive. Use them to direct support, never to stigmatize.
- Flagging more students than will actually fail is intentional, because recall is prioritized over precision.

## Field validation

Intervention timing and watchlist size were informed by an interview with a government/aided school teacher. See the **Field Interaction Record** in the project report.

## Repository contents

```
predicting_course_failure_early.py   # full pipeline (Colab-ready)
mid_term_watchlist.csv               # generated output
README.md
```

## Acknowledgements

- Dataset: P. Cortez and A. Silva, *Using Data Mining to Predict Secondary School Student Performance*, 2008. UCI Machine Learning Repository.
- Tools: pandas, scikit-learn, matplotlib.

## License

Code: MIT (or your institution's preference). Dataset: see the UCI repository terms.
