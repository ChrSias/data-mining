"""
ΕΡΩΤΗΜΑ 2: Κατηγοριοποίηση (Logistic Regression, Decision Tree, Random Forest)
Δύο σενάρια Grid Search: (Α) Holdout split, (Β) 5-Fold Cross-Validation
Προϋπόθεση: να έχει τρέξει το question1_eda.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os, warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-GUI backend (αναγκαίο με n_jobs=-1)
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay,
                              accuracy_score, precision_score, recall_score, f1_score)

warnings.filterwarnings("ignore")

_HERE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_HERE, "output")
CLEANED_CSV = os.path.join(_HERE, "cleaned_dataset.csv")


# 1. Φόρτωση και διαχωρισμός σε train pool / holdout
df = pd.read_csv(CLEANED_CSV)
print(f"Πλήρες dataset: {df.shape}")

# Στρωματοποιημένο sample 100K για όλο το model selection.
# Τα υπόλοιπα ~2.4M μένουν ως τεράστιο holdout test για αξιόπιστα μετρικά.
df_pool, df_holdout = train_test_split(
    df, train_size=100_000, stratify=df["Label"], random_state=42
)

# Αφαίρεση κλάσεων με <50 δείγματα στο train pool (δεν επαρκούν για 5-fold CV).
counts = df_pool["Label"].value_counts()
valid = counts[counts >= 50].index
removed = counts[counts < 50]
if len(removed):
    print(f"\nΑφαιρούνται σπάνιες κλάσεις (<50 δείγματα):\n{removed}")
df_pool = df_pool[df_pool["Label"].isin(valid)]
df_holdout = df_holdout[df_holdout["Label"].isin(valid)]

print(f"\nTrain pool: {df_pool.shape[0]:,} | Holdout: {df_holdout.shape[0]:,}")
print(f"Κλάσεις: {df_pool['Label'].nunique()}")


# 2. X / y και label encoding
X         = df_pool.drop("Label", axis=1)
X_holdout = df_holdout.drop("Label", axis=1)

le = LabelEncoder()
y         = le.fit_transform(df_pool["Label"])
y_holdout = le.transform(df_holdout["Label"])
classes   = list(le.classes_)


# 3. Μοντέλα και Grids υπερπαραμέτρων
# class_weight="balanced" σε όλους τους αλγορίθμους λόγω class imbalance.
# Τα prefixes "clf__" είναι για το βήμα "clf" του Pipeline (scaler -> clf).
models = {
    "LR": (LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
           {"clf__C": [0.01, 0.1, 1, 10, 100]}),
    "DT": (DecisionTreeClassifier(class_weight="balanced", random_state=42),
           {"clf__max_depth":        [5, 10, 20, None],
            "clf__min_samples_leaf": [1, 5, 20],
            "clf__criterion":        ["gini", "entropy"]}),
    "RF": (RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
           {"clf__n_estimators": [100, 200],
            "clf__max_depth":    [10, 20, None],
            "clf__max_features": ["sqrt", "log2"]}),
}


# 4. Βοηθητικές
def metrics(y_true, y_pred):
    return {
        "accuracy":   accuracy_score(y_true, y_pred),
        "precision":  precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":     recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":         f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":   f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

def save_cm(y_true, y_pred, title, filepath):
    # normalize="true": κάθε γραμμή αθροίζει 1.0 (αναγκαίο σε ανισόρροπα data).
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=classes, normalize="true",
        values_format=".2f", ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=100)
    plt.close()


results = []
cm_num = {"LR": 6, "DT": 7, "RF": 8}


# 5. ΠΡΟΣΕΓΓΙΣΗ Α: Grid Search με Holdout (train/val/test)
print(f"\n{'='*60}\n ΠΡΟΣΕΓΓΙΣΗ Α: Holdout grid search\n{'='*60}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_holdout):,}")

for name, (clf, grid) in models.items():
    print(f"\n[{name}]")
    # Pipeline: ο scaler κάνει fit μόνο στο train (όχι στο val) -> αποφυγή leakage.
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    # Manual grid search πάνω στο validation set.
    best_score, best_params = -1, None
    for params in ParameterGrid(grid):
        m = clone(pipe).set_params(**params)
        m.fit(X_train, y_train)
        score = f1_score(y_val, m.predict(X_val), average="weighted", zero_division=0)
        if score > best_score:
            best_score, best_params = score, params

    # Refit στο ΟΛΟΚΛΗΡΟ train pool με τις καλύτερες παραμέτρους.
    final = clone(pipe).set_params(**best_params).fit(X, y)
    y_pred = final.predict(X_holdout)

    print(f"Best: {best_params} | Val F1: {best_score:.4f}")
    print(classification_report(y_holdout, y_pred, target_names=classes, zero_division=0))
    results.append({"model": name, "approach": "A_Holdout", **metrics(y_holdout, y_pred)})
    save_cm(y_holdout, y_pred, f"{name} - Προσέγγιση Α (Holdout)",
            os.path.join(OUTPUT_DIR, f"{cm_num[name]:02d}_cm_{name.lower()}_holdout.png"))


# 6. ΠΡΟΣΕΓΓΙΣΗ Β: GridSearchCV με 5-Fold Cross-Validation
print(f"\n{'='*60}\n ΠΡΟΣΕΓΓΙΣΗ Β: GridSearchCV (cv=5)\n{'='*60}")

for name, (clf, grid) in models.items():
    print(f"\n[{name}]")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    gs = GridSearchCV(pipe, grid, cv=5, scoring="f1_weighted", n_jobs=-1).fit(X, y)
    y_pred = gs.predict(X_holdout)
    print(f"Best: {gs.best_params_} | CV F1: {gs.best_score_:.4f}")
    print(classification_report(y_holdout, y_pred, target_names=classes, zero_division=0))
    results.append({"model": name, "approach": "B_CV", **metrics(y_holdout, y_pred)})


# 7. Συγκριτικός πίνακας + γράφημα
res = pd.DataFrame(results)
print(f"\n{'='*60}\n ΣΥΓΚΡΙΣΗ ΜΟΝΤΕΛΩΝ\n{'='*60}")
print(res.to_string(index=False))
res.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, metric in zip(axes.flatten(), ["accuracy", "precision", "recall", "f1", "f1_macro"]):
    pivot = res.pivot(index="model", columns="approach", values=metric)
    pivot.plot(kind="bar", ax=ax, width=0.6, color=["#2196F3", "#FF5722"])
    ax.set_title(metric.upper().replace("_", " "))
    ax.set_ylim(0.5, 1.0)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(["Holdout (A)", "CV (B)"], loc="lower right")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.4f", fontsize=7, padding=2)
axes.flatten()[-1].set_visible(False)
plt.suptitle("Σύγκριση μοντέλων - Α (Holdout) vs Β (CV)", y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_comparison.png"), dpi=100, bbox_inches="tight")
plt.close()
