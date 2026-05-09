"""
train_model.py
────────────────────────────────────────────────────────────────
Trains three ML models on the synthetic (or real) traffic dataset:
  1. Random Forest     – supervised multi-class classifier
  2. Isolation Forest  – unsupervised anomaly detector
  3. One-Class SVM     – unsupervised anomaly detector (trained on normal only)

Outputs:
  models/random_forest.pkl
  models/isolation_forest.pkl
  models/oneclasssvm.pkl
  models/scaler.pkl
  models/label_encoder.pkl
  reports/classification_report.txt
  reports/confusion_matrix.png
  reports/feature_importance.png
  reports/roc_curves.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, RocCurveDisplay
)
from sklearn.preprocessing import label_binarize

# ── paths ─────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
DATA    = BASE / "data" / "traffic_dataset.csv"
MODELS  = BASE / "models";  MODELS.mkdir(exist_ok=True)
REPORTS = BASE / "reports"; REPORTS.mkdir(exist_ok=True)

FEATURES = [
    "duration", "protocol", "src_port", "dst_port",
    "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "pkt_len_mean", "pkt_len_std",
    "flow_iat_mean", "flow_iat_std", "fwd_iat_mean", "bwd_iat_mean",
    "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt",
    "psh_flag_cnt", "ack_flag_cnt", "down_up_ratio",
]

# ── load & preprocess ─────────────────────────────────────────────────
def load_data():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA}\n"
            "Run:  python data/generate_traffic.py  first."
        )
    df = pd.read_csv(DATA)
    X  = df[FEATURES].fillna(0).values
    y  = df["label"].values
    names = df["label_name"].values
    return X, y, names, df["label_name"].unique()

def split_and_scale(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    return X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te, scaler

# ── model 1: random forest ────────────────────────────────────────────
def train_random_forest(X_tr_s, y_tr, X_te_s, y_te, class_names):
    print("\n[1/3] Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tr_s, y_tr)

    y_pred = rf.predict(X_te_s)
    y_prob = rf.predict_proba(X_te_s)

    # cross-val
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(rf, X_tr_s, y_tr, cv=cv, scoring="f1_macro")
    print(f"    CV macro-F1: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")

    # text report
    report = classification_report(y_te, y_pred, target_names=class_names)
    print(report)
    (REPORTS / "classification_report.txt").write_text(report)

    # confusion matrix
    _plot_confusion(y_te, y_pred, class_names, "Random Forest")

    # feature importance
    _plot_feature_importance(rf, FEATURES)

    # ROC
    _plot_roc(y_te, y_prob, rf.classes_, class_names)

    joblib.dump(rf, MODELS / "random_forest.pkl")
    print(f"    Saved -> {MODELS/'random_forest.pkl'}")
    return rf

# ── model 2: isolation forest ─────────────────────────────────────────
def train_isolation_forest(X_tr_s, y_tr, X_te_s, y_te):
    print("\n[2/3] Training Isolation Forest (unsupervised) ...")
    # train only on normal traffic
    normal_mask = y_tr == 0
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.15,   # expected fraction of anomalies in production
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_tr_s[normal_mask])

    # predict: +1 = normal, -1 = anomaly  ->  remap to 0/1
    raw  = iso.predict(X_te_s)
    pred = (raw == -1).astype(int)           # 1 = anomaly
    true = (y_te != 0).astype(int)           # 1 = any attack

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(true, pred)
    f1  = f1_score(true, pred)
    print(f"    Anomaly detection -- Accuracy: {acc:.4f}  F1: {f1:.4f}")

    joblib.dump(iso, MODELS / "isolation_forest.pkl")
    print(f"    Saved -> {MODELS/'isolation_forest.pkl'}")
    return iso

# ── model 3: one-class SVM ────────────────────────────────────────────
def train_ocsvm(X_tr_s, y_tr, X_te_s, y_te):
    print("\n[3/3] Training One-Class SVM (unsupervised) ...")
    normal_mask = y_tr == 0
    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    oc.fit(X_tr_s[normal_mask])

    raw  = oc.predict(X_te_s)
    pred = (raw == -1).astype(int)
    true = (y_te != 0).astype(int)

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(true, pred)
    f1  = f1_score(true, pred)
    print(f"    Anomaly detection -- Accuracy: {acc:.4f}  F1: {f1:.4f}")

    joblib.dump(oc, MODELS / "oneclasssvm.pkl")
    print(f"    Saved -> {MODELS/'oneclasssvm.pkl'}")
    return oc

# ── plot helpers ──────────────────────────────────────────────────────
plt.style.use("dark_background")
PALETTE = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#c77dff"]

def _plot_confusion(y_te, y_pred, class_names, title):
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax
    )
    ax.set_title(f"{title} -- Confusion Matrix", fontsize=14, color="white")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    plt.tight_layout()
    out = REPORTS / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {out}")

def _plot_feature_importance(rf, features):
    imp = pd.Series(rf.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.cool(np.linspace(0.2, 0.9, len(imp)))
    imp.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Random Forest -- Feature Importance", fontsize=14, color="white")
    ax.set_xlabel("Importance", color="white")
    plt.tight_layout()
    out = REPORTS / "feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {out}")

def _plot_roc(y_te, y_prob, classes, class_names):
    y_bin = label_binarize(y_te, classes=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, name) in enumerate(zip(classes, class_names)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=PALETTE[i % len(PALETTE)])
    ax.plot([0,1],[0,1],"--", color="gray", alpha=0.5)
    ax.set_title("ROC Curves -- Random Forest", fontsize=14, color="white")
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = REPORTS / "roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {out}")

# ── main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AI Anomaly Detection -- Model Training Pipeline")
    print("=" * 60)

    X, y, names, class_names = load_data()
    X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te, scaler = split_and_scale(X, y)

    # sort class_names by label index for consistent ordering
    le = LabelEncoder().fit(y)
    ordered_names = [names[y == i][0] for i in le.classes_]

    joblib.dump(scaler, MODELS / "scaler.pkl")
    joblib.dump(le,     MODELS / "label_encoder.pkl")

    train_random_forest(X_tr_s, y_tr, X_te_s, y_te, ordered_names)
    train_isolation_forest(X_tr_s, y_tr, X_te_s, y_te)
    train_ocsvm(X_tr_s, y_tr, X_te_s, y_te)

    print("\n[OK] All models trained and saved to ./models/")
    print("[OK] Reports saved to ./reports/")

if __name__ == "__main__":
    main()
