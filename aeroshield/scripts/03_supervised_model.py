"""
AeroShield - 03: Supervised ML detector (C4, session 3-4 evidence)
Random Forest classifier for anomalous identity/access events, evaluated with
confusion matrix, precision, recall, F1 and ROC-AUC.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, classification_report)
import joblib

NAVY = "#1B2A4A"; BLUE = "#2E5AAC"; RED = "#B3312C"

df = pd.read_csv("data/identity_access_logs.csv", parse_dates=["timestamp"])
df["hour"] = df.timestamp.dt.hour
df["is_off_hours"] = df.hour.isin([0, 1, 2, 3, 4, 23]).astype(int)
df["is_weekend"] = df.timestamp.dt.dayofweek.isin([5, 6]).astype(int)
df["is_fail"] = (df.outcome == "fail").astype(int)

# per-user rolling behaviour features (simple UBA-style baseline deviation)
user_resource_freq = df.groupby(["user_id", "resource"]).size().rename("user_resource_freq")
df = df.merge(user_resource_freq, on=["user_id", "resource"], how="left")

FEATURES_NUM = ["hour", "is_off_hours", "is_weekend", "is_fail", "user_resource_freq"]
FEATURES_CAT = ["role", "resource", "auth_method", "location"]
TARGET = "is_anomalous"

X = df[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=821, stratify=y
)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
], remainder="passthrough")

clf = Pipeline([
    ("pre", pre),
    ("rf", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=821, n_jobs=-1)),
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred, output_dict=True)

print("Confusion matrix:\n", cm)
print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}")

# ---------------------------------------------------------------------------
# Figure: confusion matrix + ROC curve
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Normal", "Anomalous"])
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["Normal", "Anomalous"])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion matrix (Random Forest)", fontsize=10, color=NAVY, weight="bold")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max()/2 else NAVY, fontsize=12, weight="bold")

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color=BLUE, lw=2, label=f"ROC-AUC = {auc:.3f}")
axes[1].plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
axes[1].set_title("ROC curve", fontsize=10, color=NAVY, weight="bold")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/figures/fig6_supervised_model_eval.png", dpi=180, facecolor="white")
plt.close()

# Feature importance
ohe = clf.named_steps["pre"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(FEATURES_CAT))
all_names = cat_names + FEATURES_NUM
importances = clf.named_steps["rf"].feature_importances_
imp_df = pd.DataFrame({"feature": all_names, "importance": importances}).sort_values(
    "importance", ascending=False).head(12)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.barh(imp_df.feature[::-1], imp_df.importance[::-1], color=BLUE)
ax.set_title("Top feature importances (Random Forest)", fontsize=10, color=NAVY, weight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/fig7_feature_importance.png", dpi=180, facecolor="white")
plt.close()

joblib.dump(clf, "outputs/models/supervised_rf_identity.joblib")

metrics = {
    "model": "RandomForestClassifier",
    "features": FEATURES_NUM + FEATURES_CAT,
    "n_train": len(X_train), "n_test": len(X_test),
    "confusion_matrix": cm.tolist(),
    "precision": round(precision, 4), "recall": round(recall, 4),
    "f1": round(f1, 4), "roc_auc": round(auc, 4),
    "classification_report": report,
    "top_features": imp_df.to_dict(orient="records"),
}
with open("outputs/supervised_model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

print("Saved model + evaluation figures.")
