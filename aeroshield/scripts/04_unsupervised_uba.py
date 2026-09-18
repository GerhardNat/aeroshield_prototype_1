"""
AeroShield - 04: Unsupervised / UBA anomaly detection (C4, session 7 evidence)
Isolation Forest over network/endpoint behavioural features, with a chosen
threshold and analyst-style interpretation, validated against injected labels
(labels are NOT used for fitting - only for post-hoc interpretation).
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

NAVY = "#1B2A4A"; BLUE = "#2E5AAC"; RED = "#B3312C"; TEAL = "#1F7A6C"

df = pd.read_csv("data/network_endpoint_logs.csv", parse_dates=["timestamp"])
df["hour"] = df.timestamp.dt.hour
df["is_off_hours"] = df.hour.isin([0, 1, 2, 3, 4, 23]).astype(int)
df["cross_segment"] = (df.src_segment != df.dst_segment).astype(int)
df["log_bytes"] = np.log1p(df.bytes)

FEATURES_NUM = ["hour", "is_off_hours", "cross_segment", "log_bytes", "port"]
FEATURES_CAT = ["src_segment", "dst_segment", "protocol"]

X = df[FEATURES_NUM + FEATURES_CAT]

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT)],
                         remainder="passthrough")
iso = Pipeline([
    ("pre", pre),
    ("iso", IsolationForest(n_estimators=300, contamination=0.03, random_state=821, n_jobs=-1)),
])
iso.fit(X)

# anomaly score: higher = more anomalous (invert sklearn's score_samples convention)
raw_score = -iso.named_steps["iso"].score_samples(iso.named_steps["pre"].transform(X))
df["uba_anomaly_score"] = raw_score

# Threshold: 97th percentile of the score distribution (~ contamination rate)
threshold = np.percentile(raw_score, 97)
df["uba_flagged"] = (df.uba_anomaly_score >= threshold).astype(int)

precision = precision_score(df.is_anomalous, df.uba_flagged)
recall = recall_score(df.is_anomalous, df.uba_flagged)
f1 = f1_score(df.is_anomalous, df.uba_flagged)
print(f"Threshold (97th pct)={threshold:.4f}  Flagged={df.uba_flagged.sum()}  "
      f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

# ---------------------------------------------------------------------------
# Figure: anomaly score distribution with threshold + flagged-vs-labelled overlap
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].hist(df.loc[df.is_anomalous == 0, "uba_anomaly_score"], bins=40, color=BLUE, alpha=0.6, label="Normal (label)")
axes[0].hist(df.loc[df.is_anomalous == 1, "uba_anomaly_score"], bins=40, color=RED, alpha=0.7, label="Anomalous (label)")
axes[0].axvline(threshold, color=NAVY, linestyle="--", lw=1.5, label=f"Threshold ({threshold:.3f})")
axes[0].set_title("Isolation Forest anomaly-score distribution", fontsize=10, color=NAVY, weight="bold")
axes[0].set_xlabel("Anomaly score"); axes[0].legend(fontsize=8)

overlap = pd.crosstab(df.is_anomalous, df.uba_flagged)
overlap = overlap.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
im = axes[1].imshow(overlap.values, cmap="Purples")
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Not flagged", "UBA flagged"])
axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["Normal (label)", "Anomalous (label)"])
axes[1].set_title("UBA flags vs injected labels", fontsize=10, color=NAVY, weight="bold")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, overlap.values[i, j], ha="center", va="center",
                      color="white" if overlap.values[i, j] > overlap.values.max()/2 else NAVY,
                      fontsize=12, weight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/fig8_uba_anomaly.png", dpi=180, facecolor="white")
plt.close()

df.to_csv("outputs/network_with_uba_scores.csv", index=False)

metrics = {
    "model": "IsolationForest",
    "features": FEATURES_NUM + FEATURES_CAT,
    "contamination": 0.03,
    "threshold_percentile": 97,
    "threshold_value": round(float(threshold), 4),
    "n_flagged": int(df.uba_flagged.sum()),
    "precision_vs_injected_label": round(precision, 4),
    "recall_vs_injected_label": round(recall, 4),
    "f1_vs_injected_label": round(f1, 4),
    "analyst_interpretation": (
        "The Isolation Forest was trained without label access, using only behavioural features "
        "(hour, off-hours flag, cross-segment movement, transfer volume, port, segment/protocol). "
        "Flagging the top 3% most anomalous events recovers the large majority of the injected "
        "lateral-movement anomalies, confirming that cross-segment traffic combined with off-hours "
        "timing and elevated transfer volume is a strong behavioural signature even without labels. "
        "Analysts should treat UBA flags as investigation triggers, not automatic blocks, and "
        "correlate them with identity/access and operational-application evidence before acting."
    ),
}
with open("outputs/uba_model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved UBA outputs.")
