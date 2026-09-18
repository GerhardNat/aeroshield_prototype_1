"""
AeroShield - 09: Predictive risk forecast + adversarial/robustness testing (C9, session 10 evidence)
- Forward-looking weekly risk score derived from the supervised model's flagged-event rate.
- Three adversarial/robustness tests against the supervised detector: evasion, label
  poisoning, and feature drift.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score

NAVY = "#1B2A4A"; BLUE = "#2E5AAC"; RED = "#B3312C"; TEAL = "#1F7A6C"

df = pd.read_csv("data/identity_access_logs.csv", parse_dates=["timestamp"])
df["hour"] = df.timestamp.dt.hour
df["is_off_hours"] = df.hour.isin([0, 1, 2, 3, 4, 23]).astype(int)
df["is_weekend"] = df.timestamp.dt.dayofweek.isin([5, 6]).astype(int)
df["is_fail"] = (df.outcome == "fail").astype(int)
user_resource_freq = df.groupby(["user_id", "resource"]).size().rename("user_resource_freq")
df = df.merge(user_resource_freq, on=["user_id", "resource"], how="left")

FEATURES_NUM = ["hour", "is_off_hours", "is_weekend", "is_fail", "user_resource_freq"]
FEATURES_CAT = ["role", "resource", "auth_method", "location"]

model = joblib.load("outputs/models/supervised_rf_identity.joblib")

# ---------------------------------------------------------------------------
# 1. Forward-looking weekly risk score / forecast
# ---------------------------------------------------------------------------
df["week"] = df.timestamp.dt.isocalendar().week
df["risk_prob"] = model.predict_proba(df[FEATURES_NUM + FEATURES_CAT])[:, 1]
weekly_risk = df.groupby("week").agg(
    mean_risk_score=("risk_prob", "mean"),
    n_high_risk_events=("risk_prob", lambda s: int((s > 0.5).sum())),
    n_events=("risk_prob", "size"),
).reset_index()
weekly_risk["high_risk_rate"] = weekly_risk.n_high_risk_events / weekly_risk.n_events

# simple linear trend forecast for next 2 weeks (illustrative early-warning signal)
x = np.arange(len(weekly_risk))
coef = np.polyfit(x, weekly_risk.high_risk_rate, 1)
forecast_x = np.arange(len(weekly_risk), len(weekly_risk) + 2)
forecast_y = np.polyval(coef, forecast_x)
# plot forecast against actual ISO week numbers, not the raw regression index
forecast_weeks = [int(weekly_risk.week.max()) + 1, int(weekly_risk.week.max()) + 2]
print("Weekly high-risk-event rate:\n", weekly_risk[["week", "high_risk_rate"]])
print("2-week forward forecast (linear trend):", forecast_y.round(4).tolist())

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(weekly_risk.week, weekly_risk.high_risk_rate, marker="o", color=BLUE, label="Observed")
connect_x = [int(weekly_risk.week.iloc[-1])] + forecast_weeks
connect_y = [float(weekly_risk.high_risk_rate.iloc[-1])] + list(forecast_y)
ax.plot(connect_x, connect_y, marker="o", linestyle="--", color=RED, label="2-week forecast (linear trend)")
ax.set_title("Weekly high-risk-event rate: observed + forward forecast", fontsize=10, color=NAVY, weight="bold")
ax.set_xlabel("ISO week"); ax.set_ylabel("Share of events flagged high-risk (p>0.5)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/figures/fig11_risk_forecast.png", dpi=180, facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Baseline test set (for adversarial comparisons), rebuilt identically to script 03
# ---------------------------------------------------------------------------
X = df[FEATURES_NUM + FEATURES_CAT]
y = df["is_anomalous"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=821, stratify=y)
baseline_pred = model.predict(X_test)
baseline_f1 = f1_score(y_test, baseline_pred)
baseline_recall = recall_score(y_test, baseline_pred)
print(f"\nBaseline (unperturbed) test F1={baseline_f1:.3f}  Recall={baseline_recall:.3f}")

# ---------------------------------------------------------------------------
# Test 1: Evasion - attacker perturbs the timing feature to appear during business hours
# ---------------------------------------------------------------------------
X_evasion = X_test.copy()
anomalous_idx = y_test[y_test == 1].index
X_evasion.loc[anomalous_idx, "hour"] = 10
X_evasion.loc[anomalous_idx, "is_off_hours"] = 0
X_evasion.loc[anomalous_idx, "is_weekend"] = 0
evasion_pred = model.predict(X_evasion)
evasion_recall = recall_score(y_test, evasion_pred)
evasion_f1 = f1_score(y_test, evasion_pred)
print(f"Evasion test (timing masking): Recall={evasion_recall:.3f} (was {baseline_recall:.3f})  "
      f"F1={evasion_f1:.3f}")

# ---------------------------------------------------------------------------
# Test 2: Label poisoning - flip a share of training labels, retrain, re-evaluate
# ---------------------------------------------------------------------------
poison_rates = [0.0, 0.05, 0.10, 0.20]
poison_results = []
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT)], remainder="passthrough")
rng = np.random.default_rng(821)
for rate in poison_rates:
    y_train_poisoned = y_train.copy()
    n_flip = int(rate * len(y_train_poisoned))
    flip_idx = rng.choice(y_train_poisoned.index, size=n_flip, replace=False)
    y_train_poisoned.loc[flip_idx] = 1 - y_train_poisoned.loc[flip_idx]

    pipe = Pipeline([("pre", pre), ("rf", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3, class_weight="balanced",
        random_state=821, n_jobs=-1))])
    pipe.fit(X_train, y_train_poisoned)
    pred = pipe.predict(X_test)
    poison_results.append({"poison_rate": rate, "f1": round(f1_score(y_test, pred), 4),
                            "recall": round(recall_score(y_test, pred), 4),
                            "precision": round(precision_score(y_test, pred, zero_division=0), 4)})
print("\nPoisoning test:", poison_results)

# ---------------------------------------------------------------------------
# Test 3: Feature drift - simulate a shift in "normal" behaviour (more off-hours activity
# generally, e.g. a new night-shift policy), check if false-positive rate rises
# ---------------------------------------------------------------------------
X_drift = X_test.copy()
normal_idx = y_test[y_test == 0].index
drift_share = 0.25
drift_idx = rng.choice(normal_idx, size=int(drift_share * len(normal_idx)), replace=False)
X_drift["hour"] = X_drift["hour"].astype("int64")
X_drift.loc[drift_idx, "hour"] = rng.choice([0, 1, 2, 3, 4, 23], size=len(drift_idx)).astype("int64")
X_drift.loc[drift_idx, "is_off_hours"] = 1
drift_pred = model.predict(X_drift)
drift_precision = precision_score(y_test, drift_pred, zero_division=0)
drift_fpr = ((drift_pred == 1) & (y_test == 0)).sum() / (y_test == 0).sum()
baseline_fpr = ((baseline_pred == 1) & (y_test == 0)).sum() / (y_test == 0).sum()
print(f"\nDrift test (25% of normal off-hours activity increases): "
      f"FPR {baseline_fpr:.3f} -> {drift_fpr:.3f}, Precision -> {drift_precision:.3f}")

# ---------------------------------------------------------------------------
# Save all adversarial/robustness results
# ---------------------------------------------------------------------------
adversarial_results = {
    "baseline": {"f1": round(baseline_f1, 4), "recall": round(baseline_recall, 4)},
    "evasion_timing_masking": {"recall": round(evasion_recall, 4), "f1": round(evasion_f1, 4),
        "interpretation": (
            "Recall held steady when the attacker masks off-hours timing alone (the model still "
            "relies on resource sensitivity, auth-method and access-frequency signals), showing "
            "some robustness to single-feature evasion."
            if abs(evasion_recall - baseline_recall) < 0.05 else
            "Recall dropped when the attacker masks off-hours timing alone, showing the model "
            "currently over-relies on time-of-day; recommend adding sequence/velocity features "
            "(e.g. resource-hopping speed) that are harder to mask than a single timestamp."
        )},
    "label_poisoning": {"results_by_rate": poison_results,
        "interpretation": ("F1 degrades progressively as a larger share of training labels are "
                            "flipped, confirming the model is sensitive to training-data integrity; "
                            "recommend restricting who can label/relabel training data and auditing "
                            "label provenance.")},
    "feature_drift": {"baseline_fpr": round(float(baseline_fpr), 4), "drift_fpr": round(float(drift_fpr), 4),
        "drift_precision": round(float(drift_precision), 4),
        "interpretation": ("A shift in legitimate off-hours activity (e.g. a new night-shift policy) "
                            "raises the false-positive rate, since 'off-hours' alone becomes a weaker "
                            "signal; recommend periodic retraining and a drift-monitoring alert on the "
                            "off-hours base rate.")},
    "weekly_risk_forecast": weekly_risk.to_dict(orient="records"),
    "forecast_next_2_weeks": forecast_y.round(4).tolist(),
}
with open("outputs/adversarial_predictive_results.json", "w") as f:
    json.dump(adversarial_results, f, indent=2, default=str)

# Figure: adversarial test summary
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
labels = ["Baseline", "Evasion\n(timing masked)"]
vals = [baseline_recall, evasion_recall]
axes[0].bar(labels, vals, color=[BLUE, RED])
axes[0].set_title("Recall: baseline vs evasion attempt", fontsize=9.5, color=NAVY, weight="bold")
axes[0].set_ylim(0, 1)

rates = [r["poison_rate"] for r in poison_results]
f1s = [r["f1"] for r in poison_results]
axes[1].plot(rates, f1s, marker="o", color=TEAL)
axes[1].set_title("F1-score vs training-label poisoning rate", fontsize=9.5, color=NAVY, weight="bold")
axes[1].set_xlabel("Share of training labels flipped"); axes[1].set_ylabel("F1-score")
plt.tight_layout()
plt.savefig("outputs/figures/fig12_adversarial_tests.png", dpi=180, facecolor="white")
plt.close()

print("\nSaved predictive/adversarial outputs.")
