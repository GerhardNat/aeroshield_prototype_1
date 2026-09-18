"""
AeroShield - 10: Real aircraft-engine OT sensor telemetry (NASA C-MAPSS)
Data source #5 (real, not synthetic). Framed as operational-technology (OT)
sensor monitoring for airport/aviation systems: the same anomaly-detection
and early-warning methodology used elsewhere in this project (baseline
deviation -> flagged event -> analyst investigation) applied to real
degradation telemetry. Unexplained sensor drift is operationally relevant
whether its root cause is genuine mechanical wear or sensor
tampering/spoofing - both require investigation under an OT security lens.

Primary subset: FD001 (single operating condition, single fault mode -
cleanest for a clear demonstration). FD002-FD004 are downloaded and
available in data/cmapss/ for extension but not modelled here.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

NAVY = "#1B2A4A"; BLUE = "#2E5AAC"; TEAL = "#1F7A6C"; RED = "#B3312C"; AMBER = "#C98A1A"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"sensor_{i}" for i in range(1, 22)]
RUL_CAP = 125  # standard piecewise-linear RUL cap (Heimes 2008 / widely used convention)

train = pd.read_csv("data/cmapss/train_FD001.txt", sep=r"\s+", header=None, names=COLS)
test = pd.read_csv("data/cmapss/test_FD001.txt", sep=r"\s+", header=None, names=COLS)
rul_truth = pd.read_csv("data/cmapss/RUL_FD001.txt", header=None, names=["RUL"])

print(f"Train: {train.unit.nunique()} units, {len(train)} rows | Test: {test.unit.nunique()} units, {len(test)} rows")

# ---------------------------------------------------------------------------
# Drop near-constant sensors (well-documented for FD001: single operating
# condition means several sensors never vary and carry no signal)
# ---------------------------------------------------------------------------
sensor_cols = [c for c in COLS if c.startswith("sensor_")]
stds = train[sensor_cols].std()
useful_sensors = stds[stds > 1e-4].index.tolist()
dropped_sensors = [c for c in sensor_cols if c not in useful_sensors]
print(f"Dropped {len(dropped_sensors)} near-constant sensors: {dropped_sensors}")
print(f"Retained {len(useful_sensors)} informative sensors: {useful_sensors}")

# ---------------------------------------------------------------------------
# RUL labels: train = max_cycle(unit) - cycle, capped at RUL_CAP
# ---------------------------------------------------------------------------
max_cycle = train.groupby("unit").cycle.transform("max")
train["RUL"] = np.minimum(max_cycle - train.cycle, RUL_CAP)

FEATURES = useful_sensors  # deliberately excludes raw cycle number: absolute cycle count
                           # does not generalise to the official test protocol, where units
                           # are censored at an arbitrary cycle unrelated to their total life -
                           # RUL must be inferred from the sensor degradation state itself

# Rolling-window features (window=5 cycles) capture degradation trend and reduce
# single-reading sensor noise - standard practice for this benchmark.
def add_rolling_features(df, window=5):
    df = df.sort_values(["unit", "cycle"]).copy()
    roll_mean = df.groupby("unit")[useful_sensors].transform(
        lambda s: s.rolling(window, min_periods=1).mean())
    roll_std = df.groupby("unit")[useful_sensors].transform(
        lambda s: s.rolling(window, min_periods=1).std().fillna(0))
    roll_mean.columns = [f"{c}_rmean" for c in useful_sensors]
    roll_std.columns = [f"{c}_rstd" for c in useful_sensors]
    return pd.concat([df, roll_mean, roll_std], axis=1)

train = add_rolling_features(train)
test = add_rolling_features(test)
FEATURES = useful_sensors + [f"{c}_rmean" for c in useful_sensors] + [f"{c}_rstd" for c in useful_sensors]

# ---------------------------------------------------------------------------
# Figure 1: raw sensor degradation trajectories for a handful of sample units
# ---------------------------------------------------------------------------
sample_units = [1, 24, 55, 81]
show_sensors = ["sensor_2", "sensor_4", "sensor_11", "sensor_15"]
fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))
for ax, sc in zip(axes.flat, show_sensors):
    for u, c in zip(sample_units, [NAVY, BLUE, TEAL, AMBER]):
        sub = train[train.unit == u]
        ax.plot(sub.cycle, sub[sc], color=c, alpha=0.85, lw=1.2, label=f"Unit {u}")
    ax.set_title(sc, fontsize=9.5, color=NAVY, weight="bold")
    ax.set_xlabel("Cycle"); ax.set_ylabel("Reading")
axes[0, 0].legend(fontsize=7)
plt.tight_layout()
plt.savefig("outputs/figures/fig13_engine_sensor_trajectories.png", dpi=180, facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Supervised: Random Forest RUL regression (group split by unit - no leakage)
# ---------------------------------------------------------------------------
groups = train.unit
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=821)
train_idx, val_idx = next(gss.split(train, groups=groups))
X_train, X_val = train.iloc[train_idx][FEATURES], train.iloc[val_idx][FEATURES]
y_train, y_val = train.iloc[train_idx].RUL, train.iloc[val_idx].RUL

rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=4,
                            random_state=821, n_jobs=-1)
rf.fit(X_train, y_train)
val_pred = rf.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
val_rmse = mean_squared_error(y_val, val_pred) ** 0.5
print(f"Validation (held-out units): MAE={val_mae:.2f} cycles, RMSE={val_rmse:.2f} cycles")

# ---------------------------------------------------------------------------
# Official test protocol: predict RUL at the LAST observed cycle of each test
# unit, compare against RUL_FD001.txt ground truth, using the NASA/PHM08
# asymmetric scoring function (penalises late/optimistic predictions more).
# ---------------------------------------------------------------------------
last_cycle = test.loc[test.groupby("unit").cycle.idxmax()].sort_values("unit").reset_index(drop=True)
assert (last_cycle.unit.values == np.arange(1, len(last_cycle) + 1)).all(), "test units must align 1..N with RUL_FD001.txt row order"
test_pred = rf.predict(last_cycle[FEATURES])
test_pred_capped = np.minimum(test_pred, RUL_CAP)
actual = rul_truth.RUL.values

test_mae = mean_absolute_error(actual, test_pred_capped)
test_rmse = mean_squared_error(actual, test_pred_capped) ** 0.5

def phm08_score(pred, true):
    d = pred - true
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(s.sum())

score = phm08_score(test_pred_capped, actual)
print(f"Official test set (last cycle per unit, n={len(actual)}): "
      f"MAE={test_mae:.2f}  RMSE={test_rmse:.2f}  PHM08 score={score:.1f}")

# Figure: predicted vs actual RUL + error distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(actual, test_pred_capped, alpha=0.6, color=BLUE, s=18)
lims = [0, max(actual.max(), test_pred_capped.max()) + 5]
axes[0].plot(lims, lims, color=RED, linestyle="--", lw=1.2)
axes[0].set_xlabel("Actual RUL (cycles)"); axes[0].set_ylabel("Predicted RUL (cycles)")
axes[0].set_title(f"Predicted vs actual RUL (test set)\nMAE={test_mae:.1f}, RMSE={test_rmse:.1f}, PHM08={score:.0f}",
                   fontsize=9.5, color=NAVY, weight="bold")

errors = test_pred_capped - actual
axes[1].hist(errors, bins=25, color=TEAL, alpha=0.8)
axes[1].axvline(0, color=NAVY, linestyle="--", lw=1)
axes[1].set_title("Prediction error distribution\n(positive = late/optimistic prediction)",
                   fontsize=9.5, color=NAVY, weight="bold")
axes[1].set_xlabel("Predicted - actual RUL (cycles)")
plt.tight_layout()
plt.savefig("outputs/figures/fig14_engine_rul_prediction.png", dpi=180, facecolor="white")
plt.close()

# Feature importance
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(6.3, 4))
ax.barh(imp.index[::-1], imp.values[::-1], color=BLUE)
ax.set_title("Top feature importances (RUL regression)", fontsize=9.5, color=NAVY, weight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/fig15_engine_feature_importance.png", dpi=180, facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Unsupervised: degradation-onset / drift anomaly detection
# Baseline = each unit's own first 15% of cycles (assumed healthy). Score =
# mean absolute z-score across informative sensors vs that baseline. Onset =
# first cycle where a rolling-5-cycle mean score exceeds a fixed threshold.
# ---------------------------------------------------------------------------
onset_results = []
threshold = 2.5
for u, g in train.groupby("unit"):
    g = g.sort_values("cycle").reset_index(drop=True)
    n_baseline = max(5, int(0.15 * len(g)))
    baseline = g.iloc[:n_baseline][useful_sensors]
    mu, sigma = baseline.mean(), baseline.std().replace(0, 1e-6)
    z = ((g[useful_sensors] - mu) / sigma).abs().mean(axis=1)
    roll = z.rolling(5, min_periods=1).mean()
    over = roll[roll > threshold]
    onset_cycle = int(over.index[0]) + 1 if len(over) else None
    life = int(g.cycle.max())
    lead_time = (life - onset_cycle) if onset_cycle else None
    onset_results.append({"unit": int(u), "life_cycles": life, "onset_cycle": onset_cycle,
                           "lead_time_cycles": lead_time})

onset_df = pd.DataFrame(onset_results)
detected = onset_df.dropna(subset=["onset_cycle"])
detect_rate = len(detected) / len(onset_df)
mean_lead = detected.lead_time_cycles.mean()
print(f"\nDegradation-onset detection: flagged {len(detected)}/{len(onset_df)} units "
      f"({detect_rate:.0%}); mean early-warning lead time = {mean_lead:.1f} cycles before failure")
onset_df.to_csv("outputs/engine_degradation_onset.csv", index=False)

# Figure: onset-detection score trace for one sample unit + lead-time distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sample_u = 24
g = train[train.unit == sample_u].sort_values("cycle").reset_index(drop=True)
n_baseline = max(5, int(0.15 * len(g)))
baseline = g.iloc[:n_baseline][useful_sensors]
mu, sigma = baseline.mean(), baseline.std().replace(0, 1e-6)
z = ((g[useful_sensors] - mu) / sigma).abs().mean(axis=1)
roll = z.rolling(5, min_periods=1).mean()
axes[0].plot(g.cycle, roll, color=BLUE)
axes[0].axhline(threshold, color=RED, linestyle="--", lw=1.2, label=f"Threshold ({threshold})")
onset_row = onset_df[onset_df.unit == sample_u].iloc[0]
if pd.notna(onset_row.onset_cycle):
    axes[0].axvline(onset_row.onset_cycle, color=AMBER, linestyle=":", lw=1.5,
                     label=f"Detected onset (cycle {int(onset_row.onset_cycle)})")
axes[0].set_title(f"Unit {sample_u}: degradation-onset score over life "
                   f"(failed at cycle {int(onset_row.life_cycles)})", fontsize=9, color=NAVY, weight="bold")
axes[0].set_xlabel("Cycle"); axes[0].set_ylabel("Mean |z-score| vs healthy baseline")
axes[0].legend(fontsize=7.5)

axes[1].hist(detected.lead_time_cycles, bins=20, color=TEAL, alpha=0.85)
axes[1].axvline(mean_lead, color=RED, linestyle="--", lw=1.2, label=f"Mean = {mean_lead:.0f} cycles")
axes[1].set_title("Early-warning lead time across all detected units", fontsize=9.5, color=NAVY, weight="bold")
axes[1].set_xlabel("Cycles before failure that onset was flagged")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/figures/fig16_engine_onset_detection.png", dpi=180, facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------------
metrics = {
    "source": "NASA PCoE C-MAPSS (FD001) - real turbofan engine sensor telemetry",
    "train_units": int(train.unit.nunique()), "train_rows": int(len(train)),
    "test_units": int(test.unit.nunique()),
    "sensors_retained": useful_sensors, "sensors_dropped_near_constant": dropped_sensors,
    "rul_cap": RUL_CAP,
    "validation_mae_cycles": round(val_mae, 2), "validation_rmse_cycles": round(val_rmse, 2),
    "test_mae_cycles": round(test_mae, 2), "test_rmse_cycles": round(test_rmse, 2),
    "test_phm08_score": round(score, 1),
    "degradation_onset_detection": {
        "threshold_mean_abs_zscore": threshold,
        "units_flagged": int(len(detected)), "units_total": int(len(onset_df)),
        "detection_rate": round(detect_rate, 4),
        "mean_lead_time_cycles": round(float(mean_lead), 1),
    },
    "interpretation": (
        "The Random Forest RUL regressor, trained only on real NASA sensor telemetry, "
        "achieves a competitive PHM08 score for this benchmark using a simple feature set. "
        "Independently, unsupervised baseline-deviation monitoring (the same style of "
        "technique used on the synthetic identity/network logs elsewhere in this project) "
        "flags degradation onset with a substantial early-warning lead time before failure, "
        "demonstrating that this OT-monitoring approach transfers from synthetic security "
        "logs to real operational sensor data. An airport SOC could apply the same baseline-"
        "deviation logic to ground-support-equipment or engine telemetry feeds to flag "
        "anomalies worth investigating, whether the root cause turns out to be mechanical "
        "wear or sensor tampering/spoofing."
    ),
}
with open("outputs/engine_telemetry_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

print("\nSaved real-engine-telemetry outputs (figures 13-16 + metrics JSON).")
