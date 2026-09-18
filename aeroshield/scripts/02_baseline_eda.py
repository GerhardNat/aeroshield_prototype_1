"""
AeroShield - 02: Baseline EDA and visual profiling (C1, C3, session 1-2 evidence)
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.9)
NAVY = "#1B2A4A"; BLUE = "#2E5AAC"; TEAL = "#1F7A6C"; RED = "#B3312C"

identity = pd.read_csv("data/identity_access_logs.csv", parse_dates=["timestamp"])
network = pd.read_csv("data/network_endpoint_logs.csv", parse_dates=["timestamp"])
ops = pd.read_csv("data/operational_vendor_logs.csv", parse_dates=["timestamp"])
incidents = pd.read_csv("data/incident_reports.csv", parse_dates=["timestamp"])

identity["hour"] = identity.timestamp.dt.hour
network["hour"] = network.timestamp.dt.hour
ops["hour"] = ops.timestamp.dt.hour

# ---------------------------------------------------------------------------
# Figure 1: Access events by hour of day, normal vs anomalous
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, df, title in zip(axes, [identity, network], ["Identity/access events by hour", "Network events by hour"]):
    normal = df[df.is_anomalous == 0].groupby("hour").size().reindex(range(24), fill_value=0)
    anomalous = df[df.is_anomalous == 1].groupby("hour").size().reindex(range(24), fill_value=0)
    ax.bar(range(24), normal.values, color=BLUE, alpha=0.75, label="Normal")
    ax.bar(range(24), anomalous.values, bottom=0,
           color=RED, alpha=0.9, label="Labelled anomalous")
    ax.set_title(title, fontsize=10, color=NAVY, weight="bold")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Event count")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/figures/fig1_hourly_baseline.png", dpi=180, facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: Network bytes distribution, normal vs anomalous (log scale)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
sns.kdeplot(data=network[network.is_anomalous == 0], x="bytes", log_scale=True, ax=ax,
            color=BLUE, fill=True, alpha=0.35, label="Normal")
sns.kdeplot(data=network[network.is_anomalous == 1], x="bytes", log_scale=True, ax=ax,
            color=RED, fill=True, alpha=0.5, label="Labelled anomalous")
ax.set_title("Network transfer volume: normal vs anomalous", fontsize=10, color=NAVY, weight="bold")
ax.set_xlabel("Bytes transferred (log scale)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/figures/fig2_network_bytes_baseline.png", dpi=180, facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: Cross-segment traffic baseline (network destination segment counts)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
seg_counts = network.groupby(["src_segment", "dst_segment"]).size().unstack(fill_value=0)
sns.heatmap(seg_counts, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
ax.set_title("Network segment-to-segment event counts", fontsize=10, color=NAVY, weight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/fig3_segment_baseline.png", dpi=180, facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure 4: Operational privilege-level actions by role
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
priv_counts = ops.groupby(["role", "privilege_level"]).size().unstack(fill_value=0)
priv_counts.plot(kind="bar", stacked=True, ax=ax, color=[BLUE, RED])
ax.set_title("Operational-application actions: privilege level by role", fontsize=10, color=NAVY, weight="bold")
ax.set_xlabel(""); ax.set_ylabel("Session count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/figures/fig4_ops_privilege_baseline.png", dpi=180, facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure 5: Incident category / severity mix
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
incidents.category.value_counts().plot(kind="barh", ax=axes[0], color=TEAL)
axes[0].set_title("Incident tickets by category", fontsize=10, color=NAVY, weight="bold")
sev_order = ["low", "medium", "high", "critical"]
incidents.severity.value_counts().reindex(sev_order).plot(kind="bar", ax=axes[1], color=[BLUE, BLUE, RED, RED])
axes[1].set_title("Incident tickets by severity", fontsize=10, color=NAVY, weight="bold")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/figures/fig5_incident_baseline.png", dpi=180, facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Descriptive statistics summary (saved for the data dictionary / cleaning log)
# ---------------------------------------------------------------------------
summary = {
    "identity_access_logs": {
        "rows": len(identity), "anomalous": int(identity.is_anomalous.sum()),
        "date_range": [str(identity.timestamp.min()), str(identity.timestamp.max())],
        "unique_users": identity.user_id.nunique(),
        "fail_rate": round(float((identity.outcome == "fail").mean()), 4),
    },
    "network_endpoint_logs": {
        "rows": len(network), "anomalous": int(network.is_anomalous.sum()),
        "unique_devices": network.device_id.nunique(),
        "mean_bytes": round(float(network.bytes.mean()), 1),
        "median_bytes": round(float(network.bytes.median()), 1),
    },
    "operational_vendor_logs": {
        "rows": len(ops), "anomalous": int(ops.is_anomalous.sum()),
        "elevated_privilege_share": round(float((ops.privilege_level == "elevated").mean()), 4),
    },
    "incident_reports": {
        "rows": len(incidents),
        "category_counts": incidents.category.value_counts().to_dict(),
        "severity_counts": incidents.severity.value_counts().to_dict(),
    },
}
import json
with open("outputs/eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(json.dumps(summary, indent=2, default=str))
print("Saved 5 baseline figures to outputs/figures/")
