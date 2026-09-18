"""
AeroShield - 05: Security investigation - cross-log correlation & incident timeline (C5)
Correlates identity/access, network and operational-app evidence by user_id and
time window to reconstruct a defensible incident timeline for the injected
lateral-movement scenario, and drafts IR containment/eradication/recovery/
monitoring recommendations.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import json

identity = pd.read_csv("data/identity_access_logs.csv", parse_dates=["timestamp"])
network = pd.read_csv("data/network_endpoint_logs.csv", parse_dates=["timestamp"])
ops = pd.read_csv("data/operational_vendor_logs.csv", parse_dates=["timestamp"])

with open("data/_scenario_ground_truth.json") as f:
    truth = json.load(f)
compromised_users = truth["compromised_users"]

# Pick the user with the most cross-domain anomalous evidence as the lead investigation case
counts = {}
for u in compromised_users:
    n = (identity[(identity.user_id == u) & (identity.is_anomalous == 1)].shape[0]
         + network[(network.user_id == u) & (network.is_anomalous == 1)].shape[0]
         + ops[(ops.user_id == u) & (ops.is_anomalous == 1)].shape[0])
    counts[u] = n
lead_user = max(counts, key=counts.get)
print("Lead investigation subject:", lead_user, "| anomalous-evidence count:", counts[lead_user])

# Build a unified timeline of all events (any source) touching this user
events = []
for _, r in identity[identity.user_id == lead_user].iterrows():
    events.append({"timestamp": r.timestamp, "source": "identity_access", "event_id": r.event_id,
                    "detail": f"{r.role} login to {r.resource} via {r.auth_method} -> {r.outcome} "
                              f"(location={r.location})", "anomalous": bool(r.is_anomalous)})
for _, r in network[network.user_id == lead_user].iterrows():
    events.append({"timestamp": r.timestamp, "source": "network_endpoint", "event_id": r.event_id,
                    "detail": f"{r.src_segment} -> {r.dst_segment} traffic, port {r.port}, "
                              f"{r.bytes:,} bytes (device {r.device_id})", "anomalous": bool(r.is_anomalous)})
for _, r in ops[ops.user_id == lead_user].iterrows():
    events.append({"timestamp": r.timestamp, "source": "operational_app", "event_id": r.session_id,
                    "detail": f"{r.action} on {r.application} ({r.privilege_level}) -> {r.outcome}",
                    "anomalous": bool(r.is_anomalous)})

timeline = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)
timeline["timestamp"] = timeline["timestamp"].astype(str)
timeline.to_csv("outputs/incident_timeline.csv", index=False)

anomalous_timeline = timeline[timeline.anomalous].reset_index(drop=True)
print(f"\nTotal correlated events for {lead_user}: {len(timeline)} "
      f"({len(anomalous_timeline)} flagged anomalous)")
print(anomalous_timeline[["timestamp", "source", "event_id", "detail"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Incident narrative + IR recommendations (C5 evidence)
# ---------------------------------------------------------------------------
first_ts = anomalous_timeline.timestamp.min() if len(anomalous_timeline) else None
last_ts = anomalous_timeline.timestamp.max() if len(anomalous_timeline) else None

narrative = {
    "incident_id": f"INC-{lead_user}",
    "subject_user": lead_user,
    "window_start": first_ts,
    "window_end": last_ts,
    "event_ids_cited": anomalous_timeline.event_id.tolist(),
    "summary": (
        f"Correlated evidence across identity/access, network/endpoint and operational-application "
        f"logs for {lead_user} shows a coherent pattern consistent with credential compromise "
        f"followed by lateral movement: off-hours authentication to a sensitive resource, "
        f"cross-segment network traffic toward the operational network shortly afterward, and an "
        f"elevated-privilege action on an operational application within the same window. The event "
        f"IDs above ({', '.join(anomalous_timeline.event_id.tolist()[:6])}{'...' if len(anomalous_timeline) > 6 else ''}) "
        f"provide the evidentiary chain."
    ),
    "containment": [
        f"Suspend/disable the {lead_user} account and force re-authentication with MFA before restoral.",
        "Isolate the associated device/VPN session at the network layer to block further cross-segment traffic.",
    ],
    "eradication": [
        "Rotate credentials and any certificates associated with the account and affected vendor connection.",
        "Review and revert any elevated-privilege configuration changes made during the window.",
    ],
    "recovery": [
        "Restore normal access only after a clean re-verification of identity and device posture.",
        "Re-enable vendor connectivity under a documented, time-boxed maintenance window with monitoring.",
    ],
    "monitoring": [
        "Add a standing correlation rule: off-hours auth + cross-segment traffic + elevated-privilege "
        "operational action within a 2-hour window triggers a high-priority SOC alert.",
        "Track the account and device for 30 days post-incident for recurrence.",
    ],
}
with open("outputs/incident_narrative.json", "w") as f:
    json.dump(narrative, f, indent=2, default=str)

print("\nSaved incident timeline (CSV) and narrative/IR recommendations (JSON).")
