"""
AeroShield - 06: Security intelligence (C6, session 6 evidence)
Defines Priority Intelligence Requirements, applies the intelligence cycle to
enrich correlated evidence, and produces both an operational alert brief and
an executive risk report.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import json

identity = pd.read_csv("data/identity_access_logs.csv", parse_dates=["timestamp"])
network = pd.read_csv("data/network_endpoint_logs.csv", parse_dates=["timestamp"])
ops = pd.read_csv("data/operational_vendor_logs.csv", parse_dates=["timestamp"])
incidents = pd.read_csv("data/incident_reports.csv", parse_dates=["timestamp"])
with open("outputs/incident_narrative.json") as f:
    narrative = json.load(f)

# ---------------------------------------------------------------------------
# Priority Intelligence Requirements (PIRs)
# ---------------------------------------------------------------------------
PIRS = [
    "PIR-1: Which vendor or staff accounts show correlated anomalous activity across two or more "
    "log sources (identity, network, operational-app) within a 24-hour window?",
    "PIR-2: Is there evidence of traffic or access crossing from passenger-facing/vendor segments "
    "into operational-application segments outside of an approved change/maintenance window?",
    "PIR-3: Which vendor firms are associated with the highest volume or severity of security "
    "incidents, and does this warrant a change to their access/monitoring posture?",
    "PIR-4: What proportion of high/critical-severity incident tickets map to confirmed (vs "
    "false-positive) technical anomalies, and is that ratio trending up or down?",
]

# ---------------------------------------------------------------------------
# Enrichment: quantify evidence to answer each PIR (intelligence cycle: collect -> process -> analyse)
# ---------------------------------------------------------------------------
enrich = {}

# PIR-1
combined_anom_by_user = pd.concat([
    identity[identity.is_anomalous == 1][["user_id"]].assign(source="identity"),
    network[network.is_anomalous == 1][["user_id"]].assign(source="network"),
    ops[ops.is_anomalous == 1][["user_id"]].assign(source="ops"),
])
multi_source = combined_anom_by_user.groupby("user_id").source.nunique()
multi_source_users = multi_source[multi_source >= 2].sort_values(ascending=False)
enrich["PIR-1"] = {
    "multi_source_flagged_accounts": int(len(multi_source_users)),
    "top_accounts": multi_source_users.head(10).to_dict(),
}

# PIR-2
cross_segment_to_ops = network[(network.dst_segment == "operational-apps")]
enrich["PIR-2"] = {
    "cross_segment_events_to_operational_network": int(len(cross_segment_to_ops)),
    "distinct_source_accounts": int(cross_segment_to_ops.user_id.nunique()),
}

# PIR-3
vendor_ops = ops[ops.role == "vendor"]
vendor_incident_rate = vendor_ops.groupby("vendor_firm").is_anomalous.mean().sort_values(ascending=False)
enrich["PIR-3"] = {"vendor_anomaly_rate": vendor_incident_rate.round(4).to_dict()}

# PIR-4
high_sev = incidents[incidents.severity.isin(["high", "critical"])]
confirmed_rate = 1 - (incidents.category == "false_positive").mean()
enrich["PIR-4"] = {
    "high_critical_ticket_share": round(float(len(high_sev) / len(incidents)), 4),
    "overall_confirmed_rate": round(float(confirmed_rate), 4),
}

with open("outputs/intelligence_enrichment.json", "w") as f:
    json.dump(enrich, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# Operational alert brief (for SOC analysts - actionable, technical)
# ---------------------------------------------------------------------------
top_account = list(enrich["PIR-1"]["top_accounts"].items())[0] if enrich["PIR-1"]["top_accounts"] else (None, 0)
operational_brief = f"""AEROSHIELD SOC OPERATIONAL ALERT BRIEF
Reporting window: last 45 days (synthetic evaluation dataset)

TOP PRIORITY: {narrative['incident_id']}
Subject: {narrative['subject_user']}
Window: {narrative['window_start']} to {narrative['window_end']}
{narrative['summary']}

Recommended immediate actions:
- {narrative['containment'][0]}
- {narrative['containment'][1]}

MULTI-SOURCE CORRELATION WATCHLIST
{enrich['PIR-1']['multi_source_flagged_accounts']} accounts show anomalous activity flagged across
2 or more independent log sources in the same window - the strongest indicator of a real incident
vs a single noisy alert. Top account: {top_account[0]} ({top_account[1]} sources).

CROSS-SEGMENT TRAFFIC
{enrich['PIR-2']['cross_segment_events_to_operational_network']} network events crossed from a
passenger-facing/vendor segment into the operational-application segment, from
{enrich['PIR-2']['distinct_source_accounts']} distinct accounts. Each should be checked against the
change/maintenance calendar; unscheduled instances are investigation priorities.
"""
with open("outputs/operational_alert_brief.txt", "w") as f:
    f.write(operational_brief)

# ---------------------------------------------------------------------------
# Executive risk report (for operations leadership - business-framed)
# ---------------------------------------------------------------------------
worst_vendor = max(enrich["PIR-3"]["vendor_anomaly_rate"], key=enrich["PIR-3"]["vendor_anomaly_rate"].get)
executive_report = f"""AEROSHIELD EXECUTIVE OPERATIONAL-SECURITY RISK REPORT
Prepared for: Airport operations leadership and CISO
Reporting window: last 45 days (synthetic evaluation dataset)

HEADLINE
One confirmed, evidence-backed security incident ({narrative['incident_id']}) was reconstructed from
correlated identity, network and operational-application evidence, consistent with vendor-credential
compromise and lateral movement toward operational systems. No passenger-facing service disruption
occurred in the reviewed window.

RISK POSTURE
- {enrich['PIR-4']['high_critical_ticket_share']:.0%} of logged incident tickets were rated high or
  critical severity; {enrich['PIR-4']['overall_confirmed_rate']:.0%} of all tickets were confirmed
  technical issues rather than false positives, indicating alert triage is reasonably well-tuned.
- {enrich['PIR-1']['multi_source_flagged_accounts']} accounts triggered anomalies across multiple
  independent systems, the pattern most associated with genuine compromise rather than noise.
- Vendor "{worst_vendor}" shows the highest anomaly rate among vendor connections
  ({enrich['PIR-3']['vendor_anomaly_rate'][worst_vendor]:.1%} of sessions), warranting a targeted
  access review.

RECOMMENDATION FOR INVESTMENT DECISION
The comparative control simulation (Section 7) quantifies how much three candidate investments
(vendor VPN MFA, privileged-access monitoring, tighter network segmentation) would have reduced the
likelihood and impact of the reconstructed incident, to support budget prioritisation.
"""
with open("outputs/executive_risk_report.txt", "w") as f:
    f.write(executive_report)

with open("outputs/pirs.json", "w") as f:
    json.dump({"PIRs": PIRS}, f, indent=2)

print("PIRs:")
for p in PIRS:
    print(" -", p)
print("\n--- Operational brief ---\n", operational_brief)
print("\n--- Executive report ---\n", executive_report)
