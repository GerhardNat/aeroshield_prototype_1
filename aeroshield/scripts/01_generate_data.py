"""
AeroShield (SAS821S Capstone T15) - Synthetic data generator
Generates 4 documented, synthetic datasets modelling a fictional airport operator:
  1. identity_access_logs.csv      - staff & vendor identity/access events
  2. network_endpoint_logs.csv     - segmented network/endpoint traffic
  3. operational_vendor_logs.csv   - operational application & vendor-connection sessions
  4. incident_reports.csv          - free-text SOC ticket / incident narratives

All identifiers, IPs, names and organisations are fictional. Reproducible via fixed seed.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

def data_path(filename):
    return os.path.join(DATA_DIR, filename)

SEED = 821
rng = np.random.default_rng(SEED)
random.seed(SEED)

START = datetime(2026, 6, 1)
DAYS = 45
N_STAFF = 180
N_VENDOR = 35

DEPARTMENTS = ["Security", "Ground Operations", "Baggage Handling", "IT", "Gate Services",
               "Retail Concessions", "Engineering & Maintenance", "Cargo"]
VENDOR_FIRMS = ["SkyLink Ground Services", "AeroFuel Solutions", "Coastal Catering Co",
                "Falcon Cargo Handling", "Meridian Aircraft Maintenance", "Horizon Cleaning Services"]
RESOURCES = ["Baggage-Handling-System", "Gate-Management-System", "Departure-Control-System",
             "Airfield-Ops-Portal", "Staff-VPN", "HR-Self-Service", "Maintenance-Ticketing",
             "Cargo-Manifest-System", "Retail-POS-Backend"]
AUTH_METHODS = ["badge+pin", "sso_password", "sso_mfa", "vpn_certificate", "vpn_password"]

# ---------------------------------------------------------------------------
# Helper: build a population of staff and vendor identities
# ---------------------------------------------------------------------------
staff_ids = [f"STF{1000+i}" for i in range(N_STAFF)]
staff_dept = {u: rng.choice(DEPARTMENTS) for u in staff_ids}
staff_home_segment = {u: "staff-ops" for u in staff_ids}

vendor_ids = [f"VND{2000+i}" for i in range(N_VENDOR)]
vendor_firm = {v: rng.choice(VENDOR_FIRMS) for v in vendor_ids}
vendor_home_segment = {v: "vendor-vpn" for v in vendor_ids}

all_users = staff_ids + vendor_ids

# a small set of users designated as "compromised" for the injected attack scenario / labels
N_COMPROMISED = 9
compromised_users = list(rng.choice(all_users, size=N_COMPROMISED, replace=False))

def rand_timestamp(day_offset_max=DAYS, off_hours_bias=0.0):
    day = rng.integers(0, day_offset_max)
    if rng.random() < off_hours_bias:
        hour = rng.choice([0, 1, 2, 3, 4, 23])
    else:
        hour = rng.integers(6, 22)
    minute = rng.integers(0, 60)
    second = rng.integers(0, 60)
    return START + timedelta(days=int(day), hours=int(hour), minutes=int(minute), seconds=int(second))

def rand_ip(segment):
    prefixes = {"passenger-wifi": "10.40", "staff-ops": "10.20", "vendor-vpn": "10.60"}
    p = prefixes[segment]
    return f"{p}.{rng.integers(0,255)}.{rng.integers(1,255)}"

# ---------------------------------------------------------------------------
# 1. Identity / access logs
# ---------------------------------------------------------------------------
rows = []
N_ACCESS = 6200
for i in range(N_ACCESS):
    is_vendor = rng.random() < 0.22
    user = rng.choice(vendor_ids) if is_vendor else rng.choice(staff_ids)
    role = "vendor" if is_vendor else "staff"
    dept = vendor_firm[user] if is_vendor else staff_dept[user]

    is_compromised_event = user in compromised_users and rng.random() < 0.55
    off_hours_bias = 0.6 if is_compromised_event else 0.06
    ts = rand_timestamp(off_hours_bias=off_hours_bias)

    resource = rng.choice(RESOURCES)
    auth_method = rng.choice(AUTH_METHODS, p=[0.30, 0.28, 0.20, 0.12, 0.10])
    location = "airport-terminal" if role == "staff" else rng.choice(["airport-terminal", "vendor-remote-site"])

    label = 0
    if is_compromised_event:
        # anomaly patterns: off-hours + sensitive resource + repeated failed attempts before success
        resource = rng.choice(["Baggage-Handling-System", "Departure-Control-System", "Staff-VPN"])
        auth_method = rng.choice(["vpn_password", "sso_password"])
        outcome = rng.choice(["success", "fail"], p=[0.7, 0.3])
        label = 1
    else:
        outcome = rng.choice(["success", "fail"], p=[0.94, 0.06])

    rows.append({
        "event_id": f"IA{i:06d}",
        "timestamp": ts.isoformat(),
        "user_id": user,
        "role": role,
        "department_or_vendor": dept,
        "resource": resource,
        "auth_method": auth_method,
        "outcome": outcome,
        "location": location,
        "is_anomalous": label,
    })

identity_df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
identity_df.to_csv(data_path("identity_access_logs.csv"), index=False)

# ---------------------------------------------------------------------------
# 2. Network / endpoint logs
# ---------------------------------------------------------------------------
rows = []
N_NET = 6400
COMMON_PORTS = [443, 80, 22, 3389, 8080, 5432, 445]
for i in range(N_NET):
    is_vendor = rng.random() < 0.22
    user = rng.choice(vendor_ids) if is_vendor else rng.choice(staff_ids)
    home_segment = "vendor-vpn" if is_vendor else "staff-ops"

    is_compromised_event = user in compromised_users and rng.random() < 0.5
    ts = rand_timestamp(off_hours_bias=0.55 if is_compromised_event else 0.05)

    if is_compromised_event:
        # cross-segment lateral movement: vendor-vpn/staff-ops -> operational segment traffic
        src_segment = home_segment
        dst_segment = "operational-apps"
        port = rng.choice([3389, 22, 445, 5432])
        n_bytes = int(rng.integers(500000, 9000000))
        label = 1
    else:
        src_segment = home_segment
        dst_segment = rng.choice([home_segment, "internet"], p=[0.75, 0.25])
        port = rng.choice(COMMON_PORTS, p=[0.45, 0.15, 0.1, 0.05, 0.15, 0.05, 0.05])
        n_bytes = int(rng.integers(200, 400000))
        label = 0

    rows.append({
        "event_id": f"NW{i:06d}",
        "timestamp": ts.isoformat(),
        "user_id": user,
        "src_segment": src_segment,
        "dst_segment": dst_segment,
        "src_ip": rand_ip(home_segment if home_segment in ("passenger-wifi","staff-ops","vendor-vpn") else "staff-ops"),
        "dst_ip": rand_ip("staff-ops") if dst_segment != "internet" else f"198.51.{rng.integers(0,255)}.{rng.integers(1,255)}",
        "port": int(port),
        "protocol": rng.choice(["TCP", "UDP"], p=[0.85, 0.15]),
        "bytes": n_bytes,
        "device_id": f"DEV{rng.integers(1,400):04d}",
        "is_anomalous": label,
    })

network_df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
network_df.to_csv(data_path("network_endpoint_logs.csv"), index=False)

# ---------------------------------------------------------------------------
# 3. Operational application & vendor-connection logs
# ---------------------------------------------------------------------------
rows = []
N_OPS = 3100
APPS = ["Baggage-Handling-System", "Gate-Management-System", "Departure-Control-System",
        "Airfield-Ops-Portal", "Cargo-Manifest-System"]
ACTIONS = ["view_manifest", "update_schedule", "read_sensor_status", "export_report",
           "modify_routing_rule", "admin_config_change", "create_user", "download_bulk_data"]
PRIV_ACTIONS = {"modify_routing_rule", "admin_config_change", "create_user", "download_bulk_data"}

for i in range(N_OPS):
    is_vendor = rng.random() < 0.35
    user = rng.choice(vendor_ids) if is_vendor else rng.choice(staff_ids)
    role = "vendor" if is_vendor else "staff"

    is_compromised_event = user in compromised_users and rng.random() < 0.6
    ts = rand_timestamp(off_hours_bias=0.5 if is_compromised_event else 0.05)

    app = rng.choice(APPS)
    if is_compromised_event:
        action = rng.choice(list(PRIV_ACTIONS))
        privilege_level = "elevated"
        outcome = rng.choice(["success", "denied"], p=[0.65, 0.35])
        label = 1
    else:
        action = rng.choice(ACTIONS, p=[0.28, 0.18, 0.18, 0.14, 0.08, 0.05, 0.04, 0.05])
        privilege_level = "elevated" if action in PRIV_ACTIONS else "standard"
        outcome = rng.choice(["success", "denied"], p=[0.97, 0.03])
        label = 0

    rows.append({
        "session_id": f"OP{i:06d}",
        "timestamp": ts.isoformat(),
        "user_id": user,
        "role": role,
        "vendor_firm": vendor_firm.get(user, ""),
        "application": app,
        "action": action,
        "privilege_level": privilege_level,
        "outcome": outcome,
        "is_anomalous": label,
    })

ops_df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
ops_df.to_csv(data_path("operational_vendor_logs.csv"), index=False)

# ---------------------------------------------------------------------------
# 4. Incident report / SOC ticket text
# ---------------------------------------------------------------------------
CATEGORIES = ["access_abuse", "phishing", "malware", "network_intrusion", "vendor_risk",
              "data_exposure", "policy_violation", "false_positive"]

TEMPLATES = {
    "access_abuse": [
        "SOC noticed {user} ({role}) logging into {resource} at {hour}:00, well outside normal shift hours. "
        "Access originated from {segment} and included a privilege escalation attempt on {app}.",
        "Repeated failed authentication attempts for {user} against {resource} followed by a successful "
        "login from an unusual location. Recommend forcing password reset and reviewing session activity.",
    ],
    "phishing": [
        "A staff member in {dept} reported a suspicious email impersonating IT support requesting VPN "
        "credentials. Email header shows spoofed domain. No credentials confirmed compromised yet.",
        "Multiple phishing emails referencing 'urgent baggage system update' were sent to {dept} staff. "
        "One recipient clicked the link; endpoint isolated as a precaution.",
    ],
    "malware": [
        "Endpoint protection flagged suspicious executable on a workstation in {dept}. File quarantined; "
        "investigating lateral spread toward {app}.",
        "Unusual outbound traffic from device {device} to an external IP consistent with command-and-control "
        "beaconing. Device isolated pending forensic review.",
    ],
    "network_intrusion": [
        "Network monitoring flagged cross-segment traffic from {segment} toward the operational network, "
        "targeting {app} on port {port}. Volume was significantly above baseline.",
        "Unexpected lateral connection observed between vendor VPN segment and {app}; source account is {user}.",
    ],
    "vendor_risk": [
        "{vendor} technician accessed {app} outside the scheduled maintenance window listed in the service "
        "agreement. Contract compliance review requested.",
        "Vendor remote-access session from {vendor} showed an elevated-privilege configuration change on "
        "{app} without a documented change ticket.",
    ],
    "data_exposure": [
        "Bulk export of records from {app} was performed by {user}; volume far exceeds routine reporting "
        "patterns. Data-loss-prevention review initiated.",
        "A misconfigured report on {app} briefly exposed operational schedule data to an unauthorised group.",
    ],
    "policy_violation": [
        "{user} shared login credentials with a colleague in {dept} to expedite shift handover, violating "
        "access-control policy. Manager notified for corrective action.",
        "Personal USB device was connected to a workstation in {dept} in violation of endpoint policy.",
    ],
    "false_positive": [
        "Alert on {user} accessing {resource} after hours was reviewed; confirmed as an authorised "
        "maintenance window logged in advance. No further action required.",
        "Spike in traffic from {segment} was traced to a scheduled backup job on {app}; closed as benign.",
    ],
}

rows = []
N_TICKETS = 340
for i in range(N_TICKETS):
    cat = rng.choice(CATEGORIES, p=[0.16, 0.14, 0.11, 0.13, 0.13, 0.09, 0.12, 0.12])
    tpl = rng.choice(TEMPLATES[cat])
    user = rng.choice(all_users)
    role = "vendor" if user in vendor_ids else "staff"
    dept = vendor_firm.get(user, staff_dept.get(user, "Operations"))
    text = tpl.format(
        user=user, role=role, dept=dept,
        resource=rng.choice(RESOURCES), app=rng.choice(APPS),
        hour=int(rng.choice([0,1,2,3,23])), segment=rng.choice(["vendor-vpn","staff-ops","passenger-wifi"]),
        device=f"DEV{rng.integers(1,400):04d}", vendor=rng.choice(VENDOR_FIRMS),
        port=rng.choice(COMMON_PORTS),
    )
    severity = rng.choice(["low", "medium", "high", "critical"],
                           p=[0.30, 0.35, 0.25, 0.10] if cat != "false_positive" else [0.85, 0.15, 0.0, 0.0])
    ts = rand_timestamp()
    rows.append({
        "ticket_id": f"TCK{i:05d}",
        "timestamp": ts.isoformat(),
        "category": cat,
        "severity": severity,
        "description": text,
    })

incidents_df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
incidents_df.to_csv(data_path("incident_reports.csv"), index=False)

# ---------------------------------------------------------------------------
# Save the ground-truth compromised-user list (used later to validate correlation/timeline)
# ---------------------------------------------------------------------------
with open(data_path("_scenario_ground_truth.json"), "w") as f:
    json.dump({"compromised_users": compromised_users, "seed": SEED}, f, indent=2)

print("Identity/access logs:", identity_df.shape, "| anomalous:", identity_df.is_anomalous.sum())
print("Network/endpoint logs:", network_df.shape, "| anomalous:", network_df.is_anomalous.sum())
print("Operational/vendor logs:", ops_df.shape, "| anomalous:", ops_df.is_anomalous.sum())
print("Incident reports:", incidents_df.shape)
print("Combined tabular rows:", len(identity_df) + len(network_df) + len(ops_df))
print("Compromised users (ground truth):", compromised_users)
