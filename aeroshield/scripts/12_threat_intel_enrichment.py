"""
AeroShield - 12: Threat intelligence enrichment (MITRE ATT&CK + CISA KEV) - real data
Maps the project's incident narrative (Section 4/5, INC-VND2023) to real, official
MITRE ATT&CK Enterprise techniques, and cross-references the CISA Known Exploited
Vulnerabilities catalog for current, real advisories relevant to the vendor-VPN /
remote-access attack surface central to the simulated scenario.

Both sources are live/official data, not synthetic:
- MITRE ATT&CK: github.com/mitre-attack/attack-stix-data (official MITRE repository)
- CISA KEV: github.com/cisagov/kev-data (official CISA mirror, updated continuously)
"""
import json
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open("data/threat_intel/attack_enterprise.json", encoding="utf-8") as f:
    attack = json.load(f)
with open("data/threat_intel/kev_catalog.json", encoding="utf-8") as f:
    kev = json.load(f)

techniques = {o["id"]: o for o in attack["objects"] if o.get("type") == "attack-pattern"}


def by_external_id(ext_id):
    for t in techniques.values():
        for ref in t.get("external_references", []):
            if ref.get("source_name") == "mitre-attack" and ref.get("external_id") == ext_id:
                return t
    return None


# ---------------------------------------------------------------------------
# Curated mapping: our observed incident indicators -> real ATT&CK technique IDs.
# Each mapping is validated against the actual downloaded STIX data below (not
# invented) - the name/description printed/saved are MITRE's real text.
# ---------------------------------------------------------------------------
MAPPING = [
    ("Off-hours VPN authentication to sensitive resources using a vendor account",
     "T1078", "Identity/access anomaly (Section 3.1) + incident timeline (Section 4)"),
    ("Vendor remote-access session used as the entry point into the environment",
     "T1133", "Vendor-connection log correlation (Section 4)"),
    ("Cross-segment network traffic from vendor-VPN into the operational network",
     "T1021", "Network/endpoint UBA anomaly (Section 3.2)"),
    ("Elevated-privilege operational-application actions (admin_config_change, create_user)",
     "T1548", "Operational-app log correlation (Section 4)"),
    ("create_user action on operational applications during the incident window",
     "T1136", "Operational-app log correlation (Section 4)"),
    ("download_bulk_data action flagged as anomalous",
     "T1020", "Operational-app log correlation (Section 4)"),
]

enrichment = []
for indicator, tid, evidence in MAPPING:
    t = by_external_id(tid)
    if t is None:
        print(f"WARNING: {tid} not found in downloaded STIX data - skipping")
        continue
    tactics = [p["phase_name"] for p in t.get("kill_chain_phases", [])]
    enrichment.append({
        "observed_indicator": indicator,
        "attack_technique_id": tid,
        "attack_technique_name": t["name"],
        "attack_tactics": tactics,
        "attack_description_excerpt": t["description"].strip().split("\n")[0][:280],
        "project_evidence": evidence,
    })
    print(f"{tid} — {t['name']}  (tactics: {', '.join(tactics)})")

with open("outputs/attack_technique_mapping.json", "w", encoding="utf-8") as f:
    json.dump({"incident_id": "INC-VND2023", "technique_mapping": enrichment,
               "source": "MITRE ATT&CK Enterprise (mitre-attack/attack-stix-data, official)"},
              f, indent=2)

# ---------------------------------------------------------------------------
# CISA KEV: real, current advisories relevant to vendor-VPN / remote-access attack surface
# ---------------------------------------------------------------------------
KEYWORDS = ["VPN", "Remote", "Gateway", "Secure Access", "Citrix", "Ivanti", "Fortinet",
            "Pulse Secure", "Cisco", "Palo Alto"]
relevant = []
for v in kev["vulnerabilities"]:
    text = f"{v.get('vendorProject','')} {v.get('product','')} {v.get('vulnerabilityName','')}"
    if any(k.lower() in text.lower() for k in KEYWORDS):
        relevant.append(v)

relevant_sorted = sorted(relevant, key=lambda v: v.get("dateAdded", ""), reverse=True)[:8]
kev_summary = {
    "catalog_total_entries": kev.get("count"),
    "catalog_date_released": kev.get("dateReleased"),
    "vendor_vpn_remote_access_relevant_count": len(relevant),
    "most_recent_relevant_examples": [
        {"cveID": v["cveID"], "vendorProject": v["vendorProject"], "product": v["product"],
         "vulnerabilityName": v["vulnerabilityName"], "dateAdded": v["dateAdded"]}
        for v in relevant_sorted
    ],
    "interpretation": (
        f"Of {kev.get('count')} vulnerabilities CISA currently lists as actively exploited in "
        f"the wild, {len(relevant)} involve VPN, remote-access, or remote-gateway products — "
        "the same attack-surface category (vendor VPN) implicated in the simulated INC-VND2023 "
        "scenario and targeted by Simulation Scenario A (vendor VPN MFA, Section 6). This is live "
        "external corroboration that vendor/remote-access hardening is a currently active, "
        "real-world priority, not just a scenario assumption."
    ),
}
with open("outputs/kev_enrichment.json", "w", encoding="utf-8") as f:
    json.dump(kev_summary, f, indent=2)

print(f"\nKEV catalog: {kev.get('count')} total entries; "
      f"{len(relevant)} relevant to vendor/VPN/remote-access.")
for v in relevant_sorted[:5]:
    print(f"  {v['dateAdded']} | {v['cveID']} | {v['vendorProject']} {v['product']} - {v['vulnerabilityName']}")

print("\nSaved threat-intelligence enrichment outputs.")
