import json
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NAVY = "#1B2A4A"; BLUE = "#2E5AAC"; TEAL = "#1F7A6C"; RED = "#B3312C"; AMBER = "#C98A1A"

with open("outputs/attack_technique_mapping.json") as f:
    mapping = json.load(f)["technique_mapping"]
with open("outputs/kev_enrichment.json") as f:
    kev = json.load(f)

# Figure 21: ATT&CK technique mapping for INC-VND2023
fig, ax = plt.subplots(figsize=(11.5, 4.2))
labels = [f"{m['attack_technique_id']}\n{m['attack_technique_name']}" for m in mapping]
tactics_count = [len(m["attack_tactics"]) for m in mapping]
colors = [BLUE if "privilege-escalation" not in m["attack_tactics"] else RED for m in mapping]
ax.barh(labels[::-1], [1]*len(labels), color=colors[::-1])
ax.set_xlim(0, 2.3)
ax.set_xticks([])
for i, m in enumerate(mapping[::-1]):
    ax.text(1.03, i, ", ".join(m["attack_tactics"]), va="center", fontsize=7.6, color=NAVY)
ax.set_title("INC-VND2023 mapped to real MITRE ATT&CK techniques", fontsize=10, color=NAVY, weight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/fig21_attack_technique_mapping.png", dpi=180, facecolor="white")
plt.close()

# Figure 22: KEV vendor/remote-access relevant entries over time (recent)
examples = kev["most_recent_relevant_examples"]
df = pd.DataFrame(examples)
df["dateAdded"] = pd.to_datetime(df["dateAdded"])
df = df.sort_values("dateAdded")
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.scatter(df.dateAdded, range(len(df)), color=RED, s=60, zorder=3)
for i, row in enumerate(df.itertuples()):
    ax.text(row.dateAdded, i + 0.15,
            f"{row.cveID} — {row.vendorProject}", fontsize=7.3, color=NAVY)
ax.set_yticks([])
ax.set_title(f"Most recent CISA KEV entries relevant to vendor/VPN/remote-access\n"
             f"({kev['vendor_vpn_remote_access_relevant_count']} of {kev['catalog_total_entries']} total KEV entries)",
             fontsize=9.5, color=NAVY, weight="bold")
ax.set_xlabel("Date added to KEV catalog")
plt.tight_layout()
plt.savefig("outputs/figures/fig22_kev_relevant_timeline.png", dpi=180, facecolor="white")
plt.close()

print("Saved figures 21-22.")
