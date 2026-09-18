import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13.6, 7.8))
ax.set_xlim(0, 13.6)
ax.set_ylim(0, 7.8)
ax.axis("off")

navy = "#1B2A4A"
blue = "#2E5AAC"
teal = "#1F7A6C"
grey = "#7A8494"
lightbg = "#EEF2F8"
realbg = "#FBEFD8"
realedge = "#C98A1A"

def box(x, y, w, h, text, fc="#FFFFFF", ec=navy, tc=navy, fs=7.8, bold=False, lw=1.3):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, weight="bold" if bold else "normal", zorder=3)

def arrow(x1, y1, x2, y2, color=grey, lw=1.1):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                         linewidth=lw, color=color, zorder=1)
    ax.add_patch(a)

def col_title(x, text):
    ax.text(x, 7.6, text, ha="center", fontsize=9.1, weight="bold", color=navy)

# Column 1: Data sources (4 synthetic + 2 real)
srcs = ["Identity / access\nlogs (staff, vendor)", "Network / endpoint\nlogs (segmented VLANs)",
        "Operational app &\nvendor-connection logs", "Incident-report text\n(SOC tickets)"]
y0 = 6.55
for i, s in enumerate(srcs):
    box(0.15, y0 - i*0.98, 1.95, 0.78, s, fc=lightbg, ec=blue, fs=7.0)
box(0.15, y0 - 4*0.98, 1.95, 0.78, "Real aircraft-engine\ntelemetry (NASA\nC-MAPSS)", fc=realbg, ec=realedge, fs=7.0, bold=True)
box(0.15, y0 - 5*0.98, 1.95, 0.78, "Real network traffic\n(UNSW-NB15)", fc=realbg, ec=realedge, fs=7.0, bold=True)
col_title(1.1, "1. Data sources\n(4 synthetic + 2 real)")

# Column 2: Ingestion & prep
box(2.55, 2.55, 1.55, 2.8, "Ingestion,\ncleaning,\ndata dictionary\n&\nfeature\nengineering", fc="#FFFFFF", ec=teal, fs=7.4)
col_title(3.3, "2. Ingestion\n& prep")
for i in range(6):
    arrow(2.1, y0 - i*0.98 + 0.35, 2.55, 3.95)

# Column 3: Analytical modules
mods = ["Baseline / EDA", "Supervised ML detector",
        "Unsupervised / UBA\nanomaly detection", "Incident correlation\n& timeline",
        "Security intelligence\n(PIR / cycle)", "Simulation\n(control scenarios)",
        "Text mining / NLP", "Predictive & adversarial\nrobustness testing",
        "Real-telemetry RUL &\ndegradation-onset detection",
        "Real network intrusion\ndetection (UNSW-NB15)",
        "ATT&CK + KEV threat-intel\nenrichment"]
mx = 4.7
my0 = 7.25
mh = 0.535
mgap = 0.625
for i, m in enumerate(mods):
    is_real = i >= 8
    fc = realbg if is_real else lightbg
    ec = realedge if is_real else blue
    box(mx, my0 - i*mgap, 2.15, mh, m, fc=fc, ec=ec, fs=6.3)
arrow(4.1, 3.95, mx, 3.95)
col_title(mx+1.07, "3. Analytical modules")

# Column 4: Storage
box(7.5, 3.15, 1.4, 1.6, "Curated\nanalytics\nstore", fc="#FFFFFF", ec=teal, fs=7.6)
col_title(8.2, "4. Storage")
arrow(7.05, 3.95, 7.5, 3.95)

# Column 5: Prototype / dashboard
box(9.25, 2.95, 1.7, 2.0, "Decision-support\nprototype /\ndashboard\n(8 tabs)", fc="#FFFFFF", ec=navy, fs=7.6, bold=True, lw=1.6)
col_title(10.1, "5. Prototype")
arrow(8.9, 3.95, 9.25, 3.95)

# Decision outputs row (bottom)
outs = ["SOC operational\nalerts", "Executive\nrisk report", "Control-investment\nrecommendation"]
ow = 1.35
ostart = 8.55
for i, o in enumerate(outs):
    ox = ostart + i*(ow+0.18)
    box(ox, 0.5, ow, 0.85, o, fc="#FFFFFF", ec=navy, fs=6.5)
    arrow(10.1, 2.95, ox+ow/2, 1.35)
ax.text(10.5, 1.75, "6. Decision outputs", ha="center", fontsize=9.1, weight="bold", color=navy)

# Legend
box(0.15, 0.5, 0.35, 0.28, "", fc=realbg, ec=realedge, lw=1.3)
ax.text(0.62, 0.64, "Real data (NASA, UNSW-NB15, MITRE ATT&CK, CISA KEV)", ha="left", va="center", fontsize=7.3, color=navy)

fig.tight_layout(pad=0.4)
fig.savefig("architecture_diagram.png", dpi=220, facecolor="white")
print("saved")
