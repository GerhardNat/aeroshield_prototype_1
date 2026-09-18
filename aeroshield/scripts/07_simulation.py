"""
AeroShield - 07: Simulation - comparative control scenarios (C7, session 8 evidence)
Monte Carlo simulation (>=1,000 iterations/scenario) comparing baseline vs three
candidate control investments, quantifying incident likelihood and operational
impact (cost proxy in analyst-hours + estimated service-disruption minutes).
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1B2A4A"; BLUE = "#2E5AAC"; TEAL = "#1F7A6C"; RED = "#B3312C"; AMBER = "#C98A1A"

rng = np.random.default_rng(821)
N_ITER = 5000
N_ATTEMPTS_PER_YEAR = 40   # plausible credential-compromise / vendor-access-abuse attempts per year

# Baseline (no new control) probabilities, calibrated loosely from the observed anomaly rates
BASE_P_INITIAL_COMPROMISE = 0.35     # probability an attempt succeeds in obtaining a foothold
BASE_P_LATERAL_MOVEMENT = 0.55       # given foothold, probability of reaching operational segment
BASE_P_DETECTED_EARLY = 0.30         # given lateral movement, probability SOC detects before impact

# Impact distributions if an incident reaches "operational impact" stage (log-normal, minutes of
# service disruption / degraded operations) and analyst-hours to investigate + remediate
IMPACT_MINUTES_MU, IMPACT_MINUTES_SIGMA = 3.6, 0.7     # ~ median 37 min, heavy right tail
ANALYST_HOURS_MU, ANALYST_HOURS_SIGMA = 2.1, 0.5       # ~ median 8 analyst-hours

SCENARIOS = {
    "Baseline (current controls)": dict(mfa_reduction=0.0, pam_detect_boost=0.0, seg_lateral_reduction=0.0),
    "A: Vendor VPN MFA": dict(mfa_reduction=0.55, pam_detect_boost=0.0, seg_lateral_reduction=0.0),
    "B: Privileged-access monitoring": dict(mfa_reduction=0.0, pam_detect_boost=0.40, seg_lateral_reduction=0.0),
    "C: Tighter network segmentation": dict(mfa_reduction=0.0, pam_detect_boost=0.0, seg_lateral_reduction=0.50),
}

def run_scenario(params, n_iter=N_ITER):
    p_compromise = BASE_P_INITIAL_COMPROMISE * (1 - params["mfa_reduction"])
    p_lateral = BASE_P_LATERAL_MOVEMENT * (1 - params["seg_lateral_reduction"])
    p_detected_early = min(0.97, BASE_P_DETECTED_EARLY + params["pam_detect_boost"])

    incidents_per_year = np.zeros(n_iter)
    total_disruption_minutes = np.zeros(n_iter)
    total_analyst_hours = np.zeros(n_iter)

    for i in range(n_iter):
        attempts = rng.poisson(N_ATTEMPTS_PER_YEAR)
        compromised = rng.random(attempts) < p_compromise
        n_compromised = compromised.sum()
        lateral = rng.random(n_compromised) < p_lateral
        n_lateral = lateral.sum()
        detected_early = rng.random(n_lateral) < p_detected_early
        n_operational_impact = int((~detected_early).sum())

        incidents_per_year[i] = n_operational_impact
        if n_operational_impact > 0:
            total_disruption_minutes[i] = rng.lognormal(IMPACT_MINUTES_MU, IMPACT_MINUTES_SIGMA,
                                                          n_operational_impact).sum()
            total_analyst_hours[i] = rng.lognormal(ANALYST_HOURS_MU, ANALYST_HOURS_SIGMA,
                                                     n_operational_impact).sum()
        # every lateral-movement case (detected or not) still costs some analyst triage time
        total_analyst_hours[i] += n_lateral * rng.uniform(0.5, 2.0)

    return {
        "incidents_per_year": incidents_per_year,
        "disruption_minutes": total_disruption_minutes,
        "analyst_hours": total_analyst_hours,
    }

results = {name: run_scenario(p) for name, p in SCENARIOS.items()}

summary_rows = []
for name, r in results.items():
    summary_rows.append({
        "scenario": name,
        "mean_incidents_per_year": round(float(r["incidents_per_year"].mean()), 2),
        "p90_incidents_per_year": round(float(np.percentile(r["incidents_per_year"], 90)), 2),
        "mean_disruption_minutes_per_year": round(float(r["disruption_minutes"].mean()), 1),
        "mean_analyst_hours_per_year": round(float(r["analyst_hours"].mean()), 1),
    })
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv("outputs/simulation_summary.csv", index=False)

baseline_mean = summary_df.loc[0, "mean_incidents_per_year"]
summary_df["incident_reduction_vs_baseline"] = (
    (baseline_mean - summary_df.mean_incidents_per_year) / baseline_mean
).round(4)
summary_df.to_csv("outputs/simulation_summary.csv", index=False)
print("\nWith incident reduction vs baseline:\n", summary_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Figure: incidents/year distribution + mean comparison
# ---------------------------------------------------------------------------
colors = [NAVY, BLUE, TEAL, AMBER]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
for (name, r), c in zip(results.items(), colors):
    axes[0].hist(r["incidents_per_year"], bins=range(0, 12), alpha=0.55, label=name, color=c, density=True)
axes[0].set_title("Simulated operational-impact incidents / year\n(5,000 Monte Carlo iterations per scenario)",
                   fontsize=9.5, color=NAVY, weight="bold")
axes[0].set_xlabel("Incidents per year"); axes[0].legend(fontsize=6.8)

axes[1].bar(summary_df.scenario, summary_df.mean_disruption_minutes_per_year, color=colors)
axes[1].set_title("Mean annual service-disruption minutes by scenario", fontsize=9.5, color=NAVY, weight="bold")
axes[1].set_ylabel("Minutes/year")
axes[1].tick_params(axis="x", labelrotation=20, labelsize=7)
plt.tight_layout()
plt.savefig("outputs/figures/fig9_simulation_scenarios.png", dpi=180, facecolor="white")
plt.close()

with open("outputs/simulation_params.json", "w") as f:
    json.dump({"n_iterations": N_ITER, "scenarios": SCENARIOS,
               "base_probabilities": {"p_initial_compromise": BASE_P_INITIAL_COMPROMISE,
                                       "p_lateral_movement": BASE_P_LATERAL_MOVEMENT,
                                       "p_detected_early": BASE_P_DETECTED_EARLY},
               "assumptions": [
                   "Attempt volume follows a Poisson process, mean 40 credential/vendor-access "
                   "abuse attempts per year (illustrative, not derived from live data).",
                   "Impact minutes and analyst-hours follow log-normal distributions calibrated to "
                   "plausible SOC investigation/remediation effort, not measured incident data.",
                   "Scenario effects (MFA, PAM, segmentation) are applied as independent multiplicative "
                   "reductions on the relevant stage probability; combined-control scenarios are not "
                   "modelled in this milestone.",
                   "Results are for relative, comparative decision support only, not an absolute "
                   "forecast of real-world incident rates.",
               ]}, f, indent=2)

print("Saved simulation outputs.")
