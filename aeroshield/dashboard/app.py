"""
AeroShield SOC Decision-Support Dashboard (C10)
Run with: streamlit run dashboard/app.py

Integrates outputs from all analytical modules (baseline, supervised, UBA,
correlation/timeline, security intelligence, simulation, NLP, predictive/
adversarial) into a single analyst-facing view.
"""
import streamlit as st
import pandas as pd
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="AeroShield SOC Dashboard", layout="wide")

@st.cache_data
def load_json(path):
    with open(os.path.join(BASE, path)) as f:
        return json.load(f)

@st.cache_data
def load_csv(path):
    return pd.read_csv(os.path.join(BASE, path))

st.title("🛬 AeroShield — Airport Operational Security Analytics")
st.caption("Decision-support prototype · SAS821S Capstone · Topic T15 · Milestone 2 (synthetic evaluation data)")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Overview", "Incident Investigation", "Model Performance", "Simulation",
     "Intelligence Reports", "Real Engine Telemetry (NASA)",
     "Real Network Intrusion (UNSW-NB15)", "Threat Intel (ATT&CK + KEV)"]
)

# ---------------------------------------------------------------------------
with tab1:
    eda = load_json("outputs/eda_summary.json")
    eng = load_json("outputs/engine_telemetry_metrics.json")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Identity/access events", eda["identity_access_logs"]["rows"],
              f"{eda['identity_access_logs']['anomalous']} flagged anomalous")
    c2.metric("Network/endpoint events", eda["network_endpoint_logs"]["rows"],
              f"{eda['network_endpoint_logs']['anomalous']} flagged anomalous")
    c3.metric("Operational/vendor sessions", eda["operational_vendor_logs"]["rows"],
              f"{eda['operational_vendor_logs']['anomalous']} flagged anomalous")
    c4.metric("Incident tickets", eda["incident_reports"]["rows"])
    c5.metric("Real engine telemetry rows (NASA)", eng["train_rows"], f"{eng['train_units']} engine units")

    st.subheader("Behavioural baselines")
    ic1, ic2 = st.columns(2)
    ic1.image("outputs/figures/fig1_hourly_baseline.png", use_container_width=True)
    ic2.image("outputs/figures/fig3_segment_baseline.png", use_container_width=True)
    ic3, ic4 = st.columns(2)
    ic3.image("outputs/figures/fig2_network_bytes_baseline.png", use_container_width=True)
    ic4.image("outputs/figures/fig5_incident_baseline.png", use_container_width=True)

# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Correlated incident timeline")
    narrative = load_json("outputs/incident_narrative.json")
    st.error(f"**{narrative['incident_id']}** — subject `{narrative['subject_user']}` — "
             f"{narrative['window_start']} to {narrative['window_end']}")
    st.write(narrative["summary"])

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Containment**")
        for x in narrative["containment"]:
            st.markdown(f"- {x}")
        st.markdown("**Eradication**")
        for x in narrative["eradication"]:
            st.markdown(f"- {x}")
    with colB:
        st.markdown("**Recovery**")
        for x in narrative["recovery"]:
            st.markdown(f"- {x}")
        st.markdown("**Monitoring**")
        for x in narrative["monitoring"]:
            st.markdown(f"- {x}")

    st.subheader("Full correlated event timeline")
    timeline = load_csv("outputs/incident_timeline.csv")
    only_anom = st.checkbox("Show only anomalous events", value=True)
    view = timeline[timeline.anomalous] if only_anom else timeline
    st.dataframe(view, use_container_width=True, height=350)

# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Supervised detector (identity/access anomalies)")
    sup = load_json("outputs/supervised_model_metrics.json")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", sup["precision"])
    m2.metric("Recall", sup["recall"])
    m3.metric("F1-score", sup["f1"])
    m4.metric("ROC-AUC", sup["roc_auc"])
    st.image("outputs/figures/fig6_supervised_model_eval.png", use_container_width=True)
    st.image("outputs/figures/fig7_feature_importance.png", use_container_width=False, width=550)

    st.subheader("Unsupervised / UBA anomaly detection (network)")
    uba = load_json("outputs/uba_model_metrics.json")
    u1, u2, u3 = st.columns(3)
    u1.metric("Precision vs injected label", uba["precision_vs_injected_label"])
    u2.metric("Recall vs injected label", uba["recall_vs_injected_label"])
    u3.metric("Events flagged", uba["n_flagged"])
    st.image("outputs/figures/fig8_uba_anomaly.png", use_container_width=True)
    st.info(uba["analyst_interpretation"])

    st.subheader("Predictive risk forecast & adversarial robustness")
    st.image("outputs/figures/fig11_risk_forecast.png", use_container_width=False, width=550)
    st.image("outputs/figures/fig12_adversarial_tests.png", use_container_width=True)
    adv = load_json("outputs/adversarial_predictive_results.json")
    with st.expander("Adversarial test interpretations"):
        st.write("**Evasion:**", adv["evasion_timing_masking"]["interpretation"])
        st.write("**Label poisoning:**", adv["label_poisoning"]["interpretation"])
        st.write("**Feature drift:**", adv["feature_drift"]["interpretation"])

# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Comparative control-scenario simulation")
    sim = load_csv("outputs/simulation_summary.csv")
    st.dataframe(sim, use_container_width=True)
    st.image("outputs/figures/fig9_simulation_scenarios.png", use_container_width=True)
    params = load_json("outputs/simulation_params.json")
    with st.expander("Simulation assumptions"):
        for a in params["assumptions"]:
            st.markdown(f"- {a}")

# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Security intelligence outputs")
    pirs = load_json("outputs/pirs.json")
    st.markdown("**Priority Intelligence Requirements**")
    for p in pirs["PIRs"]:
        st.markdown(f"- {p}")

    st.markdown("**Operational alert brief (SOC analysts)**")
    with open(os.path.join(BASE, "outputs/operational_alert_brief.txt")) as f:
        st.code(f.read(), language=None)

    st.markdown("**Executive risk report (leadership)**")
    with open(os.path.join(BASE, "outputs/executive_risk_report.txt")) as f:
        st.code(f.read(), language=None)

    st.subheader("Text mining / NLP on incident reports")
    st.image("outputs/figures/fig10_nlp_evaluation.png", use_container_width=True)
    nlp = load_json("outputs/nlp_metrics.json")
    st.write(f"Classification accuracy: **{nlp['classification']['accuracy']}**, "
             f"macro-F1: **{nlp['classification']['macro_f1']}**")
    st.caption("Note: near-perfect text-classification accuracy reflects the template-based "
               "structure of the synthetic ticket text; real deployment text is expected to be "
               "noisier and would need a larger, more varied labelled sample.")

# ---------------------------------------------------------------------------
with tab6:
    st.subheader("Real aircraft-engine OT sensor telemetry — NASA C-MAPSS (data source 5)")
    st.caption("Source: NASA PCoE Data Set Repository — \"Turbofan Engine Degradation Simulation\" "
               "(FD001 subset). Real, not synthetic. Used to demonstrate that the same "
               "baseline-deviation monitoring approach used on the synthetic security logs "
               "transfers to real operational sensor data.")
    eng = load_json("outputs/engine_telemetry_metrics.json")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Test MAE (cycles)", eng["test_mae_cycles"])
    e2.metric("Test RMSE (cycles)", eng["test_rmse_cycles"])
    e3.metric("PHM08 score", eng["test_phm08_score"])
    e4.metric("Onset detection rate", f"{eng['degradation_onset_detection']['detection_rate']:.0%}")

    st.image("outputs/figures/fig13_engine_sensor_trajectories.png", use_container_width=True)
    st.markdown("**Remaining Useful Life (RUL) prediction — Random Forest regressor**")
    st.image("outputs/figures/fig14_engine_rul_prediction.png", use_container_width=True)
    st.image("outputs/figures/fig15_engine_feature_importance.png", use_container_width=False, width=550)

    st.markdown("**Unsupervised degradation-onset / drift detection**")
    st.image("outputs/figures/fig16_engine_onset_detection.png", use_container_width=True)
    st.info(eng["interpretation"])

# ---------------------------------------------------------------------------
with tab7:
    st.subheader("Real network intrusion detection — UNSW-NB15 (data source 6)")
    st.caption("Source: UNSW-NB15 (Moustafa & Slay, 2015) — hybrid of real background network "
               "activity and synthetic attack injection via IXIA PerfectStorm, ACCS Cyber Range "
               "Lab. Retrieved via a verified public mirror; official train/test protocol used "
               "for evaluation (a genuine generalisation test, not a random re-split).")
    unsw = load_json("outputs/unsw_nb15_metrics.json")
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Supervised Precision", unsw["supervised"]["precision"])
    u2.metric("Supervised Recall", unsw["supervised"]["recall"])
    u3.metric("Supervised ROC-AUC", unsw["supervised"]["roc_auc"])
    u4.metric("Unsupervised F1 (no labels)", unsw["unsupervised"]["f1"])

    st.image("outputs/figures/fig17_unsw_baseline.png", use_container_width=True)
    st.markdown("**Supervised classifier — official train/test protocol**")
    st.image("outputs/figures/fig18_unsw_supervised_eval.png", use_container_width=True)
    st.image("outputs/figures/fig19_unsw_feature_importance.png", use_container_width=False, width=550)
    st.markdown("**Unsupervised anomaly detection (Isolation Forest, no label access)**")
    st.image("outputs/figures/fig20_unsw_unsupervised_eval.png", use_container_width=False, width=550)
    st.info(unsw["interpretation"])

# ---------------------------------------------------------------------------
with tab8:
    st.subheader("Threat intelligence enrichment — real MITRE ATT&CK + CISA KEV")
    st.caption("Both sources are live/official data: MITRE ATT&CK Enterprise (official MITRE "
               "repository) and the CISA Known Exploited Vulnerabilities catalog (official CISA "
               "mirror, updated continuously).")

    st.markdown("**INC-VND2023 mapped to real ATT&CK techniques**")
    st.image("outputs/figures/fig21_attack_technique_mapping.png", use_container_width=True)
    attack_map = load_json("outputs/attack_technique_mapping.json")
    with st.expander("Full technique mapping detail"):
        for m in attack_map["technique_mapping"]:
            st.markdown(f"**{m['attack_technique_id']} — {m['attack_technique_name']}** "
                        f"({', '.join(m['attack_tactics'])})")
            st.caption(m["attack_description_excerpt"])
            st.write(f"Observed indicator: {m['observed_indicator']}")
            st.write(f"Evidence: {m['project_evidence']}")
            st.divider()

    st.markdown("**CISA KEV: current, real advisories relevant to vendor/VPN/remote-access**")
    kev = load_json("outputs/kev_enrichment.json")
    k1, k2 = st.columns(2)
    k1.metric("Total KEV catalog entries", kev["catalog_total_entries"])
    k2.metric("Relevant to vendor/VPN/remote-access", kev["vendor_vpn_remote_access_relevant_count"])
    st.image("outputs/figures/fig22_kev_relevant_timeline.png", use_container_width=True)
    st.info(kev["interpretation"])
