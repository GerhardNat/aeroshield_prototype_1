# AeroShield — SAS821S Capstone (Topic T15) — Milestone 2 Prototype

Multi-domain security analytics decision-support prototype for a fictional airport
operator, covering identity/access, network/endpoint, operational-application/vendor,
and incident-report text data — plus 4 REAL data sources: NASA PCoE aircraft-engine
sensor telemetry (C-MAPSS), UNSW-NB15 real network intrusion data, MITRE ATT&CK
(official technique taxonomy) and the CISA Known Exploited Vulnerabilities catalog
(live feed) — used to validate the project's methodology on genuine data and enrich
its threat-intelligence reporting with real, current context.

## Structure
```
data/                    4 synthetic source CSVs + data dictionary + ground-truth scenario file
data/cmapss/              Real NASA C-MAPSS engine telemetry (FD001-FD004) + README_SOURCE.md
data/unsw_nb15/           Real UNSW-NB15 network intrusion train/test CSVs
data/threat_intel/        Real MITRE ATT&CK STIX data + CISA KEV catalog JSON
scripts/                  01-13: data generation through threat-intel enrichment (run in order)
outputs/                  figures, metrics (JSON), model artefacts, generated reports
dashboard/app.py          Streamlit decision-support prototype (C10) — 8 tabs
```

## Run order
```
pip install -r requirements.txt
python scripts/01_generate_data.py
python scripts/02_baseline_eda.py
python scripts/03_supervised_model.py
python scripts/04_unsupervised_uba.py
python scripts/05_incident_correlation.py
python scripts/06_security_intelligence.py
python scripts/07_simulation.py
python scripts/08_text_mining_nlp.py
python scripts/09_predictive_adversarial.py
python scripts/10_real_engine_telemetry.py
python scripts/11_real_network_intrusion.py
python scripts/12_threat_intel_enrichment.py
python scripts/12b_threat_intel_figures.py
streamlit run dashboard/app.py
```

## Data note
Sources 1-4 are synthetically generated (fixed seed = 821) to represent a fictional
airport operator. No real organisation, person, or system is referenced.
Sources 5-8 are REAL data:
- data/cmapss/ — NASA PCoE Turbofan Engine Degradation Simulation (C-MAPSS)
- data/unsw_nb15/ — UNSW-NB15 network intrusion data (Moustafa & Slay, 2015)
- data/threat_intel/attack_enterprise.json — MITRE ATT&CK Enterprise (official)
- data/threat_intel/kev_catalog.json — CISA Known Exploited Vulnerabilities (official, live)
See each source's file/README for full provenance and citation.
