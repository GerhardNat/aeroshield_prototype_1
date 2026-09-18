# Source: NASA Prognostics Center of Excellence (PCoE) Data Set Repository

Dataset: "6. Turbofan Engine Degradation Simulation" (C-MAPSS)
Citation: A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set",
NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
Repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Retrieved via a public GitHub mirror (raw.githubusercontent.com/ericlrf/rul) since the
original NASA/S3 host is not reachable from this environment; contents verified against
the published dataset specification (unit counts, cycle ranges, 26-column format) before use.

This is REAL sensor telemetry (not synthetic): four sub-datasets (FD001-FD004), each a set
of turbofan engine units run from a healthy state to failure under simulated flight
conditions, using NASA's C-MAPSS dynamical engine model. 21 sensor channels + 3 operational
settings per cycle. No RUL labels are given for training data (must be derived); RUL labels
for test data are provided separately.

| Subset | Units (train) | Operating conditions | Fault modes |
|--------|---------------|----------------------|-------------|
| FD001  | 100           | 1                     | 1 (HPC degradation) |
| FD002  | 260           | 6                     | 1 (HPC degradation) |
| FD003  | 100           | 1                     | 2 (HPC + Fan degradation) |
| FD004  | 249           | 6                     | 2 (HPC + Fan degradation) |
