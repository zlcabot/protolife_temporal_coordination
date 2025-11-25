# Temporal Coordination Capacity for Proto-Life Emergence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code repository for: **"Temporal Coordination as Physical Criterion for Life's Emergence"**

## Overview

This repository provides Python implementations for computing temporal coordination capacity—a unified physical criterion distinguishing proto-life from chemistry. The framework decomposes temporal organization into three complementary modes:

- **Duration (Φ_d)**: Integration of past states via autocorrelation
- **Frequency (Φ_f)**: Projection into future via spectral organization  
- **Agency (Φ_a)**: Present-moment coordination capacity

The **balance ratio** R = Φ_a / (Φ_d + Φ_f + Φ_a) quantifies the relative emphasis on present coordination versus past-future determination. Systems cross the proto-life threshold at R ≈ 0.15.

## Installation

```bash
git clone https://github.com/[username]/protolife-temporal-coordination.git
cd protolife-temporal-coordination
pip install -r requirements.txt
```

## Repository Structure

```
protolife-temporal-coordination/
├── README.md
├── LICENSE
├── requirements.txt
├── coordination/
│   ├── __init__.py
│   ├── measures.py      # Core Φ_d, Φ_f, Φ_a, R computation
│   └── events.py        # Event detection algorithms
├── simulations/
│   ├── __init__.py
│   ├── formose.py       # Formose autocatalysis network
│   ├── glycolysis.py    # Deterministic and stochastic glycolysis
│   ├── material.py      # RNA/Protein/RNP competition
│   ├── information.py   # Analog vs digital encoding
│   ├── spatial.py       # Confinement vs diffusion
│   └── oscillators.py   # BZ and Sel'kov models
├── figures/
│   ├── figure1_coordination_hierarchy.py
│   ├── figure2_formose_phase_space.py
│   ├── figure3_material_information_spatial.py
│   └── extended_data_figure1_glycolysis.py
└── examples/
    └── basic_usage.py
```

## Quick Start

```python
from coordination.measures import compute_coordination
from coordination.events import detect_events

import numpy as np

# Generate or load your time series
t = np.linspace(0, 100, 2000)
signal = np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.random.randn(len(t))

# Detect coordination events
events = detect_events(signal)

# Compute coordination measures for each event
results = []
for start, end in events:
    window = signal[start:end]
    phi_d, phi_f, phi_a, R = compute_coordination(window)
    results.append({'phi_d': phi_d, 'phi_f': phi_f, 'phi_a': phi_a, 'R': R})

# Aggregate
mean_R = np.mean([r['R'] for r in results])
print(f"Mean balance ratio: R = {mean_R:.3f}")
```

## Core Functions

### `compute_coordination(signal, lag_window=10)`

Computes the three coordination measures and balance ratio.

**Parameters:**

- `signal`: 1D numpy array of time series data
- `lag_window`: Number of lags for autocorrelation (default: 10)

**Returns:**

- `phi_d`: Duration coordination (temporal memory)
- `phi_f`: Frequency coordination (spectral organization)
- `phi_a`: Agency (present-moment coordination)
- `R`: Balance ratio

### `detect_events(signal, method='hybrid')`

Detects coordination events in time series.

**Parameters:**

- `signal`: 1D numpy array
- `method`: Detection method ('peaks', 'zero_crossings', 'envelope', 'hybrid')

**Returns:**

- List of (start, end) index tuples for each detected event

## Reproducing Paper Results

### Table 1: Coordination Hierarchy

```bash
python -m simulations.oscillators    # BZ oscillator
python -m simulations.glycolysis     # Deterministic and stochastic
python -m simulations.formose        # Formose network
python -m simulations.material       # RNA/Protein/RNP
python -m simulations.information    # Analog vs digital
python -m simulations.spatial        # Confinement sweep
```

### Figures

```bash
python figures/figure1_coordination_hierarchy.py
python figures/figure2_formose_phase_space.py
python figures/figure3_material_information_spatial.py
python figures/extended_data_figure1_glycolysis.py
```

## Key Results

| System             | R           | Classification      |
| ------------------ | ----------- | ------------------- |
| BZ Oscillator      | 0.00        | Duration-dominant   |
| Pure RNA           | 0.05        | Memory-locked       |
| Glycolysis (det)   | 0.14 ± 0.16 | Threshold           |
| Protocell          | 0.23 ± 0.12 | Proto-life          |
| Glycolysis (stoch) | 0.32 ± 0.08 | Noise-enhanced      |
| Enzyme (proton)    | 0.38 ± 0.03 | Evolved catalysis   |
| Formose (noisy)    | 0.43 ± 0.08 | Prebiotic selection |
| RNP Hybrid         | 0.60 ± 0.04 | Material synergy    |
| Digital coding     | 0.66 ± 0.03 | Error correction    |

## Dependencies

- Python ≥ 3.8
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- Matplotlib ≥ 3.4
- COPASI/Basico ≥ 0.33 (for glycolysis models)

## Citation

If you use this code, please cite:

```bibtex
@article{temporal_coordination_2025,
  author = {[Author]},
  title = {Temporal Coordination as Physical Criterion for Life's Emergence},
  journal = {[Journal]},
  year = {2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

[Author contact information]
