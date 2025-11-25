"""
Temporal Coordination Capacity Framework

Computes coordination measures for time series data:
- Duration (Φ_d): temporal memory via autocorrelation
- Frequency (Φ_f): spectral organization via entropy
- Agency (Φ_a): present-moment coordination
- Balance ratio (R): relative emphasis on present coordination
"""

from .measures import compute_coordination, phi_duration, phi_frequency, phi_agency, balance_ratio
from .events import detect_events, detect_peaks, detect_zero_crossings, detect_envelope_peaks

__version__ = '1.0.0'
__all__ = [
    'compute_coordination',
    'phi_duration',
    'phi_frequency', 
    'phi_agency',
    'balance_ratio',
    'detect_events',
    'detect_peaks',
    'detect_zero_crossings',
    'detect_envelope_peaks'
]
