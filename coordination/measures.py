"""
Coordination Measures

Core functions for computing temporal coordination capacity:
- Duration (Φ_d): temporal memory via mean autocorrelation
- Frequency (Φ_f): spectral organization via inverse normalized entropy
- Agency (Φ_a): present-moment coordination via geometric coupling
- Balance ratio (R): relative emphasis on present coordination

Reference: "Temporal Coordination as Physical Criterion for Life's Emergence"
"""

import numpy as np
from scipy import signal as sig


def phi_duration(x, lag_window=10):
    """
    Compute duration coordination (Φ_d) via mean autocorrelation.
    
    Φ_d = (1/L) * Σ|ρ_ℓ| for ℓ = 1 to L
    
    where ρ_ℓ is the Pearson correlation at lag ℓ.
    
    Parameters
    ----------
    x : array_like
        Input time series (1D)
    lag_window : int, optional
        Number of lags to compute (default: 10)
        
    Returns
    -------
    phi_d : float
        Duration coordination in [0, 1]
        High values indicate strong temporal memory
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    
    if n < lag_window + 1:
        lag_window = n - 1
    
    if np.var(x) < 1e-10:
        return 0.0
    
    # Compute autocorrelation via FFT for efficiency
    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[n-1:]  # Keep positive lags only
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Mean absolute autocorrelation over lag window
    phi_d = np.mean(np.abs(autocorr[1:lag_window+1]))
    
    return float(np.clip(phi_d, 0, 1))


def phi_frequency(x, nperseg=None):
    """
    Compute frequency coordination (Φ_f) via inverse normalized spectral entropy.
    
    Φ_f = 1 - H/H_max
    
    where H = -Σ p(ω) log₂ p(ω) is spectral entropy computed from 
    power spectral density.
    
    Parameters
    ----------
    x : array_like
        Input time series (1D)
    nperseg : int, optional
        Length of each segment for Welch's method (default: len(x)//4)
        
    Returns
    -------
    phi_f : float
        Frequency coordination in [0, 1]
        High values indicate concentrated spectral energy (organized oscillations)
        Low values indicate flat spectrum (noise-like)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    
    if nperseg is None:
        nperseg = max(n // 4, 8)
    
    if n < nperseg:
        nperseg = n
    
    # Compute power spectral density via Welch's method
    freqs, psd = sig.welch(x, nperseg=nperseg)
    
    # Normalize to probability distribution
    psd = psd + 1e-12  # Avoid log(0)
    psd_norm = psd / np.sum(psd)
    
    # Spectral entropy
    H = -np.sum(psd_norm * np.log2(psd_norm))
    H_max = np.log2(len(psd))
    
    # Inverse normalized entropy
    phi_f = 1 - H / H_max if H_max > 0 else 0.0
    
    return float(np.clip(phi_f, 0, 1))


def phi_agency(phi_d, phi_f):
    """
    Compute agency (Φ_a) via geometric coupling of duration and frequency.
    
    Φ_a = 4 * sqrt(Φ_d(1-Φ_d) * Φ_f(1-Φ_f))
    
    This functional form ensures Φ_a maximizes when both Φ_d and Φ_f 
    achieve balanced intermediate values (~0.5). Systems with either 
    no memory (Φ_d → 0) or no spectral structure (Φ_f → 0) cannot 
    maintain organized coordination.
    
    Parameters
    ----------
    phi_d : float
        Duration coordination
    phi_f : float
        Frequency coordination
        
    Returns
    -------
    phi_a : float
        Agency in [0, 1]
        High values indicate synergistic coupling of memory and anticipation
    """
    # Geometric mean of variances
    var_d = phi_d * (1 - phi_d)
    var_f = phi_f * (1 - phi_f)
    
    phi_a = 4 * np.sqrt(var_d * var_f)
    
    return float(np.clip(phi_a, 0, 1))


def balance_ratio(phi_d, phi_f, phi_a):
    """
    Compute balance ratio (R) quantifying relative emphasis on present coordination.
    
    R = Φ_a / (Φ_d + Φ_f + Φ_a)
    
    Parameters
    ----------
    phi_d : float
        Duration coordination
    phi_f : float
        Frequency coordination
    phi_a : float
        Agency
        
    Returns
    -------
    R : float
        Balance ratio in [0, 1]
        R ≈ 0: duration-dominant (past determines future)
        R ≈ 0.15: proto-life threshold
        R > 0.15: agency-balanced (present coordination achieves causal co-equality)
    """
    total = phi_d + phi_f + phi_a
    
    if total < 1e-10:
        return 0.0
    
    R = phi_a / total
    
    return float(np.clip(R, 0, 1))


def compute_coordination(x, lag_window=10, nperseg=None):
    """
    Compute all coordination measures for a time series.
    
    Parameters
    ----------
    x : array_like
        Input time series (1D)
    lag_window : int, optional
        Number of lags for autocorrelation (default: 10)
    nperseg : int, optional
        Segment length for spectral analysis (default: len(x)//4)
        
    Returns
    -------
    phi_d : float
        Duration coordination (temporal memory)
    phi_f : float
        Frequency coordination (spectral organization)
    phi_a : float
        Agency (present-moment coordination)
    R : float
        Balance ratio (relative emphasis on present)
        
    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 1000)
    >>> x = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(len(t))
    >>> phi_d, phi_f, phi_a, R = compute_coordination(x)
    >>> print(f"R = {R:.3f}")
    """
    x = np.asarray(x, dtype=float)
    
    # Compute individual measures
    pd = phi_duration(x, lag_window=lag_window)
    pf = phi_frequency(x, nperseg=nperseg)
    pa = phi_agency(pd, pf)
    R = balance_ratio(pd, pf, pa)
    
    return pd, pf, pa, R


def compute_coordination_ensemble(events, signal, window_half=20):
    """
    Compute coordination measures across an ensemble of detected events.
    
    Parameters
    ----------
    events : list of int
        Indices of detected event centers
    signal : array_like
        Full time series
    window_half : int
        Half-width of analysis window around each event
        
    Returns
    -------
    results : dict
        Dictionary with keys 'phi_d', 'phi_f', 'phi_a', 'R', each containing
        arrays of per-event values, plus 'mean' and 'std' for each measure
    """
    signal = np.asarray(signal)
    n = len(signal)
    
    phi_d_list = []
    phi_f_list = []
    phi_a_list = []
    R_list = []
    
    for center in events:
        start = max(0, center - window_half)
        end = min(n, center + window_half)
        
        if end - start < 10:  # Minimum window size
            continue
            
        window = signal[start:end]
        pd, pf, pa, R = compute_coordination(window)
        
        phi_d_list.append(pd)
        phi_f_list.append(pf)
        phi_a_list.append(pa)
        R_list.append(R)
    
    results = {
        'phi_d': np.array(phi_d_list),
        'phi_f': np.array(phi_f_list),
        'phi_a': np.array(phi_a_list),
        'R': np.array(R_list),
        'N': len(R_list),
        'mean_R': np.mean(R_list) if R_list else 0.0,
        'std_R': np.std(R_list) if R_list else 0.0,
        'mean_phi_d': np.mean(phi_d_list) if phi_d_list else 0.0,
        'mean_phi_f': np.mean(phi_f_list) if phi_f_list else 0.0,
        'mean_phi_a': np.mean(phi_a_list) if phi_a_list else 0.0,
    }
    
    return results


if __name__ == '__main__':
    # Demo: compare organized oscillation vs noise
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    t = np.linspace(0, 20, 2000)
    
    # Organized oscillation
    oscillation = np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(len(t))
    
    # Pure noise
    noise = np.random.randn(len(t))
    
    # Compute coordination
    pd_osc, pf_osc, pa_osc, R_osc = compute_coordination(oscillation)
    pd_noise, pf_noise, pa_noise, R_noise = compute_coordination(noise)
    
    print("Organized oscillation:")
    print(f"  Φ_d = {pd_osc:.3f}, Φ_f = {pf_osc:.3f}, Φ_a = {pa_osc:.3f}, R = {R_osc:.3f}")
    
    print("\nPure noise:")
    print(f"  Φ_d = {pd_noise:.3f}, Φ_f = {pf_noise:.3f}, Φ_a = {pa_noise:.3f}, R = {R_noise:.3f}")
