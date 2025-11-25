"""
Formose Autocatalytic Network Simulation

Models competitive prebiotic chemistry:
- Autocatalytic sugar formation (proto-life pathway)
- Irreversible tar polymerization (chemical death)
- Stochastic energy fluctuations (environmental noise)

Demonstrates that environmental noise serves as physical selection mechanism.

Reference: "Temporal Coordination as Physical Criterion for Life's Emergence"
"""

import numpy as np
from scipy.integrate import odeint
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coordination.measures import compute_coordination
from coordination.events import detect_events


# Default parameters
DEFAULT_PARAMS = {
    'k_A': 0.8,      # Autocatalytic rate constant
    'k_T': 0.2,      # Tar formation rate constant  
    'k_D': 0.1,      # Sugar degradation rate constant
    'E0': 50.0,      # Mean energy flux (kT)
    'sigma': 0.4,    # Noise amplitude
    'F0': 100.0,     # Initial formaldehyde
    'A0': 1.0,       # Initial autocatalyst (small seed)
    'T0': 0.0,       # Initial tar
    'duration': 10000.0,  # Simulation duration
    'dt': 0.1,       # Time step
}


def formose_deterministic(y, t, params, E_t):
    """
    Deterministic Formose reaction ODE.
    
    Reactions:
    - F -> A (autocatalytic, rate k_A * [A] * E(t))
    - F -> T (tar formation, rate k_T * E(t))
    - A -> T (degradation, rate k_D)
    """
    F, A, T = y
    
    k_A = params['k_A']
    k_T = params['k_T']
    k_D = params['k_D']
    
    # Energy at current time
    E = E_t(t)
    
    # Rates
    r_auto = k_A * A * F * E / params['E0']  # Autocatalysis
    r_tar = k_T * F * E / params['E0']        # Tar formation
    r_deg = k_D * A                            # Degradation
    
    dF = -r_auto - r_tar
    dA = r_auto - r_deg
    dT = r_tar + r_deg
    
    return [dF, dA, dT]


def simulate_formose(params=None, noisy=True, seed=None):
    """
    Simulate Formose reaction network.
    
    Parameters
    ----------
    params : dict, optional
        Simulation parameters (uses defaults if not provided)
    noisy : bool
        If True, use stochastic energy input; if False, constant energy
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - t: time array
        - F, A, T: concentration arrays
        - R_auto: balance ratio for autocatalytic pathway
        - R_tar: balance ratio for tar pathway
        - events_auto: detected events in autocatalyst
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    if seed is not None:
        np.random.seed(seed)
    
    # Time array
    t = np.arange(0, params['duration'], params['dt'])
    n = len(t)
    
    # Energy function
    if noisy:
        # Pre-generate noise for reproducibility
        noise = np.random.randn(n)
        E_array = params['E0'] * (1 + params['sigma'] * noise)
        E_array = np.clip(E_array, 0.1 * params['E0'], 3 * params['E0'])
        
        def E_t(time):
            idx = int(time / params['dt'])
            idx = min(idx, n - 1)
            return E_array[idx]
    else:
        def E_t(time):
            return params['E0']
    
    # Initial conditions
    y0 = [params['F0'], params['A0'], params['T0']]
    
    # Integrate
    solution = odeint(formose_deterministic, y0, t, args=(params, E_t))
    F = solution[:, 0]
    A = solution[:, 1]
    T = solution[:, 2]
    
    # Compute coordination for autocatalytic pathway
    events_auto, centers_auto = detect_events(A, method='hybrid')
    
    R_auto_list = []
    for start, end in events_auto:
        if end - start >= 10:
            window = A[start:end]
            _, _, _, R = compute_coordination(window)
            R_auto_list.append(R)
    
    R_auto = np.mean(R_auto_list) if R_auto_list else 0.0
    R_auto_std = np.std(R_auto_list) if R_auto_list else 0.0
    
    # Compute coordination for tar pathway
    events_tar, centers_tar = detect_events(T, method='hybrid')
    
    R_tar_list = []
    for start, end in events_tar:
        if end - start >= 10:
            window = T[start:end]
            _, _, _, R = compute_coordination(window)
            R_tar_list.append(R)
    
    R_tar = np.mean(R_tar_list) if R_tar_list else 0.0
    
    results = {
        't': t,
        'F': F,
        'A': A,
        'T': T,
        'R_auto': R_auto,
        'R_auto_std': R_auto_std,
        'R_tar': R_tar,
        'N_events_auto': len(R_auto_list),
        'N_events_tar': len(R_tar_list),
        'events_auto': events_auto,
        'noisy': noisy,
        'params': params
    }
    
    return results


def phase_space_sweep(E0_range=(10, 200), sigma_range=(0, 0.8), 
                      n_E0=20, n_sigma=20, seed=42):
    """
    Sweep parameter space to map coordination landscape.
    
    Parameters
    ----------
    E0_range : tuple
        Range of energy flux values
    sigma_range : tuple
        Range of noise amplitudes
    n_E0, n_sigma : int
        Number of grid points
    seed : int
        Random seed
        
    Returns
    -------
    E0_grid, sigma_grid : ndarray
        Parameter grids
    R_grid : ndarray
        Coordination values at each grid point
    """
    np.random.seed(seed)
    
    E0_values = np.linspace(E0_range[0], E0_range[1], n_E0)
    sigma_values = np.linspace(sigma_range[0], sigma_range[1], n_sigma)
    
    E0_grid, sigma_grid = np.meshgrid(E0_values, sigma_values)
    R_grid = np.zeros_like(E0_grid)
    
    params = DEFAULT_PARAMS.copy()
    params['duration'] = 5000  # Shorter for sweep
    
    for i in range(n_sigma):
        for j in range(n_E0):
            params['E0'] = E0_values[j]
            params['sigma'] = sigma_values[i]
            
            results = simulate_formose(params, noisy=True, seed=seed + i*n_E0 + j)
            R_grid[i, j] = results['R_auto']
    
    return E0_grid, sigma_grid, R_grid


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    print("Simulating Formose reaction...")
    
    # Compare noisy vs constant conditions
    results_noisy = simulate_formose(noisy=True, seed=42)
    results_const = simulate_formose(noisy=False, seed=42)
    
    print(f"\nNoisy conditions:")
    print(f"  Autocatalysis R = {results_noisy['R_auto']:.3f} ± {results_noisy['R_auto_std']:.3f}")
    print(f"  Tar R = {results_noisy['R_tar']:.3f}")
    print(f"  N events = {results_noisy['N_events_auto']}")
    
    print(f"\nConstant conditions:")
    print(f"  Autocatalysis R = {results_const['R_auto']:.3f}")
    print(f"  Tar R = {results_const['R_tar']:.3f}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax1 = axes[0]
    ax1.plot(results_noisy['t'][:5000], results_noisy['A'][:5000], 'r-', label='Autocatalyst (noisy)')
    ax1.plot(results_noisy['t'][:5000], results_noisy['T'][:5000], 'gray', alpha=0.5, label='Tar (noisy)')
    ax1.set_ylabel('Concentration')
    ax1.set_title(f'Noisy: R_auto = {results_noisy["R_auto"]:.3f}')
    ax1.legend()
    
    ax2 = axes[1]
    ax2.plot(results_const['t'][:5000], results_const['A'][:5000], 'b-', label='Autocatalyst (constant)')
    ax2.plot(results_const['t'][:5000], results_const['T'][:5000], 'gray', alpha=0.5, label='Tar (constant)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Concentration')
    ax2.set_title(f'Constant: R_auto = {results_const["R_auto"]:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('formose_comparison.png', dpi=150)
    print("\nSaved formose_comparison.png")
