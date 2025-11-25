"""
Basic Usage Example

Demonstrates how to compute temporal coordination capacity
for different types of time series.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordination.measures import compute_coordination
from coordination.events import detect_events


def main():
    np.random.seed(42)
    t = np.linspace(0, 50, 5000)
    dt = t[1] - t[0]
    
    print("=" * 60)
    print("Temporal Coordination Capacity - Basic Examples")
    print("=" * 60)
    
    # Example 1: Pure noise (should have low R)
    print("\n1. Pure white noise")
    noise = np.random.randn(len(t))
    phi_d, phi_f, phi_a, R = compute_coordination(noise)
    print(f"   Φ_d = {phi_d:.3f} (duration)")
    print(f"   Φ_f = {phi_f:.3f} (frequency)")
    print(f"   Φ_a = {phi_a:.3f} (agency)")
    print(f"   R   = {R:.3f} (balance ratio)")
    print(f"   Classification: {'Duration-dominant' if R < 0.15 else 'Agency-balanced'}")
    
    # Example 2: Regular oscillation (moderate R)
    print("\n2. Regular sine wave")
    sine = np.sin(2 * np.pi * 0.2 * t)
    phi_d, phi_f, phi_a, R = compute_coordination(sine)
    print(f"   Φ_d = {phi_d:.3f}")
    print(f"   Φ_f = {phi_f:.3f}")
    print(f"   Φ_a = {phi_a:.3f}")
    print(f"   R   = {R:.3f}")
    print(f"   Classification: {'Duration-dominant' if R < 0.15 else 'Agency-balanced'}")
    
    # Example 3: Noisy oscillation (higher R due to stochastic resonance)
    print("\n3. Noisy oscillation (stochastic resonance)")
    noisy_osc = np.sin(2 * np.pi * 0.2 * t) + 0.5 * np.random.randn(len(t))
    phi_d, phi_f, phi_a, R = compute_coordination(noisy_osc)
    print(f"   Φ_d = {phi_d:.3f}")
    print(f"   Φ_f = {phi_f:.3f}")
    print(f"   Φ_a = {phi_a:.3f}")
    print(f"   R   = {R:.3f}")
    print(f"   Classification: {'Duration-dominant' if R < 0.15 else 'Agency-balanced'}")
    
    # Example 4: Event-based analysis
    print("\n4. Event-based analysis of noisy oscillation")
    events, centers = detect_events(noisy_osc, method='hybrid')
    print(f"   Detected {len(events)} events")
    
    R_values = []
    for start, end in events:
        if end - start >= 10:
            window = noisy_osc[start:end]
            _, _, _, R_event = compute_coordination(window)
            R_values.append(R_event)
    
    print(f"   Mean R across events: {np.mean(R_values):.3f} ± {np.std(R_values):.3f}")
    print(f"   Events above threshold (R > 0.15): {sum(r > 0.15 for r in R_values)}/{len(R_values)}")
    
    # Example 5: Autocatalytic-like dynamics
    print("\n5. Autocatalytic-like dynamics (logistic with noise)")
    # Simulate noisy logistic growth with oscillations
    x = np.zeros(len(t))
    x[0] = 0.1
    K = 1.0  # Carrying capacity
    r = 0.5  # Growth rate
    for i in range(1, len(t)):
        dx = r * x[i-1] * (1 - x[i-1]/K) * dt
        noise_term = 0.1 * np.sqrt(dt) * np.random.randn()
        x[i] = max(0.01, x[i-1] + dx + noise_term)
    
    phi_d, phi_f, phi_a, R = compute_coordination(x)
    print(f"   Φ_d = {phi_d:.3f}")
    print(f"   Φ_f = {phi_f:.3f}")
    print(f"   Φ_a = {phi_a:.3f}")
    print(f"   R   = {R:.3f}")
    print(f"   Classification: {'Duration-dominant' if R < 0.15 else 'Agency-balanced'}")
    
    print("\n" + "=" * 60)
    print("Proto-life threshold: R ≈ 0.15")
    print("Systems above this threshold exhibit agency-balanced dynamics")
    print("where present coordination achieves causal co-equality with")
    print("past memory and future anticipation.")
    print("=" * 60)


if __name__ == '__main__':
    main()
