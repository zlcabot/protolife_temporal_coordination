"""
Event Detection

Functions for detecting coordination events in time series data.
Event-based analysis avoids the "duration trap" where trajectory averaging 
collapses temporal heterogeneity and systematically underestimates coordination.

Reference: "Temporal Coordination as Physical Criterion for Life's Emergence"
"""

import numpy as np
from scipy import signal as sig


def detect_peaks(x, prominence_factor=0.05):
    """
    Detect peaks in time series.
    
    Parameters
    ----------
    x : array_like
        Input time series
    prominence_factor : float
        Minimum prominence as fraction of signal std (default: 0.05)
        
    Returns
    -------
    peaks : ndarray
        Indices of detected peaks
    """
    x = np.asarray(x, dtype=float)
    prominence = prominence_factor * np.std(x)
    
    peaks, properties = sig.find_peaks(x, prominence=prominence)
    
    return peaks


def detect_zero_crossings(x):
    """
    Detect zero crossings (mean crossings) in time series.
    
    Parameters
    ----------
    x : array_like
        Input time series
        
    Returns
    -------
    crossings : ndarray
        Indices of zero crossings
    """
    x = np.asarray(x, dtype=float)
    x_centered = x - np.mean(x)
    
    # Find sign changes
    signs = np.sign(x_centered)
    sign_changes = np.diff(signs)
    crossings = np.where(sign_changes != 0)[0]
    
    return crossings


def detect_envelope_peaks(x):
    """
    Detect peaks in signal envelope via Hilbert transform.
    
    Parameters
    ----------
    x : array_like
        Input time series
        
    Returns
    -------
    peaks : ndarray
        Indices of envelope peaks
    """
    x = np.asarray(x, dtype=float)
    x_centered = x - np.mean(x)
    
    # Compute analytic signal via Hilbert transform
    analytic = sig.hilbert(x_centered)
    envelope = np.abs(analytic)
    
    # Find peaks in envelope
    peaks, _ = sig.find_peaks(envelope, prominence=0.05 * np.std(envelope))
    
    return peaks


def detect_events(x, method='hybrid', window_half=20):
    """
    Detect coordination events in time series.
    
    Parameters
    ----------
    x : array_like
        Input time series
    method : str
        Detection method:
        - 'peaks': Signal peaks only
        - 'zero_crossings': Zero crossings only
        - 'envelope': Envelope peaks only
        - 'hybrid': Combination of all three (default)
    window_half : int
        Half-width of event window (default: 20)
        
    Returns
    -------
    events : list of tuple
        List of (start, end) index tuples for each detected event
    centers : ndarray
        Indices of event centers
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    
    if method == 'peaks':
        centers = detect_peaks(x)
    elif method == 'zero_crossings':
        centers = detect_zero_crossings(x)
    elif method == 'envelope':
        centers = detect_envelope_peaks(x)
    elif method == 'hybrid':
        # Combine all three methods
        peaks = detect_peaks(x)
        crossings = detect_zero_crossings(x)
        envelope = detect_envelope_peaks(x)
        
        # Merge and remove duplicates within tolerance
        all_centers = np.concatenate([peaks, crossings, envelope])
        all_centers = np.unique(all_centers)
        
        # Remove centers too close together (within window_half)
        if len(all_centers) > 1:
            centers = [all_centers[0]]
            for c in all_centers[1:]:
                if c - centers[-1] >= window_half:
                    centers.append(c)
            centers = np.array(centers)
        else:
            centers = all_centers
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert centers to event windows
    events = []
    for center in centers:
        start = max(0, center - window_half)
        end = min(n, center + window_half)
        events.append((start, end))
    
    return events, centers


def merge_overlapping_events(events, min_gap=5):
    """
    Merge overlapping or adjacent event windows.
    
    Parameters
    ----------
    events : list of tuple
        List of (start, end) tuples
    min_gap : int
        Minimum gap between events to keep them separate
        
    Returns
    -------
    merged : list of tuple
        Merged event windows
    """
    if not events:
        return []
    
    # Sort by start time
    events = sorted(events, key=lambda x: x[0])
    
    merged = [events[0]]
    for start, end in events[1:]:
        last_start, last_end = merged[-1]
        
        if start <= last_end + min_gap:
            # Merge with previous
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    
    return merged


def extract_event_windows(x, events):
    """
    Extract signal windows for each event.
    
    Parameters
    ----------
    x : array_like
        Input time series
    events : list of tuple
        List of (start, end) index tuples
        
    Returns
    -------
    windows : list of ndarray
        Signal windows for each event
    """
    x = np.asarray(x)
    windows = [x[start:end] for start, end in events]
    return windows


if __name__ == '__main__':
    # Demo: detect events in noisy oscillation
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    t = np.linspace(0, 20, 2000)
    x = np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.random.randn(len(t))
    
    events, centers = detect_events(x, method='hybrid')
    
    print(f"Detected {len(events)} events")
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(t, x, 'b-', alpha=0.7, label='Signal')
    
    # Shade event windows
    for start, end in events[:10]:  # Show first 10
        plt.axvspan(t[start], t[end], alpha=0.3, color='green')
    
    plt.scatter(t[centers[:10]], x[centers[:10]], c='red', s=50, zorder=5, label='Event centers')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.title('Event Detection Demo')
    plt.tight_layout()
    plt.savefig('event_detection_demo.png', dpi=150)
    plt.show()
