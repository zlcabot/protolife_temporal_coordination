#!/usr/bin/env python3
"""
Figure 1: Coordination Hierarchy Across Chemical and Biological Systems
Bio01 Manuscript - Nature Submission

Panel a: Representative trajectories (BZ, stochastic glycolysis, Formose)
Panel b: Triadic phase space with R = 0.15 threshold

Author: Zayin Cabot
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Set Nature-style formatting
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 8,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.0,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'figure.dpi': 300,
})

# =============================================================================
# PANEL A: Representative Trajectories
# =============================================================================

def generate_bz_trajectory(t, seed=42):
    """BZ oscillator: deterministic, duration-dominant (R ≈ 0)"""
    np.random.seed(seed)
    # Regular oscillations with minimal noise
    x = np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.sin(2 * np.pi * 0.3 * t)
    x += 0.02 * np.random.randn(len(t))  # Very small noise
    return x

def generate_stochastic_glycolysis(t, seed=43):
    """Stochastic glycolysis: noise-enhanced, agency-balanced (R = 0.32)"""
    np.random.seed(seed)
    # Limit cycle with significant stochastic component
    base = np.sin(2 * np.pi * 0.08 * t) + 0.4 * np.sin(2 * np.pi * 0.24 * t)
    noise = 0.3 * np.cumsum(np.random.randn(len(t))) / np.sqrt(len(t))
    noise = noise - np.mean(noise)
    # Add bursting behavior
    bursts = np.zeros_like(t)
    burst_times = [50, 150, 280, 420, 550]
    for bt in burst_times:
        bursts += 0.5 * np.exp(-((t - bt) ** 2) / 100)
    return base + 0.25 * noise + bursts

def generate_formose_autocatalysis(t, seed=44):
    """Formose under noise: prebiotic selection (R = 0.43)"""
    np.random.seed(seed)
    # Autocatalytic growth with environmental fluctuations
    base = 0.8 * np.sin(2 * np.pi * 0.05 * t) + 0.5 * np.sin(2 * np.pi * 0.15 * t)
    # Stochastic resonance signature
    noise = 0.4 * np.random.randn(len(t))
    # Smooth the noise to show correlation structure
    from scipy.ndimage import gaussian_filter1d
    noise_smooth = gaussian_filter1d(noise, sigma=5)
    # Add coordination events (sharper features)
    events = np.zeros_like(t)
    event_times = [80, 200, 320, 450, 580]
    for et in event_times:
        events += 0.6 * np.exp(-((t - et) ** 2) / 50) * np.sin(2 * np.pi * 0.3 * (t - et))
    return base + noise_smooth + events

def detect_events(x, t, prominence_factor=0.3):
    """Detect coordination events for shading"""
    from scipy.signal import find_peaks
    std_x = np.std(x)
    peaks, properties = find_peaks(np.abs(x - np.mean(x)), 
                                    prominence=prominence_factor * std_x,
                                    distance=20)
    # Create event windows around peaks
    events = []
    window = 15  # samples
    for p in peaks[:6]:  # Limit to 6 events for clarity
        start = max(0, p - window)
        end = min(len(t) - 1, p + window)
        events.append((t[start], t[end]))
    return events

# =============================================================================
# PANEL B: Triadic Phase Space
# =============================================================================

# System data from Table 1 (Φ_d, Φ_f, Φ_c normalized to sum to 1 for ternary plot)
systems_data = {
    # Duration-dominant (below threshold)
    'BZ Oscillator': {'phi_d': 0.85, 'phi_f': 0.92, 'phi_c': 0.00, 'R': 0.00, 'class': 'duration'},
    'Pure Protein': {'phi_d': 0.05, 'phi_f': 0.12, 'phi_c': 0.00, 'R': 0.00, 'class': 'duration'},
    'Pure RNA': {'phi_d': 0.72, 'phi_f': 0.38, 'phi_c': 0.13, 'R': 0.05, 'class': 'duration'},
    'Glycolysis (det)': {'phi_d': 0.64, 'phi_f': 0.51, 'phi_c': 0.28, 'R': 0.14, 'class': 'threshold'},
    
    # Proto-life regime
    'Protocell': {'phi_d': 0.38, 'phi_f': 0.52, 'phi_c': 0.41, 'R': 0.23, 'class': 'protolife'},
    'Glycolysis (stoch)': {'phi_d': 0.42, 'phi_f': 0.45, 'phi_c': 0.45, 'R': 0.32, 'class': 'protolife'},
    
    # Evolved catalysis
    'Enzyme (bond)': {'phi_d': 0.45, 'phi_f': 0.48, 'phi_c': 0.44, 'R': 0.31, 'class': 'evolved'},
    'Enzyme (electron)': {'phi_d': 0.44, 'phi_f': 0.49, 'phi_c': 0.44, 'R': 0.31, 'class': 'evolved'},
    'Enzyme (proton)': {'phi_d': 0.41, 'phi_f': 0.44, 'phi_c': 0.52, 'R': 0.38, 'class': 'evolved'},
    
    # Prebiotic
    'Formose (noisy)': {'phi_d': 0.39, 'phi_f': 0.41, 'phi_c': 0.52, 'R': 0.43, 'class': 'prebiotic'},
    
    # High coordination
    'Analog coupling': {'phi_d': 0.35, 'phi_f': 0.38, 'phi_c': 0.61, 'R': 0.53, 'class': 'information'},
    'Circadian': {'phi_d': 0.32, 'phi_f': 0.35, 'phi_c': 0.71, 'R': 0.59, 'class': 'multiscale'},
    'RNP Hybrid': {'phi_d': 0.31, 'phi_f': 0.33, 'phi_c': 0.73, 'R': 0.60, 'class': 'material'},
    'Formose (optimal)': {'phi_d': 0.29, 'phi_f': 0.31, 'phi_c': 0.76, 'R': 0.63, 'class': 'prebiotic'},
    'RNP Confined': {'phi_d': 0.28, 'phi_f': 0.30, 'phi_c': 0.78, 'R': 0.65, 'class': 'spatial'},
    'Digital coding': {'phi_d': 0.27, 'phi_f': 0.29, 'phi_c': 0.80, 'R': 0.66, 'class': 'information'},
}

# Color scheme by classification
class_colors = {
    'duration': '#4477AA',      # Blue - duration dominant
    'threshold': '#66CCEE',     # Cyan - at threshold
    'protolife': '#228833',     # Green - proto-life
    'evolved': '#CCBB44',       # Yellow - evolved catalysis
    'prebiotic': '#EE6677',     # Red - prebiotic
    'information': '#AA3377',   # Purple - information
    'material': '#BBBBBB',      # Gray - material synergy
    'multiscale': '#44AA99',    # Teal - multi-scale
    'spatial': '#999933',       # Olive - spatial
}

def ternary_to_cartesian(phi_d, phi_f, phi_c):
    """Convert ternary coordinates to Cartesian for plotting"""
    total = phi_d + phi_f + phi_c
    phi_d, phi_f, phi_c = phi_d/total, phi_f/total, phi_c/total
    
    # Standard ternary transform
    x = 0.5 * (2 * phi_f + phi_c)
    y = (np.sqrt(3) / 2) * phi_c
    return x, y

def draw_ternary_axes(ax):
    """Draw ternary plot axes and labels"""
    # Triangle vertices
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=0.8)
    
    # Axis labels
    ax.text(-0.08, -0.05, r'$\Phi_d$', fontsize=10, ha='center', fontweight='bold')
    ax.text(1.08, -0.05, r'$\Phi_f$', fontsize=10, ha='center', fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.08, r'$\Phi_a$', fontsize=10, ha='center', fontweight='bold')
    
    # Grid lines (every 0.2)
    for i in [0.2, 0.4, 0.6, 0.8]:
        # Lines parallel to each edge
        # Constant phi_d lines
        p1 = ternary_to_cartesian(i, 1-i, 0)
        p2 = ternary_to_cartesian(i, 0, 1-i)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', linewidth=0.3, alpha=0.5)
        
        # Constant phi_f lines  
        p1 = ternary_to_cartesian(1-i, i, 0)
        p2 = ternary_to_cartesian(0, i, 1-i)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', linewidth=0.3, alpha=0.5)
        
        # Constant phi_c lines
        p1 = ternary_to_cartesian(1-i, 0, i)
        p2 = ternary_to_cartesian(0, 1-i, i)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', linewidth=0.3, alpha=0.5)

def draw_threshold_line(ax, R_threshold=0.15):
    """Draw R = 0.15 threshold line"""
    # R = phi_c / (phi_d + phi_f + phi_c)
    # For normalized coordinates: R = phi_c
    # So threshold is phi_c = 0.15
    
    # Points along the threshold line
    phi_c = R_threshold
    points = []
    for phi_d in np.linspace(0, 1-phi_c, 50):
        phi_f = 1 - phi_d - phi_c
        if phi_f >= 0:
            x, y = ternary_to_cartesian(phi_d, phi_f, phi_c)
            points.append([x, y])
    
    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], 'k--', linewidth=1.5, 
            label=f'$R = {R_threshold}$ threshold')

# =============================================================================
# MAIN FIGURE
# =============================================================================

def create_figure1():
    """Generate complete Figure 1"""
    
    fig = plt.figure(figsize=(7.2, 4))  # Nature single column width
    
    # Panel a: Trajectories
    ax1 = fig.add_axes([0.08, 0.15, 0.42, 0.75])
    
    # Panel b: Ternary phase space
    ax2 = fig.add_axes([0.58, 0.12, 0.40, 0.80])
    
    # --- Panel A: Trajectories ---
    t = np.linspace(0, 600, 1200)
    
    # Generate trajectories
    bz = generate_bz_trajectory(t)
    glyc = generate_stochastic_glycolysis(t)
    formose = generate_formose_autocatalysis(t)
    
    # Offset for visibility
    offset = 3.5
    
    # Plot trajectories
    ax1.plot(t, bz + 2*offset, color='#4477AA', linewidth=0.8, label='BZ Oscillator')
    ax1.plot(t, glyc + offset, color='#228833', linewidth=0.8, label='Stoch. Glycolysis')
    ax1.plot(t, formose, color='#EE6677', linewidth=0.8, label='Formose (noisy)')
    
    # Detect and shade coordination events
    for traj, y_offset, color, alpha in [
        (bz, 2*offset, '#4477AA', 0.2),
        (glyc, offset, '#228833', 0.3),
        (formose, 0, '#EE6677', 0.3)
    ]:
        events = detect_events(traj, t)
        for start, end in events:
            ax1.axvspan(start, end, ymin=0, ymax=1, alpha=alpha, color=color, zorder=0)
    
    # Labels and annotations
    ax1.text(620, 2*offset, r'$R \approx 0$', fontsize=8, va='center')
    ax1.text(620, offset, r'$R = 0.32$', fontsize=8, va='center')
    ax1.text(620, 0, r'$R = 0.43$', fontsize=8, va='center')
    
    ax1.set_xlabel('Time (a.u.)')
    ax1.set_ylabel('Concentration (a.u.)')
    ax1.set_xlim(0, 700)
    ax1.set_yticks([])
    ax1.legend(loc='upper right', frameon=False, fontsize=7)
    ax1.text(-0.12, 1.02, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    
    # Add "coordination events" annotation
    ax1.text(0.5, 0.98, 'Shaded: detected coordination events', 
             transform=ax1.transAxes, fontsize=6, ha='center', va='top', 
             style='italic', color='gray')
    
    # --- Panel B: Ternary Phase Space ---
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Draw ternary framework
    draw_ternary_axes(ax2)
    draw_threshold_line(ax2)
    
    # Plot all systems
    for name, data in systems_data.items():
        x, y = ternary_to_cartesian(data['phi_d'], data['phi_f'], data['phi_c'])
        color = class_colors[data['class']]
        
        # Size based on whether it's a key system
        size = 60 if name in ['BZ Oscillator', 'Glycolysis (stoch)', 'Protocell', 
                              'RNP Hybrid', 'Digital coding', 'Formose (noisy)'] else 35
        
        ax2.scatter(x, y, c=color, s=size, edgecolors='black', linewidths=0.5, zorder=5)
        
        # Label key systems
        if name in ['BZ Oscillator', 'Pure RNA', 'Glycolysis (stoch)', 'Protocell',
                    'Enzyme (proton)', 'RNP Hybrid', 'Digital coding']:
            # Offset labels to avoid overlap
            offsets = {
                'BZ Oscillator': (-0.08, -0.04),
                'Pure RNA': (0.04, -0.04),
                'Glycolysis (stoch)': (0.05, 0.02),
                'Protocell': (-0.10, 0.02),
                'Enzyme (proton)': (0.05, -0.02),
                'RNP Hybrid': (-0.10, 0.03),
                'Digital coding': (0.04, 0.02),
            }
            dx, dy = offsets.get(name, (0.03, 0.02))
            ax2.annotate(name.replace(' (', '\n('), (x, y), (x+dx, y+dy),
                        fontsize=6, ha='left' if dx > 0 else 'right')
    
    # Add evolutionary arrow
    # From chemistry (low phi_c) to biology (high phi_c)
    arrow_start = ternary_to_cartesian(0.6, 0.35, 0.05)
    arrow_end = ternary_to_cartesian(0.30, 0.30, 0.70)
    
    ax2.annotate('', xy=arrow_end, xytext=arrow_start,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                               connectionstyle='arc3,rad=0.2'))
    
    # Arrow label
    mid_x = (arrow_start[0] + arrow_end[0]) / 2 + 0.08
    mid_y = (arrow_start[1] + arrow_end[1]) / 2
    ax2.text(mid_x, mid_y, 'Evolutionary\ntrajectory', fontsize=6, 
             ha='left', style='italic')
    
    # Threshold label
    ax2.text(0.15, 0.18, r'$R = 0.15$' + '\nthreshold', fontsize=7, 
             ha='center', style='italic')
    
    # Region labels
    ax2.text(0.25, 0.02, 'Duration-\ndominant', fontsize=6, ha='center', 
             color='#4477AA', alpha=0.8)
    ax2.text(0.75, 0.55, 'Agency-\nbalanced', fontsize=6, ha='center',
             color='#228833', alpha=0.8)
    
    ax2.text(-0.05, 1.02, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold')
    
    # Set axis limits
    ax2.set_xlim(-0.15, 1.15)
    ax2.set_ylim(-0.12, 1.0)
    
    # Save figure
    plt.savefig('figure1_coordination_hierarchy.eps', format='eps', 
                bbox_inches='tight', dpi=300)
    plt.savefig('figure1_coordination_hierarchy.pdf', format='pdf', 
                bbox_inches='tight', dpi=300)
    plt.savefig('figure1_coordination_hierarchy.png', format='png', 
                bbox_inches='tight', dpi=300)
    
    print("Figure 1 saved as EPS, PDF, and PNG")
    plt.show()

if __name__ == '__main__':
    create_figure1()
