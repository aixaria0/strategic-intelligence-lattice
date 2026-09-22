# HISTORICAL LOGIC, RECONSTRUCTED: two mistakes to retain as regression targets.
import numpy as np

def old_steering_score(own_entropy, adversary_entropy, alpha=1.0, beta=1.0):
    # The actual legacy snippet reused the SAME entropy for both values:
    return beta * own_entropy - alpha * own_entropy

def old_constant_plane(agent, max_vhunt, resolution=20):
    # A flat plane across x/y is NOT a measured entropy basin.
    X, Y = np.meshgrid(np.linspace(0, 1, resolution),
                       np.linspace(0, 1, resolution))
    Z = np.full_like(X, agent["V_hunt_prev"])
    intensity = Z / (max_vhunt or 1)
    return X, Y, Z, intensity

# Original RGB Plotly Surface(facecolor=hex_colors) was unsupported.
# Original 2D Plotly layout annotations with a z coordinate were unsupported.
