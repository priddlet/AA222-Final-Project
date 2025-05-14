import numpy as np
from scipy.integrate import solve_ivp

## Constants and params

MU_EARTH = 3.986e5    # km^3/s^2
MU_SUN = 1.327e11      # km^3/s^2
AU = 1.496e8           # km 

# Normalized units for PR3BP (sun-earth system)
mu = MU_EARTH / (MU_EARTH + MU_SUN)

# Time normalization (1 unit = 1 earth year)
T_UNIT = 2 * np.pi     # rad/year


## Dynamics: PR3BP Equations

def pr3bp_dynamics(t, state, mu):
    x, y, vx, vy = state
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    
    ax = 2 * vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
    ay = -2 * vx + y - (1 - mu)*y/r1**3 - mu*y/r2**3
    
    return [vx, vy, ax, ay]


## Delta-V and trajectory cost (TO BE IMPLEMENTED)

def delta_v_cost(control_sequence):
    # Placeholder: actual computation depends on control structure
    # E.g., minimize sum of magnitudes of control changes or total impulse?? 
    return np.sum(np.linalg.norm(control_sequence, axis=1))


## Constraints (Stubs)

def reentry_angle_constraint(final_state):
    x, y, vx, vy = final_state
    r = np.sqrt(x**2 + y**2)
    v = np.array([vx, vy])
    r_vec = np.array([x, y])
    
    cos_theta = np.dot(v, r_vec) / (np.linalg.norm(v) * r)
    angle_deg = np.degrees(np.arccos(cos_theta))

    # Acceptable corridor: ~6.5 degrees +- 1
    target_angle = 6.5
    tolerance = 1.0

    return angle_deg - target_angle  # Constraint is zero when match


def planetary_protection_constraint(traj):
    # Enforce a minimum radius from Earth (assumed at x = -mu, y = 0 in normalized units).
    min_allowed_dist = 1e-3  # Example: 1,000 km in normalized units
    
    earth_pos = np.array([-mu, 0])
    dists = np.linalg.norm(traj[:, :2] - earth_pos, axis=1)
    
    min_dist = np.min(dists)
    
    return min_allowed_dist - min_dist  # Positive => violation


def terminal_constraint(final_state):
    x, y, vx, vy = final_state
    r = np.sqrt(x**2 + y**2)
    v_mag = np.sqrt(vx**2 + vy**2)

    # In normalized units, r_earth = 1 - mu
    desired_r = 1 - mu
    desired_v = np.sqrt(1 / desired_r)  # circular velocity approximation

    return np.array([r - desired_r, v_mag - desired_v])


## Trajectory Simulation

def simulate_trajectory(state0, control_sequence, t_span, mu):
    # For shooting method: simulate dynamics with given controls
    # Here, assume zero control and just propagate
    sol = solve_ivp(pr3bp_dynamics, t_span, state0, args=(mu,), dense_output=True)
    return sol.t, sol.y.T  # time vector and state trajectory


## Optimization Problem Setup

def optimization_problem(x):
    """
    Entry point for optimizer. `x` includes discretized control parameters.
    This returns objective and constraint values.
    """
    # Unpack decision variables (e.g., control thrusts)
    # control_sequence = x.reshape((N_steps, 2))  # if 2D control
    
    # Simulate trajectory (currently no control)
    state0 = [-1.0, 0.0, 0.0, 0.8]  # Starting guess (normalized)
    t_span = (0, 6.28)  # 1 normalized period
    
    t, traj = simulate_trajectory(state0, None, t_span, mu)
    final_state = traj[-1]

    # Objective: minimize delta-V (stub)
    cost = delta_v_cost(np.zeros((len(t), 2)))  # Placeholder

    # Constraints
    c1 = reentry_angle_constraint(final_state)
    c2 = planetary_protection_constraint(traj)
    c3 = terminal_constraint(final_state)
    
    constraints = np.array([c1, c2, c3])
    
    return cost, constraints


## Main (Testing Setup)

if __name__ == "__main__":
    # Test forward propagation
    state0 = [-1.0, 0.0, 0.0, 0.8]  # Example initial state
    t_span = (0, 6.28)
    t, traj = simulate_trajectory(state0, None, t_span, mu)

    import matplotlib.pyplot as plt
    plt.plot(traj[:, 0], traj[:, 1])
    plt.title("Simulated PR3BP Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid()
    plt.show()
