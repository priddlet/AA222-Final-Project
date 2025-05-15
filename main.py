# This file actually runs the optimization problem

import numpy as np
from simulation import Problem

## Constants and params

MU_EARTH = 3.986e5    # km^3/s^2
MU_SUN = 1.327e11      # km^3/s^2
AU = 1.496e8           # km 

# Normalized units for PR3BP (sun-earth system)
mu = MU_EARTH / (MU_EARTH + MU_SUN)

# Time normalization (1 unit = 1 earth year)
T_UNIT = 2 * np.pi     # rad/year

def main():
    # Initialize the problem
    t_span = (0, 6.28)
    initial_conditions = np.array([-1.0, 0.0, 0.0, 0.8])
    problem = Problem(initial_conditions, mu, t_span)

    # Simulate the trajectory
    problem.simulate_trajectory()

    # Plot the trajectory
    problem.plot_trajectory()

if __name__ == "__main__":
    main()