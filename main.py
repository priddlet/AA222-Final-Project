# This file actually runs the optimization problem

import numpy as np
from simulation import Problem
from planet import Object

## Constants and params

MU_EARTH = 3.986e5    # km^3/s^2
MU_SUN = 1.327e11      # km^3/s^2
AU = 1.496e8           # km 

# Normalized units for PR3BP (sun-earth system)
mu = MU_EARTH / (MU_EARTH + MU_SUN)

scale_factor = 1e-3

# Time normalization (1 unit = 1 earth year)
T_UNIT = 2 * np.pi     # rad/year

G = 1

def main():
    # Initialize the problem
    t_span = (0, 70)
    initial_conditions = np.array([1, -10, 0, 1.5])
    planet1 = Object("Earth", np.array([0.0, 0.0]), 1, G, 3, "planet", "blue")
    planet2 = Object("Moon", np.array([10, 0.0]), 2, G, 6, "satellite", "gray")
    planet3 = Object("Sun", np.array([5, -5]), 3, G, 10, "planet", "yellow")
    problem = Problem(initial_conditions, [planet1, planet2, planet3], t_span)

    # Simulate the trajectory
    problem.simulate_trajectory()

    # Plot the trajectory
    problem.plot_trajectory()

if __name__ == "__main__":
    main()