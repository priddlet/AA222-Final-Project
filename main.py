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

# This is just an arbitrary sample problem
# We'll do the apollo 11 mission

# Mission:
# 1. Transition from LEO to lunar orbit
# 2. Orbit around the moon for a lil bit...
# 3. Yeet the astronauts back to earth
# 4. Transition to re-entry orbit
# Objectives:
# 1. Minimize the delta-v

# Possible approach:

# Split into smaller sub-problems
# We can make these problems constraints of the overall orbit trajectory
# 1. Transition from LEO to lunar orbit
# 2. Orbit around the moon for a lil bit...
# 3. Yeet the astronauts back to earth
# 4. Transition to re-entry orbit
# 5. Meet the re-entry orbit constraints

# And then I'm thinking we do a genetic algorithm to optimize the control sequence
# Maybe organize this by time-step, because I figure we'll just have delta-v in the tangentional direction
# And we'll have a fixed amount of delta-v total for the mission

# We'll manually find a basic control sequence that satisfies the constraints
# Because otherwise it becomes more of a reinforcement learning problem thing than an optimization problem

# So the evaluating constraints are:
# 1. The initial conditions are the initial conditions of the LEO



# We'll define this as starting in orbit around the earth
def apollo_11_mission():

def main():
    # Initialize the problem
    t_span = (0, 20)
    initial_conditions = np.array([1, -10, 3, 1])
    # Name, position, radius, G, mass, type, color, initial_velocity, dynamic
    planet1 = Object("Earth", np.array([0.0, 0.0]), 2, G, 100, "planet", "blue",[0, 0], False)
    planet2 = Object("Moon", np.array([10, 0.0]), 1, G, 20, "satellite", "gray", [0, .2], False)
    planet3 = Object("Sun", np.array([5, -5]), 1, G, 50, "planet", "yellow", [0, 0], False)
    problem = Problem(initial_conditions, [planet1, planet2, planet3], t_span)

    # Simulate the trajectory
    problem.simulate_trajectory()

    # Plot the trajectory
    problem.plot_trajectory()

if __name__ == "__main__":
    main()