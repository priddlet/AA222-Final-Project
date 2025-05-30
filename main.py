import numpy as np
from simulation import Problem
from planet import Object
from optimizer import Optimizer
import matplotlib.pyplot as plt

G = 1  # normalized gravitational constant

def create_bodies():
    earth_radius = 2
    moon_radius = 1
    sun_radius = 5
    safe_gap = 20

    earth = Object("Earth", np.array([0.0, 0.0]), earth_radius, G, 100, "planet", "blue", [0, 0], False)
    moon = Object("Moon", np.zeros(2), moon_radius, G, 20, "satellite", "gray", [0, 0.2], False)
    sun = Object("Sun", np.zeros(2), sun_radius, G, 50, "planet", "yellow", [0, 0], False)

    earth.set_protected_zone("planet")      # 2 * radius
    moon.set_protected_zone("satellite")    # 3 * radius
    sun.set_protected_zone("planet")

    moon_distance = earth.protected_zone + moon.protected_zone + safe_gap
    sun_distance = moon_distance / 2

    moon.position = np.array([moon_distance, 0.0])
    moon.x, moon.y = moon.position

    sun.position = np.array([sun_distance, -500.0])
    sun.x, sun.y = sun.position

    return earth, moon, sun

def run_phase(name, initial_conditions, t_span, bodies):
    problem = Problem(initial_conditions, bodies, t_span)
    t_n = 100  # number of time steps per phase
    problem.t_eval = np.linspace(*t_span, t_n)
    optimizer = Optimizer(problem, t_n=t_n)
    best_control = optimizer.optimize()
    problem.set_control_sequence(best_control)
    problem.simulate_trajectory()
    return problem.trajectory, problem.control_sequence, problem

def plot_combined_trajectory(phases):
    plt.figure()
    for i, (traj, _, problem) in enumerate(phases):
        for obj in problem.objects:
            circle = plt.Circle(obj.position, obj.radius, color=obj.color, fill=True, alpha=0.3)
            plt.gca().add_patch(circle)
            if obj.protected_zone:
                zone = plt.Circle(obj.position, obj.protected_zone, color=obj.color, fill=False, linestyle='--', alpha=0.4)
                plt.gca().add_patch(zone)
        plt.plot(traj[:, 0], traj[:, 1], label=f"Phase {i+1}")

    plt.scatter(phases[0][0][0, 0], phases[0][0][0, 1], color="red", label="Start")
    plt.title("Full Mission Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.show()

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
    earth, moon, sun = create_bodies()
    bodies = [earth, moon, sun]

    # Define mission segments
    t1, t2, t3 = (0, 5), (0, 5), (0, 5)
    x0 = np.array([1, -10, 3, 1])

    # Run all phases
    traj1, ctrl1, prob1 = run_phase("Earth to Moon", x0, t1, bodies)
    traj2, ctrl2, prob2 = run_phase("Lunar Orbit", traj1[-1], t2, bodies)
    traj3, ctrl3, prob3 = run_phase("Return to Earth", traj2[-1], t3, bodies)

    # Combine trajectories and controls
    full_trajectory = np.vstack([traj1, traj2, traj3])
    full_control = np.vstack([ctrl1, ctrl2, ctrl3])
    total_delta_v = np.sum(np.linalg.norm(full_control, axis=1))

    # Visualize full trajectory
    plot_combined_trajectory([(traj1, ctrl1, prob1), (traj2, ctrl2, prob2), (traj3, ctrl3, prob3)])

if __name__ == "__main__":
    main()
