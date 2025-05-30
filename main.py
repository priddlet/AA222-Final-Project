"""Main script for running the Apollo 11 mission simulation."""

import numpy as np
from simulation import Problem
from planet import Object
from optimizer import Optimizer
import matplotlib.pyplot as plt

# Normalized gravitational constant
G = 1


def create_bodies():
    """Create the celestial bodies for the simulation.
    
    Returns:
        tuple: (earth, moon, sun) objects
    """
    earth_radius = 2
    moon_radius = 1
    sun_radius = 5
    safe_gap = 20

    earth = Object("Earth", np.array([0.0, 0.0]), earth_radius, G, 100, 
                  "planet", "blue", [0, 0], False)
    moon = Object("Moon", np.zeros(2), moon_radius, G, 20, 
                 "satellite", "gray", [0, 0.2], False)
    sun = Object("Sun", np.zeros(2), sun_radius, G, 50, 
                "planet", "yellow", [0, 0], False)

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
    """Run a single mission phase.
    
    Args:
        name (str): Name of the phase
        initial_conditions (np.ndarray): Initial state
        t_span (tuple): Time span
        bodies (list): List of celestial bodies
        
    Returns:
        tuple: (trajectory, control_sequence, problem)
    """
    problem = Problem(initial_conditions, bodies, t_span)
    t_n = 100  # number of time steps per phase
    problem.t_eval = np.linspace(*t_span, t_n)
    optimizer = Optimizer(problem, t_n=t_n)
    best_control = optimizer.optimize()
    problem.set_control_sequence(best_control)
    problem.simulate_trajectory()
    return problem.trajectory, problem.control_sequence, problem


def plot_combined_trajectory(phases):
    """Plot the combined trajectory of all mission phases.
    
    Args:
        phases (list): List of (trajectory, control, problem) tuples
    """
    plt.figure()
    for i, (traj, _, problem) in enumerate(phases):
        for obj in problem.objects:
            circle = plt.Circle(obj.position, obj.radius, color=obj.color, 
                              fill=True, alpha=0.3)
            plt.gca().add_patch(circle)
            if obj.protected_zone:
                zone = plt.Circle(obj.position, obj.protected_zone, 
                                color=obj.color, fill=False, linestyle='--', 
                                alpha=0.4)
                plt.gca().add_patch(zone)
        plt.plot(traj[:, 0], traj[:, 1], label=f"Phase {i+1}")

    plt.scatter(phases[0][0][0, 0], phases[0][0][0, 1], color="red", 
               label="Start")
    plt.title("Full Mission Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.show()


def apollo_11_mission():
    """Run the Apollo 11 mission simulation.
    
    This function is currently not implemented.
    """
    raise NotImplementedError("Not implemented")


def main():
    """Main function to run the simulation."""
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
    plot_combined_trajectory([(traj1, ctrl1, prob1), 
                            (traj2, ctrl2, prob2), 
                            (traj3, ctrl3, prob3)])


if __name__ == "__main__":
    main()
