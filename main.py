"""Main script for running the Apollo 11 mission simulation."""

import numpy as np
from simulation import Problem
from planet import Object
from optimizer import GeneticOptimizer
from apollo_11 import Apollo11Mission
import matplotlib.pyplot as plt

# Normalized gravitational constant
G = 1


def create_bodies():
    """Create the celestial bodies for the simulation.
    
    Returns:
        tuple: (earth, moon) objects
    """
    earth_radius = 6.371  # Earth radius in normalized units
    moon_radius = 1.737   # Moon radius in normalized units 
    safe_gap = 384.4     # Average Earth-Moon distance in normalized units


    earth = Object("Earth", np.array([0.0, 0.0]), earth_radius, G, 100, 
                  "planet", "blue", [0, 0], False, 4 * earth_radius, 2 * earth_radius)
    moon = Object("Moon", np.zeros(2), moon_radius, G, 20, 
                 "satellite", "gray", [0, 0.2], False, 4 * moon_radius, 2 * moon_radius)

    earth.set_protected_zone("planet")      # 2 * radius
    moon.set_protected_zone("satellite")    # 3 * radius

    moon_distance = earth.protected_zone + moon.protected_zone + safe_gap

    moon.position = np.array([moon_distance, 0.0])
    moon.x, moon.y = moon.position

    return earth, moon


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
    optimizer = GeneticOptimizer(problem, t_n=t_n)
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

def run_apollo_11(earth, moon):
    apollo_11 = Apollo11Mission(earth, moon)

    # Plot initial trajectory before optimization
    apollo_11.problem.lunar_insertion_evaluate(0, 3.2, 0.5) # Get initial unoptimized trajectory
    initial_trajectory = apollo_11.problem.trajectory
    initial_control = apollo_11.problem.control_sequence
    initial_problem = apollo_11.problem
    plot_combined_trajectory([(initial_trajectory, initial_control, initial_problem)])

    """# Run the mission optimization
    apollo_11.run_mission()
    trajectory = apollo_11.trajectory
    control = apollo_11.control_sequence
    problem = apollo_11.problem
    plot_combined_trajectory([(trajectory, control, problem)])"""

def main():
    """Main function to run the simulation."""
    earth, moon = create_bodies()
    bodies = [earth, moon]

    # Define mission segments
    # TODO: Start here and modify the time scheme throughout everything
    # We want this
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
    #main()
    earth, moon = create_bodies()
    bodies = [earth, moon]
    run_apollo_11(earth, moon)
