"""Main script for running the Apollo 11 mission simulation."""

import numpy as np
from simulation import Problem
from planet import Object
from optimizer import GeneticOptimizer  
import matplotlib.pyplot as plt
#from optimizer import CrossEntropyOptimizer
from hybrid import HybridOptimizer

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
    earth_protected_zone = 1 * earth_radius
    moon_protected_zone = 2 * moon_radius
    earth_max_orbit = 4 * earth_radius
    moon_max_orbit = 10 * moon_radius

    earth = Object("Earth", np.array([0.0, 0.0]), earth_radius, G, 100, 
                  "planet", "blue", [0, 0], False, earth_max_orbit, earth_protected_zone)
    moon = Object("Moon", np.zeros(2), moon_radius, G, 20, 
                 "satellite", "gray", [0, 0.2], False, moon_max_orbit, moon_protected_zone)


    moon_distance = earth.protected_zone + moon.protected_zone + safe_gap

    moon.position = np.array([moon_distance, 0.0])
    moon.x, moon.y = moon.position

    return earth, moon


def run_phase(name, initial_conditions, t_span, bodies, num_steps=1000):
    """Run a single mission phase.
    
    Args:
        name (str): Name of the phase
        initial_conditions (np.ndarray): Initial state
        t_span (tuple): Time span
        bodies (list): List of celestial bodies
        num_steps (int): Number of timesteps in simulation
        
    Returns:
        tuple: (trajectory, control_sequence, problem)
    """
    problem = Problem(initial_conditions, bodies, t_span, num_steps=num_steps)
    optimizer = GeneticOptimizer(problem, t_n=num_steps)
    best_control = optimizer.optimize()
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
            if obj.max_orbit:
                zone = plt.Circle(obj.position, obj.max_orbit, 
                                color=obj.color, fill=False, linestyle='--', 
                                alpha=0.4)
                plt.gca().add_patch(zone)
        plt.plot(traj[:, 0], traj[:, 1], label=f"Phase {i+1}")
        

    plt.scatter(phases[0][0][0, 0], phases[0][0][0, 1], color="blue", 
               label="Start")
    
    # Plot the burn coordinates and directions with scaled arrows
    for burn in phases[0][2].control_sequence:
        plt.scatter(burn.coordinates[0], burn.coordinates[1], color="red")
        plt.arrow(burn.coordinates[0], burn.coordinates[1], 
                 burn.delta_v[0], burn.delta_v[1],
                 head_width=0.2, head_length=0.3, fc='red', ec='red',
                 label='Burn Direction')
   
    plt.title("Full Mission Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")

    # Get min and max coordinates from all trajectories
    x_coords = []
    y_coords = []
    for traj, _, _ in phases:
        x_coords.extend(traj[:, 0])
        y_coords.extend(traj[:, 1])
    
    # Add some padding (10%) around the trajectory bounds
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    padding_x = (x_max - x_min) * 0.1
    padding_y = (y_max - y_min) * 0.1
    
    plt.xlim(x_min - padding_x, x_max + padding_x)
    plt.ylim(y_min - padding_y, y_max + padding_y)
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.show()

def apollo_11_mission(earth, moon):
    mission_duration = 1700
    num_steps_per_timestep = 20
    rtol = 1e-7
    atol = 1e-7

    # Initial conditions for free return trajectory
    # Starting from low Earth orbit
    earth_radius = earth.radius
    leo_altitude = 0.2  # km above Earth's surface (in normalized units)
    leo_radius = earth_radius + leo_altitude

    # Solve for circular orbit velocity
    circular_orbit_velocity = np.sqrt(G * earth.mass / leo_radius)
    
    # Initial position and velocity for circular orbit
    initial_conditions = np.array([
        leo_radius,  # x position
        0,              # y position
        0,              # x velocity
        circular_orbit_velocity            # y velocity (circular orbit velocity ~7.8 km/s)
    ])

    apollo_11 = Problem(initial_conditions, [earth, moon], mission_duration, num_steps_per_timestep)

    # Define the three burns for the free return trajectory
    # 1. TLI (Trans-Lunar Injection) burn
    tli_delta_v = 1.54 # km/s
    tli_time = 5    # minutes after launch
    apollo_11.add_burn_to_trajectory(tli_delta_v, tli_time, rtol, atol)
    apollo_11.simulate_trajectory(rtol, atol)

    # Evaluate constraints and print them
    print("\nInitial trajectory:")
    verbose = True
    constraints, valid_trajectory = apollo_11.lunar_insertion_evaluate(verbose)
    initial_trajectory = apollo_11.trajectory
    initial_control = apollo_11.control_sequence

    # Now we're going to optimize the first part of the trajectory
    plot_combined_trajectory([(initial_trajectory, initial_control, apollo_11)])

    # Define the optimizer parameters
    x0 = np.array([tli_time, tli_delta_v])
    min_time_step = 1 / num_steps_per_timestep
    max_time_step = 10
    min_delta_v = -10
    max_delta_v = 10

    # Define the optimization parameters
    num_samples = 50
    n_best = 3
    iterations = 3
    time_variance = 1e-4
    delta_v_variance = 1e-5
    decay_rate = 0.5

    # Run the optimization
    """optimizer = CrossEntropyOptimizer(apollo_11, x0, min_time_step, max_time_step, min_delta_v, max_delta_v, rtol, atol)
    best_control = optimizer.optimize(num_samples, n_best, iterations, time_variance, delta_v_variance, decay_rate)"""
    best_control = np.array([4.99943793, 1.54261006])

    # Add the burn to the trajectory
    apollo_11.clear_control_sequence()
    apollo_11.add_burn_to_trajectory(best_control[1], best_control[0], rtol, atol)
    apollo_11.simulate_trajectory(rtol, atol)
    print("\nOptimized trajectory:")
    constraints, valid_trajectory = apollo_11.lunar_insertion_evaluate(True)


    # 2. Transition to a circular moon orbit
    tangent_point, moon_orbit_delta_v = apollo_11.find_moon_orbit_delta_v()
    r_moon = np.linalg.norm(tangent_point[:2] - moon.position)
    # Calculate velocity needed for circular orbit around the moon
    v_circular = np.sqrt(moon.G * moon.mass / r_moon)
    x0 = np.concatenate([tangent_point[:2], v_circular * tangent_point[2:4] / np.linalg.norm(tangent_point[2:4])])

    # Define a Problem object for the second part of the mission
    mission_pt2_duration = 20
    num_steps_per_timestep_pt2 = 20
    apollo_11_pt2 = Problem(x0, [earth, moon], mission_pt2_duration, num_steps_per_timestep_pt2)
    apollo_11_pt2.simulate_trajectory(rtol, atol)
    
    # Orbit around the moon n times
    n_orbits = 3

    # Could also do this as length of orbit/v_circular 
    # But I'm gonna use keplers third law just to be a badass!
    moon_orbit_time = 2 * np.pi * r_moon**1.5 / np.sqrt(moon.G * moon.mass)

    return_burn_time = 2
    return_burn_delta_v = 1
    
    moon_orbit_duration = n_orbits * moon_orbit_time
    apollo_11_pt2.add_burn_to_trajectory(return_burn_delta_v, moon_orbit_duration + return_burn_time, rtol, atol)
    apollo_11_pt2.simulate_trajectory(rtol, atol)
    initial_trajectory_pt2 = apollo_11_pt2.trajectory
    initial_control_pt2 = apollo_11_pt2.control_sequence

    # Print the inital objective function and constraints
    penalty, earth_return_trajectory = apollo_11_pt2.earth_return_evaluate(True)
    print("Initial objective function:", penalty)
    print("Initial constraints:", earth_return_trajectory)

    # Define the optimizer parameters
    x0 = np.array([return_burn_time, return_burn_delta_v])
    min_time_step = 1 / num_steps_per_timestep_pt2
    max_time_step = 10
    min_delta_v = -10
    max_delta_v = 10
    num_samples = 50
    n_best = 3
    iterations = 3
    time_variance = 1e-4
    delta_v_variance = 1e-5
    decay_rate = 0.5

    """# Run the optimization
    optimizer = CrossEntropyOptimizer(apollo_11_pt2, x0, min_time_step, max_time_step, min_delta_v, max_delta_v, rtol, atol)
    best_control = optimizer.optimize(num_samples, n_best, iterations, time_variance, delta_v_variance, decay_rate)

    # Add the burn to the trajectory
    apollo_11_pt2.clear_control_sequence()
    apollo_11_pt2.add_burn_to_trajectory(best_control[1], best_control[0], rtol, atol)
    apollo_11_pt2.simulate_trajectory(rtol, atol)
    print("\nOptimized trajectory:")
    penalty, earth_return_trajectory = apollo_11_pt2.earth_return_evaluate(True)
    print("Optimized objective function:", penalty)
    print("Optimized constraints:", earth_return_trajectory)"""

    """"# Get the final trajectory and plot it
    plot_combined_trajectory([(initial_trajectory, initial_control, apollo_11), 
                              (apollo_11.trajectory, apollo_11.control_sequence, apollo_11),
                              (apollo_11_pt2.trajectory, apollo_11_pt2.control_sequence, apollo_11_pt2)])"""
    plot_combined_trajectory([(initial_trajectory_pt2, initial_control_pt2, apollo_11_pt2)])

def main():
    """Main function to run the simulation."""
    earth, moon = create_bodies()
    bodies = [earth, moon]

    # Define mission segments with uniform timesteps
    num_steps = 1000  # Number of timesteps per phase
    t1, t2, t3 = (0, 5), (0, 5), (0, 5)
    x0 = np.array([1, -10, 3, 1])

    # Run all phases
    traj1, ctrl1, prob1 = run_phase("Earth to Moon", x0, t1, bodies, num_steps=num_steps)
    traj2, ctrl2, prob2 = run_phase("Lunar Orbit", traj1[-1], t2, bodies, num_steps=num_steps)
    traj3, ctrl3, prob3 = run_phase("Return to Earth", traj2[-1], t3, bodies, num_steps=num_steps)

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
    apollo_11_mission(earth, moon)
