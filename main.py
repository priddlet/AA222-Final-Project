"""Main script for running the Apollo 11 mission simulation."""

import numpy as np
from simulation import Problem
from planet import Object
from optimizer import GeneticOptimizer  
import matplotlib.pyplot as plt
#from optimizer import CrossEntropyOptimizer
from hybrid import HybridOptimizer
from optimizer import CrossEntropyOptimizer

# Normalized gravitational constant
G = 1


def create_bodies():
    """Create the celestial bodies for the simulation.
    
    Returns:
        tuple: (earth, moon) objects
    """
    earth_radius = 6.371  # Earth radius in normalized units
    moon_radius = 1.737   # Moon radius in normalized units 
    earth_mass = 81.284
    moon_mass = 1.23
    semi_major_axis = 389.7     # Average Earth-Moon distance in normalized units
    mu = 1.21505856096e-2
    earth_pos = np.array([-mu * semi_major_axis, 0])
    moon_pos = np.array([(1 - mu) * semi_major_axis, 0])
    earth_protected_zone = 1 * earth_radius
    moon_protected_zone = 2 * moon_radius
    earth_max_orbit = 4 * earth_radius
    moon_max_orbit = 10 * moon_radius


    earth = Object("Earth", earth_pos, earth_radius, G, earth_mass, 
                  "planet", "blue", [0, 0], False, earth_max_orbit, earth_protected_zone)
    moon = Object("Moon", moon_pos, moon_radius, G, moon_mass, 
                 "satellite", "gray", [0, 0], False, moon_max_orbit, moon_protected_zone)

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

    rtol = 1e-7
    atol = 1e-9

    problem = Problem(initial_conditions, bodies, t_span, num_steps=num_steps)
    optimizer = GeneticOptimizer(problem, t_n=num_steps)
    best_control = optimizer.optimize()
    problem.simulate_trajectory(rtol, atol)
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

def get_LEO_velocity(initial_x, initial_y, moon, earth):
    """Get the velocity needed to be in a low earth orbit in the CR3BP.
    
    Args:
        initial_x (float): x position
        initial_y (float): y position
        moon (Object): moon object  
        earth (Object): earth object   
    
    Returns:
        float: velocity
    """
    mu = moon.mass / (moon.mass + earth.mass)
    length_normalization = moon.position[0] - earth.position[0]
    time_normalization_factor = 383.0
    velocity_normalization = length_normalization / time_normalization_factor
    earth_pos_normalized = earth.position / length_normalization
    x_normalized = initial_x / length_normalization
    y_normalized = initial_y / length_normalization
    r_earth_orbit = np.sqrt((x_normalized - earth_pos_normalized[0])**2 + (y_normalized - earth_pos_normalized[1])**2)
    v_circular_inertial = np.sqrt((1 - mu) / r_earth_orbit)
    v_rotating = v_circular_inertial - x_normalized
    return v_rotating * velocity_normalization

def get_moon_orbit_velocity(initial_x, initial_y, moon, earth):
    """Get the velocity vector needed to be in a circular orbit around the Moon in CR3BP.
    
    Args:
        initial_x (float): x position (in physical units, e.g. km)
        initial_y (float): y position (in physical units, e.g. km)
        moon (Object): moon object with .mass and .position
        earth (Object): earth object with .mass and .position
    
    Returns:
        np.ndarray: velocity vector in physical units (e.g. km/s) in rotating frame
    """
    mu = moon.mass / (moon.mass + earth.mass)  # Moon's mass ratio
    length_normalization = moon.position[0] - earth.position[0]  # Earth–Moon distance
    time_normalization = 383.0  # seconds per normalized time unit
    velocity_normalization = length_normalization / time_normalization

    # Normalize positions
    moon_pos_norm = moon.position / length_normalization
    x_norm = initial_x / length_normalization
    y_norm = initial_y / length_normalization

    # Relative position to Moon
    dx = x_norm - moon_pos_norm[0]
    dy = y_norm - moon_pos_norm[1]
    r_moon_orbit = np.sqrt(dx**2 + dy**2)

    # Inertial circular velocity magnitude
    v_circular_inertial = np.sqrt(mu / r_moon_orbit)

    # Tangent direction (unit vector)
    unit_velocity_tangent = np.array([dy, dx]) / r_moon_orbit
    v_inertial_vec = v_circular_inertial * unit_velocity_tangent

    # Rotation-induced velocity at this point (since omega = 1)
    v_rot_frame = np.array([-y_norm, x_norm])

    # Subtract to get rotating frame velocity vector
    v_rotating_vec = v_inertial_vec - v_rot_frame 

    # Convert to dimensional (physical) velocity
    velocity_required = (v_rotating_vec * velocity_normalization) + np.array([-y_norm, x_norm])

    return velocity_required

def plot_mission_trajectory(initial_trajectory, initial_control, problem, color='cyan', alpha=1.0, label='Trajectory'):
    """Plot the mission trajectory with a space-like visualization.
    
    Args:
        initial_trajectory (np.ndarray): Initial trajectory data
        initial_control (list): Initial control sequence
        problem (Problem): Problem instance containing simulation data
        color (str): Color for the trajectory
        alpha (float): Opacity of the trajectory
        label (str): Label for the trajectory in the legend
    """
    # Create a new figure with a dark background for a space-like feel
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Plot trajectory
    plot_combined_trajectory([
        (initial_trajectory, initial_control, problem)
    ], color=color, alpha=alpha, label=label)
    
    # Add celestial bodies
    earth_circle = plt.Circle((problem.earth_pos_normalized[0], problem.earth_pos_normalized[1]), 
                            problem.earth.radius, color='blue', alpha=0.8, label='Earth')
    moon_circle = plt.Circle((problem.moon_pos_normalized[0], problem.moon_pos_normalized[1]), 
                           problem.moon.radius, color='gray', alpha=0.8, label='Moon')
    ax.add_patch(earth_circle)
    ax.add_patch(moon_circle)
    
    # Add grid, legend and title
    plt.grid(True, alpha=0.2)
    plt.legend(loc='upper right')
    plt.title('Earth Return Trajectory Optimization', fontsize=14, pad=20)
    
    # Add some stars in the background
    num_stars = 100
    x_stars = np.random.uniform(-2, 2, num_stars)
    y_stars = np.random.uniform(-2, 2, num_stars)
    plt.scatter(x_stars, y_stars, color='white', alpha=0.5, s=1)
    
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def apollo_11_mission(earth, moon):
    mission_duration = 700
    num_steps_per_timestep = 20
    rtol = 1e-6
    atol = 1e-6
    length_of_timestep = 383.0 / num_steps_per_timestep
    print(f"Length of timestep: {length_of_timestep} seconds")

    # Initial conditions for free return trajectory
    # Starting from low Earth orbit
    earth_radius = earth.radius
    leo_altitude = 0.2  # km above Earth's surface (in normalized units)
    leo_radius = earth_radius + leo_altitude

    # Calculate position relative to Earth's actual position
    initial_x = earth.position[0] + leo_radius
    initial_y = earth.position[1]

    # Initial position and velocity for circular orbit
    initial_conditions = np.array([
        initial_x,  # x position
        initial_y,  # y position
        0,         # x velocity
        -get_LEO_velocity(initial_x, initial_y, moon, earth) # y velocity
    ])

    apollo_11 = Problem(initial_conditions, [earth, moon], mission_duration, num_steps_per_timestep, "lunar_insertion")

    # 1. TLI (Trans-Lunar Injection) burn
    tli_delta_v = 3.15011329# km/s
    tli_time = 1.98527651
    apollo_11.add_burn_to_trajectory(tli_delta_v, tli_time, rtol, atol)
    apollo_11.simulate_trajectory(rtol, atol)

    # Evaluate constraints and print them
    print("\nInitial trajectory:")
    verbose = True
    constraints, valid_trajectory = apollo_11.evaluate(verbose)
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

    # Run the hybrid optimization
    #optimizer = HybridOptimizer(apollo_11, use_gradient_refinement=True, stage1_method="pso", initial_conditions=x0)
    optimizer = CrossEntropyOptimizer(apollo_11, x0, min_time_step, max_time_step, min_delta_v, max_delta_v, rtol, atol)
    best_control = optimizer.optimize(num_samples=50, n_best=3, iterations=3, time_variance=1e-1, delta_v_variance=1e-3, decay_rate=0.5)

    # Add the optimized burn to the trajectory
    apollo_11.clear_control_sequence()
    apollo_11.add_burn_to_trajectory(best_control[1], best_control[0], rtol, atol)
    apollo_11.simulate_trajectory(rtol, atol)
    print("\nOptimized trajectory:")
    constraints, valid_trajectory = apollo_11.evaluate(True)

    plot_combined_trajectory([(apollo_11.trajectory, apollo_11.control_sequence, apollo_11)])

    # 2. Transition to a circular moon orbit
    tangent_point, inertial_moon_orbit_delta_v1 = apollo_11.find_moon_orbit_delta_v()
    start_velocity = get_moon_orbit_velocity(tangent_point[0], tangent_point[1], moon, earth)
    r_moon = np.linalg.norm(tangent_point[:2] - moon.position)
            
    # Set up initial conditions for moon orbit
    x0 = np.array([
        tangent_point[0],  # x position
        tangent_point[1],  # y position
        start_velocity[0],
        start_velocity[1]
    ])

    # Define a Problem object for the second part of the mission
    mission_pt2_duration = 1000 # TODO: Change this as needed
    num_steps_per_timestep_pt2 = 20
    apollo_11_pt2 = Problem(x0, [earth, moon], mission_pt2_duration, num_steps_per_timestep_pt2, "earth_return")
    
    # Add the moon orbit burn
    apollo_11_pt2.simulate_trajectory(rtol, atol)

    
    # Orbit around the moon n times
    n_orbits = 3

    # Could also do this as length of orbit/v_circular 
    # But I'm gonna use keplers third law just to be a badass!
    moon_orbit_time = 2 * np.pi * r_moon**1.5 / np.sqrt(moon.G * moon.mass)

    # TODO: Find good initial conditions for the return burn AFTER optimizing the first part of the trajectory
    return_burn_time = moon_orbit_time * 1.1  # after 1 full orbit + buffer
    v_circular = np.linalg.norm(x0[2:4])  # current moon orbit velocity
    v_escape = np.sqrt(2 * moon.G * moon.mass / r_moon)
    delta_v_guess = v_escape - v_circular  # or maybe just 0.7 to 1.2 km/s

    
    moon_orbit_duration = n_orbits * moon_orbit_time
    apollo_11_pt2.add_burn_to_trajectory(delta_v_guess, moon_orbit_duration + return_burn_time, rtol, atol)
    apollo_11_pt2.simulate_trajectory(rtol, atol)
    initial_trajectory_pt2 = apollo_11_pt2.trajectory
    initial_control_pt2 = apollo_11_pt2.control_sequence

    plot_combined_trajectory([(initial_trajectory_pt2, initial_control_pt2, apollo_11_pt2)])

    # Print the inital objective function and constraints
    penalty, earth_return_trajectory = apollo_11_pt2.evaluate(True)
    print("\nInitial objective function:", penalty)
    print("Initial constraints:", earth_return_trajectory)

    # Define the optimizer parameters
    x0 = np.array([return_burn_time, delta_v_guess])

    # Run the hybrid optimization
    #optimizer = HybridOptimizer(apollo_11_pt2, use_gradient_refinement=True, stage1_method="pso", initial_conditions=x0)
    #best_control = optimizer.optimize()
    optimizer = CrossEntropyOptimizer(apollo_11_pt2, x0, min_time_step, max_time_step, min_delta_v, max_delta_v, rtol, atol)
    best_control = optimizer.optimize(num_samples=50, n_best=3, iterations=3, time_variance=1e-1, delta_v_variance=1e-3, decay_rate=0.5)


    # Add the optimized burn to the trajectory
    apollo_11_pt2.clear_control_sequence()
    apollo_11_pt2.add_burn_to_trajectory(best_control[1], best_control[0], rtol, atol)
    apollo_11_pt2.simulate_trajectory(rtol, atol)
    print("\nOptimized trajectory:")
    constraints, valid_trajectory = apollo_11_pt2.evaluate(True)

    print("\nOptimized objective function:", penalty)
    print("Optimized constraints:", earth_return_trajectory)

    # Plot initial and optimized trajectories
    plot_mission_trajectory(initial_trajectory_pt2, initial_control_pt2, apollo_11_pt2, 
                          color='red', alpha=0.3, label='Initial Trajectory')
    plot_mission_trajectory(apollo_11_pt2.trajectory, apollo_11_pt2.control_sequence, apollo_11_pt2,
                          color='cyan', alpha=1.0, label='Optimized Trajectory')

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
