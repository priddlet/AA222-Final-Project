import matplotlib.pyplot as plt
import numpy as np


def plot_results(apollo11, apollo11_pt2, burn1_unoptimized, burn1_optimized, burn2_unoptimized, burn2_optimized, start_point):
    """Plot the final trajectory of the mission.
    
    Args:
        burn1_unoptimized (tuple): Unoptimized burn1
        burn1_optimized (tuple): Optimized burn1
        burn2_unoptimized (tuple): Unoptimized burn2
        burn2_optimized (tuple): Optimized burn2
    """
    plot_pt(apollo11, burn1_unoptimized, burn1_optimized)
    plot_pt(apollo11_pt2, burn2_unoptimized, burn2_optimized)
    unoptimized_trajectory = combine_trajectories(apollo11, apollo11_pt2, burn1_unoptimized, burn2_unoptimized, start_point)
    optimized_trajectory = combine_trajectories(apollo11, apollo11_pt2, burn1_optimized, burn2_optimized, start_point)
    plot_complete_trajectories(unoptimized_trajectory, optimized_trajectory, apollo11.objects[0], apollo11.objects[1])


def combine_trajectories(apollo11, apollo11_pt2,burn1, burn2, start_point):
    """Combine the trajectories of the two Apollo 11 missions.
    
    Args:
        apollo11 (Problem): Apollo 11 problem instance
        apollo11_pt2 (Problem): Apollo 11 part 2 problem instance
    """
    rtol = 1e-7
    atol = 1e-9
    apollo11.clear_control_sequence()
    apollo11.add_burn_to_trajectory(burn1[1], burn1[0], rtol, atol)
    apollo11.simulate_trajectory(rtol, atol)
    apollo11_pt2.clear_control_sequence()
    apollo11_pt2.add_burn_to_trajectory(burn2[1], burn2[0], rtol, atol)
    apollo11_pt2.simulate_trajectory(rtol, atol)
    first_point = start_point
    print(first_point)
    trajectory = None
    min_dist = np.inf
    for i in range(len(apollo11.trajectory)):
        dist = np.linalg.norm(apollo11.trajectory[i][:2] - first_point)
        if dist < min_dist:
            min_dist = dist
            min_index = i
    trajectory = apollo11.trajectory[:min_index]
    # Find the closest point to LEO
    leo_distance = np.inf
    leo_index = 0
    for i in range(len(apollo11_pt2.trajectory)):
        dist = np.linalg.norm(apollo11_pt2.trajectory[i][:2] - apollo11_pt2.objects[0].position)
        if dist < leo_distance:
            leo_distance = dist
            leo_index = i
    trajectory = np.concatenate((trajectory, apollo11_pt2.trajectory[:leo_index]))
    return trajectory

def plot_complete_trajectories(unoptimized_trajectory, optimized_trajectory,earth,moon):
    """Plot the complete trajectories of the unoptimized and optimized missions.
    
    Args:
        unoptimized_trajectory (np.ndarray): Unoptimized trajectory
        optimized_trajectory (np.ndarray): Optimized trajectory
    """
    plt.figure(figsize=(10, 10))
    
    # Plot Earth
    earth_circle = plt.Circle(earth.position, earth.radius, color=earth.color, fill=True, alpha=0.3)
    earth_protected = plt.Circle(earth.position, earth.protected_zone, color=earth.color, fill=False, linestyle='--', alpha=0.4)
    earth_orbit = plt.Circle(earth.position, earth.max_orbit, color=earth.color, fill=False, linestyle='--', alpha=0.4)
    
    # Plot Moon  
    moon_circle = plt.Circle(moon.position, moon.radius, color=moon.color, fill=True, alpha=0.3)
    moon_protected = plt.Circle(moon.position, moon.protected_zone, color=moon.color, fill=False, linestyle='--', alpha=0.4)
    moon_orbit = plt.Circle(moon.position, moon.max_orbit, color=moon.color, fill=False, linestyle='--', alpha=0.4)
    
    # Add all patches
    ax = plt.gca()
    for patch in [earth_circle, earth_protected, earth_orbit, 
                 moon_circle, moon_protected, moon_orbit]:
        ax.add_patch(patch)

    # Plot unoptimized trajectory
    plt.plot(unoptimized_trajectory[:, 0], unoptimized_trajectory[:, 1], 'r--', label='Unoptimized', alpha=0.7)

    # Plot optimized trajectory
    plt.plot(optimized_trajectory[:, 0], optimized_trajectory[:, 1], 'g-', label='Optimized', alpha=0.7)
    
    # Set equal aspect ratio and limits
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x (normalized units)')
    plt.ylabel('y (normalized units)')
    plt.title('Earth-Moon System with Trajectories')
    plt.legend()
    
    # Set reasonable axis limits
    margin = 1.2 * max(earth.max_orbit, moon.max_orbit)
    plt.xlim(earth.position[0] - margin, moon.position[0] + margin)
    plt.ylim(-margin, margin)
    
    plt.show()

def plot_pt(apollo11, burn1_unoptimized, burn1_optimized):
    """Plot the Earth-Moon system with trajectories.
    
    Args:
        apollo11 (Problem): Apollo 11 problem instance
        burn1_unoptimized (tuple): Unoptimized first burn parameters (time, delta_v)
        burn1_optimized (tuple): Optimized first burn parameters (time, delta_v)
    """
    earth = apollo11.objects[0]
    moon = apollo11.objects[1]
    
    plt.figure(figsize=(10, 10))
    
    # Plot Earth
    earth_circle = plt.Circle(earth.position, earth.radius, color=earth.color, fill=True, alpha=0.3)
    earth_protected = plt.Circle(earth.position, earth.protected_zone, color=earth.color, fill=False, linestyle='--', alpha=0.4)
    earth_orbit = plt.Circle(earth.position, earth.max_orbit, color=earth.color, fill=False, linestyle='--', alpha=0.4)
    
    # Plot Moon  
    moon_circle = plt.Circle(moon.position, moon.radius, color=moon.color, fill=True, alpha=0.3)
    moon_protected = plt.Circle(moon.position, moon.protected_zone, color=moon.color, fill=False, linestyle='--', alpha=0.4)
    moon_orbit = plt.Circle(moon.position, moon.max_orbit, color=moon.color, fill=False, linestyle='--', alpha=0.4)
    
    # Add all patches
    ax = plt.gca()
    for patch in [earth_circle, earth_protected, earth_orbit, 
                 moon_circle, moon_protected, moon_orbit]:
        ax.add_patch(patch)

    # Plot unoptimized trajectory
    rtol = 1e-7
    atol = 1e-9
    apollo11.clear_control_sequence()
    apollo11.add_burn_to_trajectory(burn1_unoptimized[1], burn1_unoptimized[0], rtol, atol)
    apollo11.simulate_trajectory(rtol, atol)
    plt.plot(apollo11.trajectory[:, 0], apollo11.trajectory[:, 1], 'r--', label='Unoptimized', alpha=0.7)
    results = apollo11.evaluate_trajectory(rtol, atol)


    # Plot optimized trajectory
    apollo11.clear_control_sequence()
    apollo11.add_burn_to_trajectory(burn1_optimized[1], burn1_optimized[0], rtol, atol)
    apollo11.simulate_trajectory(rtol, atol)
    plt.plot(apollo11.trajectory[:, 0], apollo11.trajectory[:, 1], 'g-', label='Optimized', alpha=0.7)
    results = apollo11.evaluate_trajectory(rtol, atol)

    # Set equal aspect ratio and limits
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x (normalized units)')
    plt.ylabel('y (normalized units)')
    plt.title('Earth-Moon System with Trajectories')
    plt.legend()
    
    # Set reasonable axis limits
    margin = 1.2 * max(earth.max_orbit, moon.max_orbit)
    plt.xlim(earth.position[0] - margin, moon.position[0] + margin)
    plt.ylim(-margin, margin)
    
    plt.show()