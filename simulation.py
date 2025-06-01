"""Simulation code for the 3-body problem.

This module defines a Problem class that contains the initial conditions and dynamics
of the system, along with functions to solve the system using the 4th order Runge-Kutta method.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from planet import Object


class Problem:
    """Class representing a 3-body problem simulation.
    
    Attributes:
        initial_conditions (np.ndarray): Initial state vector
        objects (list[Object]): List of celestial objects
        t_span (tuple): Time span of simulation
        t (np.ndarray): Time points
        trajectory (np.ndarray): State trajectory
        control_sequence (list[Burn]): Control inputs [delta_vx, delta_vy, time] for each burn
        t_eval (np.ndarray): Time points for evaluation
        num_steps (int): Number of timesteps in simulation
        earth_pos (np.ndarray): Earth position
        leo_radius (float): Low Earth orbit radius
        sun_pos (np.ndarray): Sun position
        sun_radius (float): Sun radius
        target_angle (float): Target reentry angle
        reentry_angle_tolerance (float): Reentry angle tolerance
        min_allowed_dist (float): Minimum allowed distance
        max_terminal_speed (float): Maximum terminal speed
    """
    
    def __init__(self, initial_conditions, objects: list[Object], mission_duration, num_steps_per_timestep):
        """Initialize the problem.
        
        Args:
            initial_conditions (np.ndarray): Initial state vector
            objects (list[Object]): List of celestial objects
            mission_duration (int): Duration of mission
            num_steps_per_timestep (int): Number of timesteps per timestep
        """
        self.initial_conditions = initial_conditions
        self.objects = objects
        self.mission_duration = mission_duration
        self.num_steps_per_timestep = num_steps_per_timestep
        self.num_steps = num_steps_per_timestep * mission_duration
        self.t_span = (0, mission_duration)
        self.t = None
        self.trajectory = None
        self.control_sequence = []  # Three burns, each with [delta_vx, delta_vy, time]
        self.t_eval = np.linspace(*self.t_span, self.num_steps)  # Uniform timesteps

        # Earth position in normalized units
        self.earth_pos = np.array([0, 0])
        self.leo_radius = 1  # 1 - self.mu

        # Sun position in normalized units
        self.sun_pos = np.array([5, 0])
        self.sun_radius = 1

        # Set the constraint constants
        self.target_angle = 6.5  # Acceptable corridor: ~6.5 degrees +- 1
        self.reentry_angle_tolerance = 1.0
        self.min_allowed_dist = 1e-3
        self.max_terminal_speed = np.sqrt(1 / self.leo_radius)

    class Burn:
        def __init__(self, delta_v, time, time_index):
            self.coordinates = None
            self.delta_v = delta_v
            self.time = time
            self.time_index = time_index
        def set_coordinates(self, coordinates):
            self.coordinates = coordinates
        def get_direction(self):
            return self.delta_v / np.linalg.norm(self.delta_v)
        def get_magnitude(self):
            return np.linalg.norm(self.delta_v)

    
    def pr3bp_dynamics(self, t, state):
        """Dynamics equations for the PR3BP.
        
        Args:
            t (float): Time (required by solve_ivp but not used)
            state (np.ndarray): State vector [x, y, vx, vy]
            
        Returns:
            np.ndarray: State derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt]
        """
        x, y, vx, vy = state
        a = np.zeros(2)
        for obj in self.objects:
            a += obj.get_gravitational_acceleration(x, y)
        
        # Propagate the orbits of the other objects
        earth = self.objects[0]
        for obj in self.objects[1:]:
            obj.propagate_position(obj.get_gravitational_acceleration(earth.x, earth.y), t)

        return np.array([vx, vy, a[0], a[1]])
    
    
    def simulate_trajectory(self, rtol_sim, atol_sim):
        """Simulate the trajectory of the system by handling each burn separately.
        
        Updates self.trajectory and self.t with the simulation results.
        """
        # Initialize trajectory storage
        all_trajectories = []
        all_times = []
        
        # Start with initial conditions
        current_state = self.initial_conditions
        current_time_index = 0
        
        # Simulate between each burn
        for burn in self.control_sequence:
            # Simulate up to the burn time
            t_span = (self.t_eval[current_time_index], self.t_span[1])
            t_eval = self.t_eval[current_time_index:]

            sol= solve_ivp(self.pr3bp_dynamics, t_span, current_state,
                                t_eval=t_eval, method='RK45', rtol=rtol_sim, atol=atol_sim)

            if not sol.success:
                raise RuntimeError(f"Trajectory integration failed: {sol.message}")

            all_trajectories.append(np.transpose(sol.y))
            all_times.append(sol.t)
                        
            # Apply burn
            current_state = sol.y.T[-1].copy()
            current_state[2:4] += burn.delta_v  # Add delta-v to velocity
            current_time_index = burn.time_index

            # Update the location of the burn
            burn.set_coordinates(current_state[:2])
        
        # Simulate final segment
        t_span = (self.t_eval[current_time_index], self.t_span[1])
        t_eval = self.t_eval[current_time_index:]
        sol = solve_ivp(self.pr3bp_dynamics, t_span, current_state,
                       t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
        
        # Store final segment
        all_trajectories.append(np.transpose(sol.y))
        all_times.append(sol.t)
        
        # Combine all trajectories
        self.trajectory = np.vstack(all_trajectories)
        self.t = np.concatenate(all_times)


    def lunar_insertion_evaluate(self,verbose):
        """Evaluate the lunar insertion constraint.
        
        Args:
            verbose (bool): Whether to print verbose output
        """

        # Evaluate the constraints

        # The first constraint is the free return trajectory
        # We'll give it an inital valid trajectory so we only have to worry about optimizing ours

        # Make sure that we complete one orbit around the moon
        # This will control the sign of our constraint variable
        # First find the closest approach to the moon
        moon = self.objects[1]
        moon_x_displacement = self.trajectory[:, 0] - moon.position[0]
        moon_y_displacement = self.trajectory[:, 1] - moon.position[1]
        completed_moon_orbit = False
        free_return_trajectory = False
        finish_time = np.inf

        # Check if the trajectory crosses the moon's x-axis twice
        first_cross_index = None
        second_cross_index = None
        last_sign = moon_x_displacement[0] > 0
        for i in range(len(moon_x_displacement)):
            if moon_x_displacement[i] > 0 != last_sign:
                if first_cross_index is None:
                    first_cross_index = i
                else:
                    second_cross_index = i
                    break
            last_sign = moon_x_displacement[i] > 0
        
        # Check if at those two points the y displacement is on opposite sides of the moon
        if first_cross_index is not None and second_cross_index is not None:
            first_cross_y = moon_y_displacement[first_cross_index]
            second_cross_y = moon_y_displacement[second_cross_index]
            completed_moon_orbit = ((first_cross_y > 0 and second_cross_y > 0)
                                    or (first_cross_y < 0 and second_cross_y < 0))

        # Then, if we did make it around the moon, check if we complete an orbit around the earth afterwards
        if completed_moon_orbit:
            earth = self.objects[0]
            earth_x_displacement = self.trajectory[:, 0] - earth.position[0]
            earth_y_displacement = self.trajectory[:, 1] - earth.position[1]
            earth_x_displacement = earth_x_displacement[second_cross_index:]
            earth_y_displacement = earth_y_displacement[second_cross_index:]
            last_sign = earth_x_displacement[0] > 0
            earth_first_cross_index = None
            earth_second_cross_index = None
            for i in range(second_cross_index, len(earth_x_displacement)):
                if earth_x_displacement[i] > 0 != last_sign:
                    if earth_first_cross_index is None:
                        earth_first_cross_index = i
                    else:
                        earth_second_cross_index = i
                        break
                last_sign = earth_x_displacement[i] > 0
            
            if earth_first_cross_index is not None and earth_second_cross_index is not None:
                earth_first_cross_y = earth_y_displacement[earth_first_cross_index]
                earth_second_cross_y = earth_y_displacement[earth_second_cross_index]
                free_return_trajectory = ((earth_first_cross_y > 0 and earth_second_cross_y > 0)
                                           or (earth_first_cross_y < 0 and earth_second_cross_y < 0))

        # Time constraint for completing the whole trajectory should also be captured by these constraints

        # Then we want our metric to be the combined error from our target min distance constraints
        constraints = self.planetary_orbit_constraint()

        # Then we'll create a penalty function that will penalize the constraints

        #Fixed constraint penalties
        penalty = 0
        if not completed_moon_orbit:
            penalty += 1e6
        if not free_return_trajectory:
            penalty += 1e6

        #Variable constraint penalties
        dist_penalty = 1
        penalty += dist_penalty * np.sum(constraints)

        # And then we'll add the total delta-v which is our objective function
        delta_v_penalty = 100
        loc, moon_orbit_delta_v = self.find_moon_orbit_delta_v()
        penalty += delta_v_penalty * (self.total_delta_v_constraint() + moon_orbit_delta_v)

        # Add the time penalty
        if free_return_trajectory:
            # Get the time it takes to complete the orbit around the earth
            finish_time = self.t_eval[earth_second_cross_index]
            time_penalty = 1
            penalty += time_penalty * finish_time
        else:
            penalty += 1e6
        
        valid_trajectory = completed_moon_orbit and free_return_trajectory and np.all(constraints <= 0)

        if verbose:
            print("Valid trajectory:", valid_trajectory)
            print("Completed moon orbit:", completed_moon_orbit)
            print("Completed free return trajectory:", free_return_trajectory)
            print("Time to complete trajectory:", finish_time)
            print("Distance from earth protected zone:", constraints[0])
            print("Distance from moon protected zone:", constraints[1])
            print("Total Delta-v used:", self.total_delta_v_constraint())
            print("Penalty:", penalty)
        return penalty, valid_trajectory
    
    def find_moon_orbit_delta_v(self):
        """Find the delta-v required to transition to a circular moon orbit."""
        # We need to find the point in the trajectory where the velocity is tangential to the moon
        # We know that is they are tangential, then the dot product of the velocity and the displacement vector is 0
        moon = self.objects[1]
        moon_x_displacement = self.trajectory[:, 0] - moon.position[0]
        moon_y_displacement = self.trajectory[:, 1] - moon.position[1]
        moon_orbit_velocity = self.trajectory[:, 2:4]
        
        # Find the point in the trajectory where the velocity is tangential to the moon
        tangent_point = None
        for i in range(len(moon_x_displacement)):
            unit_velocity = moon_orbit_velocity[i] / np.linalg.norm(moon_orbit_velocity[i])
            displacement = np.array([moon_x_displacement[i], moon_y_displacement[i]])
            if abs(np.dot(unit_velocity, displacement)) < 1e-6:
                tangent_point = self.trajectory[i]
                break
        if tangent_point is None:
            raise ValueError("No tangent point found")
        
        # Calculate the delta_v required for a circular orbit around the moon
        moon_orbit_radius = np.linalg.norm(tangent_point[:2] - moon.position)
        moon_orbit_velocity = np.sqrt(moon.mu / moon_orbit_radius)
        delta_v_required = moon_orbit_velocity - np.linalg.norm(tangent_point[2:4])
        return tangent_point, delta_v_required
    
    
    def planetary_orbit_constraint(self):
        """Calculate planetary orbit constraint violations.
        
        Returns:
            np.ndarray: Constraint violations for each object
        """
        dist_errors = []
        # Calculate the minimum distance to each object
        for obj in self.objects:
            dists = np.linalg.norm(self.trajectory[:, :2] - obj.position, axis=1)
            min_dist = np.min(dists)
            # Enforce a minimum radius from each object
            dist_error = obj.protected_zone - min_dist
            max_dist_error = -1 * (obj.max_orbit - min_dist)
            dist_errors.append(dist_error)
            dist_errors.append(max_dist_error)
        return np.array(dist_errors)
    
    def add_burn_to_trajectory(self, delta_v_amount, time, rtol, atol):
        """Apply a delta-v burn to the trajectory.
        
        Args:
            delta_v_amount (float): Amount of delta-v
            time (float): Time
        """
        # Get velocity direction at burn time
        if self.trajectory is None:
            self.simulate_trajectory(rtol, atol)
            
        # Get state at burn time
        time_idx = np.abs(self.t_eval - time).argmin()
        effective_burn_time = self.t_eval[time_idx]
        state = self.trajectory[time_idx]
        vx, vy = state[2:4]
        
        # Normalize velocity vector to get direction
        v_mag = np.sqrt(vx**2 + vy**2)
        if v_mag > 0:
            v_dir = np.array([vx/v_mag, vy/v_mag])
        else:
            v_dir = np.array([1, 0])  # Default direction if velocity is zero

        # Add the burn to the control sequence
        self.control_sequence.append(self.Burn(
            np.array([delta_v_amount * v_dir[0],  delta_v_amount * v_dir[1]]),
            effective_burn_time,
            time_idx
        ))

    def earth_return_evaluate(self, verbose):
        """Evaluate the earth return constraint.
        
        Args:
            verbose (bool): Whether to print verbose output
        """
        # Make sure we cross over the earth's x-axis twice
        earth = self.objects[0]
        earth_x_displacement = self.trajectory[:, 0] - earth.position[0]
        earth_y_displacement = self.trajectory[:, 1] - earth.position[1]
        earth_first_cross_index = None
        earth_second_cross_index = None
        last_sign = earth_x_displacement[0] > 0
        for i in range(len(earth_x_displacement)):
            if earth_x_displacement[i] > 0 != last_sign:
                if earth_first_cross_index is None:
                    earth_first_cross_index = i
                else:
                    earth_second_cross_index = i
                    break
            last_sign = earth_x_displacement[i] > 0
        
        if earth_first_cross_index is not None and earth_second_cross_index is not None:
            earth_first_cross_y = earth_y_displacement[earth_first_cross_index]
            earth_second_cross_y = earth_y_displacement[earth_second_cross_index]
            earth_return_trajectory = ((earth_first_cross_y > 0 and earth_second_cross_y > 0)
                                           or (earth_first_cross_y < 0 and earth_second_cross_y < 0))
        
        # Then we want to find the closest point to our end conditions
        # First find the closest point to LEO
        leo_radius = self.leo_radius
        x_displacement = self.trajectory[:, 0] - earth.position[0]
        y_displacement = self.trajectory[:, 1] - earth.position[1]
        distance = np.sqrt(x_displacement**2 + y_displacement**2)
        leo_index = np.argmin(distance - leo_radius)

        # Then at that point we'll evaluate our end conditions
        r_error, delta_v_error = self.terminal_error(self.trajectory[leo_index])
        
        # We also want to know the time it takes to complete the trajectory
        finish_time = self.t_eval[leo_index]

        # Penalty function
        penalty = 0

        # Fixed constraint penalties
        penalty += 1e6 * earth_return_trajectory

        # Then we have our variable constraint penalties
        penalty += 50 * r_error

        # Then we'll add the total delta-v which is our objective function and the time penalty
        time_penalty = 1
        penalty += time_penalty * finish_time

        delta_v_penalty = 100
        penalty += delta_v_penalty * (delta_v_error + self.total_delta_v_constraint())


        if verbose:
            print("Earth return trajectory:", earth_return_trajectory)
            print("Time to complete trajectory:", finish_time)
            print("Total Delta-v used:", self.total_delta_v_constraint())
            print("Penalty:", penalty)
        return penalty, earth_return_trajectory
        

        
    def clear_control_sequence(self):
        """Clear the control sequence."""
        self.control_sequence = []
        self.trajectory = None
        self.t = None

    
    def total_delta_v_constraint(self):
        """Calculate total delta-v constraint violation.
        
        Returns:
            float: Total delta-v used
        """
        total_delta_v = 0
        for burn in self.control_sequence:
            total_delta_v += burn.get_magnitude()
        return total_delta_v
    
    
    def reentry_angle_error(self, final_state):
        """Calculate reentry angle constraint violation.
        
        Returns:
            np.ndarray: Constraint violation
        """
        x, y, vx, vy = final_state
        earth = self.objects[0]
        r = np.sqrt((x - earth.position[0])**2 + (y - earth.position[1])**2)
        v = np.array([vx, vy])
        r_vec = np.array([x - earth.position[0], y - earth.position[1]])
        
        # Calculate the angle between the velocity vector and the position vector
        cos_theta = np.dot(v, r_vec) / (np.linalg.norm(v) * r)
        angle_deg = np.degrees(np.arccos(cos_theta))

        # Calculate the error in angle
        angle_error = abs(angle_deg - 90 - self.target_angle) - self.reentry_angle_tolerance

        return angle_error  # Constraint is positive when outside of tolerance
    
    def terminal_error(self, final_state):
        """Calculate terminal constraint violations.
        
        Returns:
            np.ndarray: Constraint violations [distance error, speed error]
        """
        x, y, vx, vy = final_state
        earth = self.objects[0]
        # Calculate the distance to the earth
        r = np.sqrt((x - earth.position[0])**2 + (y - earth.position[1])**2)
        v_mag = np.linalg.norm(np.array([vx, vy]))

        # Calculate the error in distance and speed
        r_error = r - self.leo_radius
        v_error = v_mag - self.max_terminal_speed

        return r_error, v_error
    
    def evaluate_constraints(self):
        """Evaluate all constraints.
        
        Returns:
            np.ndarray: Combined constraint violations
        """
        if self.trajectory is None or len(self.trajectory) == 0:
            return np.ones(10) * 1e6  # heavy penalty

        try:
            reentry_angle_error = self.reentry_angle_constraint()
            planetary_orbit_error = self.planetary_orbit_constraint()
            terminal_error = self.terminal_constraint()
        except Exception as e:
            print("Constraint eval failed:", e)
            return np.ones(10) * 1e6  # heavy penalty

        return np.concatenate([reentry_angle_error, planetary_orbit_error, terminal_error], axis=0)
    

    def plot_trajectory(self):
        """Plot the trajectory of the system."""
        # Makes sure the trajectory is simulated
        if self.trajectory is None:
            raise ValueError("Trajectory not simulated yet")
        
        # Plot trajectories of dynamic objects
        for obj in self.objects:    
            if obj.dynamic:
                plt.plot(obj.trajectory[:,0], obj.trajectory[:, 1], 
                        color=obj.color, label=obj.name)

        # Plot Earth and Sun
        for obj in self.objects:    
            circle = plt.Circle(obj.position, obj.radius, color=obj.color, 
                              label=obj.name)
            plt.gca().add_patch(circle)
            
            if obj.protected_zone is not None:
                pzone = plt.Circle(obj.position, obj.protected_zone, 
                                 color=obj.color, linestyle='--', fill=False, 
                                 alpha=0.3)
                plt.gca().add_patch(pzone)
        
        plt.legend()

        # Plot the trajectory
        plt.plot(self.trajectory[:, 0], self.trajectory[:, 1])

        # Plot the initial conditions
        plt.scatter(self.initial_conditions[0], self.initial_conditions[1], 
                   color="red", label="Initial Conditions")

        plt.title("Simulated Satellite Trajectory")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid()
        plt.show()


