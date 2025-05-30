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
            t_span (tuple): Time span of simulation
            num_steps (int): Number of timesteps in simulation
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
            t_span = (self.t_eval[current_time_index], burn.time)
            t_eval= self.t_eval[current_time_index:burn.time_index]
            
            # Simulate without control
            sol = solve_ivp(self.pr3bp_dynamics, t_span, current_state,
                            t_eval=t_eval, method='RK45', rtol=rtol_sim, atol=atol_sim)
            
            # Store trajectory up to burn
            all_trajectories.append(sol.y.T)
            all_times.append(sol.t)
            
            # Apply burn
            current_state = sol.y.T[-1].copy()
            current_state[2:4] += burn.delta_v  # Add delta-v to velocity
            current_time_index = burn.time_index

            # Update the location of the burn
            burn.set_coordinates(current_state[:2])
            
            print(f"Applied burn at t={burn.time}:")
            print(f"  Delta-v: {burn.delta_v}")
            print(f"  New velocity: {current_state[2:4]}")
        
        # Simulate final segment
        t_span = (self.t_eval[current_time_index], self.t_span[1])
        t_eval = self.t_eval[current_time_index:]
        sol = solve_ivp(self.pr3bp_dynamics, t_span, current_state,
                       t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
        
        # Store final segment
        all_trajectories.append(sol.y.T)
        all_times.append(sol.t)
        
        # Combine all trajectories
        self.trajectory = np.vstack(all_trajectories)
        self.t = np.concatenate(all_times)


    def lunar_insertion_evaluate(self, delta_v_amount, time):
        """Evaluate the lunar insertion constraint.
        
        Args:
            delta_v_amount (float): Amount of delta-v
            time (float): Time
        """

        # Evaluate the constraints

        # The first constraint is the free return trajectory
        # We'll give it an inital valid trajectory so we only have to worry about optimizing ours

        # Make sure that we complete one orbit around the moon
        # This will control the sign of our constraint variable
        # First find the closest approach to the moon
        moon = self.objects[1]
        moon_distances = np.linalg.norm(self.trajectory[:, :2] - moon.position, axis=1)
        earth_distances = np.linalg.norm(self.trajectory[:, :2] - self.earth_pos, axis=1)
        # We need to make sure that this changes sign exactly twice

        completed_moon_orbit = not np.where(np.diff(np.signbit(moon_distances)))[0] == 2
        # This one I think this is 3 because in order to gaurantee the free return trajectory
        # 1. It needs to leave the earth
        # 2. It comes back to the earth
        # 3. It does an orbit around the earth
        completed_earth_orbit = not np.where(np.diff(np.signbit(earth_distances)))[0] == 3


        # Time constraint for completing the whole trajectory should also be captured by these constraints
        
        # Then we want our metric to be the combined error from our target min distance constraints
        constraints = self.planetary_orbit_constraint()

        # Then we'll create a penalty function that will penalize the constraints

        #Fixed constraint penalties
        penalty = 0
        if completed_moon_orbit:
            penalty += 1e6
        if completed_earth_orbit:
            penalty += 1e6

        #Variable constraint penalties
        dist_penalty = 1
        penalty += dist_penalty * np.sum(constraints)

        # And then we'll add the total delta-v which is our objective function
        delta_v_penalty = 10
        penalty += delta_v_penalty * self.total_delta_v_constraint()


        print(completed_earth_orbit, completed_moon_orbit, constraints, self.total_delta_v_constraint())
        return penalty
    
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
        

    
    def total_delta_v_constraint(self):
        """Calculate total delta-v constraint violation.
        
        Returns:
            float: Total delta-v used
        """
        total_delta_v = 0
        for burn in self.control_sequence:
            total_delta_v += burn.get_magnitude()
        return total_delta_v
    
    
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
            max_dist_error = -1 * (obj.ideal_max_orbit - min_dist)
            dist_errors.append(dist_error)
            dist_errors.append(max_dist_error)
        return np.array(dist_errors)
    
    
    def reentry_angle_constraint(self):
        """Calculate reentry angle constraint violation.
        
        Returns:
            np.ndarray: Constraint violation
        """
        x, y, vx, vy = self.trajectory[-1]
        earth = self.objects[0]
        r = np.sqrt((x - earth.position[0])**2 + (y - earth.position[1])**2)
        v = np.array([vx, vy])
        r_vec = np.array([x - earth.position[0], y - earth.position[1]])
        
        # Calculate the angle between the velocity vector and the position vector
        cos_theta = np.dot(v, r_vec) / (np.linalg.norm(v) * r)
        angle_deg = np.degrees(np.arccos(cos_theta))

        # Calculate the error in angle
        angle_error = abs(angle_deg - 90 - self.target_angle) - self.reentry_angle_tolerance

        return np.array([angle_error])  # Constraint is positive when outside of tolerance
    
    def terminal_constraint(self):
        """Calculate terminal constraint violations.
        
        Returns:
            np.ndarray: Constraint violations [distance error, speed error]
        """
        x, y, vx, vy = self.trajectory[-1]
        earth = self.objects[0]
        # Calculate the distance to the earth
        r = np.sqrt((x - earth.position[0])**2 + (y - earth.position[1])**2)
        v_mag = np.linalg.norm(np.array([vx, vy]))

        # Calculate the error in distance and speed
        r_error = r - self.leo_radius
        v_error = v_mag - self.max_terminal_speed

        return np.array([r_error, v_error])
    
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


