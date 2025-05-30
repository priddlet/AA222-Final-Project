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
        control_sequence (np.ndarray): Control inputs
        t_eval (np.ndarray): Time points for evaluation
        earth_pos (np.ndarray): Earth position
        leo_radius (float): Low Earth orbit radius
        sun_pos (np.ndarray): Sun position
        sun_radius (float): Sun radius
        target_angle (float): Target reentry angle
        reentry_angle_tolerance (float): Reentry angle tolerance
        min_allowed_dist (float): Minimum allowed distance
        max_terminal_speed (float): Maximum terminal speed
    """
    
    def __init__(self, initial_conditions, objects: list[Object], t_span):
        """Initialize the problem.
        
        Args:
            initial_conditions (np.ndarray): Initial state vector
            objects (list[Object]): List of celestial objects
            t_span (tuple): Time span of simulation
        """
        self.initial_conditions = initial_conditions
        self.objects = objects
        self.t_span = t_span
        self.t = None
        self.trajectory = None
        self.control_sequence = None
        self.t_eval = np.linspace(*t_span, 4000) # Intervals of the timespan

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

        self.burn1 = [None, None]
        self.burn2 = [None, None]
    
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
    
    def set_burn1(self, delta_v_amount, index):
        """Set the first burn.
        
        Args:
            delta_v_amount (float): Amount of delta-v
            index (int): Index of the burn
        """
        self.burn1 = [delta_v_amount, index]
    
    def simulate_trajectory(self):
        """Simulate the trajectory of the system.
        
        Updates self.trajectory and self.t with the simulation results.
        """
        if self.control_sequence is None:
            # Default: no control
            self.control_sequence = np.zeros((len(self.t_eval), 2))

        # Create interpolation of control inputs over t_eval
        control_interp = lambda t: np.array([
            np.interp(t, self.t_eval, self.control_sequence[:, 0]),
            np.interp(t, self.t_eval, self.control_sequence[:, 1])
        ])

        def dynamics_with_control(t, state):
            x, y, vx, vy = state
            a = np.zeros(2)
            for obj in self.objects:
                a += obj.get_gravitational_acceleration(x, y)

            # Apply control acceleration (assumed already normalized units)
            delta_v = control_interp(t)
            return np.array([vx, vy, a[0] + delta_v[0], a[1] + delta_v[1]])

        sol = solve_ivp(dynamics_with_control, self.t_span, self.initial_conditions,
                       t_eval=self.t_eval, method='RK45', rtol=1e-8, atol=1e-10)
    
        self.trajectory = sol.y.T
        self.t = sol.t

    def set_control_sequence(self, control_sequence):
        """Set the control sequence for the system.
        
        Args:
            control_sequence (np.ndarray): Control sequence
        """
        self.control_sequence = control_sequence

    def lunar_insertion_evaluate(self, delta_v_amount, time):
        """Evaluate the lunar insertion constraint.
        
        Args:
            delta_v_amount (float): Amount of delta-v
            time (float): Time
        """
        # Propagate the trajectory with the delta-v burst
        self.burst_to_trajectory(delta_v_amount, time)

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
    
    def burst_to_trajectory(self, delta_v_amount, time):
        """Apply a delta-v burst and propagate the trajectory.
        
        Args:
            delta_v_amount (float): Amount of delta-v
            time (float): Time
            duration (float): Duration
        """
        # Simulate trajectory with delta-v applied in velocity direction
        if self.trajectory is None:
            self.simulate_trajectory()
            
        # Get state at burn time
        time_idx = np.abs(self.t - time).argmin()
        state = self.trajectory[time_idx]
        vx, vy = state[2:4]
        
        # Normalize velocity vector to get direction
        v_mag = np.sqrt(vx**2 + vy**2)
        if v_mag > 0:
            v_dir = np.array([vx/v_mag, vy/v_mag])
        else:
            v_dir = np.array([1, 0])  # Default direction if velocity is zero

        # Apply delta-v scaled by direction
        self.control_sequence[time_idx] = delta_v_amount * v_dir
        self.burn1 = [delta_v_amount, time_idx]

        # Store original control sequence
        original_control = self.control_sequence.copy()
        
        # Simulate with burn
        self.set_control_sequence(self.control_sequence)
        self.simulate_trajectory()
        
        # Restore original control sequence
        self.set_control_sequence(original_control)
    
    def total_delta_v_constraint(self):
        """Calculate total delta-v constraint violation.
        
        Returns:
            np.ndarray: Constraint violation
        """
        return np.sum(np.linalg.norm(self.control_sequence, axis=1))
    
    def get_burn1_coordinates(self):
        """Get the coordinates of the first burn.
        
        Returns:
            np.ndarray: Coordinates of the first burn
        """
        return self.trajectory[self.burn1[1]]

    def get_burn1_direction(self):
        """Get the direction of the first burn.
        
        Returns:
            np.ndarray: Direction of the first burn
        """
        return self.control_sequence[self.burn1[1]]
    
    
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


