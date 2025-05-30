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
        self.t_eval = np.linspace(*t_span, 100)

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

    def delta_v_cost(self):
        """Calculate the delta-v cost.
        
        Returns:
            float: Total delta-v cost
        """
        return np.sum(np.linalg.norm(self.control_sequence, axis=1))
    
    def planetary_protection_constraint(self):
        """Calculate planetary protection constraint violations.
        
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
            dist_errors.append(dist_error)
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
            planetary_protection_error = self.planetary_protection_constraint()
            terminal_error = self.terminal_constraint()
        except Exception as e:
            print("Constraint eval failed:", e)
            return np.ones(10) * 1e6  # heavy penalty

        return np.concatenate([reentry_angle_error, planetary_protection_error, terminal_error], axis=0)

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


