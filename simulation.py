# This file contains the simulation code for the 3 body problem
# It defines a Problem class that contains the initial conditions and the dynamics of the system
# It also contains the functions to solve the system using the 4th order Runge-Kutta method

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Problem:
    # Initialize the problem with the initial conditions
    def __init__(self, initial_conditions, mu, t_span):
        """
        Inputs:
            initial_conditions: initial conditions of the system
            mu: gravitational parameter of the system
            t_span: time span of the simulation
        """
        self.initial_conditions = initial_conditions
        self.mu = mu
        self.t_span = t_span
        self.t = None
        self.trajectory = None
        self.control_sequence = None

        # Earth position in normalized units
        self.earth_pos = np.array([-self.mu, 0])
        self.leo_radius = 1 - self.mu

        # Sun position in normalized units
        self.sun_pos = np.array([1 - self.mu, 0])
        self.sun_radius = 1

        # Set the constraint constants

        # Reentry angle constraint
        # Acceptable corridor: ~6.5 degrees +- 1
        self.target_angle = 6.5
        self.reentry_angle_tolerance = 1.0
        
        # Planetary protection constraint
        self.min_allowed_dist = 1e-3

        # Terminal speed constraint
        self.max_terminal_speed = np.sqrt(1 / self.leo_radius)
    
    # Contains the dynamics equations for the PR3BP
    def pr3bp_dynamics(self, t, state, mu):
        """
        Inputs:
            t: time
            state: state of the system
            mu: gravitational parameter of the system
        Outputs:
            dstate/dt: derivative of the state
        """
        x, y, vx, vy = state
        r1 = np.sqrt((x + self.mu)**2 + y**2)
        r2 = np.sqrt((x - 1 + self.mu)**2 + y**2)
        
        ax = 2 * vy + x - (1 - self.mu)*(x + self.mu)/r1**3 - self.mu*(x - 1 + self.mu)/r2**3
        ay = -2 * vx + y - (1 - self.mu)*y/r1**3 - self.mu*y/r2**3
        
        return [vx, vy, ax, ay]
    
    # Simulate the trajectory of the system
    # Updates the self.trajectory and self.t
    # TODO: add control input
    def simulate_trajectory(self):
        """
        Inputs:
            t_span: time span of the simulation
            control_sequence: control sequence for the system
        """
        # For shooting method: simulate dynamics with given controls
        # Here, assume zero control and just propagate
        sol = solve_ivp(self.pr3bp_dynamics, self.t_span, self.initial_conditions, args=(self.mu,), dense_output=True)
        self.trajectory = sol.y.T # state trajectory
        self.t = sol.t # time vector
    
    # Set the control sequence for the system
    def set_control_sequence(self, control_sequence):
        """
        Inputs:
            control_sequence: control sequence for the system
        """
        self.control_sequence = control_sequence

    # Cost function for the delta-v
    def delta_v_cost(self):
        """
        Outputs:
            cost: cost of the delta-v for a given control sequence
        """
        return np.sum(np.linalg.norm(self.control_sequence, axis=1))
    
    # Placeholder: define avoidance/safe zones
    def planetary_protection_constraint(self):
        # Enforce a minimum radius from Earth
        dists = np.linalg.norm(self.trajectory[:, :2] - self.earth_pos, axis=1)
        
        # Find the minimum distance
        min_dist = np.min(dists)

        # Calculate the error in distance
        dist_error = self.min_allowed_dist - min_dist

        return dist_error  # Constraint is positive when outside of tolerance
    
    # Constrain angle relative to earth entry corridor
    def reentry_angle_constraint(self):
        x, y, vx, vy = self.trajectory[-1]
        r = np.sqrt((x - self.earth_pos[0])**2 + (y - self.earth_pos[1])**2)
        v = np.array([vx, vy])
        r_vec = np.array([x - self.earth_pos[0], y - self.earth_pos[1]])
        
        # Calculate the angle between the velocity vector and the position vector
        cos_theta = np.dot(v, r_vec) / (np.linalg.norm(v) * r)
        angle_deg = np.degrees(np.arccos(cos_theta))

        # Calculate the error in angle
        angle_error = (-1 * self.reentry_angle_tolerance) + abs(angle_deg - self.target_angle)

        return angle_error  # Constraint is positive when outside of tolerance
    
    # Constrain terminal error
    def terminal_constraint(self):
        x, y, vx, vy = self.trajectory[-1]

        # Calculate the distance to the earth
        r = np.sqrt((x - self.earth_pos[0])**2 + (y - self.earth_pos[1])**2)
        v_mag = np.linalg.norm(np.array([vx, vy]))

        # Calculate the error in distance and speed
        r_error = r - self.leo_radius
        v_error = v_mag - self.max_terminal_speed

        return r_error, v_error
    
    # Evaluate the constraints
    def evaluate_constraints(self):
        """
        Outputs:
            constraints: (ndarray) constraints of the system
        """
        reentry_angle_error = self.reentry_angle_constraint()
        planetary_protection_error = self.planetary_protection_constraint()
        terminal_r_error, terminal_v_error = self.terminal_constraint()

        return np.array([reentry_angle_error, planetary_protection_error, terminal_r_error, terminal_v_error])

    # Plot the trajectory of the system
    def plot_trajectory(self):
        """
        Inputs:
            t: time vector
            y: state trajectory
        """
        # Makes sure the trajectory is simulated
        if self.trajectory is None:
            raise ValueError("Trajectory not simulated yet")
        
        # TODO: Plot the constraints
        # TODO: Plot the planets

        # Plot Earth and Sun
        earth_circle = plt.Circle(self.earth_pos, 0.02, color='blue', label='Earth')
        sun_circle = plt.Circle((0,0), 0.04, color='yellow', label='Sun')
        plt.gca().add_patch(earth_circle)
        plt.gca().add_patch(sun_circle)
        plt.legend()

        # Plot the trajectory
        plt.plot(self.trajectory[:, 0], self.trajectory[:, 1])
        plt.title("Simulated PR3BP Trajectory")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid()
        plt.show()
    


