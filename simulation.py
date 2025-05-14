# This file contains the simulation code for the 3 body problem
# It defines a Problem class that contains the initial conditions and the dynamics of the system
# It also contains the functions to solve the system using the 4th order Runge-Kutta method

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Problem:
    # Initialize the problem with the initial conditions
    def __init__(self, initial_conditions, mu, t_span, constraints):
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

        # Set the constraint constants
        self.max_reentry_angle = constraints[0]
        self.max_planetary_protection_violation = constraints[1]
        self.max_terminal_error = constraints[2]
    
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
        """
        Outputs:
            constraint: constraint of the planetary protection
        """
        return 0.0
    
    # Constrain angle relative to earth entry corridor
    def reentry_angle_constraint(self):
        """
        Outputs:
            constraint: constraint of the reentry angle
        """
        return self.trajectory[-1, 0] - self.max_reentry_angle
    
    # Constrain terminal error
    def terminal_constraint(self):
        """
        Outputs:
            constraint: constraint of the terminal condition
        """
        return self.trajectory[-1, 0] - self.max_terminal_error
    
    # Evaluate the constraints
    def evaluate_constraints(self):
        """
        Outputs:
            constraints: (ndarray) constraints of the system
        """
        return np.array([self.planetary_protection_constraint(), self.reentry_angle_constraint(), self.terminal_constraint()])

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
        # TODO: Mark the initial conditions
        
        # Plot the trajectory
        plt.plot(self.trajectory[:, 0], self.trajectory[:, 1])
        plt.title("Simulated PR3BP Trajectory")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid()
        plt.show()
    


