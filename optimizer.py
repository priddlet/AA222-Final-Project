# This file contains an Optimizer class which optimizes a Problem instance

import numpy as np
from simulation import Problem

class Optimizer:
    # Initialize the optimizer with a Problem instance
    def __init__(self, problem: Problem):
        self.problem = problem
        self.control_sequence = None
        self.best_cost = np.inf
    
    def evaluate_trajectory(self, control_sequence):
        """
        Inputs:
            control_sequence: control sequence for the system
        Outputs:
            cost: cost of the control sequence
            constraints: constraints of the control sequence
        """
        self.problem.set_control_sequence(control_sequence)
        self.problem.simulate_trajectory()
        cost = self.problem.delta_v_cost()
        constraints = self.problem.evaluate_constraints()
        return cost, constraints
    
    # Optimize the control sequence
    # TODO: Actually implement this!
    def optimize(self):
        """
        Outputs:
            control_sequence: optimized control sequence
        """
        pass

