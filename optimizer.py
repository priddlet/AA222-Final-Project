"""Genetic algorithm optimizer for trajectory optimization."""

import numpy as np
from simulation import Problem
import random

# Default optimization parameters
POP_SIZE = 30
GEN = 50
MUT_RATE = 0.1
N = 100


class Optimizer:
    """Genetic algorithm optimizer for trajectory optimization.
    
    Attributes:
        problem (Problem): The optimization problem to solve
        population_size (int): Size of the genetic algorithm population
        generations (int): Number of generations to evolve
        mutation_rate (float): Probability of mutation
        t_n (int): Number of time steps
        control_dim (tuple): Dimensions of control sequence
        control_sequence (np.ndarray): Best control sequence found
        best_cost (float): Best cost found
        best_sequence (np.ndarray): Best control sequence found
    """
    
    def __init__(self, problem: Problem, population_size=POP_SIZE, generations=GEN, 
                 mutation_rate=MUT_RATE, t_n=N):
        """Initialize the optimizer.
        
        Args:
            problem (Problem): The optimization problem to solve
            population_size (int): Size of the genetic algorithm population
            generations (int): Number of generations to evolve
            mutation_rate (float): Probability of mutation
            t_n (int): Number of time steps
        """
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.t_n = t_n
        self.control_dim = (t_n, 2)  # delta-v in 2D
        self.control_sequence = None
        self.best_cost = np.inf
        self.best_sequence = None

    def evaluate_trajectory(self, control_sequence):
        """Evaluate the cost of a control sequence.
        
        Args:
            control_sequence (np.ndarray): Control sequence to evaluate
            
        Returns:
            float: Total cost including constraint violations
        """
        self.problem.set_control_sequence(control_sequence)
        try:
            self.problem.simulate_trajectory()
            cost = self.problem.delta_v_cost()
            constraints = self.problem.evaluate_constraints()
            penalty = np.sum(np.maximum(0, constraints)**2)
            return cost + 1e5 * penalty
        except Exception as e:
            print("Simulation failed:", e)
            return 1e10  # heavy penalty

    def initialize_population(self):
        """Initialize random population of control sequences.
        
        Returns:
            list: List of random control sequences
        """
        return [np.random.uniform(-0.01, 0.01, self.control_dim) 
                for _ in range(self.population_size)]

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents.
        
        Args:
            parent1 (np.ndarray): First parent
            parent2 (np.ndarray): Second parent
            
        Returns:
            np.ndarray: Child solution
        """
        alpha = np.random.rand()
        return alpha * parent1 + (1 - alpha) * parent2

    def mutate(self, individual):
        """Apply mutation to an individual.
        
        Args:
            individual (np.ndarray): Individual to mutate
            
        Returns:
            np.ndarray: Mutated individual
        """
        noise = np.random.normal(0, 0.005, individual.shape)
        mask = np.random.rand(*individual.shape) < self.mutation_rate
        return np.where(mask, individual + noise, individual)

    def select_parents(self, scores, population, num_parents):
        """Select best parents based on scores.
        
        Args:
            scores (list): List of fitness scores
            population (list): List of individuals
            num_parents (int): Number of parents to select
            
        Returns:
            list: Selected parents
        """
        sorted_indices = np.argsort(scores)
        return [population[i] for i in sorted_indices[:num_parents]]

    def optimize(self):
        """Run the genetic algorithm optimization.
        
        Returns:
            np.ndarray: Best control sequence found
        """
        population = self.initialize_population()

        for gen in range(self.generations):
            scores = [self.evaluate_trajectory(ind) for ind in population]
            best_idx = np.argmin(scores)
            if scores[best_idx] < self.best_cost:
                self.best_cost = scores[best_idx]
                self.best_sequence = population[best_idx]

            print(f"Gen {gen}: Best Cost = {self.best_cost:.4f}")

            parents = self.select_parents(scores, population, 
                                        num_parents=self.population_size // 2)
            children = []

            while len(children) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                children.append(child)

            population = children

        self.control_sequence = self.best_sequence
        self.problem.set_control_sequence(self.best_sequence)
        self.problem.simulate_trajectory()
        return self.best_sequence

