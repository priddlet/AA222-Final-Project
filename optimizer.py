"""Genetic algorithm optimizer for trajectory optimization."""

import numpy as np
from simulation import Problem
import random

# Default optimization parameters
POP_SIZE = 30
GEN = 50
MUT_RATE = 0.1
N = 100


class GeneticOptimizer:
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


class CrossEntropyOptimizer:
    def __init__(self, problem, x0, min_time_step, max_time_step, min_delta_v, max_delta_v, rtol, atol):
        self.problem = problem
        self.x0 = x0
        self.x_length = len(x0)
        # x = [delta_v_amount, time]
        self.min_time_step = min_time_step
        self.max_time_step = max_time_step
        self.min_delta_v = min_delta_v
        self.max_delta_v = max_delta_v
        self.rtol = rtol
        self.atol = atol
        self.best, self.valid = self.problem.lunar_insertion_evaluate(False)

    def optimize(self, num_samples, n_best, iterations, time_variance, delta_v_variance, decay_rate):
        """Run cross entropy optimization.
        
        Args:
            num_samples (int): Number of samples to generate per iteration
            n_best (int): Number of best samples to use for updating distribution
            iterations (int): Number of iterations to run
            time_variance (float): Variance for time sampling
            delta_v_variance (float): Variance for delta-v sampling
            decay_rate (float): Rate at which variance decays each iteration
            
        Returns:
            np.ndarray: Best parameters found [time, delta_v]
        """
        # Initialize mean at current best solution
        mean = self.x0.copy()  # [time, delta_v]
        variance = np.array([time_variance, delta_v_variance])
        
        for i in range(iterations):
            # Generate samples for time and delta_v
            time_samples = np.random.normal(mean[0], np.sqrt(variance[0]), size=(num_samples, 1))
            delta_v_samples = np.random.normal(mean[1], np.sqrt(variance[1]), size=(num_samples, 1))
            
            # Clip samples to valid ranges
            time_samples = np.clip(time_samples, self.min_time_step, self.max_time_step)
            delta_v_samples = np.clip(delta_v_samples, self.min_delta_v, self.max_delta_v)
            
            # Combine samples
            samples = np.hstack((time_samples, delta_v_samples))
            
            # Evaluate all samples
            scores = []
            valid_samples = []
            
            for sample in samples:
                print(f"Evaluating sample: time={sample[0]:.4f}, delta_v={sample[1]:.4f}")
                
                # Reset simulation state and add new burn
                self.problem.clear_control_sequence()
                self.problem.add_burn_to_trajectory(sample[1], sample[0], self.rtol, self.atol)
                self.problem.simulate_trajectory(self.rtol, self.atol)
                #self.problem.plot_trajectory()
                
                # Evaluate trajectory
                score, valid_trajectory = self.problem.lunar_insertion_evaluate(False)
                
                if valid_trajectory:
                    print(f"Valid trajectory found in iteration {i}")
                    scores.append(score)
                    valid_samples.append(sample)
            
            # Handle case where no valid trajectories are found
            if len(scores) == 0:
                print(f"No valid trajectories found in iteration {i}")
                variance = variance * decay_rate
                continue
                
            # Get best samples
            best_indices = np.argsort(scores)[:n_best]
            best_samples = np.array([valid_samples[i] for i in best_indices])
            best_score = scores[best_indices[0]]
            
            # Update distribution parameters
            mean = np.mean(best_samples, axis=0)
            variance = variance * decay_rate
            
            # Update overall best if improved
            if best_score < self.best:
                self.best = best_score
                self.x0 = best_samples[0]
                print(f"Updated best score to {self.best:.4f} with parameters {self.x0}")

            print(f"Iteration {i}: Best Score = {self.best:.4f}")
            
        return self.x0
        


