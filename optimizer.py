"""Genetic algorithm and cross entropy optimizers for trajectory optimization."""

import numpy as np
from simulation import Problem
import random
from scipy.optimize import minimize

# Default optimization parameters
POP_SIZE = 30
GEN = 50
MUT_RATE = 0.1

class TrajectoryOptimizationProblem:
    def __init__(self, problem, rtol=1e-7, atol=1e-9):
        self.problem = problem
        self.rtol = rtol
        self.atol = atol

    def objective(self, control_sequence):
        """Evaluate the cost of a control sequence."""
        self.problem.set_control_sequence(control_sequence)
        try:
            self.problem.simulate_trajectory(self.rtol, self.atol)
            cost = self.problem.total_delta_v_constraint()
            constraints = self.problem.evaluate_constraints()
            penalty = np.sum(np.maximum(0, constraints)**2)
            return cost + 1e5 * penalty, True
        except Exception as e:
            print("Evaluation failed:", e)
            return 1e10, False

class GeneticOptimizer:
    """Genetic algorithm optimizer for trajectory optimization."""

    def __init__(self, problem: Problem, population_size=POP_SIZE, generations=GEN, 
                 mutation_rate=MUT_RATE):
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.control_dim = (2,)  # [time, delta_v]
        self.control_sequence = None
        self.best_cost = np.inf
        self.best_sequence = None

    def evaluate_trajectory(self, control_sequence):
        """Evaluate the cost of a control sequence."""
        self.problem.set_control_sequence(control_sequence)
        try:
            self.problem.simulate_trajectory(rtol=1e-7, atol=1e-9)
            cost = self.problem.total_delta_v_constraint()
            constraints = self.problem.evaluate_constraints()
            penalty = np.sum(np.maximum(0, constraints)**2)
            return cost + 1e5 * penalty
        except Exception as e:
            print("Simulation failed:", e)
            return 1e10

    def initialize_population(self):
        return [np.random.uniform(-0.01, 0.01, self.control_dim) 
                for _ in range(self.population_size)]

    def crossover(self, parent1, parent2):
        alpha = np.random.rand()
        return alpha * parent1 + (1 - alpha) * parent2

    def mutate(self, individual):
        noise = np.random.normal(0, 0.005, individual.shape)
        mask = np.random.rand(*individual.shape) < self.mutation_rate
        return np.where(mask, individual + noise, individual)

    def select_parents(self, scores, population, num_parents):
        sorted_indices = np.argsort(scores)
        return [population[i] for i in sorted_indices[:num_parents]]

    def optimize(self):
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
        self.problem.simulate_trajectory(rtol=1e-7, atol=1e-9)
        return self.best_sequence

class CrossEntropyOptimizer:
    def __init__(self, problem, x0, min_time_step, max_time_step, min_delta_v, max_delta_v, rtol=1e-7, atol=1e-9):
        self.problem = problem
        self.x0 = x0
        self.x_length = len(x0)
        self.min_time_step = min_time_step
        self.max_time_step = max_time_step
        self.min_delta_v = min_delta_v
        self.max_delta_v = max_delta_v
        self.rtol = rtol
        self.atol = atol
        self.best, self.valid = self.problem.lunar_insertion_evaluate(False)

    def evaluate_objective(self, x):
        self.problem.clear_control_sequence()
        self.problem.add_burn_to_trajectory(x[1], x[0], self.rtol, self.atol)
        try:
            self.problem.simulate_trajectory(self.rtol, self.atol)
            score, valid = self.problem.evaluate(False)
            return score if valid else 1e10
        except:
            return 1e10

    def optimize(self, num_samples, n_best, iterations, time_variance, delta_v_variance, decay_rate, use_local_refinement=True):
        mean = self.x0.copy()
        variance = np.array([time_variance, delta_v_variance])

        for i in range(iterations):
            time_samples = np.random.normal(mean[0], np.sqrt(variance[0]), size=(num_samples, 1))
            delta_v_samples = np.random.normal(mean[1], np.sqrt(variance[1]), size=(num_samples, 1))

            samples = np.hstack((time_samples, delta_v_samples))
            scores = []
            valid_samples = []

            for n, sample in enumerate(samples):
                print(f"Evaluating sample {n}: time={sample[0]:.4f}, delta_v={sample[1]:.4f}")

                self.problem.clear_control_sequence()
                self.problem.add_burn_to_trajectory(sample[1], sample[0], self.rtol, self.atol)
                try:
                    self.problem.simulate_trajectory(self.rtol, self.atol)
                    score, valid = self.problem.evaluate(False)
                    if valid:
                        scores.append(score)
                        valid_samples.append(sample)
                        print(f"  ✓ Valid trajectory with score {score:.4f}")
                except:
                    continue

            if len(scores) == 0:
                print(f"No valid trajectories found in iteration {i}")
                variance *= decay_rate
                continue

            best_indices = np.argsort(scores)[:n_best]
            best_samples = np.array([valid_samples[j] for j in best_indices])
            best_score = scores[best_indices[0]]
            mean = np.mean(best_samples, axis=0)
            variance *= decay_rate

            if best_score < self.best:
                self.best = best_score
                self.x0 = best_samples[0]
                print(f"  ✳ Updated best score to {self.best:.4f} with parameters {self.x0}")

            print(f"Iteration {i}: Current best score = {self.best:.4f}")

        if use_local_refinement:
            print("Starting local refinement...")
            bounds = [(self.min_time_step, self.max_time_step), (self.min_delta_v, self.max_delta_v)]
            
            # First stage: Nelder-Mead (more robust but slower)
            print("Stage 1: Nelder-Mead optimization...")
            res_nm = minimize(self.evaluate_objective, self.x0, 
                            method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res_nm.success and res_nm.fun < self.best:
                self.best = res_nm.fun
                self.x0 = res_nm.x
                print(f"✓ Nelder-Mead improved score to {self.best:.4f}")
                
                # Second stage: BFGS (faster but needs good initial guess)
                print("Stage 2: BFGS refinement...")
                res_bfgs = minimize(self.evaluate_objective, self.x0,
                                  method='BFGS',
                                  options={'maxiter': 50})
                
                if res_bfgs.success and res_bfgs.fun < self.best:
                    self.best = res_bfgs.fun
                    self.x0 = res_bfgs.x
                    print(f"✓ BFGS further improved score to {self.best:.4f}")
            else:
                print("✗ Local refinement did not improve the score")

        return self.x0
