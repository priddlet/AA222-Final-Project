# This file contains an Optimizer class which optimizes a Problem instance

import numpy as np
from simulation import Problem
import random

POP_SIZE = 30
GEN = 50
MUT_RATE = 0.1
N = 100

class Optimizer:
    def __init__(self, problem: Problem, population_size=POP_SIZE, generations=GEN, mutation_rate=MUT_RATE, t_n=N):
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
        return [np.random.uniform(-0.01, 0.01, self.control_dim) for _ in range(self.population_size)]

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

            parents = self.select_parents(scores, population, num_parents=self.population_size // 2)
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

