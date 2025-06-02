import numpy as np
from optimizer import GeneticOptimizer, TrajectoryOptimizationProblem
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class CrossEntropyOptimizerWrapper:
    def __init__(self, problem):
        self.problem = problem
        self.traj_problem = TrajectoryOptimizationProblem(problem)

    def optimize(self, initial_guess, num_samples=50, n_best=5, iterations=5, decay=0.7, log_history=False):
        mean = initial_guess.copy()
        std_dev = 0.01 * np.ones_like(initial_guess)
        best_scores = []

        for i in range(iterations):
            samples = np.random.normal(loc=mean, scale=std_dev, size=(num_samples,) + mean.shape)
            scores = []
            valid_samples = []

            for sample in samples:
                score, valid = self.traj_problem.objective(sample)
                if valid:
                    scores.append(score)
                    valid_samples.append(sample)

            if len(valid_samples) == 0:
                std_dev *= decay
                continue

            sorted_indices = np.argsort(scores)
            best_samples = np.array([valid_samples[j] for j in sorted_indices[:n_best]])
            mean = np.mean(best_samples, axis=0)
            best_scores.append(scores[sorted_indices[0]])
            std_dev *= decay

        if log_history:
            plt.plot(best_scores)
            plt.title("CEM Convergence")
            plt.xlabel("Iteration")
            plt.ylabel("Best Score")
            plt.grid()
            plt.show()

        return mean


class ParticleSwarmOptimizerWrapper:
    def __init__(self, problem, swarm_size=30, max_iter=10):
        self.problem = problem
        self.traj_problem = TrajectoryOptimizationProblem(problem)
        self.swarm_size = swarm_size
        self.max_iter = max_iter

    def optimize(self, dim_shape):
        inertia = 0.5
        cognitive = 1.5
        social = 1.5

        positions = np.random.uniform(-0.01, 0.01, (self.swarm_size, 2))
        velocities = np.zeros_like(positions)
        personal_best = positions.copy()
        personal_best_scores = np.array([self.traj_problem.objective(p)[0] for p in positions])
        
        global_best_idx = np.argmin(personal_best_scores)
        global_best = personal_best[global_best_idx].copy()

        for iter in range(self.max_iter):
            for i in range(self.swarm_size):
                r1 = np.random.rand(2)
                r2 = np.random.rand(2)
                velocities[i] = (
                    inertia * velocities[i]
                    + cognitive * r1 * (personal_best[i] - positions[i])
                    + social * r2 * (global_best - positions[i])
                )
                positions[i] += velocities[i]

                score, valid = self.traj_problem.objective(positions[i])
                if score < personal_best_scores[i]:
                    personal_best[i] = positions[i].copy()
                    personal_best_scores[i] = score

            global_best_idx = np.argmin(personal_best_scores)
            global_best = personal_best[global_best_idx].copy()
            print(f"[PSO] Iter {iter}: Best Score = {personal_best_scores[global_best_idx]:.4f}")

        return global_best


class HybridOptimizer:
    def __init__(self, problem, use_gradient_refinement=True, stage1_method="ga", initial_conditions=None):
        self.problem = problem
        self.use_gradient_refinement = use_gradient_refinement
        self.stage1_method = stage1_method
        self.initial_conditions = initial_conditions

    def optimize(self):
        print("[Stage 1] Global Search")
        if self.stage1_method == "ga":
            stage1 = GeneticOptimizer(self.problem)
            if self.initial_conditions is not None:
                stage1.best_sequence = self.initial_conditions
            best_sequence = stage1.optimize()
        elif self.stage1_method == "pso":
            stage1 = ParticleSwarmOptimizerWrapper(self.problem)
            if self.initial_conditions is not None:
                stage1.global_best = self.initial_conditions
            best_sequence = stage1.optimize(dim_shape=(2,))
        else:
            raise ValueError("Unknown stage 1 method: choose 'ga' or 'pso'")

        print("[Stage 2] Cross Entropy Refinement")
        cem = CrossEntropyOptimizerWrapper(self.problem)
        if self.initial_conditions is not None:
            best_sequence = cem.optimize(self.initial_conditions, log_history=True)
        else:
            best_sequence = cem.optimize(best_sequence, log_history=True)

        if self.use_gradient_refinement:
            print("[Stage 3] Gradient-Based Refinement")
            try:
                best_sequence = self.gradient_refine(best_sequence)
            except Exception as e:
                print("Gradient refinement failed:", e)

        self.problem.set_control_sequence(best_sequence)
        self.problem.simulate_trajectory(rtol=1e-7, atol=1e-9)
        return best_sequence

    def gradient_refine(self, initial_sequence):
        def flattened_obj(x_flat):
            x = x_flat.reshape(initial_sequence.shape)
            traj_problem = TrajectoryOptimizationProblem(self.problem)
            cost, _ = traj_problem.objective(x)
            return cost

        result = minimize(flattened_obj, initial_sequence.flatten(),
                          method="L-BFGS-B", options={'maxiter': 100})

        return result.x.reshape(initial_sequence.shape)
