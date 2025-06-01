"""Simulation code for the 3-body problem.

This module defines a Problem class that contains the initial conditions and dynamics
of the system, along with functions to solve the system using the 4th order Runge-Kutta method.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from planet import Object


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Problem:
    def __init__(self, initial_conditions, mu, mission_duration, num_steps_per_timestep):
        self.initial_conditions = initial_conditions
        self.mu = mu

        self.mission_duration = mission_duration
        self.num_steps_per_timestep = num_steps_per_timestep
        self.num_steps = mission_duration * num_steps_per_timestep
        self.t_span = (0, mission_duration)
        self.t_eval = np.linspace(*self.t_span, self.num_steps)

        self.t = None
        self.trajectory = None
        self.control_sequence = []

        self.leo_radius = 0.1
        self.max_terminal_speed = 1.0
        self.target_angle = 6.5
        self.reentry_angle_tolerance = 1.0
        self.moon_radius = 0.027
        self.earth_radius = 0.034

    class Burn:
        def __init__(self, delta_v, time, time_index):
            self.delta_v = delta_v
            self.time = time
            self.time_index = time_index
            self.coordinates = None

        def set_coordinates(self, coordinates):
            self.coordinates = coordinates

        def get_direction(self):
            return self.delta_v / np.linalg.norm(self.delta_v)

        def get_magnitude(self):
            return np.linalg.norm(self.delta_v)

    def cr3bp_dynamics(self, t, state):
        x, y, vx, vy = state
        mu = self.mu

        r1 = np.sqrt((x + mu)**2 + y**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2)

        ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
        ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3

        return np.array([vx, vy, ax, ay])

    def simulate_trajectory(self, rtol=1e-7, atol=1e-9):
        all_trajectories = []
        all_times = []

        current_state = self.initial_conditions.copy()
        current_time_index = 0

        for burn in self.control_sequence:
            # Simulate up to the burn time
            t_span = (self.t_eval[current_time_index], burn.time)
            t_eval = self.t_eval[current_time_index:burn.time_index]

            sol = solve_ivp(self.cr3bp_dynamics, t_span, current_state,
                            t_eval=t_eval, method='RK45', rtol=rtol, atol=atol)

            all_trajectories.append(sol.y.T)
            all_times.append(sol.t)

            current_state = sol.y.T[-1].copy()
            current_state[2:4] += burn.delta_v
            burn.set_coordinates(current_state[:2])
            current_time_index = self.t_eval.searchsorted(burn.time)

        t_span = (self.t_eval[current_time_index], self.t_eval[-1])
        t_eval = self.t_eval[current_time_index:]
        sol = solve_ivp(self.pr3bp_dynamics, t_span, current_state,
                       t_eval=t_eval, method='RK45', rtol=rtol_sim, atol=atol_sim)
        
        # Store final segment
        all_trajectories.append(np.transpose(sol.y))
        all_times.append(sol.t)

        self.trajectory = np.vstack(all_trajectories)
        self.t = np.concatenate(all_times)

    def add_burn_to_trajectory(self, delta_v_amount, time, direction=None, rtol=1e-7, atol=1e-9):
        if self.trajectory is None:
            self.simulate_trajectory(rtol, atol)

        time_idx = np.abs(self.t_eval - time).argmin()
        effective_burn_time = self.t_eval[time_idx]
        state = self.trajectory[time_idx]
        vx, vy = state[2:4]

        if direction is not None:
            v_dir = direction / np.linalg.norm(direction)
        else:
            v_mag = np.sqrt(vx**2 + vy**2)
            v_dir = np.array([vx, vy]) / v_mag if v_mag > 0 else np.array([1, 0])

        self.control_sequence.append(self.Burn(delta_v_amount * v_dir, effective_burn_time, time_idx))

    def clear_control_sequence(self):
        self.control_sequence = []
        self.trajectory = None
        self.t = None

    def evaluate_constraints(self):
        if self.trajectory is None:
            return np.ones(6) * 1e6

        final_state = self.trajectory[-1]
        r_vec = final_state[:2]
        v_vec = final_state[2:4]

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        r_error = abs(r - self.leo_radius)
        v_error = abs(v - self.max_terminal_speed)

        cos_theta = np.dot(r_vec, v_vec) / (r * v)
        angle_deg = np.degrees(np.arccos(cos_theta))
        angle_error = abs(angle_deg - 90 - self.target_angle)

        moon_dist = np.linalg.norm(r_vec - np.array([1 - self.mu, 0]))
        moon_violation = 0 if moon_dist > self.moon_radius else 1e5

        earth_dist = np.linalg.norm(r_vec + np.array([self.mu, 0]))
        earth_violation = 0 if earth_dist > self.earth_radius else 1e5

        total_delta_v = sum(burn.get_magnitude() for burn in self.control_sequence)

        return np.array([r_error, v_error, angle_error, moon_violation, earth_violation, total_delta_v])

    def plot_trajectory(self):
        if self.trajectory is None:
            raise ValueError("Trajectory not simulated yet")

        plt.figure()
        plt.plot(self.trajectory[:, 0], self.trajectory[:, 1], label="Spacecraft Trajectory")

        plt.scatter([-self.mu, 1 - self.mu], [0, 0], color=['blue', 'gray'], label="Primaries")

        for burn in self.control_sequence:
            plt.scatter(*burn.coordinates, color='red')
            plt.arrow(burn.coordinates[0], burn.coordinates[1],
                      burn.delta_v[0], burn.delta_v[1],
                      head_width=0.01, head_length=0.02, fc='red', ec='red')

        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.axis("equal")
        plt.grid()
        plt.title("CR3BP Trajectory")
        plt.show()
