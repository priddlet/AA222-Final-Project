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
        control_sequence (list[Burn]): Control inputs [delta_vx, delta_vy, time] for each burn
        t_eval (np.ndarray): Time points for evaluation
        num_steps (int): Number of timesteps in simulation
        earth_pos (np.ndarray): Earth position
        leo_radius (float): Low Earth orbit radius
        sun_pos (np.ndarray): Sun position
        sun_radius (float): Sun radius
        target_angle (float): Target reentry angle
        reentry_angle_tolerance (float): Reentry angle tolerance
        min_allowed_dist (float): Minimum allowed distance
        max_terminal_speed (float): Maximum terminal speed
    """
    
    def __init__(self, initial_conditions, objects: list[Object], mission_duration, num_steps_per_timestep,type):
        """Initialize the problem.
        
        Args:
            initial_conditions (np.ndarray): Initial state vector
            objects (list[Object]): List of celestial objects
            mission_duration (int): Duration of mission
            num_steps_per_timestep (int): Number of timesteps per timestep
        """
        self.initial_conditions = initial_conditions
        self.objects = objects
        self.mission_duration = mission_duration
        self.num_steps_per_timestep = num_steps_per_timestep
        self.num_steps = num_steps_per_timestep * mission_duration
        self.t_span = (0, mission_duration)
        self.t = None
        self.trajectory = None
        self.control_sequence = []  # Three burns, each with [delta_vx, delta_vy, time]
        self.t_eval = np.linspace(*self.t_span, self.num_steps)  # Uniform timesteps

        # Earth position in normalized units
        self.earth = self.objects[0]
        self.moon = self.objects[1]
        self.leo_radius = self.earth.radius + 0.2 
        self.earth_pos_normalized = None
        self.moon_pos_normalized = None

        #self.mu = self.moon.mass / (self.earth.mass + self.moon.mass) # This is based on how the cr3bp is defined
        self.mu = 1.21505856096e-2

        # Set the constraint constants
        self.target_angle = 6.5  # Acceptable corridor: ~6.5 degrees +- 1
        self.reentry_angle_tolerance = 1.0
        self.min_allowed_dist = 1e-3
        self.max_terminal_speed = np.sqrt(1 / self.leo_radius)

        # Set the type of problem
        if type == "lunar_insertion":
            self.evaluate = self.lunar_insertion_evaluate
        elif type == "earth_return":
            self.evaluate = self.earth_return_evaluate

    class Burn:
        def __init__(self, delta_v, time, time_index):
            self.coordinates = None
            self.delta_v = delta_v
            self.time = time
            self.time_index = time_index
        def set_coordinates(self, coordinates):
            self.coordinates = coordinates
        def get_direction(self):
            return self.delta_v / np.linalg.norm(self.delta_v)
        def get_magnitude(self):
            return np.linalg.norm(self.delta_v)
        
    def cr3bp_dynamics(self, t, state):
        # Distance to the moon and earth
        x, y, vx, vy = state
        r1 = np.linalg.norm(np.array([x + self.mu, y])) 
        r2 = np.linalg.norm(np.array([x - 1 + self.mu, y])) 

        ax = (2 * vy) + x - (((1 - self.mu) * (x + self.mu)) / r1**3) - ((self.mu * (x - 1 + self.mu)) / r2**3)
        ay = (-2 * vx) + y - (((1 - self.mu) * y) / r1**3) - ((self.mu * y) / r2**3)

        return np.array([vx, vy, ax, ay])

    
    def simulate_trajectory(self, rtol, atol):
        """Simulate the trajectory of the system by handling each burn separately.

        Updates self.trajectory and self.t with the simulation results.
        """
        import numpy as np
        from scipy.integrate import solve_ivp

        all_trajectories = []
        all_times = []

        length_normalization = self.moon.position[0] - self.earth.position[0]
        time_normalization_factor = 383.0
        velocity_normalization = length_normalization / time_normalization_factor
        self.earth_pos_normalized = self.earth.position / length_normalization
        self.moon_pos_normalized = self.moon.position / length_normalization

        initial_position = self.initial_conditions[:2] / length_normalization
        initial_velocity = self.initial_conditions[2:4] / velocity_normalization
        current_state = np.concatenate([initial_position, initial_velocity])
        current_time_index = 0

        if len(self.control_sequence) > 0:
            for burn in self.control_sequence:
                if burn.time_index <= current_time_index:
                    continue  # Skip degenerate burns

                t0 = self.t_eval[current_time_index] / time_normalization_factor
                t1 = burn.time / time_normalization_factor

                t_span = (t0, t1)
                t_eval = self.t_eval[current_time_index:burn.time_index] / time_normalization_factor

                result = solve_ivp(self.cr3bp_dynamics, t_span, current_state,
                                t_eval=t_eval, method='RK45', rtol=rtol, atol=atol)

                all_trajectories.append(result.y.T)
                all_times.append(result.t)

                current_state = result.y[:, -1].copy()
                current_state[2:4] += burn.delta_v / velocity_normalization
                current_time_index = burn.time_index
                burn.set_coordinates(current_state[:2] * length_normalization)

        t0_final = self.t_eval[current_time_index] / time_normalization_factor
        t1_final = self.t_eval[-1] / time_normalization_factor

        t_span_final = (t0_final, t1_final)
        t_eval_final = self.t_eval[current_time_index:] / time_normalization_factor

        result_final = solve_ivp(self.cr3bp_dynamics, t_span_final, current_state,
                                t_eval=t_eval_final, method='RK45', rtol=rtol, atol=atol)

        all_trajectories.append(result_final.y.T)
        all_times.append(result_final.t)

        trajectory_array = np.vstack(all_trajectories)
        trajectory_array[:, :2] *= length_normalization
        trajectory_array[:, 2:] *= velocity_normalization
        self.trajectory = trajectory_array
        self.t = np.concatenate(all_times) * time_normalization_factor

    

    def set_control_sequence(self, control_sequence):
        """Set the control sequence for the trajectory.
        
        Args:
            control_sequence (np.ndarray): Control sequence in format [time, delta_v]
        """
        self.clear_control_sequence()
        time = control_sequence[0]
        delta_v_amount = control_sequence[1]
        self.add_burn_to_trajectory(delta_v_amount, time, rtol=1e-7, atol=1e-9)

    def add_burn_to_trajectory(self, delta_v_amount, time, rtol, atol):
        """Apply a delta-v burn to the trajectory.
        
        Args:
            delta_v_amount (float): Amount of delta-v
            time (float): Time of the burn
            rtol (float): Relative tolerance for simulation
            atol (float): Absolute tolerance for simulation
        """
        # Get velocity direction at burn time
        if self.trajectory is None:
            self.simulate_trajectory(rtol, atol)
        
        # Get state at burn time
        time_idx = np.abs(self.t_eval - time).argmin()
        effective_burn_time = self.t_eval[time_idx]
        state = self.trajectory[time_idx]
        vx, vy = state[2:4]
        
        # Normalize velocity vector to get direction
        v_mag = np.sqrt(vx**2 + vy**2)
        if v_mag > 0:
            v_dir = np.array([vx/v_mag, vy/v_mag])
        else:
            v_dir = np.array([1, 0])  # Default direction if velocity is zero

        # Add the burn to the control sequence
        self.control_sequence.append(self.Burn(
            np.array([delta_v_amount * v_dir[0], delta_v_amount * v_dir[1]]),
            effective_burn_time,
            time_idx
        ))

    def lunar_insertion_evaluate(self,verbose):
        """Evaluate the lunar insertion constraint.
        
        Args:
            verbose (bool): Whether to print verbose output
        """
        moon = self.objects[1]
        earth = self.objects[0]

        # Calculate all displacements at once
        moon_displacement = self.trajectory[:, :2] - moon.position
        earth_displacement = self.trajectory[:, :2] - earth.position
        
        # Calculate distances to moon and earth
        moon_distances = np.linalg.norm(moon_displacement, axis=1)
        earth_distances = np.linalg.norm(earth_displacement, axis=1)

        # Find closest approach to moon
        closest_approach_index = np.argmin(moon_distances)
        
        # Check for moon orbit completion using vectorized operations
        sign_x = np.sign(moon_displacement[:, 0])
        sign_y = np.sign(moon_displacement[:, 1])
        
        # Find crossings of moon's vertical axis
        x_crossings = np.where(np.diff(sign_x) != 0)[0]
        if len(x_crossings) >= 2:
            first_cross_index = x_crossings[0]
            second_cross_index = x_crossings[1]
            
            # Check if orbit crossed to other side in y-direction
            first_y = moon_displacement[first_cross_index, 1]
            second_y = moon_displacement[second_cross_index, 1]
            completed_moon_orbit = np.sign(first_y) != np.sign(second_y)
            
            if completed_moon_orbit:
                first_cross_distance = moon_distances[first_cross_index]
                second_cross_distance = moon_distances[second_cross_index]
                
                # Check for free return trajectory
                earth_x_post_moon = earth_displacement[second_cross_index:, 0]
                earth_y_post_moon = earth_displacement[second_cross_index:, 1]
                
                # Find Earth crossings after moon loop
                y_crossings = np.where(np.diff(np.sign(earth_y_post_moon)) != 0)[0]
                if len(y_crossings) >= 2:
                    earth_first_cross = y_crossings[0]
                    earth_second_cross = y_crossings[1]
                    
                    x1 = earth_x_post_moon[earth_first_cross]
                    x2 = earth_x_post_moon[earth_second_cross]
                    free_return_trajectory = np.sign(x1) != np.sign(x2)
                    
                    if free_return_trajectory:
                        finish_time = self.t_eval[second_cross_index + earth_second_cross]
                    else:
                        finish_time = np.inf
                else:
                    free_return_trajectory = False
                    finish_time = np.inf
            else:
                free_return_trajectory = False
                finish_time = np.inf
                first_cross_distance = np.inf
                second_cross_distance = np.inf
        else:
            completed_moon_orbit = False
            free_return_trajectory = False
            finish_time = np.inf
            first_cross_distance = np.inf
            second_cross_distance = np.inf

        # Calculate constraints
        constraints = self.planetary_orbit_constraint()

        # Calculate penalty
        penalty = 0
        
        # Fixed constraint penalties
        if not completed_moon_orbit:
            penalty += 1e6
        if not free_return_trajectory:
            penalty += 1e6

        # Variable constraint penalties
        penalty += np.sum(constraints)  # dist_penalty = 1
        
        # Delta-v penalty
        loc, moon_orbit_delta_v = self.find_moon_orbit_delta_v()
        penalty += 100 * (self.total_delta_v_constraint() + moon_orbit_delta_v)
        penalty += first_cross_distance + second_cross_distance

        # Time penalty
        if free_return_trajectory:
            penalty += finish_time
        else:
            penalty += 1e6
        
        valid_trajectory = completed_moon_orbit and free_return_trajectory and np.all(constraints <= 0)

        if verbose:
            print("Valid trajectory:", valid_trajectory)
            print("Completed moon orbit:", completed_moon_orbit)
            print("Completed free return trajectory:", free_return_trajectory)
            print("Time to complete trajectory:", finish_time)
            print("Distance from earth protected zone:", constraints[0])
            print("Distance from moon protected zone:", constraints[1])
            print("Total Delta-v used:", self.total_delta_v_constraint())
            print("Penalty:", penalty)
            
        return penalty, valid_trajectory
    
    def find_moon_orbit_delta_v(self):
        moon = self.objects[1]
        closest_approach_index = np.argmin(np.linalg.norm(self.trajectory[:, :2] - moon.position, axis=1))
        moon_x_displacement = self.trajectory[:, 0] - moon.position[0]
        moon_y_displacement = self.trajectory[:, 1] - moon.position[1]
        moon_orbit_velocity = self.trajectory[:, 2:4]
        search_radius = 200

        min_dot_req = 1e-3
        min_dot_value = 1
        best_index = None

        i_min = max(0, closest_approach_index - search_radius)
        i_max = min(len(self.trajectory), closest_approach_index + search_radius)

        for i in range(i_min, i_max):
            vel = moon_orbit_velocity[i]
            disp = np.array([moon_x_displacement[i], moon_y_displacement[i]])
            unit_velocity = vel / np.linalg.norm(vel)
            unit_displacement = disp / np.linalg.norm(disp)

            dot = abs(np.dot(unit_velocity, unit_displacement))  # 0 = tangent
            if dot <= min_dot_req:
                best_index = i
                break
            if dot < min_dot_value:
                min_dot_value = dot
                best_index = i
        tangent_point = self.trajectory[best_index]
        moon_orbit_radius = np.linalg.norm(tangent_point[:2] - moon.position)
        moon_orbit_speed = np.sqrt(self.mu / moon_orbit_radius)
        delta_v_required = moon_orbit_speed - np.linalg.norm(tangent_point[2:4])
        return tangent_point, delta_v_required
        
    
    def planetary_orbit_constraint(self):
        """Calculate planetary orbit constraint violations.
        
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
    
    def earth_return_evaluate(self, verbose):
        """Evaluate the earth return constraint.
        
        Args:
            verbose (bool): Whether to print verbose output
        """
        return_trajectory = False

        # First we want to make sure we have a valid trajectory
        earth = self.objects[0]

        # Displacement from Earth
        earth_x_displacement = self.trajectory[:, 0] - earth.position[0]
        earth_y_displacement = self.trajectory[:, 1] - earth.position[1]

        # Detect x-axis crossings after Moon loop
        sign_x = np.sign(earth_x_displacement)

        earth_first_cross_index = None
        earth_second_cross_index = None

        for i in range(1, len(sign_x)):
            if sign_x[i] != sign_x[i - 1] and sign_x[i] != 0 and sign_x[i - 1] != 0:
                if earth_first_cross_index is None:
                    earth_first_cross_index = i
                else:
                    earth_second_cross_index = i
                    break

        # Check that both Earth crossings are on the same side in x-direction
        if earth_first_cross_index is not None and earth_second_cross_index is not None:
            y1 = earth_y_displacement[earth_first_cross_index]
            y2 = earth_y_displacement[earth_second_cross_index]

            # Check that the trajectory crosses the x axis in the opposite direction
            return_trajectory = np.sign(y1) != np.sign(y2)
        
        # Then we want to find the closest point to our end conditions
        # First find the closest point to LEO
        leo_radius = self.leo_radius
        x_displacement = self.trajectory[:, 0] - earth.position[0]
        y_displacement = self.trajectory[:, 1] - earth.position[1]
        distance = np.sqrt(x_displacement**2 + y_displacement**2)
        leo_index = np.argmin(distance - leo_radius)

        # Then at that point we'll evaluate our end conditions
        r_error, delta_v_error = self.terminal_error(self.trajectory[leo_index])
        reentry_angle_error = self.reentry_angle_error(self.trajectory[leo_index])

        # We also want to know the time it takes to complete the trajectory
        finish_time = self.t_eval[leo_index]

        # Penalty function
        penalty = 0

        # Fixed constraint penalties
        penalty += 1e6 * (1 - return_trajectory)

        # Then we have our variable constraint penalties
        penalty += 50 * r_error

        # TODO: Add reentry angle penalty if it isn't too complicated

        # Then we'll add the total delta-v which is our objective function and the time penalty
        time_penalty = 1
        penalty += time_penalty * finish_time

        delta_v_penalty = 100
        penalty += delta_v_penalty * (delta_v_error + self.total_delta_v_constraint())


        if verbose:
            print("Earth return trajectory:", return_trajectory)
            print("Time to complete trajectory:", finish_time)
            print("Total Delta-v used:", self.total_delta_v_constraint())
            print("Penalty:", penalty)
        return penalty, return_trajectory
        
    def find_valid_trajectory(self, orbit_time):
        # Find the first point in the trajectory where the velocity is along the y axis
        # Find the index closest to the specified orbit time
        closest_time_index = np.argmin(np.abs(self.t_eval - orbit_time))
        
        # Find the index of the first point where the velocity is along the y axis
        perigee_index = None
        # Find the next index where we cross the moon's x axis
        last_sign = np.sign(self.trajectory[closest_time_index, 1] - self.objects[1].position[1])
        for i in range(closest_time_index, len(self.trajectory)):
            new_sign = np.sign(self.trajectory[i, 1] - self.objects[1].position[1])
            if new_sign != last_sign:
                perigee_index = i
                break
            last_sign = new_sign

        perigee_time = self.t_eval[perigee_index]
        leo_radius = self.leo_radius
        r = np.linalg.norm(self.trajectory[perigee_index][:2] - self.objects[1].position)
        r_prime = np.linalg.norm(self.trajectory[perigee_index][:2] - self.objects[0].position + leo_radius)
        a = (r + r_prime) / 2
        v_req = np.sqrt(self.mu * (2/r - 1/a))
        v_cur = np.linalg.norm(self.trajectory[perigee_index][2:4])
        x_loc = self.trajectory[perigee_index][0] / (self.objects[1].position[0] - self.objects[0].position[0])
        delta_v = v_req - v_cur 
        return perigee_time, -delta_v
        
        

        

        
    def clear_control_sequence(self):
        """Clear the control sequence."""
        self.control_sequence = []
        self.trajectory = None
        self.t = None

    
    def total_delta_v_constraint(self):
        """Calculate total delta-v constraint violation.
        
        Returns:
            float: Total delta-v used
        """
        total_delta_v = 0
        for burn in self.control_sequence:
            total_delta_v += burn.get_magnitude()
        return total_delta_v
    
    
    def reentry_angle_error(self, final_state):
        """Calculate reentry angle constraint violation.
        
        Returns:
            np.ndarray: Constraint violation
        """
        x, y, vx, vy = final_state
        earth = self.objects[0]
        r = np.sqrt((x - earth.position[0])**2 + (y - earth.position[1])**2)
        v = np.array([vx, vy])
        r_vec = np.array([x - earth.position[0], y - earth.position[1]])
        
        # Calculate the angle between the velocity vector and the position vector
        cos_theta = np.dot(v, r_vec) / (np.linalg.norm(v) * r)
        angle_deg = np.degrees(np.arccos(cos_theta))

        # Calculate the error in angle
        angle_error = abs(angle_deg - 90 - self.target_angle) - self.reentry_angle_tolerance

        return angle_error  # Constraint is positive when outside of tolerance
    
    def terminal_error(self, final_state):
        """Calculate terminal constraint violations.
        
        Returns:
            np.ndarray: Constraint violations [distance error, speed error]
        """
        x, y, vx, vy = final_state
        earth = self.objects[0]
        # Calculate the distance to the earth
        r = np.sqrt((x - earth.position[0])**2 + (y - earth.position[1])**2)
        v_mag = np.linalg.norm(np.array([vx, vy]))

        # Calculate the error in distance and speed
        r_error = r - self.leo_radius
        v_error = v_mag - self.max_terminal_speed

        return r_error, v_error
    
    def evaluate_constraints(self):
            """Evaluate all constraints.

            Returns:
                np.ndarray: Combined constraint violations
            """
            if self.trajectory is None or len(self.trajectory) == 0:
                return np.ones(10) * 1e6  # heavy penalty

            try:
                reentry_angle_error = np.array([self.reentry_angle_error(self.trajectory[-1])])
                planetary_orbit_error = self.planetary_orbit_constraint()
                terminal_error = self.terminal_error(self.trajectory[-1])
            except Exception as e:
                print("Constraint eval failed:", e)
                return np.ones(10) * 1e6  # heavy penalty

            return np.concatenate([reentry_angle_error, planetary_orbit_error, terminal_error], axis=0)
    

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


