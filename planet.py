import numpy as np

class Object:
    def __init__(self, name, position, radius, G, mass, type, color, initial_velocity, dynamic):
        self.name = name
        self.position = position
        self.x = position[0]
        self.y = position[1]
        self.radius = radius
        self.G = G
        self.mass = mass
        self.color = color
        self.protected_zone = None
        self.velocity = initial_velocity
        self.trajectory = self.position
        self.dynamic = dynamic
        self.last_t = 0

    def get_gravitational_acceleration(self, x, y):
        r = np.linalg.norm(self.position - np.array([x, y]))
        a = self.mass * (self.position - np.array([x, y])) / r**3
        return a
    
    
    def propagate_position(self, a, cur_t):
        if self.dynamic:
            delta_t = cur_t - self.last_t
            # Use RK4 to propagate position and velocity
            k1v = a
            k1r = self.velocity
            
            k2v = a  # Assuming constant acceleration over small timestep
            k2r = self.velocity + 0.5 * k1v * delta_t
            
            k3v = a
            k3r = self.velocity + 0.5 * k2v * delta_t
            
            k4v = a
            k4r = self.velocity + k3v * delta_t
            
            # Update velocity and position using weighted average of slopes
            self.velocity -= (k1v + 2*k2v + 2*k3v + k4v) * delta_t / 6
            self.position += (k1r + 2*k2r + 2*k3r + k4r) * delta_t / 6
            self.x = self.position[0]
            self.y = self.position[1]
            self.trajectory = np.vstack((self.trajectory, np.array([self.x, self.y])))
            self.last_t = cur_t

    def get_distance(self, x, y):
        return np.linalg.norm(self.position - np.array([x, y]))
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
    
    def get_radius(self):
        return self.radius
    
    def set_protected_zone(self, type):
        if type == "planet":
            self.protected_zone = 1e3 # km
        elif type == "satellite":
            self.protected_zone = 50 #km
        else:
            raise ValueError("Invalid protected zone type")

