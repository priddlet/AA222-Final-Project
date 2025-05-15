import numpy as np

class Planet:
    def __init__(self, name, position, radius, G, mass):
        self.name = name
        self.position = position
        self.x = position[0]
        self.y = position[1]
        self.radius = radius
        self.G = G
        self.mass = mass

    def get_gravitational_force(self, other_body):
        r = np.linalg.norm(self.position - other_body.position)
        return self.G * self.mass * other_body.mass / r**2
    
    def get_gravitational_acceleration(self, x, y, vx, vy):
        r = np.linalg.norm(self.position - np.array([x, y]))
        a = self.mass * (self.position - np.array([x, y])) / r**3
        return a

    def get_distance(self, x, y):
        return np.linalg.norm(self.position - np.array([x, y]))
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
    
    def get_radius(self):
        return self.radius
