import numpy as np

class Object:
    def __init__(self, name, position, radius, G, mass, type, color):
        self.name = name
        self.position = position
        self.x = position[0]
        self.y = position[1]
        self.radius = radius
        self.G = G
        self.mass = mass
        self.color = color
        self.protected_zone = None

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
    
    def set_protected_zone(self, type):
        if type == "planet":
            self.protected_zone = 1e3 # km
        elif type == "satellite":
            self.protected_zone = 50 #km
        else:
            raise ValueError("Invalid protected zone type")

