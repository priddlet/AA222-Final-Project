import numpy as np


class Object:
    """A class representing a celestial object in the simulation.
    
    Attributes:
        name (str): Name of the object
        position (np.ndarray): 2D position vector [x, y]
        radius (float): Radius of the object
        G (float): Gravitational constant
        mass (float): Mass of the object
        color (str): Color for visualization
        protected_zone (float): Protected zone radius around the object
        velocity (np.ndarray): Initial velocity vector
        trajectory (np.ndarray): History of positions
        dynamic (bool): Whether the object moves
        last_t (float): Last time step
    """
    
    def __init__(self, name, position, radius, G, mass, type, color, initial_velocity, dynamic, max_orbit, protected_zone):
        """Initialize a celestial object.
        
        Args:
            name (str): Name of the object
            position (np.ndarray): Initial position [x, y]
            radius (float): Radius of the object
            G (float): Gravitational constant
            mass (float): Mass of the object
            type (str): Type of object (e.g., "planet", "satellite")
            color (str): Color for visualization
            initial_velocity (np.ndarray): Initial velocity vector
            dynamic (bool): Whether the object moves
            max_orbit (float): Maximum orbit radius
        """
        self.name = name
        self.position = position
        self.x = position[0]
        self.y = position[1]
        self.radius = radius
        self.G = G
        self.mass = mass
        self.color = color
        self.protected_zone = protected_zone
        self.velocity = initial_velocity
        self.trajectory = self.position
        self.dynamic = dynamic
        self.last_t = 0
        self.max_orbit = max_orbit
        self.mu = G * self.mass

    def get_gravitational_acceleration(self, x, y):
        """Calculate gravitational acceleration at point (x,y).
        
        Args:
            x (float): x-coordinate
            y (float): y-coordinate
            
        Returns:
            np.ndarray: Acceleration vector [ax, ay]
        """
        r = np.linalg.norm(self.position - np.array([x, y]))
        a = self.mass * (self.position - np.array([x, y])) / r**3
        return a
    
    def propagate_position(self, a, cur_t):
        """Propagate object position using RK4 integration.
        
        Args:
            a (np.ndarray): Acceleration vector
            cur_t (float): Current time
        """
        if self.dynamic:
            delta_t = cur_t - self.last_t
            
            # RK4 integration
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
        """Calculate distance from object to point (x,y).
        
        Args:
            x (float): x-coordinate
            y (float): y-coordinate
            
        Returns:
            float: Distance to point
        """
        return np.linalg.norm(self.position - np.array([x, y]))
    
    def get_x(self):
        """Get x-coordinate.
        
        Returns:
            float: x-coordinate
        """
        return self.x
    
    def get_y(self):
        """Get y-coordinate.
        
        Returns:
            float: y-coordinate
        """
        return self.y
    
    def get_radius(self):
        """Get object radius.
        
        Returns:
            float: Radius
        """
        return self.radius
    
    def set_protected_zone(self, type):
        """Set protected zone radius based on object type.
        
        Args:
            type (str): Type of object ("planet" or "satellite")
            
        Raises:
            ValueError: If invalid type is provided
        """
        if type == "planet":
            self.protected_zone = 2 * self.radius  # 2× body radius
        elif type == "satellite":
            self.protected_zone = 3 * self.radius  # 3× for small objects
        else:
            raise ValueError("Invalid protected zone type")

