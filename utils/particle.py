from utils.map import Map


class Particle:
    """
    A class representing a particle that moves in a map according to given speed and bearing sequences.
    It also includes methods for Particle Swarm Optimization (PSO) to evaluate and update its position based on personal
    and global bests.
    """

    def __init__(self, init_pos, speed, bearing, work_map):
        """
        Initialize the Particle with its initial position, speed, and bearing.

        Args:
            init_pos (tuple): The initial position of the particle (latitude, longitude).
            speed (list): A list of speed values for each timestep.
            bearing (list): A list of bearing values for each timestep.
            work_map (Map): The map object used for evaluating fitness.
        """
        self.pos = init_pos
        self.pos_history = [init_pos]
        self.fitness_history = []
        self.speed = speed
        self.bearing = bearing
        self.work_map = work_map
        self.last_distance = 0.0
        self.last_speed = 0.0
        self.last_bearing = 0.0
        self.cum_distance = 0.0
        self.c1 = 0.1
        self.c2 = 0.1
        self.pbest = None
        self.last_pso_fitness = None
        self.best_pso_fitness = None

    def move(self, timestep=1):
        """
        Move the particle based on its speed and bearing for the given timestep.

        Args:
            timestep (int, optional): The time interval for the movement. Defaults to 1.
        """
        if self.speed is not None and len(self.speed) > 0 and self.bearing is not None and len(self.bearing) > 0:
            self.last_speed = self.speed.pop(0)
            self.last_bearing = self.bearing.pop(0)
            self.last_distance = self.last_speed * timestep
            self.cum_distance = self.cum_distance + self.last_distance
            self.pos = Map.destination_point(self.pos[0], self.pos[1], self.last_distance, self.last_bearing)

    def save_pos(self):
        """
        Save the current position and its fitness to the history.
        """
        self.pos_history.append(self.pos)
        self.fitness_history.append(self.work_map.get_policy_prob([self.pos])[0])

    def is_finished(self):
        """
        Check if the particle has finished its movement based on speed and bearing lists.

        Returns:
            bool: True if the particle has no more speed or bearing values, False otherwise.
        """
        return self.speed is None or len(self.speed) == 0 or self.bearing is None or len(self.bearing) == 0

    def reset_cumulative_distance(self):
        """
        Reset the cumulative distance traveled by the particle.
        """
        self.cum_distance = 0.0

    def get_cumulative_distance(self):
        """
        Get the cumulative distance traveled by the particle.

        Returns:
            float: The cumulative distance traveled.
        """
        return self.cum_distance

    def pso_start(self):
        """
        Initialize PSO parameters for the particle.
        """
        self.best_pso_fitness = -1
        self.pso_evaluate()
        self.last_speed = 0
        self.last_bearing = 0

    def pso_evaluate(self):
        """
        Evaluate the particle's current position and update personal best if necessary.
        """
        self.last_pso_fitness = self.work_map.get_policy_prob([self.pos])[0]
        if self.last_pso_fitness > self.best_pso_fitness:
            self.best_pso_fitness = self.last_pso_fitness
            self.pbest = self.pos

    def pso_move(self, gbest_pos, gbest_fitness, timestep=1):
        """
        Move the particle based on PSO rules considering personal and global bests.
        Args:
            gbest_pos (tuple): The global best position (latitude, longitude).
            gbest_fitness (float): The fitness value of the global best
            timestep (int, optional): The time interval for the movement. Defaults to 1.
        """
        new_pos = self.pos
        if self.last_speed > 0:
            distance = self.last_speed * timestep
            new_pos = Map.destination_point(self.pos[0], self.pos[1], distance, self.last_bearing)
        if self.pbest != self.pos:
            pbest_distance = Map.haversine(new_pos[0], new_pos[1], self.pbest[0], self.pbest[1])
            pbest_bearing = Map.get_bearing(new_pos[0], new_pos[1], self.pbest[0], self.pbest[1])
            new_pos = Map.destination_point(new_pos[0], new_pos[1], self.c1 * pbest_distance, pbest_bearing)
        if gbest_pos is not None and gbest_pos != self.pos and gbest_fitness > self.best_pso_fitness:
            gbest_distance = Map.haversine(new_pos[0], new_pos[1], gbest_pos[0], gbest_pos[1])
            gbest_bearing = Map.get_bearing(new_pos[0], new_pos[1], gbest_pos[0], gbest_pos[1])
            new_pos = Map.destination_point(new_pos[0], new_pos[1], self.c2 * gbest_distance, gbest_bearing)
        if new_pos != self.pos:
            self.last_speed = Map.haversine(self.pos[0], self.pos[1], new_pos[0], new_pos[1])
            self.last_bearing = Map.get_bearing(self.pos[0], self.pos[1], new_pos[0], new_pos[1])
            self.pos = new_pos

    def pso_set_pbest(self):
        """
        Set the particle's position to its personal PSO best position.
        """
        self.pos = self.pbest
