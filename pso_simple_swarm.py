import csv
import os
import numpy as np
from tqdm import tqdm
from environment.functions import CEC_functions


class PSOSwarm:

    def __init__(self, fun_num=6, w=0.729844, c1=2.05 * 0.729844, c2=2.05 * 0.729844, replacement_threshold=1.0, swarm_size=10, dim=2, rangeF=100):
        self.X = None  # Current Position of particles
        self.V = None  # Current Velocity of particles
        self.P = None  # Best Position of particles

        # PSO Parameters
        self.fun_num = fun_num
        self.w = w  # Inertia weight to prevent velocities becoming too large
        self.c1 = c1  # Social component Learning Factor
        self.c2 = c2  # Cognitive component Learning Factor
        self.pbest_replacement_threshold = replacement_threshold

        # Non-Adjustable PSO Parameters
        self.swarm_size = swarm_size
        self.dimension = dim
        self.rangeF = rangeF
        self.obs_per_episode = 100
        self.episodes = 20
        self.current_eps = 0

        # Tracked Locations
        self.tracked_locations = np.zeros((self.episodes, self.obs_per_episode, self.swarm_size, self.dimension))
        self.tracked_velocities = np.zeros((self.episodes, self.obs_per_episode, self.swarm_size, self.dimension))
        self.tracked_best_locations = np.zeros((self.episodes, self.obs_per_episode, self.swarm_size, self.dimension))
        self.tracked_valuations = np.zeros((self.episodes, self.obs_per_episode, self.swarm_size))

        # Set Constraints for clamping position and limiting velocity
        self.Vmin, self.Vmax = -1 * self.rangeF, self.rangeF
        self.Xmin, self.Xmax = -1 * 100, 100

        self.function = None
        self.val = None
        self.gbest_val = None
        self.gbest_pos = None
        self.pbest_val = None

        # Initialize the swarm's positions velocities and best solutions
        self._initialize()

    def reinitialize(self):
        self._initialize()

    def _initialize(self):
        # Initialize 3 matrices for Current Position, Velocity, and Position of particles' best solution
        self.X = np.random.uniform(low=self.Xmin, high=self.Xmax, size=(self.swarm_size, self.dimension))
        self.V = np.full((self.swarm_size, self.dimension), 0)
        self.P = self.X

        self.function = CEC_functions(dim=self.dimension, fun_num=self.fun_num)

        self.update_swarm_valuations_and_bests()

    def update_swarm_valuations_and_bests(self):
        self.val = self.eval(self.P)  # Vector of particle's current value of its position based on Function Eval
        self.pbest_val = self.val  # Vector of particle's best value of a visited position.
        self.gbest_pos = self.P[np.argmin(self.pbest_val)]  # Vector of globally best visited position.
        self.gbest_val = np.min(self.pbest_val)  # Single value of the global best position

    def get_tracked_locations_and_valuations(self):
        return self.tracked_locations, self.tracked_velocities, self.tracked_best_locations, self.tracked_valuations

    def save_tracked_data(self):
        path = f'data/f{self.fun_num}/locations_at_step_0/'
        if not os.path.exists(path):
            os.makedirs(path)

        np.save(path + 'swarm_locations.npy', self.tracked_locations)
        np.save(path + 'swarm_velocities.npy', self.tracked_velocities)
        np.save(path + 'swarm_best_locations.npy', self.tracked_best_locations)
        np.save(path + 'swarm_evaluations.npy', self.tracked_valuations)
        threshold, c1, c2, w, rangeF = self.get_meta_data()
        meta_data = []
        for _ in range(self.episodes):
            meta_data.append([0, 'Simulated Data', threshold, c1, c2, w, rangeF])
        meta_data_headers = ['Action', 'Action Name', 'Replacement Threshold', 'Global Best Gravity', 'Individual Best Gravity', 'Inertia Weight', 'Velocity Range']

        with open(path + 'meta_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(meta_data_headers)
            writer.writerows(meta_data)

    def get_current_best_fitness(self):
        return self.gbest_val

    def get_meta_data(self):
        return self.pbest_replacement_threshold, self.c1, self.c2, self.w, self.rangeF

    def eval(self, X):
        return self.function.Y_matrix(np.array(X).astype(float))

    def update_velocities(self, leader):
        # Generate social and cognitive components of velocity matrix update using np.random
        social = self.c1 * np.random.uniform(size=(self.swarm_size, self.dimension)) * (leader - self.X)
        cognitive = self.c2 * np.random.uniform(size=(self.swarm_size, self.dimension)) * (self.P - self.X)

        # Update new velocity with old velocity*inertia plus component matrices
        self.V = self.w * self.V + social + cognitive
        self.V = np.clip(self.V, self.Vmin, self.Vmax)

    def update_position(self):
        # Clamp position inside boundary and reflect them in case they are out of the boundary based on:
        # S. Helwig, J. Branke, and S. Mostaghim, "Experimental Analysis of Bound Handling Techniques in Particle Swarm Optimization," IEEE TEC: 17(2), 2013, pp. 259-271

        self.X = self.X + self.V # Add velocity to current position

        # Identify positions outside the boundaries, and reflect the out-of-bounds positions
        out_of_bounds = np.logical_or(self.X < self.Xmin, self.X > self.Xmax)
        self.X = np.where(out_of_bounds, np.sign(self.X) * 2 * self.Xmax - self.X, self.X)
        self.V = np.where(out_of_bounds, 0, self.V)  # Out-of-bounds velocities are set to 0

        # Use function evaluation for each particle (vector) in Swarm to provide value for each position in X.
        self.val = self.eval(self.X)

    def update_gbest(self):
        self.gbest_pos = self.P[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)

    def update_pbest_with_random(self):
        pbest_change = (self.val - self.pbest_val) / np.abs(self.pbest_val)
        improved_particles = pbest_change < 0

        # Allow exploitative search, per standard for better solutions
        self.P = np.where(improved_particles[:, np.newaxis], self.X, self.P)
        self.pbest_val = np.where(improved_particles, self.val, self.pbest_val)

        # Promote Exploratory Search allowing for non-elitist selection
        non_elitist_improvements = np.logical_and(pbest_change >= 0, np.random.uniform(size=self.swarm_size) > self.pbest_replacement_threshold + pbest_change)
        self.P = np.where(non_elitist_improvements[:, np.newaxis], self.X, self.P)
        self.pbest_val = np.where(non_elitist_improvements, self.val, self.pbest_val)

    def update_pbest(self):
        improved_particles = self.pbest_replacement_threshold * self.val < self.pbest_val
        self.P = np.where(improved_particles[:, np.newaxis], self.X, self.P)
        self.pbest_val = np.where(improved_particles, self.val, self.pbest_val)

    def optimize(self):
        for j in tqdm(range(self.episodes)):
            for i in range(self.obs_per_episode):
                self.update_velocities(self.gbest_pos)  # Input global leader particle position
                self.update_position()
                self.update_pbest()
                self.update_gbest()

                self.tracked_locations[j][i] = self.X
                self.tracked_velocities[j][i] = self.V
                self.tracked_best_locations[j][i] = self.P
                self.tracked_valuations[j][i] = self.val



