import environment.functions as functions
import numpy as np
import os
from surface import Surface
from color_utils import get_distinct_colors, lighten_color
from pso_simple_swarm import PSOSwarm


class SwarmSimulator:
    def __init__(self, fun_num, dim=2):
        self.fun_num = fun_num
        self.dim = dim
        self.obj_f = None
        self.data_dir = None
        self.step_data_dir = None
        self.step = None
        self.ep = None

        self.positions = None
        self.ep_positions = None
        self.velocities = None
        self.ep_velocities = None
        self.swarm_best_positions = None
        self.ep_swarm_best_positions = None
        self.valuations = None
        self.ep_valuations = None
        self.meta_data = None
        self.ep_meta_data = None

        self.num_particles = None
        self.dark_colors = None
        self.light_colors = None

        self.min_explored = None
        self.surface = None

        self.set_function_number(fun_num)
        self.load_swarm_data_for_step(step=250)
        self.load_swarm_data_for_episode(episode=0)

    def set_function_number(self, fun_num):
        self.fun_num = fun_num
        self.data_dir = f'data/f{fun_num}/'
        self.obj_f = functions.CEC_functions(dim=self.dim, fun_num=fun_num)
        self.surface = Surface(eval_function=self.eval)

    def get_available_functions(self):
        # Mark the functions that have data available
        available_functions = []
        for i in range(1, 29):
            if os.path.exists(f'data/f{i}/'):
                available_functions.append({'label': f'Function {i} (Data Available)', 'value': i})
            else:
                available_functions.append({'label': f'Function {i}', 'value': i})

        return available_functions

    def get_available_steps(self):
        if not os.path.exists(self.data_dir):
            print(f'No data exists for Function {self.fun_num}.')
            return []

        # List only directories
        data_directories = os.listdir(self.data_dir)
        data_directories = [directory for directory in data_directories if os.path.isdir(self.data_dir + directory)]

        steps = sorted([int(directory.split('_')[-1]) for directory in data_directories])
        labelled_steps = []
        for i in steps:
            if i != 0:
                labelled_steps.append({'label': f'Step {i}', 'value': i})
            else:
                labelled_steps.append({'label': f'Step {i} (Simulated Data)', 'value': i})

        return labelled_steps

    def get_available_episodes(self, step):
        if not os.path.exists(self.step_data_dir):
            print(f'No data path exists to function {self.fun_num}.')
            return []

        if self.ep_positions is None:
            print(f'No data exists for function {self.fun_num}.')
            return []

        # return [{'label': f'Episode {i + 1}', 'value': i} for i in range(self.positions.shape[0])]
        metadata = self.meta_data
        options = []
        for i in range(self.positions.shape[0]):
            label = 'Episode {} - Action: {}'.format(i + 1, metadata[i]['Action_Name'])
            options.append({'label': label, 'value': i})

        return options

    def eval(self, X):
        return self.obj_f.Y_matrix(np.array(X).astype(float))

    def generate_simulated_swarm_data(self, c1, c2, w, rangeF, threshold):
        pso_simple_swarm = PSOSwarm(fun_num=self.fun_num, dim=self.dim, replacement_threshold=threshold, c1=c1, c2=c2, w=w, rangeF=rangeF)
        pso_simple_swarm.optimize()
        pso_simple_swarm.save_tracked_data()
        return

    def load_swarm_data_for_step(self, step=250):
        # There is typically 20 episodes with 100 timesteps, 10 particles, and 2 dimensions
        self.step_data_dir = self.data_dir + f'locations_at_step_{step}/'
        self.step = step

        # Verify that the data exists for the given step
        if not os.path.exists(self.step_data_dir):
            raise ValueError(f'No data exists for step {step}.')

        self.positions = np.load(self.step_data_dir + 'swarm_locations.npy')  # Shape is  (episodes, time_steps, particles, dimensions)
        self.velocities = np.load(self.step_data_dir + 'swarm_velocities.npy')  # Shape is (time_steps, particles, dimensions)
        self.swarm_best_positions = np.load(self.step_data_dir + 'swarm_best_locations.npy')  # Shape is (time_steps, particles, dim)
        self.valuations = np.load(self.step_data_dir + 'swarm_evaluations.npy')  # Shape is (time_steps, particles)
        self.meta_data = np.genfromtxt(self.step_data_dir + 'meta_data.csv', delimiter=',', dtype=None, names=True, encoding='utf-8')

        return self.positions, self.velocities, self.swarm_best_positions, self.valuations, self.meta_data

    def load_swarm_data_for_episode(self, episode=0):
        self.ep = episode
        # Verify that enough data exists for the requested episode
        if episode >= len(self.positions):
            raise ValueError(f'No data exists for episode {episode}.')
        self.ep_positions = self.positions[episode]  # Shape is (time_steps, particles, dimensions)
        self.ep_velocities = self.velocities[episode]  # Shape is (time_steps, particles, dimensions)
        self.ep_swarm_best_positions = self.swarm_best_positions[episode]  # Shape is (time_steps, particles, dimensions)
        self.ep_valuations = self.valuations[episode]  # Shape is (time_steps, particles)
        self.ep_meta_data = self.meta_data[episode]  # Meta data for the first episode

        self.load_particle_data_for_episode()

        return self.ep_positions, self.ep_velocities, self.ep_swarm_best_positions, self.ep_valuations, self.ep_meta_data

    def load_particle_data_for_episode(self):
        self.calculate_current_min_explored()
        self.num_particles = self.ep_positions.shape[1]

        self.dark_colors = get_distinct_colors(self.num_particles)
        self.light_colors = [lighten_color(color, amount=0.5) for color in self.dark_colors]

    def calculate_current_min_explored(self):
        # Calculate the minimum value explored by the swarm and store the position and value at each time step.
        X1_min = None
        X2_min = None
        min_valuation = None

        min_explored = []

        for t in range(len(self.ep_valuations)):
            current_positions = self.ep_positions[t]
            current_valuations = self.ep_valuations[t]
            min_valuation_t = np.min(current_valuations)

            if min_valuation is None or min_valuation_t < min_valuation:
                min_valuation = min_valuation_t
                min_valuation_index = np.argmin(current_valuations)
                X1_min = current_positions[min_valuation_index, 0]
                X2_min = current_positions[min_valuation_index, 1]

            min_explored.append([X1_min, X2_min, min_valuation])

        self.min_explored = min_explored
        return min_explored
