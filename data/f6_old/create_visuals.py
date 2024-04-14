import numpy as np
from mpl_toolkits.mplot3d import Axes3D


import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib import animation
from matplotlib.animation import FFMpegWriter, FuncAnimation
import warnings
warnings.filterwarnings('ignore')
import environment.functions as functions

# Load your data
# There is typically 20 episodes with 100 timesteps, 10 particles, and 2 dimensions
# We will use the first episode for this example
positions = np.load('data/swarm_locations.npy')  # Shape is  (episodes, time_steps, particles, dimensions)
positions = positions[0]  # Shape is (time_steps, particles, dimensions)
valuations = np.load('data/swarm_evaluations.npy')  # Shape is (time_steps, particles)
valuations = valuations[0]  # Shape is (time_steps, particles)

# Function

obj_f = functions.CEC_functions(dim=2, fun_num=6)
def eval(X):
    """
    The input X is a 2D array of shape (n, dim) where n is the number of points to evaluate and dim is the number of dimensions.
    The output should be a 1D array of shape (n,) where each element is the value of the function at the corresponding point.
    """
    return obj_f.Y_matrix(np.array(X).astype(float))


def animate_episode(positions, valuations, animation_path='data/animation.mp4'):
    plt.close()
    # Create a meshgrid for the background
    x1 = np.linspace(-100, 100, 100)
    x2 = np.linspace(-100, 100, 100)
    X1, X2 = np.meshgrid(x1, x2)

    points = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    Z_flat = eval(points)
    Z = Z_flat.reshape(X1.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([-1000, 0])

    # Plot the surface
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

    # Function to update the plot for each frame
    def update(frameId, ax, positions, valuations):
        ax.clear()
        ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

        # Extract the positions for the current timestep
        current_positions = positions[frameId]
        x1s = current_positions[:, 0]
        x2s = current_positions[:, 1]

        # Extract the valuations for the current timestep
        current_valuations = valuations[frameId]
        zs = current_valuations

        ax.scatter(x1s, x2s, zs, color='r')  # Plot current particle positions

    # Creating the Animation object
    # frames = range(positions.shape[0])
    frames = len(range(positions.shape[0]))
    anim = animation.FuncAnimation(fig, update, frames=frames, fargs=(ax, positions, valuations), repeat=False)

    anim.save(animation_path, writer=FFMpegWriter(fps=10))
    plt.show()

animate_episode(positions, valuations)
