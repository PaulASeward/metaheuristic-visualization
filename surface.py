import numpy as np
import math
import plotly.graph_objs as go


class Surface:
    def __init__(self, eval_function):
        self.eval_function = eval_function

        self.X, self.Y, self.points = self.generate_points()
        self.Z = self.eval_function(self.points).reshape(self.X.shape)

        # Calculate the z range, and shift the scale to positive values starting from 1
        self.z_min, self.z_max = np.min(self.Z), np.max(self.Z)
        self.z_range = self.z_max - self.z_min
        self.shift = int(abs(min(self.z_min, 0)) if self.z_min <= 0 else - abs(max(self.z_min, 0)))  # Shift the scale to positive values starting from 1 so we can use non-linear scaling
        self.marks = self.generate_nonlinear_marks()
        self.z_visible_max = None
        self.update_z_visible_max(int(self.marks[200]))

        self.fig = self.generate_surface()

    def update_z_visible_max(self, z_visible_max, linear=True):
        if linear:
            z_visible_max = self.linear_to_nonlinear_value(z_visible_max)
        self.z_visible_max = z_visible_max

    def generate_points(self):
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.ravel(), Y.ravel()], axis=-1)

        return X, Y, points

    def generate_nonlinear_marks(self):
        nonlinear_marks = {}
        exponential_ceiling = math.ceil(math.log10(self.z_range))
        linear_steps = list(
            range(0, (exponential_ceiling + 1) * 100, 100))  # Linear steps for the slider ex) [0,100,200,300,400]

        for i, val in enumerate(linear_steps):
            nonlinear_label = (10 ** i) - self.shift
            nonlinear_marks[val] = f'{nonlinear_label}'

        return nonlinear_marks

    def linear_to_nonlinear_value(self, linear_value):
        # Convert linear value to the original scale
        exponent = int(linear_value) / 100
        nonlinear_value = (10 ** exponent) - self.shift
        return int(nonlinear_value)

    def update_or_add_trace(self, fig, name, x, y, z, mode, size, color, opacity=1.0, showlegend=True):
        trace_found = False
        # Check if trace already exists
        for trace in fig.data:
            if trace.name == name:
                trace_found = True
                # Update the trace data
                trace.x = x
                trace.y = y
                trace.z = z
                # Update other properties as needed
                trace.marker.size = size
                trace.marker.color = color
                trace.marker.opacity = opacity
                trace.showlegend = showlegend
                break
        if not trace_found:
            # Add a new trace if not found
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode=mode,
                marker=dict(size=size, color=color, opacity=opacity),
                name=name,
                showlegend=showlegend
            ))

        return fig

    def clear_traces(self):
        traces_list = list(self.fig.data)

        # Collect indices of traces that are not Surface traces
        indices_to_remove = [i for i, trace in enumerate(traces_list) if not isinstance(trace, go.Surface)]

        # Remove the collected indices from the list in reverse order to avoid shifting problems
        for index in reversed(indices_to_remove):
            del traces_list[index]

        # Reassign the modified list of traces back to the figure data
        self.fig.data = tuple(traces_list)

        return self.fig

    def plot_particles(self, fig, selected_particles, timestep, positions, valuations, swarm_best_positions, velocities, min_explored, dark_colors, light_colors, show_particle_bests=True, show_velocity_trails=True, show_prev_position=True):
        for i in selected_particles:   # Add trace for each particle
            current_positions = positions[timestep, i, :]

            fig = self.update_or_add_trace(fig, f'Particle {i + 1}', [current_positions[0]], [current_positions[1]],[valuations[timestep, i]], 'markers', 3, dark_colors[i % len(dark_colors)])
            if show_particle_bests:
                fig = self.plot_particle_bests(fig, timestep, i, swarm_best_positions, light_colors)

            if show_prev_position and timestep > 0:
                prev_positions = positions[timestep - 1, i, :]
                fig = self.update_or_add_trace(fig, f'Particle {i + 1} Prev', [prev_positions[0]], [prev_positions[1]], [valuations[timestep - 1, i]], 'markers', 3, light_colors[i % len(light_colors)], 0.50, showlegend=True)

            if show_velocity_trails:
                fig = self.plot_velocity_mult_trails(fig, i, timestep, positions, velocities, valuations, dark_colors)

        # Add Previous Current Minimum Explored
        fig = self.plot_current_swarm_best(fig, timestep, min_explored)

        self.fig = fig

        return fig

    def plot_particle_bests(self, fig, timestep, particle_idx, swarm_best_positions, light_colors, size=5):
        if timestep >= 1:
            particles_best_position = swarm_best_positions[timestep - 1, particle_idx, :]

            fig = self.update_or_add_trace(fig, f'Particle {particle_idx + 1} Best', [particles_best_position[0]], [particles_best_position[1]], self.eval_function([particles_best_position]), 'markers', size, light_colors[particle_idx % len(light_colors)])

        return fig

    def plot_velocity_mult_trails(self, fig, particle_idx, timestep, positions, velocities, valuations, colors):
        trail_length = 5  # Number of trail points to create
        decay_factor = 0.2  # Factor by which the trail decays

        if timestep > 1:
            particle_position = positions[timestep, particle_idx, :]
            particle_velocity = velocities[timestep - 1, particle_idx, :]
            for j in range(trail_length):
                factor = (trail_length - j) / trail_length
                trail_pos = particle_position - particle_velocity * (1 - factor)
                fig = self.update_or_add_trace(fig, f'Particle {particle_idx + 1} Trail {j}', [trail_pos[0]], [trail_pos[1]], [valuations[timestep, particle_idx]], 'markers', 3 * factor, colors[particle_idx % len(colors)], opacity=max(0.1, 1 - decay_factor * j), showlegend=False)

        return fig

    def plot_velocity_single_trail(self, fig, particle_idx, timestep, positions, velocities, valuations, colors):
        if timestep > 1:
            trail_length = 5  # Number of trail points to create
            decay_factor = 0.2  # Factor by which the trail decays

            trail_points_x = []
            trail_points_y = []
            trail_points_z = []
            trail_sizes = []
            trail_opacities = []

            for j in range(trail_length):
                factor = (trail_length - j) / trail_length
                trail_pos = positions[timestep, particle_idx, :] - velocities[timestep-1, particle_idx, :] * (1 - factor)
                trail_points_x.append(trail_pos[0])
                trail_points_y.append(trail_pos[1])
                trail_points_z.append(valuations[timestep, particle_idx])
                trail_sizes.append(3 * factor)
                trail_opacities.append(max(0.1, 1 - decay_factor * j))

            fig = self.update_or_add_trace(fig, f'Particle {particle_idx + 1} Trail', trail_points_x, trail_points_y, trail_points_z, 'markers', trail_sizes, colors[particle_idx % len(colors)], 0.50, showlegend=True)

        return fig

    def plot_current_swarm_best(self, fig, timestep, min_explored, color='black', size=5):
        if timestep >= 1:
            min_explored_t_x1, min_explored_t_x2, min_explored_t_z = min_explored[timestep - 1]

            fig = self.update_or_add_trace(fig, f'Current Swarm Min', [min_explored_t_x1], [min_explored_t_x2], [min_explored_t_z], 'markers', size, color)
        return fig

    def get_surface(self):
        return self.fig

    def generate_surface(self):
        fig = go.Figure()

        fig.add_trace(go.Surface(z=self.Z, x=self.X, y=self.Y, colorscale='Viridis', opacity=0.7, cmin=self.z_min, cmax=self.z_visible_max))

        fig = self.update_layout(fig)

        self.fig = fig

        return fig

    def update_layout(self, fig):
        fig.update_layout(
            legend=dict(
                x=0,
                y=1,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
                bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=2,
                uirevision='constant'
            ),
            scene=dict(
                xaxis=dict(nticks=4, range=[-100, 100]),
                yaxis=dict(nticks=4, range=[-100, 100]),
                zaxis=dict(nticks=4, range=[self.z_min, self.z_visible_max]),
                uirevision='constant'
            ),
            autosize=False,
            width=1200,
            height=800
        )
        return fig


