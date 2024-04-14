# Data Visualization of Metaheuristic Algorithms: PSO

This project is a Dash web application designed to visualize the behavior of the algorithm, Particle Swarm Optimization (PSO). 

The PSO algorithm typically goes as follows:
1. Initialize a swarm of particles with random positions and velocities.
2. Evaluate the fitness of each particle.
3. Update the particle's velocity and position based on its best position and the best position of the swarm.
4. Repeat steps 2 and 3 until a stopping criterion is met.

How the swarm behaves in the search space is crucial to understanding the algorithm's performance. This application allows users to visualize the swarm's behavior in the search space of 28 benchmark functions. The user can generate swarm data for each function and inspect how the swarm explores and exploits the search space.

### PSO Implementation Details

This implementation uses a Swarm of 10 particles in 2 dimensions, following a budget of 20000 Function Evaluations:
- Function Evaluation Budget: 2 dimensions x 10 000/dim = 20, 000 Function Evaluations
- Observations Per Episode: 10 Number of Swarm Observation Intervals per Episode x 10 Number of Observations in each Interval  = 100  Observations per Episode
- Function Evaluation Allocation: 10 Swarm Size (# Particles) x 20 Episodes x 100 Observations per Episode  = 20000 Function Evaluations

The evaluating function maps these 2-D points into the z-axis and 3rd Dimension. Surfaces are generated for each function with limits of [100, 100] for each dimension. These benchmark functions have varying characteristics with more details available in environment/functions.py

## Features

- Ability to quickly load all 28 function surfaces of the search space.
- Ability to generate swarm data and visualize for each function. Some data may exist for limited functions. However, the generate button will generate data the selected function with hyperparameters instantly as well.
- Interactive visualization tool to inspect how the swarm manages efficient exploration and exploitation of the search space.
- Non-Linear Scaling: While exploring the search space, it is easy to gain an overview of the swarm's behavior, however, as the swarm converges, the visualization may become less informative. To counter this, the visualization tool uses a non-linear scaling method to zoom in on the swarm as it converges and explores more detailed nuances of the function topology required for effective exploitation.

## Installation

Ensure you have Python 3.6 or newer installed. It's recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
# deactivate to exit the virtual environment
```

Install the required dependencies:

```bash 
pip install -r requirements.txt
```

If this virtual environment is already created and requirements installed into it, we can just activate to use it:

```bash
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

## Running the Application

To start the Dash server and run the application, execute:
    
```bash
python3 app.py
```

The application will be available at http://127.0.0.1:8050/


## Project Structure
    app.py: The main Dash application.
    color_utils.py : Methods used to generate colors for the swarm particles.
    data: Data of functions already loaded, and where the swarm data is saved to.
    environment: Configuration of the benchmark functions and the swarm.
    pso_simple_swarm.py: Simplified implementation of the Particle Swarm Optimization algorithm.
    surface.py: Methods used to generate and evaluated the surface of the functions.
    swarm_simulator.py: Methods used to load/simulate the swarm behavior.
