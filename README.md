# Data Visualization of Metaheuristics

This project is a Dash web application designed to visualize the behavior of the algorithm Particle Swarm Optimization. 
## Features

- Data, From pre-loaded training experiments with actions and descriptions.
- Ability to view the topology of all 28 functions.
- Ability to generate swarm data and visualize for each function.
- Interactive visualization tool to inspect how the swarm manages efficient exploration and exploitation of the search space.

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
    data: Data of functions already loaded, and where the swarm data is saved to.
    environment: Configuration of the benchmark functions and the swarm.
    pso_simple_swarm.py: Simplified implementation of the Particle Swarm Optimization algorithm.
    surface.py: Methods used to generate and evaluated the surface of the functions.
    swarm_simulator.py: Methods used to load/simulate the swarm behavior.
