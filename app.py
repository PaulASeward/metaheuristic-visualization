import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from swarm_simulator import SwarmSimulator

# Load data
swarm = SwarmSimulator(fun_num=6)

app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='function-selector',
            options=swarm.get_available_functions(),
            clearable=False,
            placeholder="Select Function"
        ),
    ], style={'margin-bottom': '20px'}),
    html.Div([
        dcc.Dropdown(
            id='step-selector',
            options=[],
            clearable=False,
            placeholder="Select Step in Experiment"
        ),
    ], style={'margin-bottom': '20px'}),
    html.Div([
        dcc.Dropdown(
            id='episode-selector',
            options=[],
            clearable=False,
            placeholder="Select Episode"
        ),
    ], style={'margin-bottom': '20px'}),
    html.Div([
        html.Button('Generate Simulated Data', id='btn-generate', n_clicks=0),
        html.Div([
            html.Label('Inertia Weight:'),
            dcc.Input(
                id='inertia-weight-input',
                type='number',
                min=0.1,
                max=1.0,
                step=0.01,
                value=0.729844,  # default value from your PSO implementation
                style={'margin-right': '20px'}
            ),
            html.Label('Gravity to Individual Best (Cognitive):'),
            dcc.Input(
                id='cognitive-component-input',
                type='number',
                min=0,
                max=4,
                step=0.1,
                value=1.491038,  # default value from your PSO implementation
                style={'margin-right': '20px'}
            ),
            html.Label('Gravity to Swarm Best (Social):'),
            dcc.Input(
                id='social-component-input',
                type='number',
                min=0,
                max=4,
                step=0.1,
                value=1.491038,  # default value from your PSO implementation
                style={'margin-right': '20px'}
            ),
            html.Label('Velocity Range:'),
            dcc.Input(
                id='rangeF-input',
                type='number',
                min=10,
                max=100,
                step=1,
                value=100,  # default value from your PSO implementation
                style={'margin-right': '20px'}
            ),
            html.Label('Replacement Threshold:'),
            dcc.Input(
                id='threshold-input',
                type='number',
                min=0,
                max=1,
                step=0.1,
                value=1.0,  # default value from your PSO implementation
            )
        ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'}),
    ], style={'margin-bottom': '20px'}),

    html.Div(id='episode-metadata-display', style={'margin-top': '20px'}),
    html.Div([
        html.Div([
            dcc.Graph(id='3d-swarm-visualization'),
        ], style={'width': '85%', 'display': 'inline-block'}),
        html.Div([
            html.Label('Max Z Value:', style={'margin-right': '10px'}),
            dcc.Slider(
                id='z-max-slider',
                min=0,
                max=500,
                step=1,
                value=int(swarm.surface.marks[200]),
                marks=swarm.surface.marks,
                vertical=True,
                verticalHeight=400
            ),
        ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            html.Label('Display a Particle\'s Best Positions:', style={'margin-right': '10px'}),
            dcc.Dropdown(
                id='particle-best-position',
                options=[
                    {'label': 'Yes', 'value': True},
                    {'label': 'No', 'value': False},
                ],
                value=False,
                clearable=False,
                placeholder="Display a Particle's Best Positions"
            ),
            html.Label('Display a Particle\'s Velocity Trail:', style={'margin-right': '10px', 'margin-top': '10px'}),
            dcc.Dropdown(
                id='particle-trail',
                options=[
                    {'label': 'Yes', 'value': True},
                    {'label': 'No', 'value': False},
                ],
                value=False,
                clearable=False,
                placeholder="Display a Particle's Velocity Trail"
            ),
            html.Label('Display a Particle\'s Last Position:', style={'margin-right': '10px', 'margin-top': '10px'}),
            dcc.Dropdown(
                id='previous-position',
                options=[
                    {'label': 'Yes', 'value': True},
                    {'label': 'No', 'value': False},
                ],
                value=False,
                clearable=False,
                placeholder="Display a Particle's Previous Position"
            ),
        ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'margin-bottom': '20px'}),
        html.Div([
            html.Label('Particle Selection (Can also select/deselect by clicking on the legend):', style={'margin-right': '10px', 'margin-left': '10px'}),
            dcc.Dropdown(
                id='particle-selector',
                options=[{'label': f'Particle {i + 1}', 'value': i} for i in range(swarm.ep_positions.shape[1])],
                placeholder="Focus on Single Particle Behavior",
                multi=True,
                value=[i for i in range(swarm.ep_positions.shape[1])]
            ),
        ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top', 'margin-bottom': '20px'}),
    ], style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            # Timestep Slider
            html.Label('Adjust Timestep:', style={'margin-bottom': '10px'}),
            dcc.Slider(
                id='timestep-slider',
                min=0,
                max=len(swarm.ep_positions) - 1,
                value=0,
                marks={i: str(i) for i in range(0, len(swarm.ep_positions), 10)},
                step=1,
            ),
        ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Playback Controls
        html.Div([
            html.Div([
                html.Button('Previous', id='btn-previous', n_clicks=0),
                html.Button('Next', id='btn-next', n_clicks=0),
                html.Button('Play', id='btn-play', n_clicks=0),
                html.Button('Stop', id='btn-stop', n_clicks=0),
            ], style={'width': '50%', 'display': 'flex', 'margin-bottom': '10px', 'justify-content': 'space-around'}),
            html.Div([
                dcc.Dropdown(
                    id='speed-selector',
                    options=[
                        {'label': '1x Speed', 'value': 500},
                        {'label': '1.25x Speed', 'value': 400},
                        {'label': '1.5x Speed', 'value': 333},
                        {'label': '1.75x Speed', 'value': 285},
                        {'label': '2x Speed', 'value': 250},
                        {'label': '5x Speed', 'value': 100},
                        {'label': '10x Speed', 'value': 50}
                    ],
                    value=500,  # Default to 1x speed
                    clearable=False,
                    placeholder="Select Playback Speed"
                )
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex'}),

    # Auto-stepper Interval
    dcc.Interval(
        id='auto-stepper',
        interval=500,  # in milliseconds
        n_intervals=0,
        disabled=True  # Start disabled
    ),
])


@app.callback(
    [
        Output('step-selector', 'options'),
        Output('episode-selector', 'options'),
        Output('z-max-slider', 'marks',),
        Output('z-max-slider', 'value'),
        Output('episode-metadata-display', 'children'),
    ],
    [
        Input('function-selector', 'value'),
        Input('step-selector', 'value'),
        Input('episode-selector', 'value'),
        Input('btn-generate', 'n_clicks')
    ],
    [
        State('function-selector', 'value'),
        State('step-selector', 'value'),
        State('inertia-weight-input', 'value'),
        State('cognitive-component-input', 'value'),
        State('social-component-input', 'value'),
        State('rangeF-input', 'value'),
        State('threshold-input', 'value')
    ]
)
def update_swarm(fun_num_input, step_input, ep_input, btn_n_clicks, current_function, current_step, w, c1, c2, rangeF, threshold):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'btn-generate':
        swarm.generate_simulated_swarm_data(c1=c1, c2=c2, w=w, rangeF=rangeF, threshold=threshold)
        return swarm.get_available_steps(), swarm.get_available_episodes(0), dash.no_update, dash.no_update, f"Simulated data has been generated for Function {current_function}."

    if triggered_id == 'function-selector':
        swarm.set_function_number(fun_num_input)
        marks = swarm.surface.marks
        swarm.surface.clear_traces()
        swarm.surface.generate_surface()
        return swarm.get_available_steps(), [], marks, int(marks[200]), dash.no_update

    if triggered_id == 'step-selector':
        swarm.load_swarm_data_for_step(step=step_input)
        return dash.no_update, swarm.get_available_episodes(step_input), dash.no_update, dash.no_update, dash.no_update

    if triggered_id == 'episode-selector':
        swarm.load_swarm_data_for_episode(episode=ep_input)

        if ep_input is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Select an episode to view metadata."

        metadata = swarm.meta_data
        episode_row = metadata[ep_input]

        if metadata is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "No metadata is available for the selected episode."

        # Create a HTML table with metadata
        table_header = html.Thead(html.Tr([html.Th(col) for col in metadata.dtype.names]))
        table_body = html.Tbody([html.Tr([html.Td(episode_row[col]) for col in metadata.dtype.names])])
        table = html.Table([table_header, table_body], style={'width': '100%'})

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, table


@app.callback(
    Output('3d-swarm-visualization', 'figure'),
    [Input('timestep-slider', 'value'),
     Input('z-max-slider', 'value'),
     Input('particle-best-position', 'value'),
     Input('particle-trail', 'value'),
     Input('previous-position', 'value'),
     Input('particle-selector', 'value')],
    [State('particle-best-position', 'value'),
     State('particle-trail', 'value'),
     State('previous-position', 'value'),
     State('particle-selector', 'value')],
)
def update_figure(selected_timestep, slider_value, display_pbest_input, display_trail_input, display_prev_posn_input, selected_particles_input, display_best_positions, display_trail, display_prev_posn, selected_particles):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    display_best_positions = display_pbest_input if display_pbest_input is not None else display_best_positions
    display_trail = display_trail_input if display_trail_input is not None else display_trail
    display_prev_posn = display_prev_posn_input if display_prev_posn_input is not None else display_prev_posn
    selected_particles = selected_particles_input if selected_particles_input is not None else selected_particles

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'timestep-slider':
        fig = swarm.surface.get_surface()
    else:
        swarm.surface.update_z_visible_max(slider_value)
        swarm.surface.clear_traces()
        fig = swarm.surface.generate_surface()

    fig = swarm.surface.plot_particles(fig, selected_particles, selected_timestep, swarm.ep_positions, swarm.ep_valuations, swarm.ep_swarm_best_positions, swarm.ep_velocities, swarm.min_explored, swarm.dark_colors, swarm.light_colors, display_best_positions, display_trail, display_prev_posn)

    return fig


@app.callback(
    Output('timestep-slider', 'value'),
    [Input('btn-previous', 'n_clicks'),
     Input('btn-next', 'n_clicks'),
     Input('btn-play', 'n_clicks'),
     Input('btn-stop', 'n_clicks'),
     Input('auto-stepper', 'n_intervals')],
    [State('timestep-slider', 'value')]
)
def update_slider(btn_previous, btn_next, btn_play, btn_stop, n_intervals, current_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-previous' and current_value > 0:
        return current_value - 1
    elif button_id == 'btn-next' and current_value < len(swarm.ep_positions) - 1:
        return current_value + 1
    elif button_id == 'auto-stepper':
        if current_value < len(swarm.ep_positions) - 1:
            return current_value + 1
        else:
            return 0  # Loop back to start, adjust as needed
    # Play and Stop buttons don't directly update the slider, so we ignore them here

    return dash.no_update

@app.callback(
    Output('auto-stepper', 'disabled'),
    [Input('btn-play', 'n_clicks'),
     Input('btn-stop', 'n_clicks')],
    [State('auto-stepper', 'disabled')]
)
def control_auto_stepper(btn_play, btn_stop, is_disabled):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_disabled

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'btn-play':
        return False
    elif button_id == 'btn-stop':
        return True

    return is_disabled


@app.callback(
    Output('auto-stepper', 'interval'),
    [Input('speed-selector', 'value')]
)
def update_interval(speed):
    return speed

if __name__ == '__main__':
    app.run_server(debug=True)
