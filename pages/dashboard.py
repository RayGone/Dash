import dash
from plotly import express as px
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from utilities import difficulty, difficulty_color_map, graph_config, DropDown, \
    task_priority, task_priority_color_map, filterByColumn, isDebug, topContracts
    
import pandas as pd
import time

map_styles = ['open-street-map', 'carto-darkmatter']
states = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']

dash.register_page(__name__, path='/', name='Dashboard', title='Dashboard App', location='sidebar')

layout = dbc.Row(
    dbc.Col([
        dbc.Row([
            dbc.Col(html.Div(dbc.Button("Refresh", id='refresh', class_name="ms-3 btn btn-sm rounded-0", size='sm'), className='w-100 d-inline-flex flex-row-reverse'), sm=12, align='center', class_name='d-print-none'),
            dbc.Col([dbc.Label("Task Priority: ",class_name="me-2 font-weight-light")], xs=3, md=2, lg=1, align='center'),
            dbc.Col([DropDown("plist", value_list=task_priority, persistance=True)], md=4, lg=5, xs=9, align='center'),
            dbc.Col([dbc.Label("Task Difficulty: ",class_name="me-2 font-weight-light")], md=2, lg=1, xs=3, align='center'),
            dbc.Col([DropDown("diff-list", value_list=difficulty, persistance=True)], md=4, lg=5, xs=9, align='center')
        ], align='center', justify='start', key='row2', class_name='d-print-none g-2'),
        html.Br(),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody(dcc.Loading(dcc.Graph(id='chart1', className='bg-primary', config=graph_config, style={"minHeight":"200px"})))
                ]), lg=6, md=12),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody(dcc.Loading(dcc.Graph(id='chart2', className='bg-primary', config=graph_config, style={"minHeight":"200px"})))
                ]), lg=6, md=12)
        ], class_name='g-2', key='row3'),
        html.Br(),
        dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            dbc.Row([
                                dbc.Col(dbc.Label("Show Assignments: ", class_name='display-7 pe-2'), xs=4, md=2),
                                dbc.Col([
                                    dcc.RadioItems(id='chart3-options', value='all', inline=True,
                                        options=[{'label': 'All', 'value': 'all'},
                                                {'label': 'Top 10', 'value': '10'},
                                                {'label': 'Top 20', 'value': '20'}], 
                                        labelClassName='me-2 border border-start-0 pe-2 py-1', inputClassName='form-check-input me-2')
                                    ], md=4, xs=8),
                                
                                dbc.Col(dbc.Label("State: ", class_name='display-7 pe-2'), xs=4, md=2),
                                dbc.Col([
                                    DropDown("chart3-states", value_list=states, default_value="All", persistance=True)
                                ], align='center', md=4, xs=8),
                            ], className='g-1 d-print-none', align='center', justify='start')),
                        dbc.CardBody([
                            dcc.Loading(dcc.Graph(id='chart3a', className='bg-primary', config=graph_config)),
                            html.Hr(),
                            dcc.Loading(dcc.Graph(id='chart3b', className='bg-primary', config=graph_config))
                        ], style={"minHeight":"300px"})
                    ]),
                width=12),
            ], class_name="g-2", key='row4', align='center', justify='start', style={"minHeight":"300px"}),
        
        html.Hr(),
        dbc.Row([
            dbc.Col(html.H2("Summary", className='display-5 text-center'), width=12)
        ], key='row5', align='center', justify='center', class_name='mb-2'),
        html.Hr(),
        
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody(dcc.Loading(dcc.Graph(id='chart4', className='bg-primary', config=graph_config)))
                ]), lg=6, md=12),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody(dcc.Loading(dcc.Graph(id='chart5', className='bg-primary', config=graph_config)))
                ]), lg=6, md=12)
        ], class_name='g-2', key='row6', style={"minHeight":"300px"}),
        html.Br(),
        dbc.Row(
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(id='chart6', className='bg-primary', config=graph_config))))),
            key='row7', align='center', justify='start', style={"minHeight":"300px"})
    ], width=12)
)

@dash.callback(Output('chart1', 'figure'), Input('refresh', 'n_clicks'), 
              Input('theme', 'data'), Input("plist", 'value'), Input("diff-list", 'value'))
def chart1(_, mode, priority, difficulty):
    ## Or Make an API call here
    data = pd.read_csv('data.csv')
    data = filterByColumn(data, 'Task Priority', priority)
        
    data = data[['Staff', 'Difficulty', 'Created']].groupby(["Staff", 'Difficulty'])['Created'].count().to_frame().reset_index()
    data = filterByColumn(data, 'Difficulty', difficulty)
        
    data = data.rename(columns={'Created':'Count'})
    data = data.sort_values(by=['Staff'], axis=0)
    fig = px.bar(data, x='Staff', y='Count', color='Difficulty', template=mode,
                 color_discrete_map=difficulty_color_map, title="Staff Assigned and Difficulty")
    # fig.layout.template = template
    fig.update_layout({"barcornerradius": 4})
    return fig
    
@dash.callback(Output('chart2', 'figure'), Input('refresh', 'n_clicks'), Input('theme', 'data'), 
              Input("plist", 'value'),  Input("diff-list", 'value'))
def chart2(_, mode, priority, difficulty):
    ## Or Make an API call here
    data = pd.read_csv('data.csv')
    data = filterByColumn(data, 'Task Priority', priority)
    data = filterByColumn(data, 'Difficulty', difficulty)
        
    data = data['Work Type'].value_counts().to_frame().reset_index()
    fig = px.bar(data, x='Work Type', y='count', template=mode, text_auto=True,
                 color_discrete_map=difficulty_color_map, title="Assignment Types")
    fig.update_layout({
        "barcornerradius": 4,
        "margin":{"r":10}})
    return fig
    raise dash.exceptions.PreventUpdate()

@dash.callback(Output('chart3a', 'figure'), Input('refresh', 'n_clicks'), Input('theme', 'data'), 
              Input('chart3-options', 'value'), Input('chart3-states', 'value'), Input("plist", 'value'),  Input("diff-list", 'value'))
def chart3a(_, mode, show, state, priority, difficulty):
    ## Or Make an API call here
    data = pd.read_csv('data.csv')    
    data = filterByColumn(data, 'Task Priority', priority)
    data = filterByColumn(data, 'Difficulty', difficulty)    
    data = data['Contract'].value_counts().to_frame().reset_index().sort_values(by='Contract')
    total = data.shape[0]
    
    sites = pd.read_csv('sites.csv')
    data = data.merge(sites, how='inner', right_on='sites', left_on='Contract')
    if state != 'all':
        data = data[data['state'].str.lower() == state.lower()]
    
    data, ts = topContracts(data, show)
    data = data.rename(columns={'count':'Assignments'})
    
    fig = px.bar(data, x='Contract', y='Assignments', template=mode,
                 color_discrete_map=difficulty_color_map, title="Contract Assignements{}<br><sub>Total Contracts: {}</sub>".format(ts, total))
    fig.update_layout({"barcornerradius": 4})
    fig.update_layout(
        # xaxis_title_font=dict(size=8),  # X-axis title size
        xaxis_tickfont=dict(size=10),    # X-axis tick labels size
        margin={"r":10},
    )
    return fig

@dash.callback(Output('chart3b', 'figure', allow_duplicate=True), Input('refresh', 'n_clicks'), Input('theme', 'data'), 
               Input('chart3-options', 'value'), Input('chart3-states', 'value'), Input("plist", 'value'),  Input("diff-list", 'value'),
               prevent_initial_call='initial_duplicate')
def chart3b(_, mode, show, state, priority, difficulty):
    ## Or Make an API call here
    data = pd.read_csv('data.csv')    
    sites = pd.read_csv('sites.csv')
    
    data = filterByColumn(data, 'Task Priority', priority)
    data = filterByColumn(data, 'Difficulty', difficulty)
        
    data = data['Contract'].value_counts().to_frame().reset_index().sort_values(by='Contract')
    
    sites = sites.merge(data, how='inner', left_on='sites', right_on='Contract')
    if state != 'all':
        sites = sites[sites['state'].str.lower() == state.lower()]
    
    sites, ts = topContracts(sites, show)
    sites = sites.rename(columns={'count':'Assignments'})
    
    map_style = map_styles[1] if 'dark' in mode else map_styles[0]
    map = px.scatter_map(sites,lat='latitude', lon='longitude', size='Assignments', zoom=1, color='Assignments',
                         hover_name='Contract', size_max=30,hover_data=['Contract', 'Assignments', 'city'], opacity=0.8,
                         map_style=map_style, title="Contract Locations", template=mode) 
    map.update_layout(margin={"r":0,"t":50,"l":0,"b":10}, map_zoom=3, template=mode) #map_center=australian_center,
    # map.update_traces(cluster=dict(enabled=True, size=25), marker=dict(sizemode='diameter', color='royalblue', opacity=0.4, size=5))
    
    return map

@dash.callback(Output('chart3b', 'figure', allow_duplicate=True), Input('chart3b', 'clickData'),
            prevent_initial_call='initial_duplicate')
def chart3b_partial1_onclick(clickData):
    time.sleep(0.2) # Simulate a delay for the click event also to allow another callback to run first.
    if clickData is None:
        raise dash.exceptions.PreventUpdate()
    
    patched_figure = Patch()
    patched_figure["layout"]["map"]['zoom'] = 12
    patched_figure["layout"]["map"]['center'] = {"lat": clickData['points'][0]['lat'], "lon": clickData['points'][0]['lon']}
    return patched_figure

@dash.callback(Output('chart4', 'figure'), Input('refresh', 'n_clicks'), Input('theme', 'data'))
def chart4(_, mode):
    ## Or Make an API call here
    data = pd.read_csv('data.csv')
    total = data.shape[0]
    
    data = data['Task Priority'].value_counts().to_frame().reset_index().sort_values(by='Task Priority')
    fig = px.bar(data, x='Task Priority', y='count', template=mode, text_auto=True,
                 color_discrete_map=difficulty_color_map, title="Priority Counts <br><sub>Total: {}</sub>".format(total))
    fig.update_layout({"barcornerradius": 4, "margin":{"r":10}})
    return fig

@dash.callback(Output('chart5', 'figure'), Input('refresh', 'n_clicks'), Input('theme', 'data'))
def chart5(_, mode):
    ## Or Make an API call here
    data = pd.read_csv('data.csv')
    total = data.shape[0]
    
    data = data['Difficulty'].value_counts().to_frame().reset_index()
    fig = px.bar(data, x='Difficulty', y='count', template=mode, text_auto=True, category_orders={"Difficulty":difficulty},
                 color_discrete_map=difficulty_color_map, title="Difficulty Counts <br><sub>Total: {}</sub>".format(total))
    fig.update_layout({"barcornerradius": 4,  "margin":{"r":10}})
    return fig

@dash.callback(Output('chart6', 'figure'), Input('refresh', 'n_clicks'), Input('theme', 'data'))
def chart6(_, mode):
    ## Or Make an API call here
    history = pd.read_csv('summary_history.csv')
    history = history.melt(id_vars='Date', value_vars=["Immediate","Priority 1", "Priority 2", "Priority 3"], var_name='Task Priority', value_name='Count')
    
    fig = px.line(history, title="Priority Summary History", x = "Date", y="Count", color="Task Priority", markers=True, template=mode, color_discrete_map=task_priority_color_map)
    fig.update_layout({"margin":{"r":10}})
    return fig
    
