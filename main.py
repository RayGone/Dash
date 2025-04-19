from dotenv import load_dotenv
load_dotenv()

# Import required libraries
import os
import datetime

import pandas as pd
import dash
from dash import html
from plotly import express as px
from dash import dcc, clientside_callback
from dash.dependencies import Input, Output, State

##========================================
####=======================================
######======================================
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from utilities import difficulty, difficulty_color_map, graph_config, DropDown, \
    task_priority, task_priority_color_map, filterByColumn, isDebug

theme = dbc.themes.CERULEAN
load_figure_template(['cerulean', 'cerulean_dark'])
template = 'cerulean'

dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")
app = dash.Dash(
    __name__,
    external_stylesheets=[theme, dbc.icons.FONT_AWESOME, dbc_css],
    title="Dashboard App"
)

# from plotly.io import templates
# print(list(templates))
### Alternative is to use Dash Leaflet - But on App Engine additional packages may limit Free Tier usage.
australian_center = {"lat": -23.7206836, "lon": 133.785374}
default_map = px.scatter_map(lat=[-23.7206836], lon=[133.785374], zoom=1, map_style="open-street-map", title="Contributor Locations", template=template+"_dark") 
default_map.update_layout(margin={"r":0,"t":50,"l":0,"b":10}, map_zoom=3, mapbox_center=australian_center)

app.layout = dbc.Container([
        dcc.Store(id='theme', data="plotly_white"),
        dbc.Row([
            dbc.Col(),
            dbc.Col(html.H1("Dashboard", className='display-3 text-center')),
            dbc.Col(html.Div([
                    html.A(href="https://github.com/RayGone/Dash", target="_blank", title='Check Github', className='fa-brands fa-github fa-bounce me-3', style={"fontSize":"20px", "cursor": "pointer"}),
                    dbc.Label(class_name="fa fa-moon pe-2", html_for="switch"),
                    dbc.Switch( id="switch", value=True, class_name="d-inline-block", persistence=True),
                    dbc.Label(class_name="fa fa-sun", html_for="switch"),
                    
                    dbc.Button("Refresh", id='refresh', class_name="ms-3", color='primary', size='sm')
                ], className='d-inline-block float-end d-print-none', style={"whiteSpace":"nowrap"}), align='center')
        ],align='center', justify='between', key='row1', class_name='sticky-top shadow-sm mb-3 bg-body'),

        dbc.Row([
            dbc.Col(dbc.Label("Task Priority: ",class_name="me-2"), class_name="col-md-1 col-sm-2"), 
            dbc.Col([DropDown("plist", value_list=task_priority, persistance=True)], class_name='col-md-5 col-sm-10'),
            dbc.Col(dbc.Label("Task Difficulty: ",class_name="me-2"), class_name="col-md-1 col-sm-2"), 
            dbc.Col([DropDown("diff-list", value_list=difficulty, persistance=True)], class_name='col-md-5 col-sm-10')
        ], align='center', justify='start', key='row2', class_name='d-print-none'),
        html.Br(),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody(dcc.Loading(dcc.Graph(id='chart1', className='bg-primary', config=graph_config)))
                ]), class_name="col-lg-6", width=6, sm=12),
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(id='chart2', className='bg-primary', config=graph_config)))),
                class_name='col-lg-6', width=6, sm=12)
        ], class_name='g-2', key='row3', style={"minHeight":"300px"}),
        html.Br(),
        dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Label("Show Assignments: ", className='display-7 pe-2'),
                            dcc.RadioItems(id='chart3a-options', value='all', inline=True,
                                options=[{'label': 'All', 'value': 'all'},
                                        {'label': 'Top 10', 'value': '10'},
                                        {'label': 'Top 20', 'value': '20'}], 
                                labelClassName='me-2 border border-start-0 pe-2 py-1', inputClassName='form-check-input me-2')], class_name='d-inline-flex align-items-center'),
                        dbc.CardBody(dcc.Loading(dcc.Graph(id='chart3a', className='bg-primary', config=graph_config)))
                    ]),
                width=12),
                dbc.Col(
                    dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(id='chart3b', figure=default_map, className='bg-primary', config=graph_config)))),
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
                ]), class_name="col-lg-6"),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody(dcc.Loading(dcc.Graph(id='chart5', className='bg-primary', config=graph_config)))
                ]), class_name="col-lg-6", width=6, sm=12)
        ], class_name='g-2', key='row6', style={"minHeight":"300px"}),
        html.Br(),
        dbc.Row(
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(id='chart6', className='bg-primary', config=graph_config))))),
            key='row7', align='center', justify='start', style={"minHeight":"300px"}),
        html.Div(className='mb-5', style={'height': '20px', 'width':"100%"})
    ], fluid=True, class_name='dbc')


@app.callback(Output("theme","data"), Input("switch", "value"))
def themeMode(mode):
    return template if mode else template+'_dark'

@app.callback(Output('chart1', 'figure'), Input('refresh', 'n_clicks'), 
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
    
@app.callback(Output('chart2', 'figure'), Input('refresh', 'n_clicks'), Input('theme', 'data'), 
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

@app.callback(Output('chart3a', 'figure'), Input('refresh', 'n_clicks'), Input('theme', 'data'), 
              Input('chart3a-options', 'value'), Input("plist", 'value'),  Input("diff-list", 'value'))
def chart3a(_, mode, show, priority, difficulty):
    ## Or Make an API call here
    data = pd.read_csv('data.csv')    
    data = filterByColumn(data, 'Task Priority', priority)
    data = filterByColumn(data, 'Difficulty', difficulty)    
    data = data['Contract'].value_counts().to_frame().reset_index().sort_values(by='Contract')
    total = data.shape[0]
    
    ts = ''
    if show != 'all':
        data = data.sort_values(by='count', ascending=False)
        if show == '10':
            ts = ' - Top 10'
            data = data.iloc[:10]
        elif show == '20':
            ts = ' - Top 20'
            data = data.iloc[:20]
    
    fig = px.bar(data, x='Contract', y='count', template=mode,
                 color_discrete_map=difficulty_color_map, title="Contract Assignements{}<br><sub>Total Contracts: {}</sub>".format(ts, total))
    fig.update_layout({"barcornerradius": 4})
    fig.update_layout(
        # xaxis_title_font=dict(size=8),  # X-axis title size
        xaxis_tickfont=dict(size=10),    # X-axis tick labels size
        margin={"r":10},
    )
    return fig

@app.callback(Output('chart3b', 'figure'), Input('refresh', 'n_clicks'), Input('theme', 'data'), 
              Input("plist", 'value'),  Input("diff-list", 'value'))
def chart3b(_, mode, priority, difficulty):
    ## Or Make an API call here
    data = pd.read_csv('data.csv')    
    data = filterByColumn(data, 'Task Priority', priority)
    data = filterByColumn(data, 'Difficulty', difficulty)
        
    data = data['Contract'].value_counts().to_frame().reset_index().sort_values(by='Contract')
    fig = px.bar(data, x='Contract', y='count', template=mode,
                 color_discrete_map=difficulty_color_map, title="Contract Assignements<br><sub>Total Contracts: {}</sub>".format(data.shape[0]))
    fig.update_layout({"barcornerradius": 4})
    fig.update_layout(
        # xaxis_title_font=dict(size=8),  # X-axis title size
        xaxis_tickfont=dict(size=10),    # X-axis tick labels size
    )
    raise dash.exceptions.PreventUpdate()
    return fig

@app.callback(Output('chart4', 'figure'), Input('refresh', 'n_clicks'), Input('theme', 'data'))
def chart4(_, mode):
    ## Or Make an API call here
    data = pd.read_csv('data.csv')
    total = data.shape[0]
    
    data = data['Task Priority'].value_counts().to_frame().reset_index().sort_values(by='Task Priority')
    fig = px.bar(data, x='Task Priority', y='count', template=mode, text_auto=True,
                 color_discrete_map=difficulty_color_map, title="Priority Counts <br><sub>Total: {}</sub>".format(total))
    fig.update_layout({"barcornerradius": 4})
    return fig

@app.callback(Output('chart5', 'figure'), Input('refresh', 'n_clicks'), Input('theme', 'data'))
def chart5(_, mode):
    ## Or Make an API call here
    data = pd.read_csv('data.csv')
    total = data.shape[0]
    
    data = data['Difficulty'].value_counts().to_frame().reset_index()
    fig = px.bar(data, x='Difficulty', y='count', template=mode, text_auto=True, category_orders={"Difficulty":difficulty},
                 color_discrete_map=difficulty_color_map, title="Difficulty Counts <br><sub>Total: {}</sub>".format(total))
    fig.update_layout({"barcornerradius": 4})
    return fig

@app.callback(Output('chart6', 'figure'), Input('refresh', 'n_clicks'), Input('theme', 'data'))
def chart5(_, mode):
    ## Or Make an API call here
    history = pd.read_csv('summary_history.csv')
    history = history.melt(id_vars='Date', value_vars=["Immediate","Priority 1", "Priority 2", "Priority 3"], var_name='Task Priority', value_name='Count')
    
    fig = px.line(history, title="Priority Summary History", x = "Date", y="Count", color="Task Priority", markers=True, template=mode, color_discrete_map=task_priority_color_map)
    return fig
    

clientside_callback(
    """
    (switchOn) => {
       document.documentElement.setAttribute('data-bs-theme', switchOn ? 'light' : 'dark');
       return window.dash_clientside.no_update
    }
    """,
    Output("switch", "id"),
    Input("switch", "value"),
)

server = app.server
if __name__ == "__main__":
    app.run(debug=isDebug())