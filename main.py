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
from utilities import difficulty, difficulty_color_map, graph_config, DropDown, task_priority, \
    filterByColumn, isDebug

theme = dbc.themes.BOOTSTRAP

dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")
app = dash.Dash(
    __name__,
    external_stylesheets=[theme, dbc.icons.FONT_AWESOME, dbc_css]
)

from plotly.io import templates
print(list(templates))

app.layout = dbc.Container([
        dcc.Store(id='theme', data="plotly_white"),
        dbc.Row([
            dbc.Col(),
            dbc.Col(html.H1("Dashboard", className='display-3 text-center')),
            dbc.Col(html.Div([
                    dbc.Label(class_name="fa fa-moon pe-2", html_for="switch"),
                    dbc.Switch( id="switch", value=True, class_name="d-inline-block", persistence=True),
                    dbc.Label(class_name="fa fa-sun", html_for="switch"),
                    
                    dbc.Button("Refresh", id='refresh', class_name="ms-3", color='primary', size='sm')
                ], className='d-inline-block float-end', style={"whiteSpace":"nowrap"}), align='center')
        ],align='center', justify='between', key='row1'),
        dbc.Row(dbc.Col(html.Hr())),
        dbc.Row([
            dbc.Col(dbc.Label("Task Priority: ",class_name="me-2"), class_name="col-md-1 col-sm-2"), 
            dbc.Col([DropDown("plist", value_list=task_priority, persistance=True)], class_name='col-md-5 col-sm-10'),
            dbc.Col(dbc.Label("Task Difficulty: ",class_name="me-2"), class_name="col-md-1 col-sm-2"), 
            dbc.Col([DropDown("diff-list", value_list=difficulty, persistance=True)], class_name='col-md-5 col-sm-10')
        ], align='center', justify='start', key='row2'),
        html.Br(),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody(dcc.Loading(dcc.Graph(id='chart1', className='bg-primary', config=graph_config)))
                ]), class_name="col-lg-6"),
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(id='chart2', className='bg-primary', config=graph_config)))),
                class_name='col-lg-6')
        ], class_name='g-2', key='row3'),
        html.Br(),
        dbc.Row(
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(id='chart3', className='bg-primary', config=graph_config))))),
            key='row4', align='center', justify='start')
    ], fluid=True, class_name='dbc')


@app.callback(Output("theme","data"), Input("switch", "value"))
def themeMode(mode):
    return 'plotly' if mode else 'plotly_dark'

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
    fig = px.bar(data, x='Work Type', y='count', template=mode,
                 color_discrete_map=difficulty_color_map, title="Assignment Types")
    fig.update_layout({"barcornerradius": 4})
    return fig
    raise dash.exceptions.PreventUpdate()

@app.callback(Output('chart3', 'figure'), Input('refresh', 'n_clicks'), Input('theme', 'data'), 
              Input("plist", 'value'),  Input("diff-list", 'value'))
def chart3(_, mode, priority, difficulty):
    ## Or Make an API call here
    data = pd.read_csv('data.csv')    
    data = filterByColumn(data, 'Task Priority', priority)
    data = filterByColumn(data, 'Difficulty', difficulty)
        
    data = data['Contractor'].value_counts().to_frame().reset_index()
    fig = px.bar(data, x='Contractor', y='count', template=mode,
                 color_discrete_map=difficulty_color_map, title="Contractor Assignements")
    fig.update_layout({"barcornerradius": 4})
    fig.update_layout(
        xaxis_title_font=dict(size=8),  # X-axis title size
        xaxis_tickfont=dict(size=10),    # X-axis tick labels size
    )
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

if __name__ == "__main__":
    app.run(debug=isDebug())