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
        ], class_name='g-2', key='row3'),
        html.Br(),
        dbc.Row(
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(id='chart3', className='bg-primary', config=graph_config))))),
            key='row4', align='center', justify='start'),
        
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
        ], class_name='g-2', key='row6'),
        html.Br(),
        dbc.Row(
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(id='chart6', className='bg-primary', config=graph_config))))),
            key='row7', align='center', justify='start'),
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
        
    data = data['Contract'].value_counts().to_frame().reset_index().sort_values(by='Contract')
    fig = px.bar(data, x='Contract', y='count', template=mode,
                 color_discrete_map=difficulty_color_map, title="Contract Assignements<br><sub>Total Contracts: {}</sub>".format(data.shape[0]))
    fig.update_layout({"barcornerradius": 4})
    fig.update_layout(
        # xaxis_title_font=dict(size=8),  # X-axis title size
        xaxis_tickfont=dict(size=10),    # X-axis tick labels size
    )
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