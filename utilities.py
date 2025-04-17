import os
from dash import dcc

def isDebug():
    return bool(int(os.getenv("debug")))

difficulty = ["Critical", "High", "Medium", "Low"]
difficulty_color_map = {
    'Critical': 'darkred',
    'High': 'indianred',
    'Medium': 'orange',
    'Low': 'cornflowerblue'
}

graph_config = {
    'displaylogo':False,
    'modeBarButtonsToRemove': ['toggleSpikelines', "pan2d", "lasso2d"]
}

task_priority = ['Immediate', 'Priority 1', 'Priority 2', 'Priority 3', 'Hold Off']
task_priority_color_map = {
    'Priority 1': 'indianred',
    'Priority 2': 'orange',
    'Priority 3': 'cornflowerblue',
    'Immediate': 'darkred'
}

def DropDown(id, value_list=[], placeholder="", style={}, className="", default_value = "all", isSearchable=True, persistance=False ):
    return dcc.Dropdown(id=id,
        options=[{'label': 'All', 'value': 'all'}]+[{'label': status.upper(), 'value': status.lower()} for status in value_list],
        value=default_value.lower(),
        placeholder=placeholder,
        searchable=isSearchable,
        persistence=persistance,
        style=style,
        className=className
    )
    
def filterByColumn(DF, column: str, search: str):
    if column in DF.columns and search!='all':
        return DF[DF[column].str.lower().str.contains(search)]
    return DF

def arrayMatchAny(search, inArray):
    for s in inArray:
        if s == str(search):
            return True

    return False