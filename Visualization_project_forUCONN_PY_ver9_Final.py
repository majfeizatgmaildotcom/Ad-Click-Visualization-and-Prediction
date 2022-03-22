import numpy as np
import pandas as pd
import dash
#import dash_core_components as dcc
from dash import dcc
#import dash_html_components as html
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import plotly.express as px

import dash_bootstrap_components as dbc

import base64, random, subprocess, time
#from lifelines import KaplanMeierFitter
#from lifelines.statistics import logrank_test
import plotly.figure_factory as ff

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import dabest
from scipy import stats
from scipy.stats import ttest_ind
import flask
import os


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#app = dash.Dash()

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
df_all = pd.read_csv('/home/maf20025/Visualization_project/advertising.csv')
#df.head(2)
#df=df_all[0:20:]
df=df_all
df.head()



import json

def linkGeoJason(geo_world, df): 
    found = []
    missing = []
    countries_geo = []
    # For simpler acces, setting "zone" as index in a temporary dataFrame
    tmp = df.set_index('Country')
    # Looping over the custom GeoJSON file
    for country_i in geo_world['features']:

        # Country name detection
        country_name = country_i['properties']['name'] 

        # Checking if that country_i is in the dataset
        if country_name in tmp.index:

            # Adding country_i to our "Matched/found" countries
            found.append(country_name)

            # Getting information from both GeoJSON file and dataFrame
            geometry = country_i['geometry']

            # Adding 'id' information for further match between map and data 
            countries_geo.append({
                'type': 'Feature',
                'geometry': geometry,
                'id':country_name
            })

        # Else, adding the country_i to the missing countries
        else:
            missing.append(country_name)
    # Displaying metrics
    #print(f'Countries found    : {len(found)}')
    #print(f'Countries not found: {len(missing)}')
            
    geo_world_ok = {'type': 'FeatureCollection', 'features': countries_geo}        
            
    return geo_world_ok

def linkGeoJason_update(geo_world, df, conv_dict): 
        
    # Instanciating necessary lists
    found = []
    missing = []
    countries_geo = []

    # For simpler acces, setting "zone" as index in a temporary dataFrame
    tmp = df.set_index('Country')

    # Looping over the custom GeoJSON file
    for country in geo_world['features']:

        # Country name detection
        country_name = country['properties']['name']

        # Eventual replacement with our transition dictionnary
        country_name = conv_dict[country_name] if country_name in conv_dict.keys() else country_name
        go_on = country_name in tmp.index

        # If country is in original dataset or transition dictionnary
        if go_on:

            # Adding country to our "Matched/found" countries
            found.append(country_name)

            # Getting information from both GeoJSON file and dataFrame
            geometry = country['geometry']

            # Adding 'id' information for further match between map and data 
            countries_geo.append({
                'type': 'Feature',
                'geometry': geometry,
                'id':country_name
            })

        # Else, adding the country to the missing countries
        else:
            missing.append(country_name)

    # Displaying metrics
    #print(f'Countries found    : {len(found)}')
    #print(f'Countries not found: {len(missing)}')
    geo_world_ok = {'type': 'FeatureCollection', 'features': countries_geo}
    
    return geo_world_ok

# Loading geojson from (DATA_PATH depending on your local settings)
Data_File_Path ='/home/maf20025/Visualization_project/'
world_path = Data_File_Path + 'custom.geo.json'   
with open(world_path) as f:
    geo_world = json.load(f)
    
geo_world_ok = linkGeoJason(geo_world, df)    

#Key are in the Json, values are in the df
conv_dict = {
     'United States' : 'United States of America' ,
     'Russia' : 'Russian Federation'
}

geo_world_ok = linkGeoJason_update(geo_world, df, conv_dict)
dffage =set(sorted(df.Age))
dfCountry =set(sorted(df.Country))
col_options = [dict(label=x, value=x) for x in dfCountry]
#All = col_options

bins_DailySpent = round(np.sqrt(df['Daily Time Spent on Site'].count()));
binSize_1_2 = [10, bins_DailySpent, 50, 100]


# Set the bandwidth sizes.
Bandwidth_1_3 = [0.1, 0.5, 0.9]



bins_DailySpent_age = round(np.sqrt(df['Age'].count()));
binSize_2_2 = [10, bins_DailySpent_age, 50, 100]


# Set the bandwidth sizes.
Bandwidth_2_3 = [0.1, 0.5, 0.9]

GaussianMixture_n_components = [1, 2, 3, 4]

Bandwidth_4_2 = [0.1, 0.5, 0.9]

Bandwidth_5_2 =  [0.1, 0.5, 0.9]


def symbol_clickAd(df):
    a = []
    for i in range(len(df)):
        if df['Clicked on Ad'][i]==0:
            a.append('circle')
        else:
            a.append('x-open')
    return a


app.layout = html.Div([
    
        dbc.Row(
            
                [
    
                dbc.Col(html.H4("Majid Feiz"),
                   width={'size': 12,  "offset": 0})
                    
                    
                ]
            ),
    
    
    
        dbc.Row(
            
                [
    
                dbc.Col(html.H6("Recorded video dashboard presentation ->"),
                   width={'size': 12,  "offset": 0})
                    
                    
                ]
            ),
    
    
        dbc.Row(
            
                [
                
                dbc.Col(html.Video(                    
                        controls = True,
                       id = 'movie_player',
                        ## creat a new folder called static and place your video inside the static folder
                       src = "/static/Visualization_Presentation_Final_rev2.mp4",
                       autoPlay=True), width={'size': 12,  "offset": 1})
                    
                       
                    
                ]
            ),
        
        dbc.Row(
                [
                    
                dbc.Col(html.H1("Dashboard: Customer Ad-Click Visualization", className = 'text-center text-primary'), #and Prediction
                   width={'size': 12,  "offset": 0}),
                    #justify = 'center'
            
                ], justify = 'center',
                
    
            ),
    
    
    #html.Br(),
    
    html.Label('Age'),
    dcc.RangeSlider(id = 'SliderAge', min=min(df.Age), max=max(df.Age), 
                    value =[min(df.Age),max(df.Age)] , step = 1, marks = {i: i for i in range(max(df.Age))}),    # [i for i in range(19,61)]  
    html.Br(),
    html.Br(),

    #dcc.Dropdown(id="CountryDR", value='', options=col_options),
            dbc.Row(
                    [
                    
                    
    
                    dbc.Col([html.Label('People Who Clicked in the AD (1) or Not (0). Default Set to Have Both Data'),
                    #dcc.Dropdown(id="ClicedAD", value='', options=[0, 1]),
                    
                            dcc.Dropdown(id='ClicedAD',
                            options=[
                                     {'label': 'Clicked in', 'value': 1},
                                     {'label': 'Not Clicked in', 'value': 0},
                            ],
                            #optionHeight=35,                    #height/space between dropdown options
                            value=None ,                           #dropdown value selected automatically when page loads
                            #disabled=False,                     #disable dropdown value selection
                            #multi=True,                         #allow multiple dropdown values to be selected
                            #searchable=True,                    #allow user-searching of dropdown values
                            #search_value='',                    #remembers the value searched in dropdown
                            placeholder='Both Clicked Ad and not Clicked Ad. Otherwise select any option below: ',     #gray, default text shown when no option is selected
                            #clearable=True,                     #allow user to removes the selected value
                            style={'width':"100%"},             #use dictionary to define CSS styles of your dropdown
                            # className='select_box',           #activate separate CSS document in assets folder
                            # persistence=True,                 #remembers dropdown value. Used with persistence_type
                            # persistence_type='memory'         #remembers dropdown value selected until...
                            ),                                  #'memory': browser tab is refreshed
                                                                       #'session': browser tab is closed
                                                                #'local': browser cookies are deleted
                            ],
                            
                            
                            width={'size': 3, 'offset': 0}),
                ]),
    
    html.Div([
                 
        dbc.Row(
            
            [
        
    
           
                
           dbc.Col(dcc.Graph(id="graph-Geoplot", figure={}),
                    width=6, #lg={'size': 6,  "offset": 0, 'order': 'first'}
                   ),
                
                
                
           dbc.Col(dcc.Graph(id="graph-output0", figure={}, clickData= None),
                    width=6, #lg={'size': 6,  "offset": 0, 'order': 'first'}
                   ),

                    html.Br(),
                    html.Br(),
    
            ]
        )
    
    ]),
#---------------------------------
        html.Div([
                 
        dbc.Row(
            
            [
        
    
           
                
           dbc.Col(dcc.Graph(id="graph-scatterMatrix", figure={}),
                    width=6, #lg={'size': 6,  "offset": 0, 'order': 'first'}
                   ),
                
                
                
           dbc.Col(dcc.Graph(id="graph-CorrHeatmap", figure={}, clickData= None),
                    width=6, #lg={'size': 6,  "offset": 0, 'order': 'first'}
                   ),

                    html.Br(),
                    html.Br(),
    
            ]
        )
    
    ]),
                
                
#-----------------------  
    html.Div([
            
        dbc.Row(
            
            [
                
        


                dbc.Col([
                    

                        html.P("Bin Sizes:" 
                              # sytle ={'textDecoration':'underline'}
                              ),

                        dcc.RadioItems(
                                id="histogram_binSize",
                                      #options=[{"value": x, "label": x} for x in binSize_1_2],
                                      options=[
                                      {'label': str(binSize_1_2[0])+'  ', 'value': binSize_1_2[0]},
                                      {'label': str(binSize_1_2[1]) + ' (Square-root choice)  ', 'value': binSize_1_2[1]},
                                      {'label': str(binSize_1_2[2])+'  ', 'value':binSize_1_2[2]},
                                      {'label': str(binSize_1_2[3])+'  ', 'value':binSize_1_2[3]}
                                              ],
                                      value=binSize_1_2[1],
                                      labelStyle={"display": "inline-block"}),
                                    
                        ], width ={'size': 4,  "offset": 4}),
                
                dbc.Col([
                    
                    
                    html.P("Bandwidth Sizes:" 
                              # sytle ={'textDecoration':'underline'}
                              ),

                    
                        
                    dcc.RadioItems(
                                id="kde_bw_selection",
                                options=[{"value": x, "label": x} for x in Bandwidth_1_3],
                                value=Bandwidth_1_3[1],
                                labelStyle={"display": "inline-block"}),
                    ], width ={'size': 4,  "offset": 0}) #'order': 'first'
                
            
        
             ]
        )
    
    ]),           
        
        
        
        
        #-----------
    
    html.Div([
            
        dbc.Row(
            
            [    

           dbc.Col(dcc.Graph(id="graph-output1_1", figure={}), # figure= fig (is the out put of the return)
                    width=4, #lg={'size': 6,  "offset": 0, 'order': 'first'}
                   ),

                    html.Br(),
                    html.Br(),
                

          
                
            dbc.Col(dcc.Graph(id="graph-output1_2", figure={}), # figure= fig (is the out put of the return)
                    width=4, #lg={'size': 6,  "offset": 0, 'order': 'first'}
                   ),

                    html.Br(),
                    html.Br(),
                
            dbc.Col(dcc.Graph(id="graph-output1_3", figure={}), # figure= fig (is the out put of the return)
                    width=4, #lg={'size': 6,  "offset": 0, 'order': 'first'}
                   ),

                    html.Br(),
                    html.Br()     
                
                
                    ]
        )
    
    ]),           
                 
                
            #----------    
                
    html.Div([
            
        dbc.Row(
            
            [
                
        


                dbc.Col([
                    

                        html.P("Bin Sizes:" 
                              # sytle ={'textDecoration':'underline'}
                              ),

                        dcc.RadioItems(
                                id="histogram_binSize2",
                                      #options=[{"value": x, "label": x} for x in binSize_2_2],
                                      options=[
                                      {'label': str(binSize_2_2[0])+'  ', 'value': binSize_2_2[0]},
                                      {'label': str(binSize_2_2[1]) + ' (Square-root choice)  ', 'value': binSize_2_2[1]},
                                      {'label': str(binSize_2_2[2])+'  ', 'value':binSize_2_2[2]},
                                      {'label': str(binSize_2_2[3])+'  ', 'value':binSize_2_2[3]}
                                              ],
                                      value=binSize_2_2[1],
                                      labelStyle={"display": "inline-block"}),
                                    
                        ], width ={'size': 4,  "offset": 4}),
                
                dbc.Col([
                    
                    
                    html.P("Bandwidth Sizes:" 
                              # sytle ={'textDecoration':'underline'}
                              ),

                    
                        
                    dcc.RadioItems(
                                id="kde_bw_selection2",
                                options=[{"value": x, "label": x} for x in Bandwidth_2_3],
                                value=Bandwidth_2_3[1],
                                labelStyle={"display": "inline-block"}),
                    ], width ={'size': 4,  "offset": 0}) #'order': 'first'
                
            
        
             ]
        )
    
    ]),           

#-----------
    
    html.Div([
            
        dbc.Row(
            
            [    

           dbc.Col(dcc.Graph(id="graph-output2_1", figure={}), # figure= fig (is the out put of the return)
                    width=4, #lg={'size': 6,  "offset": 0, 'order': 'first'}
                   ),

                    html.Br(),
                    html.Br(),
                

          
                
            dbc.Col(dcc.Graph(id="graph-output2_2", figure={}), # figure= fig (is the out put of the return)
                    width=4, #lg={'size': 6,  "offset": 0, 'order': 'first'}
                   ),

                    html.Br(),
                    html.Br(),
                
            dbc.Col(dcc.Graph(id="graph-output2_3", figure={}), # figure= fig (is the out put of the return)
                    width=4, #lg={'size': 6,  "offset": 0, 'order': 'first'}
                   ),

                    html.Br(),
                    html.Br()     
                
                
                    ]
        )
    
    ]),           

    
#------------------- 
    
    
    html.Div([
            
        dbc.Row(
            
            [
                
        


                dbc.Col(dcc.RadioItems(
                                  id="id_GaussianMixture_n_components",
                                  #options=[{"value": x, "label": x} for x in GaussianMixture_n_components],
                                  options=[
                                  {'label': '1 Cluster  ', 'value': 1},
                                  {'label': '2 Cluster(Recommonded by Silhouettes Mehod  ', 'value': 2},
                                  {'label': '3 Cluster  ', 'value': 3},
                                  {'label': '4 Cluster  ', 'value': 4}
                                  ],
                                  value= 2,
                                  labelStyle={"display": "flex"}
                                ),
                                  width ={'size': 12,  "offset": 1})
        
             ]
        )
    
    ]),           
        

        html.Div([
            
        dbc.Row(
            
            [    

           dbc.Col(dcc.Graph(id="graph-output3_1", figure={}), # figure= fig (is the out put of the return)
                    width=6, #{'size': 6,  "offset": 3} #'order': 'first'
                   ),

                    html.Br(),
                    html.Br(),
                
                
           dbc.Col(dcc.Graph(id="graph-output3_2", figure={}), # figure= fig (is the out put of the return)
                    width=6, #{'size': 6,  "offset": 3} #'order': 'first'
                   ),

                    html.Br(),
                    html.Br()

                
                    ]
        )
    
    ]),           
    
 #----------------------------------   
    #Hypothesis for age 

    html.Div([
            
        dbc.Row(
            
            [
               
                
              dbc.Col([
                  
                html.H4("Interactive Hypothesis Testing of Customers's Daily Time Spent on Site by Selected Cluster Groups", 
                        className = 'text-left bg-success text-white'),
                        #width ={'size': 12,  "offset": 6}), #'order': 'first'

                html.H5(
                            id="the_output_print_of_pvalue52_id",
                            children='pre-value'),
                        ],
                            width ={'size': 12,  "offset": 6}), #'order': 'first'
                
                html.Br(),
                
                
               # dbc.Col(ddc.Input(
               #             id="the_input_pvalue52_id",
               #             vale=),
               #             width ={'size': 12,  "offset": 6}), #'order': 'first'
                
                
                
            dbc.Col([
                    
                    
                    html.P("Bandwidth Sizes:" 
                              # sytle ={'textDecoration':'underline'}
                              ),

                    
                        
                    dcc.RadioItems(
                                id="kde_bw_selection52",
                                options=[{"value": x, "label": x} for x in Bandwidth_5_2],
                                value=Bandwidth_5_2[1],
                                labelStyle={"display": "inline-block"}),
                    ], width ={'size': 12,  "offset": 6}) #'order': 'first' 

        
             ]
        )
    
    ]),           
        
 
    html.Div([
            
        dbc.Row(
            
            [    

           dbc.Col(dcc.Graph(id="graph-output5_1", figure={}), # figure= fig (is the out put of the return)
                    width=6, #{'size': 6,  "offset": 3} #'order': 'first'
                   ),

                    html.Br(),
                    html.Br(),
                
                
           dbc.Col(dcc.Graph(id="graph-output5_2", figure={}), # figure= fig (is the out put of the return)
                    width=6, #{'size': 6,  "offset": 3} #'order': 'first'
                   ),

                    html.Br(),
                    html.Br()

                
                    ]
        )
    
    ]),      
    
    #--------------------------------   
    #Hypothesis for gender 
    
        html.Div([
            
        dbc.Row(
            
            [
                
                
                dbc.Col([

                        html.H4("Interactive Hypothesis Testing of Customers's Age per Gender", 
                                className = 'text-left bg-success text-white'),
                                #width ={'size': 12,  "offset": 6}), #'order': 'first'

                        html.H5(
                                    id="the_output_print_of_pvalue42_id",
                                    children='pre-value'),
                                ],
                                    width ={'size': 12,  "offset": 6}), #'order': 'first'
                
                html.Br(),  
                
                
                
                
                
                
                
                dbc.Col([
                    
                    
                    html.P("Bandwidth Sizes:" 
                              # sytle ={'textDecoration':'underline'}
                              ),

                    
                        
                    dcc.RadioItems(
                                id="kde_bw_selection42",
                                options=[{"value": x, "label": x} for x in Bandwidth_4_2],
                                value=Bandwidth_4_2[1],
                                labelStyle={"display": "inline-block"}),
                    ], width ={'size': 12,  "offset": 6}) #'order': 'first' 
                
        
             ]
        )
    
    ]),           
        
 
            html.Div([
            
        dbc.Row(
            
            [    

           dbc.Col(dcc.Graph(id="graph-output4_1", figure={}), # figure= fig (is the out put of the return)
                    width=6, #{'size': 6,  "offset": 3} #'order': 'first'
                   ),

                    html.Br(),
                    html.Br(),
                
                
           dbc.Col(dcc.Graph(id="graph-output4_2", figure={}), # figure= fig (is the out put of the return)
                    width=6, #{'size': 6,  "offset": 3} #'order': 'first'
                   ),

                    html.Br(),
                    html.Br()

                
                    ]
        )
    
    ]),      
    
    
    
    
    
    
    
#----
    
])

#---------------------------------------------------------

    
@app.callback(
    Output(component_id = 'graph-Geoplot',  component_property = 'figure'),
    Input(component_id = 'SliderAge',  component_property = 'value'),
    Input(component_id = 'ClicedAD',  component_property = 'value')
)

def update_my_graph3(val_choose_from_slider, ClicedAD_option_select):
        
    if ClicedAD_option_select == 1:
        if val_choose_from_slider:
            dff3 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff3[dff3['Clicked on Ad'] == ClicedAD_option_select]
            dff_avg = dff_with_click.groupby(['Country']).mean().reset_index()
            fig3 = px.choropleth(dff_avg, geojson=geo_world_ok, locations='Country', height = 700,color = dff_avg.Age, projection='natural earth') #  color = 'Age'   hover_name='Age
                                       
            fig3.update_layout(title={
                            'text': "<b>Avg Age of Customer Ad-Clicks by Country</b>",
                            'y':0.92,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'})
            
            
            return fig3
    
    
    
    if ClicedAD_option_select == 0:
        if val_choose_from_slider:
            dff3 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff3[dff3['Clicked on Ad'] == ClicedAD_option_select]
            dff_avg = dff_with_click.groupby(['Country']).mean().reset_index()
            fig3 = px.choropleth(dff_avg, geojson=geo_world_ok, locations='Country', height = 700,color = dff_avg.Age, projection='natural earth') #  color = 'Age'   hover_name='Age'
            
            
            fig3.update_layout(title={
                    'text': "<b>Avg Age of Customer Ad-Clicks by Country</b>",
                    'y':0.92,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            
            return fig3
    
    
    
    
    if val_choose_from_slider:
        dff3 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
        dff_avg = dff3.groupby(['Country']).mean().reset_index()
        fig3 = px.choropleth(dff_avg, geojson=geo_world_ok, locations='Country', height = 700, color = dff_avg.Age, projection='natural earth') #  color = 'Age'   hover_name='Age'
        
        
        fig3.update_layout(title={
                'text': "<b>Avg Age of Customer Ad-Clicks by Country</b>",
                'y':0.92,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        
        return fig3

#----------------------------

@app.callback(
    Output(component_id = 'graph-output0',  component_property = 'figure'),
    Input(component_id = 'SliderAge',  component_property = 'value'),
    Input(component_id = 'ClicedAD',  component_property = 'value')
)

def update_my_graph0(val_choose_from_slider, ClicedAD_option_select):
    

    
    if ClicedAD_option_select == 1:
            
        if val_choose_from_slider:
            dff0 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            #dff_country = dff0[dff0.Country == CountryDR_select]
            #fig0 = px.treemap(dff_country, path=['Country','City'], values='Daily Time Spent on Site', hover_name='Country', color = 'Age')
            #fig0 = px.scatter(dff0, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
            dff_with_click = dff0[dff0['Clicked on Ad'] == ClicedAD_option_select]
            fig0 = px.treemap(dff_with_click, path=['Country','City'], values='Daily Time Spent on Site', hover_name='Country', color = 'Age',height = 700)
            
            fig0.update_layout(title={
                'text': "<b>Tree Map of Daily Time Spent on Site of Customer Ad-Clicks by Country</b>",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            
            return fig0
            
            
    if ClicedAD_option_select == 0:
            
        if val_choose_from_slider:
            dff0 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            #dff_country = dff0[dff0.Country == CountryDR_select]
            #fig0 = px.treemap(dff_country, path=['Country','City'], values='Daily Time Spent on Site', hover_name='Country', color = 'Age')
            #fig0 = px.scatter(dff0, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
            dff_with_click = dff0[dff0['Clicked on Ad'] == ClicedAD_option_select]
            fig0 = px.treemap(dff_with_click, path=['Country','City'], values='Daily Time Spent on Site', hover_name='Country', color = 'Age',height = 700)
            
            fig0.update_layout(title={
                'text': "<b>Tree Map of Daily Time Spent on Site of Customer Ad-Clicks by Country</b>",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            
            return fig0
    
    if val_choose_from_slider:
        dff0 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
        #dff_country = dff0[dff0.Country == CountryDR_select]
        #fig0 = px.treemap(dff_country, path=['Country','City'], values='Daily Time Spent on Site', hover_name='Country', color = 'Age')
        #fig0 = px.scatter(dff0, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
        fig0 = px.treemap(dff0, path=['Country','City'], values='Daily Time Spent on Site', hover_name='Country', color = 'Age',height = 700)
        
        fig0.update_layout(title={
                'text': "<b>Tree Map of Daily Time Spent on Site of Customer Ad-Clicks by Country</b>",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        
        
        return fig0
        
 
 #-----------------------this area is for scatter matrix and heat map
@app.callback(
    Output(component_id = 'graph-scatterMatrix',  component_property = 'figure'),
    Input(component_id = 'SliderAge',  component_property = 'value'),
    Input(component_id = 'ClicedAD',  component_property = 'value')
)

def update_my_graph_scatmatrix(val_choose_from_slider, ClicedAD_option_select):
        
    if ClicedAD_option_select == 1:
        if val_choose_from_slider:
            
            dff0 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff0[dff0['Clicked on Ad'] == ClicedAD_option_select]
            
            #dff_with_click_scatter_Matrix = dff_with_click[['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage']]

            fig_scatMat = px.scatter_matrix(dff_with_click,dimensions=['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage'],
                                            height = 700,  color ='Clicked on Ad')  #alpha=0.3,               
            fig_scatMat.update_layout(title={
                            'text': "<b>Scatter Marix for Customr Ad-Clicks by Country</b>",
                            'y':0.94,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'})

            fig_scatMat.update_traces(diagonal_visible=False)
            
            
            return fig_scatMat
    
    
    
    if ClicedAD_option_select == 0:
        if val_choose_from_slider:

            dff0 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff0[dff0['Clicked on Ad'] == ClicedAD_option_select]
            
            #dff_with_click_scatter_Matrix = dff_with_click[['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage']]

            fig_scatMat = px.scatter_matrix(dff_with_click,dimensions=['Daily Time Spent on Site','Age','Area Income', 'Daily Internet Usage'],
                                            height = 700,  color ='Clicked on Ad')  #alpha=0.3, 
            fig_scatMat.update_layout(title={
                            'text': "<b>Scatter Marix for Customr Ad-Clicks by Country</b>",
                            'y':0.94,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'})
            
            fig_scatMat.update_traces(diagonal_visible=False)

            return fig_scatMat
    
    
    
    
    if val_choose_from_slider:        
        
        dff0 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]

        #dff_with_click_scatter_Matrix = dff0[['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage']]

        fig_scatMat = px.scatter_matrix(dff0,dimensions=['Daily Time Spent on Site','Age','Area Income', 'Daily Internet Usage'],
                                            height = 700, color ='Clicked on Ad')  #alpha=0.3,       
        fig_scatMat.update_layout(title={
                        'text': "<b>Scatter Marix for Customr Ad-Clicks by Country</b>",
                        'y':0.94,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})
        
        fig_scatMat.update_traces(diagonal_visible=False)

        return fig_scatMat

#----------------------------

@app.callback(
    Output(component_id = 'graph-CorrHeatmap',  component_property = 'figure'),
    Input(component_id = 'SliderAge',  component_property = 'value'),
    Input(component_id = 'ClicedAD',  component_property = 'value')
)

def update_my_graphCorrMat(val_choose_from_slider, ClicedAD_option_select):
    

    
    if ClicedAD_option_select == 1:
            
        if val_choose_from_slider:
            dff0 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            #dff_country = dff0[dff0.Country == CountryDR_select]
            #fig0 = px.treemap(dff_country, path=['Country','City'], values='Daily Time Spent on Site', hover_name='Country', color = 'Age')
            #fig0 = px.scatter(dff0, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
            dff_with_click = dff0[dff0['Clicked on Ad'] == ClicedAD_option_select]
            
            dff_with_click_scatter_Matrix = dff_with_click[['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage']]
            
            corrMatrix = dff_with_click_scatter_Matrix.corr().round(2)
            #fig = px.imshow(corrMatrix)
            z_Matcor = np.array(corrMatrix)
            
            x_Matcor = ['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage']
            y_Matcor = ['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage']
            fig_heat = ff.create_annotated_heatmap(z_Matcor,  x=x_Matcor, y=y_Matcor, colorscale='rainbow',showscale=True)
 
            fig_heat.update_layout(title={
                'text': "<b>Correlation Heat Map of the scatter matrix</b>",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
                height=700,
                legend_title="Correlation",
                showlegend=True)
    
            fig_heat.update_xaxes(side="bottom")

            
            return fig_heat
            
            
    if ClicedAD_option_select == 0:
            
        if val_choose_from_slider:
            dff0 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            #dff_country = dff0[dff0.Country == CountryDR_select]
            #fig0 = px.treemap(dff_country, path=['Country','City'], values='Daily Time Spent on Site', hover_name='Country', color = 'Age')
            #fig0 = px.scatter(dff0, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
            dff_with_click = dff0[dff0['Clicked on Ad'] == ClicedAD_option_select]
            dff_with_click_scatter_Matrix = dff_with_click[['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage']]
            
            corrMatrix = dff_with_click_scatter_Matrix.corr().round(2)
            z_Matcor = np.array(corrMatrix)
            
            x_Matcor = ['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage']
            y_Matcor = ['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage']
            fig_heat = ff.create_annotated_heatmap(z_Matcor,  x=x_Matcor, y=y_Matcor, colorscale='rainbow',showscale=True)
 
            fig_heat.update_layout(title={
                'text': "<b>Correlation Heat Map of the scatter matrix</b>",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
                height=700,
                legend_title="Correlation",
                showlegend=True)
    
            fig_heat.update_xaxes(side="bottom")
            
            return fig_heat
    
    if val_choose_from_slider:
        dff0 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
        #dff_country = dff0[dff0.Country == CountryDR_select]
        #fig0 = px.treemap(dff_country, path=['Country','City'], values='Daily Time Spent on Site', hover_name='Country', color = 'Age')
        #fig0 = px.scatter(dff0, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
        dff_with_click_scatter_Matrix= dff0[['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage']]
            
        corrMatrix = dff_with_click_scatter_Matrix.corr().round(2)
        z_Matcor = np.array(corrMatrix)
            
        x_Matcor = ['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage']
        y_Matcor = ['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage']
        fig_heat = ff.create_annotated_heatmap(z_Matcor,  x=x_Matcor, y=y_Matcor, colorscale='rainbow',showscale=True)
 
        fig_heat.update_layout(title={
                'text': "<b>Correlation Heat Map of the scatter matrix</b>",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
                height=700,
                legend_title="Correlation",
                showlegend=True)
        
        fig_heat.update_xaxes(side="bottom")
            
        return fig_heat


#scatter matrix and heat map ended
 #-----------------------

#this is the area I am working

@app.callback(
    Output(component_id = 'graph-output1_1',  component_property = 'figure'),
    Input(component_id = 'SliderAge',  component_property = 'value'), 
     Input(component_id = 'ClicedAD',  component_property = 'value')
)

def update_my_graph1_1(val_choose_from_slider, ClicedAD_option_select):
    

    if ClicedAD_option_select == 1:
        if val_choose_from_slider:
                  
            #dff = df[df['Age'].isin(val_choose_from_slider)]
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            fig1 = px.scatter(dff_with_click, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age', color ='Clicked on Ad' )
            
            
            
            fig1.update_layout(title={
                'text': "<b>Scatter plot of Daily Time Spent on Site vs. Daily Internet Usage</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            
            
            return fig1
        
    if ClicedAD_option_select == 0:
        if val_choose_from_slider:
                  
            #dff = df[df['Age'].isin(val_choose_from_slider)]
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            fig1 = px.scatter(dff_with_click, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age', color ='Clicked on Ad' )
            
            fig1.update_layout(title={
                'text': "<b>Scatter plot of Daily Time Spent on Site vs. Daily Internet Usage</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            
            return fig1
        

    if val_choose_from_slider:
                  
        #dff = df[df['Age'].isin(val_choose_from_slider)]
        dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
        #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
        #dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
        fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age', color ='Clicked on Ad' )
        
        fig1.update_layout(title={
                'text': "<b>Scatter plot of Daily Time Spent on Site vs. Daily Internet Usage</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        
        return fig1
         

        
@app.callback(
    Output(component_id = 'graph-output1_2',  component_property = 'figure'),
     Input(component_id = 'SliderAge',  component_property = 'value'), 
     Input(component_id = 'ClicedAD',  component_property = 'value'),
     Input("histogram_binSize", "value")
    
)

def update_my_graph1_2(val_choose_from_slider, ClicedAD_option_select, histogram_binSize):
    

    if ClicedAD_option_select == 1:
        if val_choose_from_slider:
                  
            #dff = df[df['Age'].isin(val_choose_from_slider)]
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            #fig1 = px.scatter(dff_with_click, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age', color ='Clicked on Ad' )
    
            fig_histo = px.histogram(dff_with_click['Daily Internet Usage'], 
                       title="Interactive Daily Internet Usage Histogram",
                       nbins= histogram_binSize) #, x="Height",
            
            
            fig_histo.update_layout(
                title={
                    'text': "<b>Interactive Daily Internet Usage Histogram</b>",
                    'y':0.9,
                    'x':0.45,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                    xaxis_title="Daily Internet Usage",
                    yaxis_title="Frequency of Occurrence",
                    legend_title="Variable"
                                        )
            
            
            return fig_histo

        
    if ClicedAD_option_select == 0:
        if val_choose_from_slider:
                  
            #dff = df[df['Age'].isin(val_choose_from_slider)]
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
           
            fig_histo = px.histogram(dff_with_click['Daily Internet Usage'], 
                       title="Interactive Daily Internet Usage Histogram",
                       nbins= histogram_binSize) #, x="Height",
            
            fig_histo.update_layout(
                title={
                    'text': "<b>Interactive Daily Internet Usage Histogram</b>",
                    'y':0.9,
                    'x':0.45,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                    xaxis_title="Daily Internet Usage",
                    yaxis_title="Frequency of Occurrence",
                    legend_title="Variable"
                                        )
            
            
            
            return fig_histo
        

    if val_choose_from_slider:
                  
        #dff = df[df['Age'].isin(val_choose_from_slider)]
        dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
        #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
        #dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
        fig_histo = px.histogram(dff1['Daily Internet Usage'], 
                       title="Interactive Daily Internet Usage Histogram",
                       nbins= histogram_binSize) #, x="Height",
        
        fig_histo.update_layout(
                title={
                    'text': "<b>Interactive Daily Internet Usage Histogram</b>",
                    'y':0.9,
                    'x':0.45,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                    xaxis_title="Daily Internet Usage",
                    yaxis_title="Frequency of Occurrence",
                    legend_title="Variable"
                                        )
        
        
        return fig_histo

        
        
@app.callback(
     Output(component_id = 'graph-output1_3',  component_property = 'figure'),
     Input(component_id = 'SliderAge',  component_property = 'value'), 
     Input(component_id = 'ClicedAD',  component_property = 'value'),
     Input("kde_bw_selection", "value")   
)

def update_my_graph1_3(val_choose_from_slider, ClicedAD_option_select, BW):
    

    if ClicedAD_option_select == 1:
        if val_choose_from_slider:
                  
            #dff = df[df['Age'].isin(val_choose_from_slider)]
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            #fig1 = px.scatter(dff_with_click, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age', color ='Clicked on Ad' )
    
            x_ = np.array(dff_with_click['Daily Internet Usage'])
            hist_data = [x_]
            group_labels = ['Daily Internet Usage'] # name of the dataset
            fig_kde = ff.create_distplot(hist_data, group_labels, bin_size=BW)
            
            
            fig_kde.update_layout(
                title={
                    'text': "<b>Interactive Daily Internet Usage KDE</b>",
                    'y':0.9,
                    'x':0.45,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                    xaxis_title="Daily Internet Usage",
                    yaxis_title="Frequency of Occurrence(KDE)",
                    legend_title="Variable"
                                        )
            

            
            return fig_kde 

        
    if ClicedAD_option_select == 0:
        if val_choose_from_slider:
                  
            #dff = df[df['Age'].isin(val_choose_from_slider)]
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
           
            
            x_ = np.array(dff_with_click['Daily Internet Usage'])
            hist_data = [x_]
            group_labels = ['Daily Internet Usage'] # name of the dataset
            fig_kde = ff.create_distplot(hist_data, group_labels, bin_size=BW) #kde_bw_selection
            
            fig_kde.update_layout(
                title={
                    'text': "<b>Interactive Daily Internet Usage KDE</b>",
                    'y':0.9,
                    'x':0.45,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                    xaxis_title="Daily Internet Usage",
                    yaxis_title="Frequency of Occurrence(KDE)",
                    legend_title="Variable"
                                        )
            
            return fig_kde 
        

    if val_choose_from_slider:
                  
        #dff = df[df['Age'].isin(val_choose_from_slider)]
        dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
        #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
        #dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
        x_ = np.array(dff1['Daily Internet Usage'])
        hist_data = [x_]
        group_labels = ['Daily Internet Usage'] # name of the dataset
        fig_kde = ff.create_distplot(hist_data, group_labels, bin_size=BW)
        
        fig_kde.update_layout(
                title={
                    'text': "<b>Interactive Daily Internet Usage KDE</b>",
                    'y':0.9,
                    'x':0.45,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                    xaxis_title="Daily Internet Usage",
                    yaxis_title="Frequency of Occurrence(KDE)",
                    legend_title="Variable"
                                        )
        
        return fig_kde        
        
        
        
        
       
         
        
#-----------------------

@app.callback(
    Output(component_id = 'graph-output2_1',  component_property = 'figure'),
    Input(component_id = 'SliderAge',  component_property = 'value'),
    Input(component_id = 'ClicedAD',  component_property = 'value')
)

def update_my_graph2_1(val_choose_from_slider, ClicedAD_option_select):     
    
    if ClicedAD_option_select == 1:
        if val_choose_from_slider:
            dff2 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff2[dff2['Clicked on Ad'] == ClicedAD_option_select]
            fig2 = px.scatter(dff_with_click, x='Daily Time Spent on Site', y ='Age', hover_name='Age', color ='Clicked on Ad')
            
            fig2.update_layout(title={
                'text': "<b>Scatter plot Customers' Age vs. Daily Time Spent on Site</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            
            return fig2
 
    if ClicedAD_option_select == 0:
        if val_choose_from_slider:
            dff2 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff2[dff2['Clicked on Ad'] == ClicedAD_option_select]
            fig2 = px.scatter(dff_with_click, x='Daily Time Spent on Site', y ='Age', hover_name='Age', color ='Clicked on Ad')
            
            fig2.update_layout(title={
                'text': "<b>Scatter plot Customers' Age vs. Daily Time Spent on Site</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            
            
            return fig2
    
 
    if val_choose_from_slider:
        dff2 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
        fig2 = px.scatter(dff2, x='Daily Time Spent on Site', y ='Age', hover_name='Age', color ='Clicked on Ad')
        
        fig2.update_layout(title={
                'text': "<b>Scatter plot Customers' Age vs. Daily Time Spent on Site</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        
        
        return fig2


    
    
@app.callback(
    Output(component_id = 'graph-output2_2',  component_property = 'figure'),
     Input(component_id = 'SliderAge',  component_property = 'value'), 
     Input(component_id = 'ClicedAD',  component_property = 'value'),
     Input("histogram_binSize2", "value")
    
)

def update_my_graph2_2(val_choose_from_slider, ClicedAD_option_select, histogram_binSize):
    

    if ClicedAD_option_select == 1:
        if val_choose_from_slider:
                  
            #dff = df[df['Age'].isin(val_choose_from_slider)]
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            #fig1 = px.scatter(dff_with_click, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age', color ='Clicked on Ad' )
    
            fig_histo = px.histogram(dff_with_click['Daily Time Spent on Site'], 
                       title="Interactive Daily Time Spent on Site Histogram",
                       nbins= histogram_binSize) #, x="Height",
            
            
            fig_histo.update_layout(
                title={
                    'text': "<b>Interactive Daily Time Spent on Site Histogram</b>",
                    'y':0.9,
                    'x':0.45,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                    xaxis_title="Daily Time Spent on Site",
                    yaxis_title="Frequency of Occurrence",
                    legend_title="Variable"
                                        )
            
            
            
            return fig_histo

        
    if ClicedAD_option_select == 0:
        if val_choose_from_slider:
                  
            #dff = df[df['Age'].isin(val_choose_from_slider)]
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
           
            fig_histo = px.histogram(dff_with_click['Daily Time Spent on Site'], 
                       title="Interactive Daily Time Spent on Site Histogram",
                       nbins= histogram_binSize) #, x="Height",
            
            fig_histo.update_layout(
                title={
                    'text': "<b>Interactive Daily Time Spent on Site Histogram</b>",
                    'y':0.9,
                    'x':0.45,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                    xaxis_title="Daily Time Spent on Site",
                    yaxis_title="Frequency of Occurrence",
                    legend_title="Variable"
                                        )
            
            
            return fig_histo
        

    if val_choose_from_slider:
                  
        #dff = df[df['Age'].isin(val_choose_from_slider)]
        dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
        #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
        #dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
        fig_histo = px.histogram(dff1['Daily Time Spent on Site'], 
                       title="Interactive Daily Time Spent on Site Histogram",
                       nbins= histogram_binSize) #, x="Height",
        
        fig_histo.update_layout(
                title={
                    'text': "<b>Interactive Daily Time Spent on Site Histogram</b>",
                    'y':0.9,
                    'x':0.45,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                    xaxis_title="Daily Time Spent on Site",
                    yaxis_title="Frequency of Occurrence",
                    legend_title="Variable"
                                        )

        return fig_histo

        
        
@app.callback(
    Output(component_id = 'graph-output2_3',  component_property = 'figure'),
     Input(component_id = 'SliderAge',  component_property = 'value'), 
     Input(component_id = 'ClicedAD',  component_property = 'value'),
     Input("kde_bw_selection2", "value")
    
)

def update_my_graph2_3(val_choose_from_slider, ClicedAD_option_select, BW):
    

    if ClicedAD_option_select == 1:
        if val_choose_from_slider:
                  
            #dff = df[df['Age'].isin(val_choose_from_slider)]
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            #fig1 = px.scatter(dff_with_click, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age', color ='Clicked on Ad' )
    
            x_ = np.array(dff_with_click['Daily Time Spent on Site'])
            hist_data = [x_]
            group_labels = ['Daily Time Spent on Site'] # name of the dataset
            fig_kde = ff.create_distplot(hist_data, group_labels, bin_size=BW)
            
            fig_kde.update_layout(
                title={
                    'text': "<b>Interactive Daily Time Spent on Site KDE</b>",
                    'y':0.9,
                    'x':0.45,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                    xaxis_title="Daily Time Spent on Site",
                    yaxis_title="Frequency of Occurrence(KDE)",
                    legend_title="Variable"
                                        )
            
            
            
            return fig_kde 

        
    if ClicedAD_option_select == 0:
        if val_choose_from_slider:
                  
            #dff = df[df['Age'].isin(val_choose_from_slider)]
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
           
            
            x_ = np.array(dff_with_click['Daily Time Spent on Site'])
            hist_data = [x_]
            group_labels = ['Daily Time Spent on Site'] # name of the dataset
            fig_kde = ff.create_distplot(hist_data, group_labels, bin_size=BW) #kde_bw_selection
            
            
            fig_kde.update_layout(
                title={
                    'text': "<b>Interactive Daily Time Spent on Site KDE</b>",
                    'y':0.9,
                    'x':0.45,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                    xaxis_title="Daily Time Spent on Site",
                    yaxis_title="Frequency of Occurrence(KDE)",
                    legend_title="Variable"
                                        )
            
            return fig_kde 
        

    if val_choose_from_slider:
                  
        #dff = df[df['Age'].isin(val_choose_from_slider)]
        dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
        #fig1 = px.scatter(dff1, x='Daily Internet Usage', y ='Daily Time Spent on Site',hover_name='Age')
        #dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
        x_ = np.array(dff1['Daily Time Spent on Site'])
        hist_data = [x_]
        group_labels = ['dff1.Daily Time Spent on Site'] # name of the dataset
        fig_kde = ff.create_distplot(hist_data, group_labels, bin_size=BW)
        
        
        fig_kde.update_layout(
                title={
                    'text': "<b>Interactive Daily Time Spent on Site KDE</b>",
                    'y':0.9,
                    'x':0.45,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                    xaxis_title="Daily Time Spent on Site",
                    yaxis_title="Frequency of Occurrence(KDE)",
                    legend_title="Variable"
                                        )
        
        
        
        return fig_kde    
    
    
    
    
#-------------------

@app.callback(
    Output(component_id = 'graph-output3_1',  component_property = 'figure'),
     Input(component_id = 'SliderAge',  component_property = 'value'), 
     Input(component_id = 'ClicedAD',  component_property = 'value'),
     Input("id_GaussianMixture_n_components", "value")
    
)

def update_my_graph3_1(val_choose_from_slider, ClicedAD_option_select, GaussianM_n_component):



    if ClicedAD_option_select == 1:
        if val_choose_from_slider:
                  

            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            gmm = GaussianMixture(GaussianM_n_component)
            X_DailyTimeOnSite=np.array([dff_with_click.Age , dff_with_click['Daily Time Spent on Site'], dff_with_click['Clicked on Ad']]).T
            gmm.fit(X_DailyTimeOnSite[:,:2])
            labels = gmm.predict(X_DailyTimeOnSite[:,:2])
            frame = pd.DataFrame(X_DailyTimeOnSite)
            #predictions from gmm
            frame['Cluster'] = labels
            #a=symbol_clickAd(X_DailyTimeOnSite)
            #frame['Symbol'] = np.array(a)
            #frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster','Symbol']
            frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster']
            plt.figure(figsize=(7,7))
            #color=['blue', 'green']
            fig31 = px.scatter(frame, x='Daily Time Spent on Site', y ='Age', color='Cluster', symbol = 'Clicked on Ad') #,color=color[k]
            fig31.update_layout(legend_title_side="top left",legend_x=1.2, legend_y=1)
            fig31.update_traces(marker=dict(size=15, line=dict(width=2)))
            fig31.update_layout(title={
                'text': "<b>Scatter plot Customers' Age vs. Daily Time Spent on Site Clustered based on Gaussian Mixture Model</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            
            return fig31

        
    if ClicedAD_option_select == 0:
        if val_choose_from_slider:
                  
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
           
            gmm = GaussianMixture(GaussianM_n_component)
            X_DailyTimeOnSite=np.array([dff_with_click.Age , dff_with_click['Daily Time Spent on Site'], dff_with_click['Clicked on Ad']]).T
            gmm.fit(X_DailyTimeOnSite[:,:2])
            labels = gmm.predict(X_DailyTimeOnSite[:,:2])
            frame = pd.DataFrame(X_DailyTimeOnSite)
            #predictions from gmm
            frame['Cluster'] = labels
            #a=symbol_clickAd(X_DailyTimeOnSite)
            #frame['Symbol'] = np.array(a)
            #frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster','Symbol']
            frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster']
            plt.figure(figsize=(7,7))
            #color=['blue', 'green']
            fig31 = px.scatter(frame, x='Daily Time Spent on Site', y ='Age', color='Cluster', symbol = 'Clicked on Ad') #,color=color[k]
            fig31.update_layout(legend_title_side="top left",legend_x=1.2, legend_y=1)
            fig31.update_traces(marker=dict(size=15, line=dict(width=2)))
            fig31.update_layout(title={
                'text': "<b>Scatter plot Customers' Age vs. Daily Time Spent on Site Clustered based on Gaussian Mixture Model</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            return fig31
        

    if val_choose_from_slider:
                  

        dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
        gmm = GaussianMixture(GaussianM_n_component)
        X_DailyTimeOnSite=np.array([dff1.Age , dff1['Daily Time Spent on Site'], dff1['Clicked on Ad']]).T
        gmm.fit(X_DailyTimeOnSite[:,:2])
        labels = gmm.predict(X_DailyTimeOnSite[:,:2])
        frame = pd.DataFrame(X_DailyTimeOnSite)
        #predictions from gmm
        frame['Cluster'] = labels
        #a=symbol_clickAd(X_DailyTimeOnSite)
        #frame['Symbol'] = np.array(a)
        #frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster','Symbol']
        frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster']
        plt.figure(figsize=(7,7))
        #color=['blue', 'green']
        #display(frame)
        fig31 = px.scatter(frame, x='Daily Time Spent on Site', y ='Age', color='Cluster', symbol = 'Clicked on Ad') #,color=color[k] marker_symbol='Symbol'
        fig31.update_layout(legend_title_side="top left",legend_x=1.2, legend_y=1)
        fig31.update_traces(marker=dict(size=15, line=dict(width=2)))
        fig31.update_layout(title={
                'text': "<b>Scatter plot Customers' Age vs. Daily Time Spent on Site Clustered based on Gaussian Mixture Model</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        return fig31


        #color='DarkSlateGrey')
    
    
#-------------------

@app.callback(
    Output(component_id = 'graph-output3_2',  component_property = 'figure'),
     Input(component_id = 'SliderAge',  component_property = 'value'), 
     Input(component_id = 'ClicedAD',  component_property = 'value')
    
)

def update_my_graph3_2(val_choose_from_slider, ClicedAD_option_select):

    

    if ClicedAD_option_select == 1:
        if val_choose_from_slider:
                  

            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            fig32 = px.density_contour(dff_with_click, x="Daily Time Spent on Site", y="Age", marginal_x="histogram", marginal_y="histogram")
            fig32.update_layout(title={
                'text': "<b>2D Density Contours with Histogram plot of Customers' Age vs. Daily Time Spent on Site</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            
            
            return fig32

        
    if ClicedAD_option_select == 0:
        if val_choose_from_slider:
                  
            #fig32 = plt.figure()
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            fig32 = px.density_contour(dff_with_click, x="Daily Time Spent on Site", y="Age", marginal_x="histogram", marginal_y="histogram")
            
            fig32.update_layout(title={
                'text': "<b>2D Density Contours with Histogram plot of Customers' Age vs. Daily Time Spent on Site</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            
            return fig32

        

    if val_choose_from_slider:
                  
        #fig32 = plt.figure()
        dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
        fig32 = px.density_contour(dff1, x="Daily Time Spent on Site", y="Age", marginal_x="histogram", marginal_y="histogram")
        
        fig32.update_layout(title={
                'text': "<b>2D Density Contours with Histogram plot of Customers' Age vs. Daily Time Spent on Site</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        
        
        return fig32   

    
#------------
#Hypothesis test 1 for gender

@app.callback(
    Output(component_id = 'graph-output4_1',  component_property = 'figure'),
     Input(component_id = 'SliderAge',  component_property = 'value'), 
     Input(component_id = 'ClicedAD',  component_property = 'value') 
)

def update_my_graph4_1(val_choose_from_slider, ClicedAD_option_select):
    
    def adding_new_col(dataFrame_,newname):
        a = []
        for i in range(len(dataFrame_)):
            if df['Male'][i] == 0:
                a.append('Female')
            else:
                a.append('Male')
        dataFrame_[newname] = np.array(a)
        return dataFrame_

    

    if ClicedAD_option_select == 1:
        if val_choose_from_slider:
                  

            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            dff_with_click = adding_new_col(dff_with_click,'Gender')
            fig41 = px.violin(dff_with_click, y="Age", color="Gender", box=True, points="all")

            fig41.update_layout(title={
                'text': "<b>Distribution of Customers' Age per Gender</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            return fig41

        
    if ClicedAD_option_select == 0:
        if val_choose_from_slider:
                  
            #fig32 = plt.figure()
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            dff_with_click = adding_new_col(dff_with_click,'Gender')
            fig41 = px.violin(dff_with_click, y="Age", color="Gender", box=True, points="all")
            
            fig41.update_layout(title={
                'text': "<b>Distribution of Customers' Age per Gender</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            
            return fig41

        

    if val_choose_from_slider:
                  
        #fig32 = plt.figure()
        dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
        dff1 = adding_new_col(dff1,'Gender')
        fig41 = px.violin(dff1, y="Age", color="Gender", box=True, points="all")
        
        fig41.update_layout(title={
                'text': "<b>Distribution of Customers' Age per Gender</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        
        return fig41

    
    #-----------------------
    
    
@app.callback(
    Output(component_id = 'graph-output4_2',  component_property = 'figure'),
    Output(component_id = 'the_output_print_of_pvalue42_id',  component_property = 'children'),
     Input(component_id = 'SliderAge',  component_property = 'value'), 
     Input(component_id = 'ClicedAD',  component_property = 'value'),
     Input("kde_bw_selection42", "value")
    
)



def update_my_graph4_2(val_choose_from_slider, ClicedAD_option_select, BW):
    
    
    def adding_new_col(dataFrame_, newname):
        a = []
        for i in range(len(dataFrame_)):
            if df['Male'][i] == 0:
                a.append('Female')
            else:
                a.append('Male')
        dataFrame_[newname] = np.array(a)
        return dataFrame_

    

    if ClicedAD_option_select == 1:
        if val_choose_from_slider:
                  

            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            
            #dff_with_click = adding_new_col(dff_with_click,'Gender')
            #data_male = dff_with_click['Age'].where(dff_with_click['Gender'] == 'Male').dropna()
            
            
            data_male = dff_with_click['Age'].where(dff_with_click['Male'] == 1).dropna()
            kde_male = stats.gaussian_kde(data_male, bw_method=BW)
            data_x_male = np.linspace(10, 90)
            data_y_male = kde_male.evaluate(data_x_male)
            dic_male = {'data_x' : data_x_male, 'data_y' : data_y_male, 'Gender Age' : 'Male'}
            df_age_male = pd.DataFrame(dic_male)
            
            #data_female = dff_with_click['Age'].where(dff_with_click['Gender'] == 'Female').dropna()
            data_female = dff_with_click['Age'].where(dff_with_click['Male'] == 0).dropna()
            kde_fe = stats.gaussian_kde(data_female, bw_method=BW)
            data_x_female = np.linspace(10, 90)
            data_y_female = kde_fe.evaluate(data_x_female)
            dic_female = {'data_x' : data_x_female, 'data_y' : data_y_female, 'Gender Age' : 'Female'}
            df_age_female = pd.DataFrame(dic_female)
            
            
            
            df_age_both_male_female = pd.concat([df_age_male, df_age_female])
    
            fig42 = px.line(df_age_both_male_female, x = 'data_x', y = 'data_y',  color='Gender Age', labels={'data_x': 'Age', 'data_y': 'KDE'})
            
            #a , b = ttest_ind(data_y_male, data_y_female, equal_var=False)
            #print(a, b)
            
            
            fig42.update_layout(title={
                'text': "<b>KDE of Customers' Age per Gender</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            

            a , b_male_female = ttest_ind(data_y_male, data_y_female, equal_var=True)
            b_pvalue_Gender = "P-value Age's per Gender Male vs. Female = {:0.6}".format(b_male_female) 

            
            return fig42, b_pvalue_Gender


        
    if ClicedAD_option_select == 0:
        if val_choose_from_slider:
                  
            #fig32 = plt.figure()
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            
            #dff_with_click = adding_new_col(dff_with_click,'Gender')
            #data_male = dff_with_click['Age'].where(dff_with_click['Gender'] == 'Male').dropna()
            
            
            data_male = dff_with_click['Age'].where(dff_with_click['Male'] == 1).dropna()
            kde_male = stats.gaussian_kde(data_male, bw_method=BW)
            data_x_male = np.linspace(10, 90)
            data_y_male = kde_male.evaluate(data_x_male)
            dic_male = {'data_x' : data_x_male, 'data_y' : data_y_male, 'Gender Age' : 'Male'}
            df_age_male = pd.DataFrame(dic_male)
            
            #data_female = dff_with_click['Age'].where(dff_with_click['Gender'] == 'Female').dropna()
            data_female = dff_with_click['Age'].where(dff_with_click['Male'] == 0).dropna()
            kde_fe = stats.gaussian_kde(data_female, bw_method=BW)
            data_x_female = np.linspace(10, 90)
            data_y_female = kde_fe.evaluate(data_x_female)
            dic_female = {'data_x' : data_x_female, 'data_y' : data_y_female, 'Gender Age' : 'Female'}
            df_age_female = pd.DataFrame(dic_female)

            df_age_both_male_female = pd.concat([df_age_male, df_age_female])
    
            fig42 = px.line(df_age_both_male_female, x = 'data_x', y = 'data_y',  color='Gender Age', labels={'data_x': 'Age', 'data_y': 'KDE'})
            
            
            fig42.update_layout(title={
                'text': "<b>KDE of Customers' Age per Gender</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            
            a , b_male_female = ttest_ind(data_y_male, data_y_female, equal_var=True)
            b_pvalue_Gender = "P-value Age's per Gender Male vs. Female = {:0.6}".format(b_male_female) 

            
            return fig42, b_pvalue_Gender

        

    if val_choose_from_slider:
                  
        #fig32 = plt.figure()
        dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            
        #dff_with_click = adding_new_col(dff1,'Gender')
        #data_male = dff1['Age'].where(dff1['Gender'] == 'Male').dropna()
            
            
        data_male = dff1['Age'].where(dff1['Male'] == 1).dropna()
        kde_male = stats.gaussian_kde(data_male, bw_method=BW)
        data_x_male = np.linspace(10, 90)
        data_y_male = kde_male.evaluate(data_x_male)
        dic_male = {'data_x' : data_x_male, 'data_y' : data_y_male, 'Gender Age' : 'Male'}
        df_age_male = pd.DataFrame(dic_male)
            
        #data_female = dff1['Age'].where(dff1['Gender'] == 'Female').dropna()
        data_female = dff1['Age'].where(dff1['Male'] == 0).dropna()
        kde_fe = stats.gaussian_kde(data_female, bw_method=BW)
        data_x_female = np.linspace(10, 90)
        data_y_female = kde_fe.evaluate(data_x_female)
        dic_female = {'data_x' : data_x_female, 'data_y' : data_y_female, 'Gender Age' : 'Female'}
        df_age_female = pd.DataFrame(dic_female)

        df_age_both_male_female = pd.concat([df_age_male, df_age_female])
    
        fig42 = px.line(df_age_both_male_female, x = 'data_x', y = 'data_y',  color='Gender Age', labels={'data_x': 'Age', 'data_y': 'KDE'})
        
        fig42.update_layout(title={
                'text': "<b>KDE of Customers' Age per Gender</b>",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        
        a , b_male_female = ttest_ind(data_y_male, data_y_female, equal_var=True)
        b_pvalue_Gender = "P-value Age's per Gender Male vs. Female = {:0.6}".format(b_male_female) 

            
        return fig42, b_pvalue_Gender

 #------hypo for ages   

@app.callback(
    Output(component_id = 'graph-output5_1',  component_property = 'figure'),
     Input(component_id = 'SliderAge',  component_property = 'value'), 
     Input(component_id = 'ClicedAD',  component_property = 'value'),
     Input("id_GaussianMixture_n_components", "value")
)

def update_my_graph5_1(val_choose_from_slider, ClicedAD_option_select, Gausi_nComp51 ):

    

    if ClicedAD_option_select == 1:
        if val_choose_from_slider:
                  

            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            
            gmm2 = GaussianMixture(Gausi_nComp51)
            X_DailyTimeOnSite=np.array([dff_with_click.Age , dff_with_click['Daily Time Spent on Site'], dff_with_click['Clicked on Ad']]).T
            gmm2.fit(X_DailyTimeOnSite[:,:2])
            labels = gmm2.predict(X_DailyTimeOnSite[:,:2])
            frame = pd.DataFrame(X_DailyTimeOnSite)
            #predictions from gmm
            frame['Cluster'] = labels
            #a=symbol_clickAd(X_DailyTimeOnSite)
            #frame['Symbol'] = np.array(a)
            #frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster','Symbol']
            frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster']
            plt.figure(figsize=(7,7))
            #color=['blue', 'green']
            #fig51 = px.scatter(frame, x='Daily Time Spent on Site', y ='Age', color='Cluster', symbol = 'Clicked on Ad') #,color=color[k]
            #fig51.update_layout(legend_title_side="top left",legend_x=1.2, legend_y=1)
            #fig51.update_traces(marker=dict(size=15, line=dict(width=2)))
            
            fig51 = px.violin(frame, y="Daily Time Spent on Site", color="Cluster", box=True, points="all")
            
            fig51.update_layout(title={
                        'text': "<b>Distribution of Daily Time Spent on Site per Selected GMM Cluster</b>",
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})
            
            return fig51

        
    if ClicedAD_option_select == 0:
        if val_choose_from_slider:
                  
            #fig32 = plt.figure()
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            gmm2 = GaussianMixture(Gausi_nComp51)
            X_DailyTimeOnSite=np.array([dff_with_click.Age , dff_with_click['Daily Time Spent on Site'], dff_with_click['Clicked on Ad']]).T
            gmm2.fit(X_DailyTimeOnSite[:,:2])
            labels = gmm2.predict(X_DailyTimeOnSite[:,:2])
            frame = pd.DataFrame(X_DailyTimeOnSite)
            #predictions from gmm
            frame['Cluster'] = labels
            #a=symbol_clickAd(X_DailyTimeOnSite)
            #frame['Symbol'] = np.array(a)
            #frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster','Symbol']
            frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster']
            plt.figure(figsize=(7,7))
            #color=['blue', 'green']
            #fig51 = px.scatter(frame, x='Daily Time Spent on Site', y ='Age', color='Cluster', symbol = 'Clicked on Ad') #,color=color[k]
            #fig51.update_layout(legend_title_side="top left",legend_x=1.2, legend_y=1)
            #fig51.update_traces(marker=dict(size=15, line=dict(width=2)))
            
            fig51 = px.violin(frame, y="Daily Time Spent on Site", color="Cluster", box=True, points="all")
            
            
            fig51.update_layout(title={
                        'text': "<b>Distribution of Daily Time Spent on Site per Selected GMM Cluster</b>",
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})
            
            return fig51

        

    if val_choose_from_slider:
                  
        #fig32 = plt.figure()
        dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
        gmm2 = GaussianMixture(Gausi_nComp51)
        X_DailyTimeOnSite=np.array([dff1.Age , dff1['Daily Time Spent on Site'], dff1['Clicked on Ad']]).T
        gmm2.fit(X_DailyTimeOnSite[:,:2])
        labels = gmm2.predict(X_DailyTimeOnSite[:,:2])
        frame = pd.DataFrame(X_DailyTimeOnSite)
        #predictions from gmm
        frame['Cluster'] = labels
        #a=symbol_clickAd(X_DailyTimeOnSite)
        #frame['Symbol'] = np.array(a)
        #frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster','Symbol']
        frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster']
        plt.figure(figsize=(7,7))
        #color=['blue', 'green']
        #fig51 = px.scatter(frame, x='Daily Time Spent on Site', y ='Age', color='Cluster', symbol = 'Clicked on Ad') #,color=color[k]
        #fig51.update_layout(legend_title_side="top left",legend_x=1.2, legend_y=1)
        #fig51.update_traces(marker=dict(size=15, line=dict(width=2)))
            
        fig51 = px.violin(frame, y="Daily Time Spent on Site", color="Cluster", box=True, points="all")
        
        fig51.update_layout(title={
                        'text': "<b>Distribution of Daily Time Spent on Site per Selected GMM Cluster</b>",
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})
        
        return fig51

    
    #-----------------------
   # "the_output_print_of_pvalue52_id",the_input_pvalue52_id
    
@app.callback(
    Output(component_id = 'graph-output5_2',  component_property = 'figure'),
    Output(component_id = 'the_output_print_of_pvalue52_id',  component_property = 'children'),
     Input(component_id = 'SliderAge',  component_property = 'value'), 
     Input(component_id = 'ClicedAD',  component_property = 'value'),
     Input("kde_bw_selection52", "value"),
     Input("id_GaussianMixture_n_components", "value")        
)

def update_my_graph5_2(val_choose_from_slider, ClicedAD_option_select, BW, Gausi_nComp ):

    

    if ClicedAD_option_select == 1:
        if val_choose_from_slider:
                  

            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            
            
            gmm2 = GaussianMixture(Gausi_nComp)
            X_DailyTimeSiteCluster=np.array([dff_with_click.Age , dff_with_click['Daily Time Spent on Site'], dff_with_click['Clicked on Ad']]).T
            gmm2.fit(X_DailyTimeSiteCluster[:,:2])
            labels = gmm2.predict(X_DailyTimeSiteCluster[:,:2])
            frame = pd.DataFrame(X_DailyTimeSiteCluster)
            #predictions from gmm
            frame['Cluster'] = labels
            #a=symbol_clickAd(X_DailyTimeSiteCluster)
            #frame['Symbol'] = np.array(a)
            #frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster','Symbol']
            frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster']
            #plt.figure(figsize=(7,7))

            if Gausi_nComp == 1:
                
                data_cl0 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 0).dropna()
                kde_cl0 = stats.gaussian_kde(data_cl0, bw_method=BW) 
                data_x_cl0 = np.linspace(10, 90)
                data_y_cl0 = kde_cl0.evaluate(data_x_cl0)
                dic_cl0 = {'data_x' : data_x_cl0, 'data_y' : data_y_cl0, 'Group Cluster' : 0}
                df_TimeSpentSite_cl0 = pd.DataFrame(dic_cl0)
                df_TimeSpentSite_both_cl3_cl2_cl1_cl0 = pd.concat([df_TimeSpentSite_cl0])
                b = []
            
            if Gausi_nComp == 2:
                
                data_cl0 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 0).dropna()
                kde_cl0 = stats.gaussian_kde(data_cl0, bw_method=BW) 
                data_x_cl0 = np.linspace(10, 90)
                data_y_cl0 = kde_cl0.evaluate(data_x_cl0)
                dic_cl0 = {'data_x' : data_x_cl0, 'data_y' : data_y_cl0, 'Group Cluster' : 0}
                df_TimeSpentSite_cl0 = pd.DataFrame(dic_cl0)
            
            
                data_cl1 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 1).dropna()
                kde_cl1 = stats.gaussian_kde(data_cl1, bw_method=BW)
                data_x_cl1 = np.linspace(10, 90)
                data_y_cl1 = kde_cl1.evaluate(data_x_cl1)
                dic_cl1 = {'data_x' : data_x_cl1, 'data_y' : data_y_cl1, 'Group Cluster' : 1}
                df_TimeSpentSite_cl1 = pd.DataFrame(dic_cl1)
            
                df_TimeSpentSite_both_cl3_cl2_cl1_cl0 = pd.concat([df_TimeSpentSite_cl0, df_TimeSpentSite_cl1])
                a , b01 = ttest_ind(data_y_cl0, data_y_cl1, equal_var=False)
                b = "P-value cluster 0 vs 1 = {:0.2f}".format(b01)
                
            if Gausi_nComp == 3:
                data_cl0 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 0).dropna()
                kde_cl0 = stats.gaussian_kde(data_cl0, bw_method=BW) 
                data_x_cl0 = np.linspace(10, 90)
                data_y_cl0 = kde_cl0.evaluate(data_x_cl0)
                dic_cl0 = {'data_x' : data_x_cl0, 'data_y' : data_y_cl0, 'Group Cluster' : 0}
                df_TimeSpentSite_cl0 = pd.DataFrame(dic_cl0)
            
            
                data_cl1 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 1).dropna()
                kde_cl1 = stats.gaussian_kde(data_cl1, bw_method=BW)
                data_x_cl1 = np.linspace(10, 90)
                data_y_cl1 = kde_cl1.evaluate(data_x_cl1)
                dic_cl1 = {'data_x' : data_x_cl1, 'data_y' : data_y_cl1, 'Group Cluster' : 1}
                df_TimeSpentSite_cl1 = pd.DataFrame(dic_cl1)
                
                data_cl2 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 2).dropna()
                kde_cl2 = stats.gaussian_kde(data_cl2, bw_method=BW)
                data_x_cl2 = np.linspace(10, 90)
                data_y_cl2 = kde_cl2.evaluate(data_x_cl2)
                dic_cl2 = {'data_x' : data_x_cl2, 'data_y' : data_y_cl2, 'Group Cluster' : 2}
                df_TimeSpentSite_cl2 = pd.DataFrame(dic_cl2)
                
                df_TimeSpentSite_both_cl3_cl2_cl1_cl0 = pd.concat([df_TimeSpentSite_cl0, df_TimeSpentSite_cl1, df_TimeSpentSite_cl2])
                a , b01 = ttest_ind(data_y_cl0, data_y_cl1, equal_var=False)
                a , b02 = ttest_ind(data_y_cl0, data_y_cl2, equal_var=False)
                a , b12 = ttest_ind(data_y_cl1, data_y_cl2, equal_var=False)
                b = "P-value cluster 0 vs 1 = {:0.2f}, \
                  P-value cluster 0 vs 2 = {:0.2f},    \
                  P-value cluster 1 vs 2 = {:0.2f}".format(b01,b02,b12)
            
            if Gausi_nComp == 4:
                
                data_cl0 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 0).dropna()
                kde_cl0 = stats.gaussian_kde(data_cl0, bw_method=BW) 
                data_x_cl0 = np.linspace(10, 90)
                data_y_cl0 = kde_cl0.evaluate(data_x_cl0)
                dic_cl0 = {'data_x' : data_x_cl0, 'data_y' : data_y_cl0, 'Group Cluster' : 0}
                df_TimeSpentSite_cl0 = pd.DataFrame(dic_cl0)
            
            
                data_cl1 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 1).dropna()
                kde_cl1 = stats.gaussian_kde(data_cl1, bw_method=BW)
                data_x_cl1 = np.linspace(10, 90)
                data_y_cl1 = kde_cl1.evaluate(data_x_cl1)
                dic_cl1 = {'data_x' : data_x_cl1, 'data_y' : data_y_cl1, 'Group Cluster' : 1}
                df_TimeSpentSite_cl1 = pd.DataFrame(dic_cl1)
                
                data_cl2 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 2).dropna()
                kde_cl2 = stats.gaussian_kde(data_cl2, bw_method=BW)
                data_x_cl2 = np.linspace(10, 90)
                data_y_cl2 = kde_cl2.evaluate(data_x_cl2)
                dic_cl2 = {'data_x' : data_x_cl2, 'data_y' : data_y_cl2, 'Group Cluster' : 2}
                df_TimeSpentSite_cl2 = pd.DataFrame(dic_cl2)
            
                data_cl3 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 3).dropna()
                kde_cl3 = stats.gaussian_kde(data_cl3, bw_method=BW)
                data_x_cl3 = np.linspace(10, 90)
                data_y_cl3 = kde_cl3.evaluate(data_x_cl3)
                dic_cl3 = {'data_x' : data_x_cl3, 'data_y' : data_y_cl3, 'Group Cluster' : 3}
                df_TimeSpentSite_cl3 = pd.DataFrame(dic_cl3)
            
                df_TimeSpentSite_both_cl3_cl2_cl1_cl0 = pd.concat([df_TimeSpentSite_cl0, df_TimeSpentSite_cl1, df_TimeSpentSite_cl2, df_TimeSpentSite_cl3])
                
                a , b01 = ttest_ind(data_y_cl0, data_y_cl1, equal_var=False)
                a , b02 = ttest_ind(data_y_cl0, data_y_cl2, equal_var=False)
                a , b03 = ttest_ind(data_y_cl0, data_y_cl3, equal_var=False)
                a , b12 = ttest_ind(data_y_cl1, data_y_cl2, equal_var=False)
                a , b13 = ttest_ind(data_y_cl1, data_y_cl3, equal_var=False)
                a , b23 = ttest_ind(data_y_cl2, data_y_cl3, equal_var=False)
                b = "P-value cluster 0 vs 1 = {:0.2f}, \
                      P-value cluster 0 vs 2 = {:0.2f}, \
                      P-value cluster 0 vs 3 = {:0.2f}, \
                      P-value cluster 1 vs 2 = {:0.2f}, \
                      P-value cluster 1 vs 2 = {:0.2f}, \
                      P-value cluster 2 vs 3 = {:0.2f}".format(b01,b02,b03,b12,b13,b23)
    
            fig52 = px.line(df_TimeSpentSite_both_cl3_cl2_cl1_cl0, x = 'data_x', y = 'data_y',  color='Group Cluster',
                            labels={'data_x': 'Daily Time Spent on Site', 'data_y': 'KDE'}
                            #title = 'Daily Time Spent on Site KDE per Selected Cluster'
                           )
            fig52.update_layout(title={
                            'text': "<b>Daily Time Spent on Site KDE per Selected GMM Cluster</b>",
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'})
            return fig52, b
        
    if ClicedAD_option_select == 0:
        if val_choose_from_slider:
                  
            #fig32 = plt.figure()
            dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
            dff_with_click = dff1[dff1['Clicked on Ad'] == ClicedAD_option_select]
            gmm2 = GaussianMixture(Gausi_nComp)
            X_DailyTimeSiteCluster=np.array([dff_with_click.Age , dff_with_click['Daily Time Spent on Site'], dff_with_click['Clicked on Ad']]).T
            gmm2.fit(X_DailyTimeSiteCluster[:,:2])
            labels = gmm2.predict(X_DailyTimeSiteCluster[:,:2])
            frame = pd.DataFrame(X_DailyTimeSiteCluster)
            #predictions from gmm
            frame['Cluster'] = labels
            #a=symbol_clickAd(X_DailyTimeSiteCluster)
            #frame['Symbol'] = np.array(a)
            #frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster','Symbol']
            frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster']
            
            if Gausi_nComp == 1:
                
                data_cl0 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 0).dropna()
                kde_cl0 = stats.gaussian_kde(data_cl0, bw_method=BW) 
                data_x_cl0 = np.linspace(10, 90)
                data_y_cl0 = kde_cl0.evaluate(data_x_cl0)
                dic_cl0 = {'data_x' : data_x_cl0, 'data_y' : data_y_cl0, 'Group Cluster' : 0}
                df_TimeSpentSite_cl0 = pd.DataFrame(dic_cl0)
                df_TimeSpentSite_both_cl3_cl2_cl1_cl0 = pd.concat([df_TimeSpentSite_cl0])
                b = []
            
            if Gausi_nComp == 2:
                
                data_cl0 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 0).dropna()
                kde_cl0 = stats.gaussian_kde(data_cl0, bw_method=BW) 
                data_x_cl0 = np.linspace(10, 90)
                data_y_cl0 = kde_cl0.evaluate(data_x_cl0)
                dic_cl0 = {'data_x' : data_x_cl0, 'data_y' : data_y_cl0, 'Group Cluster' : 0}
                df_TimeSpentSite_cl0 = pd.DataFrame(dic_cl0)
            
            
                data_cl1 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 1).dropna()
                kde_cl1 = stats.gaussian_kde(data_cl1, bw_method=BW)
                data_x_cl1 = np.linspace(10, 90)
                data_y_cl1 = kde_cl1.evaluate(data_x_cl1)
                dic_cl1 = {'data_x' : data_x_cl1, 'data_y' : data_y_cl1, 'Group Cluster' : 1}
                df_TimeSpentSite_cl1 = pd.DataFrame(dic_cl1)
            
                df_TimeSpentSite_both_cl3_cl2_cl1_cl0 = pd.concat([df_TimeSpentSite_cl0, df_TimeSpentSite_cl1])
                a , b01 = ttest_ind(data_y_cl0, data_y_cl1, equal_var=False)
                b = "P-value for cluster 0 vs 1 {:0.2f}".format(b01)
                
            if Gausi_nComp == 3:
                data_cl0 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 0).dropna()
                kde_cl0 = stats.gaussian_kde(data_cl0, bw_method=BW) 
                data_x_cl0 = np.linspace(10, 90)
                data_y_cl0 = kde_cl0.evaluate(data_x_cl0)
                dic_cl0 = {'data_x' : data_x_cl0, 'data_y' : data_y_cl0, 'Group Cluster' : 0}
                df_TimeSpentSite_cl0 = pd.DataFrame(dic_cl0)
            
            
                data_cl1 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 1).dropna()
                kde_cl1 = stats.gaussian_kde(data_cl1, bw_method=BW)
                data_x_cl1 = np.linspace(10, 90)
                data_y_cl1 = kde_cl1.evaluate(data_x_cl1)
                dic_cl1 = {'data_x' : data_x_cl1, 'data_y' : data_y_cl1, 'Group Cluster' : 1}
                df_TimeSpentSite_cl1 = pd.DataFrame(dic_cl1)
                
                data_cl2 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 2).dropna()
                kde_cl2 = stats.gaussian_kde(data_cl2, bw_method=BW)
                data_x_cl2 = np.linspace(10, 90)
                data_y_cl2 = kde_cl2.evaluate(data_x_cl2)
                dic_cl2 = {'data_x' : data_x_cl2, 'data_y' : data_y_cl2, 'Group Cluster' : 2}
                df_TimeSpentSite_cl2 = pd.DataFrame(dic_cl2)
                
                df_TimeSpentSite_both_cl3_cl2_cl1_cl0 = pd.concat([df_TimeSpentSite_cl0, df_TimeSpentSite_cl1, df_TimeSpentSite_cl2])
                a , b01 = ttest_ind(data_y_cl0, data_y_cl1, equal_var=False)
                a , b02 = ttest_ind(data_y_cl0, data_y_cl2, equal_var=False)
                a , b12 = ttest_ind(data_y_cl1, data_y_cl2, equal_var=False)
                b = "P-value cluster 0 vs 1 = {:0.2f}, \
                  P-value cluster 0 vs 2 = {:0.2f}, \
                  P-value cluster 1 vs 2 = {:0.2f},".format(b01,b02,b12)
            
            if Gausi_nComp == 4:
                
                data_cl0 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 0).dropna()
                kde_cl0 = stats.gaussian_kde(data_cl0, bw_method=BW) 
                data_x_cl0 = np.linspace(10, 90)
                data_y_cl0 = kde_cl0.evaluate(data_x_cl0)
                dic_cl0 = {'data_x' : data_x_cl0, 'data_y' : data_y_cl0, 'Group Cluster' : 0}
                df_TimeSpentSite_cl0 = pd.DataFrame(dic_cl0)
            
            
                data_cl1 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 1).dropna()
                kde_cl1 = stats.gaussian_kde(data_cl1, bw_method=BW)
                data_x_cl1 = np.linspace(10, 90)
                data_y_cl1 = kde_cl1.evaluate(data_x_cl1)
                dic_cl1 = {'data_x' : data_x_cl1, 'data_y' : data_y_cl1, 'Group Cluster' : 1}
                df_TimeSpentSite_cl1 = pd.DataFrame(dic_cl1)
                
                data_cl2 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 2).dropna()
                kde_cl2 = stats.gaussian_kde(data_cl2, bw_method=BW)
                data_x_cl2 = np.linspace(10, 90)
                data_y_cl2 = kde_cl2.evaluate(data_x_cl2)
                dic_cl2 = {'data_x' : data_x_cl2, 'data_y' : data_y_cl2, 'Group Cluster' : 2}
                df_TimeSpentSite_cl2 = pd.DataFrame(dic_cl2)
            
                data_cl3 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 3).dropna()
                kde_cl3 = stats.gaussian_kde(data_cl3, bw_method=BW)
                data_x_cl3 = np.linspace(10, 90)
                data_y_cl3 = kde_cl3.evaluate(data_x_cl3)
                dic_cl3 = {'data_x' : data_x_cl3, 'data_y' : data_y_cl3, 'Group Cluster' : 3}
                df_TimeSpentSite_cl3 = pd.DataFrame(dic_cl3)
            
                df_TimeSpentSite_both_cl3_cl2_cl1_cl0 = pd.concat([df_TimeSpentSite_cl0, df_TimeSpentSite_cl1, df_TimeSpentSite_cl2, df_TimeSpentSite_cl3])
                
                a , b01 = ttest_ind(data_y_cl0, data_y_cl1, equal_var=False)
                a , b02 = ttest_ind(data_y_cl0, data_y_cl2, equal_var=False)
                a , b03 = ttest_ind(data_y_cl0, data_y_cl3, equal_var=False)
                a , b12 = ttest_ind(data_y_cl1, data_y_cl2, equal_var=False)
                a , b13 = ttest_ind(data_y_cl1, data_y_cl3, equal_var=False)
                a , b23 = ttest_ind(data_y_cl2, data_y_cl3, equal_var=False)
                b = "P-value cluster 0 vs 1 = {:0.2f}, \
                      P-value cluster 0 vs 2 = {:0.2f}, \
                      P-value cluster 0 vs 3 = {:0.2f}, \
                      P-value cluster 1 vs 2 = {:0.2f}, \
                      P-value cluster 1 vs 2 = {:0.2f}, \
                      P-value cluster 2 vs 3 = {:0.2f}".format(b01,b02,b03,b12,b13,b23)
    
            fig52 = px.line(df_TimeSpentSite_both_cl3_cl2_cl1_cl0, x = 'data_x', y = 'data_y',  color='Group Cluster',
                            labels={'data_x': 'Daily Time Spent on Site', 'data_y': 'KDE'}
                            #title = 'Daily Time Spent on Site KDE per Selected Cluster'
                           )
            fig52.update_layout(title={
                            'text': "<b>Daily Time Spent on Site KDE per Selected GMM Cluster</b>",
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'})
            return fig52, b

        

    if val_choose_from_slider:
                  
        #fig32 = plt.figure()
        dff1 = df[(df.Age >= val_choose_from_slider[0]) & (df.Age <= val_choose_from_slider[1])]
        gmm2 = GaussianMixture(Gausi_nComp)
        X_DailyTimeSiteCluster=np.array([dff1.Age , dff1['Daily Time Spent on Site'], dff1['Clicked on Ad']]).T
        gmm2.fit(X_DailyTimeSiteCluster[:,:2])
        labels = gmm2.predict(X_DailyTimeSiteCluster[:,:2])
        frame = pd.DataFrame(X_DailyTimeSiteCluster)
        #predictions from gmm
        frame['Cluster'] = labels
        frame.columns = ['Age', 'Daily Time Spent on Site', 'Clicked on Ad','Cluster']
        
        if Gausi_nComp == 1:
            
            data_cl0 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 0).dropna()
            kde_cl0 = stats.gaussian_kde(data_cl0, bw_method=BW) 
            data_x_cl0 = np.linspace(10, 90)
            data_y_cl0 = kde_cl0.evaluate(data_x_cl0)
            dic_cl0 = {'data_x' : data_x_cl0, 'data_y' : data_y_cl0, 'Group Cluster' : 0}
            df_TimeSpentSite_cl0 = pd.DataFrame(dic_cl0)
            df_TimeSpentSite_both_cl3_cl2_cl1_cl0 = pd.concat([df_TimeSpentSite_cl0])
            b = []
        
        if Gausi_nComp == 2:
            
            data_cl0 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 0).dropna()
            kde_cl0 = stats.gaussian_kde(data_cl0, bw_method=BW) 
            data_x_cl0 = np.linspace(10, 90)
            data_y_cl0 = kde_cl0.evaluate(data_x_cl0)
            dic_cl0 = {'data_x' : data_x_cl0, 'data_y' : data_y_cl0, 'Group Cluster' : 0}
            df_TimeSpentSite_cl0 = pd.DataFrame(dic_cl0)
        
        
            data_cl1 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 1).dropna()
            kde_cl1 = stats.gaussian_kde(data_cl1, bw_method=BW)
            data_x_cl1 = np.linspace(10, 90)
            data_y_cl1 = kde_cl1.evaluate(data_x_cl1)
            dic_cl1 = {'data_x' : data_x_cl1, 'data_y' : data_y_cl1, 'Group Cluster' : 1}
            df_TimeSpentSite_cl1 = pd.DataFrame(dic_cl1)
        
            df_TimeSpentSite_both_cl3_cl2_cl1_cl0 = pd.concat([df_TimeSpentSite_cl0, df_TimeSpentSite_cl1])
            a , b01 = ttest_ind(data_y_cl0, data_y_cl1, equal_var=False)
            b = "P-value cluster 0 vs 1 = {:0.2f}".format(b01)
            
        if Gausi_nComp == 3:
            data_cl0 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 0).dropna()
            kde_cl0 = stats.gaussian_kde(data_cl0, bw_method=BW) 
            data_x_cl0 = np.linspace(10, 90)
            data_y_cl0 = kde_cl0.evaluate(data_x_cl0)
            dic_cl0 = {'data_x' : data_x_cl0, 'data_y' : data_y_cl0, 'Group Cluster' : 0}
            df_TimeSpentSite_cl0 = pd.DataFrame(dic_cl0)
        
        
            data_cl1 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 1).dropna()
            kde_cl1 = stats.gaussian_kde(data_cl1, bw_method=BW)
            data_x_cl1 = np.linspace(10, 90)
            data_y_cl1 = kde_cl1.evaluate(data_x_cl1)
            dic_cl1 = {'data_x' : data_x_cl1, 'data_y' : data_y_cl1, 'Group Cluster' : 1}
            df_TimeSpentSite_cl1 = pd.DataFrame(dic_cl1)
            
            data_cl2 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 2).dropna()
            kde_cl2 = stats.gaussian_kde(data_cl2, bw_method=BW)
            data_x_cl2 = np.linspace(10, 90)
            data_y_cl2 = kde_cl2.evaluate(data_x_cl2)
            dic_cl2 = {'data_x' : data_x_cl2, 'data_y' : data_y_cl2, 'Group Cluster' : 2}
            df_TimeSpentSite_cl2 = pd.DataFrame(dic_cl2)
            
            df_TimeSpentSite_both_cl3_cl2_cl1_cl0 = pd.concat([df_TimeSpentSite_cl0, df_TimeSpentSite_cl1, df_TimeSpentSite_cl2])
            a , b01 = ttest_ind(data_y_cl0, data_y_cl1, equal_var=False)
            a , b02 = ttest_ind(data_y_cl0, data_y_cl2, equal_var=False)
            a , b12 = ttest_ind(data_y_cl1, data_y_cl2, equal_var=False)
            b = "P-value cluster 0 vs 1 = {:0.2f}, \
                  P-value cluster 0 vs 2 = {:0.2f}, \
                  P-value cluster 1 vs21 = {:0.2f}".format(b01,b02,b12)
        
        if Gausi_nComp == 4:
            
            data_cl0 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 0).dropna()
            kde_cl0 = stats.gaussian_kde(data_cl0, bw_method=BW) 
            data_x_cl0 = np.linspace(10, 90)
            data_y_cl0 = kde_cl0.evaluate(data_x_cl0)
            dic_cl0 = {'data_x' : data_x_cl0, 'data_y' : data_y_cl0, 'Group Cluster' : 0}
            df_TimeSpentSite_cl0 = pd.DataFrame(dic_cl0)
        
        
            data_cl1 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 1).dropna()
            kde_cl1 = stats.gaussian_kde(data_cl1, bw_method=BW)
            data_x_cl1 = np.linspace(10, 90)
            data_y_cl1 = kde_cl1.evaluate(data_x_cl1)
            dic_cl1 = {'data_x' : data_x_cl1, 'data_y' : data_y_cl1, 'Group Cluster' : 1}
            df_TimeSpentSite_cl1 = pd.DataFrame(dic_cl1)
            
            data_cl2 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 2).dropna()
            kde_cl2 = stats.gaussian_kde(data_cl2, bw_method=BW)
            data_x_cl2 = np.linspace(10, 90)
            data_y_cl2 = kde_cl2.evaluate(data_x_cl2)
            dic_cl2 = {'data_x' : data_x_cl2, 'data_y' : data_y_cl2, 'Group Cluster' : 2}
            df_TimeSpentSite_cl2 = pd.DataFrame(dic_cl2)
        
            data_cl3 = frame['Daily Time Spent on Site'].where(frame['Cluster'] == 3).dropna()
            kde_cl3 = stats.gaussian_kde(data_cl3, bw_method=BW)
            data_x_cl3 = np.linspace(10, 90)
            data_y_cl3 = kde_cl3.evaluate(data_x_cl3)
            dic_cl3 = {'data_x' : data_x_cl3, 'data_y' : data_y_cl3, 'Group Cluster' : 3}
            df_TimeSpentSite_cl3 = pd.DataFrame(dic_cl3)
        
            df_TimeSpentSite_both_cl3_cl2_cl1_cl0 = pd.concat([df_TimeSpentSite_cl0, df_TimeSpentSite_cl1, df_TimeSpentSite_cl2, df_TimeSpentSite_cl3])
            
            a , b01 = ttest_ind(data_y_cl0, data_y_cl1, equal_var=False)
            a , b02 = ttest_ind(data_y_cl0, data_y_cl2, equal_var=False)
            a , b03 = ttest_ind(data_y_cl0, data_y_cl3, equal_var=False)
            a , b12 = ttest_ind(data_y_cl1, data_y_cl2, equal_var=False)
            a , b13 = ttest_ind(data_y_cl1, data_y_cl3, equal_var=False)
            a , b23 = ttest_ind(data_y_cl2, data_y_cl3, equal_var=False)
            b = "P-value cluster 0 vs 1 = {:0.2f}, \
                  P-value cluster 0 vs 2 = {:0.2f}, \
                  P-value cluster 0 vs 3 = {:0.2f}, \
                  P-value cluster 1 vs 2 = {:0.2f}, \
                  P-value cluster 1 vs 2 = {:0.2f}, \
                  P-value cluster 2 vs 3 = {:0.2f}".format(b01,b02,b03,b12,b13,b23)

        fig52 = px.line(df_TimeSpentSite_both_cl3_cl2_cl1_cl0, x = 'data_x', y = 'data_y',  color='Group Cluster',
                        labels={'data_x': 'Daily Time Spent on Site', 'data_y': 'KDE'}
                        #title = 'Daily Time Spent on Site KDE per Selected Cluster'
                       )
        fig52.update_layout(title={
                        'text': "<b>Daily Time Spent on Site KDE per Selected GMM Cluster</b>",
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})
        return fig52, b
    

server = app.server
@server.route('/static/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(os.path.join(root_dir, 'static'), path)   
    
    
    
if __name__=='__main__':
    #app.run_server(debug=False, host="127.0.0.1", port=8076, threaded=True)
      app.run_server(debug=False, host="0.0.0.0", port=8076, threaded=True)