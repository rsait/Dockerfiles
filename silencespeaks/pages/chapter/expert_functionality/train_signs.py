from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import base64
import dash
import cv2
import numpy as np
import pandas as pd
from pages.chapter.user_functionality import help_functions
from pages.chapter.user_functionality import plot_functions
import pickle as pkl
from pages.chapter.expert import navbar_expert

dash.register_page(__name__, path='/expert/signs')

configurations = ['4','50','58','59','73','74','77','78']
signs = ['BIEN','CONTENTO','HOMRE','MUJER','OYENTE']

layout = html.Div([
    navbar_expert,
    html.Img(id='video', src='/user/configurations/video_mediapipe_feed'),
])