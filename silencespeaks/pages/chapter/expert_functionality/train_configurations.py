from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash
import numpy as np
import pickle as pkl
from pages.chapter.expert import navbar_expert
from pages.chapter.expert_functionality import medoid_functions
from pages.chapter.user_functionality import help_functions
import plotly.graph_objects as go
import base64

#from pages.chapter.user_functionality.configurations import prediction


dash.register_page(__name__, path='/expert/configurations')

from global_ import configurations
#configurations = ['4','50','58','59','73','74','77','78']
configuration = '4'
data = pkl.load(open('dataset/configs/' + configuration + '/itsaso_newCam.pkl','rb'))
medoid = medoid_functions.get_shape_medoid(data)
medoids = [help_functions.transform_data(medoid_functions.get_shape_medoid(
           pkl.load(open('dataset/configs/' + config + '/itsaso_newCam.pkl','rb')))) for config in configurations]
# pkl.dump(medoids, open('dataset/configs/itsaso_newCam_medoids.pkl','wb'))
fingers = ['thumb','index','middle','ring','pinky']

layout = html.Div([
    navbar_expert,
    html.Div([
        html.Img(id='video', src='/user/configurations/video_mediapipe_feed'),
        #html.Br(),
        html.H3('Choose a method to perform the classification'),
        dcc.RadioItems(
            id = 'classify-method',
            options=[
                {'label':'Medoids','value':'medoid'},
                {'label':'Classifiers','value':'classifier'},
            ],
            #value='conf',
            inputStyle={"margin-right": "5px", 'cursor': 'pointer', 'margin-left':'20px'}
        )
    ], style = {'width':'49%','display':'inline-block'}),
    html.Div([
        html.Div([
            html.H3('Choose features to train the classifier')
        ], id='div-classifier', style={'display':'none'}),
        html.Div([
            html.H3('MEDOID'),
            dcc.Dropdown(
                id='config-medoid',
                options=[dict((('label',config), ('value',config))) for config in configurations],
                # [
                #     {'label': '4', 'value':'4'},
                #     {'label': '50', 'value':'50'},
                #     {'label': '58', 'value':'58'},
                #     {'label': '59', 'value':'59'},
                #     {'label': '73', 'value':'73'},
                #     {'label': '74', 'value':'74'},
                #     {'label': '77', 'value':'77'},
                #     {'label': '78', 'value':'78'}
                # ],
                # placeholder='Select configuration...',
                value = '4',
                style = {'width':'130%'}
            ),
            html.Div(id='textarea-classif', style={'whiteSpace': 'pre'}),
            dcc.Graph(
                id='graph-medoid',
                figure=medoid_functions.obtain_graph(medoid_functions.get_shape_medoid(pkl.load(open('dataset/configs/4/itsaso_newCam.pkl','rb')))),
                responsive=True,
                style={
                    'width': '130%',
                    'height': '130%',
                    'display':'inline-block'
                }
            ),
            html.Div(html.Img(id='img-config-medoid',src='data:image/png;base64,{}'.format(base64.b64encode(open("dataset/configs/img/4.png", 'rb').read()).decode()),style={'height':'25%', 'width':'25%'})),
                    
            dcc.Interval('interval-prediction',
                        interval=0.5*1000,
                        n_intervals=0
            ),
        ], id='div-medoid', style={'display':'none'})
    ], style = {'display':'inline-block', 'width':'49%'}),
    html.Div(id='textarea-prediction-output', style={'whiteSpace': 'pre'}),
])

@callback([Output('div-classifier','style'), Output('div-medoid','style')],
           Input('classify-method','value'))
def classify_method(value):
    
    show = {'display':'inline-block', 'width':'49%'}
    hide = {'display':'none', 'width':'49%'}

    if value == 'medoid':
        return hide, show
    elif value == 'classifier':
        return show, hide
    else:
        return hide, hide


@callback([Output('graph-medoid','figure'),Output('graph-medoid','style'),
           Output('img-config-medoid','src')],
           Input('config-medoid','value'), prevent_initial_call=True)
def show_medoid_graph(configuration):

    data = pkl.load(open('dataset/configs/' + configuration + '/itsaso_newCam.pkl','rb'))
    medoid = medoid_functions.get_shape_medoid(data)
    style = {
                'width': '100%',
                'height': '100%',
                'display':'inline-block'
            }

    src_img = 'data:image/png;base64,{}'.format(base64.b64encode(open("dataset/configs/img/"+configuration+".png", 'rb').read()).decode())
    return medoid_functions.obtain_graph(medoid), style, src_img

@callback([Output('textarea-prediction-output','children'), Output('textarea-prediction-output','style')],
          Input('interval-prediction','n_intervals'),
          [State('classify-method','value'), State('config-medoid','value'), State('textarea-prediction-output','style')])
def make_prediction_medoid(interval, classify_method, config_value, actual_style):
    if classify_method == 'medoid':
        from global_ import landmarks
        if landmarks is not None:
            landmarks_to_predict = help_functions.transform_data(landmarks)

            dists = [medoid_functions.procrustes_disparity(actual_medoid, landmarks_to_predict) for actual_medoid in medoids]
            min_index = int(np.where(dists == np.amin(dists))[0])
            result = configurations[min_index]

            if result == config_value:
                style = {'backgroundColor':'#99FF99'}
            else:
                style = {'backgroundColor':'#FF9999'}

            # #tr1, tr2 = medoid_functions.procrustes_disparity_transformed_matrices(medoids[min_index],landmarks_to_predict)
            # for finger in range(5):
            #     medoid_finger = medoids[min_index][finger*4+1:finger*4+5,:]
            #     landmark_finger = landmarks_to_predict[finger*4+1:finger*4+4+1,:] 
            #     print(fingers[finger])
            #     print(medoid_finger)
            #     print(landmark_finger)
            #     close = np.allclose(medoid_finger,landmark_finger, rtol=0, atol=1e-01) #rtol=1e-03, atol=1e-01)
            #     print(close)
            #     print('--------------------------------')
            #     if not close:
            #         result = result + ' - Please pay attention to your ' + fingers[finger] + ' finger'

            return result, style

    return '', actual_style