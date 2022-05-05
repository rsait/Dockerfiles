from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import base64
import dash
import cv2
import numpy as np
import pandas as pd
import pickle as pkl
from pages.chapter.expert import navbar_expert

dash.register_page(__name__, path='/expert/record')

#configurations = ['4','50','58','59','73','74','77','78']
#configurations = ['{}'.format(x) for x in range(1,43)]
from global_ import configurations
signs = ['BIEN','CONTENTO','HOMRE','MUJER','OYENTE']

layout = html.Div([
    navbar_expert,
    html.Img(id='video', src='/user/configurations/video_mediapipe_feed'),
    html.Br(),
    html.Br(),
    html.Div([
        html.H4('What do you want to record?'),
        dcc.RadioItems(
            id = 'record-what',
            options=[
                {'label':'Configurations','value':'conf'},
                {'label':'Signs','value':'sign'},
            ],
            value='conf',
            inputStyle={"margin-right": "5px", 'cursor': 'pointer', 'margin-left':'20px'}
        )
    ]),
    html.Br(),
    html.Div([
        html.H3('Select class to record'),
        dcc.Dropdown(
            id='record-config',
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
            value='1',
        ),
        dcc.Dropdown(
            id='record-sign',
            options=[
                {'label': 'BIEN', 'value':'BIEN'},
                {'label': 'CONTENTO', 'value':'CONTENTO'},
                {'label': 'HOMBRE', 'value':'HOMBRE'},
                {'label': 'MUJER', 'value':'MUJER'},
                {'label': 'OYENTE', 'value':'OYENTE'},
            ],
            value='BIEN',
        ),
        html.Br(),
        html.Div([
            html.H4('Do you also want to save the video from which landmarks are extracted?'),
            dcc.RadioItems(
                id = 'record-landmarks-video',
                options=[
                    {'label':'Yes','value':'Yes'},
                    {'label':'No','value':'No'},
                ],
                value='No',
                inputStyle={"margin-right": "5px", 'cursor': 'pointer', 'margin-left':'20px'}
            )
        ]),
        html.Br(),
        dcc.Input(
            id='input-on-submit', 
            type='text',
            placeholder='Enter filename and click record button...',
            style={'float': 'left','margin': 'auto','margin-right':'20px', 'width':'30%'}
        ),
        dbc.Button(children='Record',
            id='record-button',
            color='primary',
        ),
        html.Br(),
        html.Div(id='textarea-record-output', style={'whiteSpace': 'pre'}),
        dcc.Store('record-or-not', data={'record':False}),
        dcc.Store('saved-landmarks',data={'land':[],'sign':[],'vid':[]}),
        dcc.Interval('interval-recording',
            interval=0.5*1000,
            n_intervals=0)
    ])
])

@callback([Output('record-config','style'),Output('record-sign','style')],
            Input('record-what','value'))
def change_dropdown(value):
    show = {'display':'block', 'width':'45%'}
    hide = {'display':'none'}
    if (value=='conf'):
        return show, hide
    else:
        return hide, show

@callback(Output('record-button','disabled'), Input('input-on-submit','value'))
def enable_button(input_name):
    if input_name is None or input_name=='':
        return True
    else:
        return False

@callback([Output('record-or-not','data'),Output('record-button','color'),
           Output('record-button','children'), Output('input-on-submit','disabled')],
           Input('record-button','n_clicks'),
          prevent_initial_call=True)
def record_landmarks(n):
    if(n%2 == 1):
        recording = {'record':True}
        color = 'danger'
        text_button = 'Recording'
        input_disabled = True
    else:
        recording = {'record':False}
        color = 'primary'
        text_button = 'Record'
        input_disabled = False

    return recording, color, text_button, input_disabled

@callback([Output('textarea-record-output','children'), Output('saved-landmarks','data')],
          Input('interval-recording','n_intervals'),
          [State('input-on-submit','value'), State('record-landmarks-video','value'),
          State('record-or-not','data'), State('saved-landmarks','data'),
          State('record-what','value'),State('record-config','value'),State('record-sign','value')])
def save_recordings(interval, filename_landmarks, record_video, recording, saved_landmarks, what, selected_conf, selected_sign):

    if (recording['record']==True):
        text = 'Recording. Frame: ' + str(np.array(saved_landmarks['land']).shape[0]) + ' - File: ' + filename_landmarks + '.'

        from global_ import landmarks
        if landmarks is not None:
            saved_landmarks['land'].append(landmarks)
            if record_video == 'Yes':
                from global_ import frame
                saved_landmarks['vid'].append(list(frame))

            # Video == 25 frames
            if what =='sign' and np.array(saved_landmarks['land']).shape[0]==25:
                saved_landmarks['sign'].append(saved_landmarks['land'])
                saved_landmarks['land'] = []                
                text = 'Recording. Video: ' + str(np.array(saved_landmarks['sign']).shape[0]) + ' - Frame: ' + str(np.array(saved_landmarks['land']).shape[0]) + ' - File: ' + filename_landmarks + '.'
        
    else:
        text = 'Not recording.'
        if saved_landmarks['land']:
            text = 'Landmarks saved in ' + filename_landmarks + '.pkl file.'

            if what=='conf':
                file_path = 'dataset/configs/' + selected_conf + '/' + filename_landmarks
                pkl.dump(np.array(saved_landmarks['land']), open(file_path + '.pkl','wb'))
            else:
                file_path = 'dataset/signs/' + selected_sign + '/' + filename_landmarks
                pkl.dump(np.array(saved_landmarks['sign']), open(file_path + '.pkl','wb'))

            if record_video == 'Yes':
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(file_path + '.avi', fourcc, 5, (640,480))
                for actual_frame in saved_landmarks['vid']:
                    #ret, jpeg = cv2.imencode('.jpg', np.array(actual_frame))
                    out.write(cv2.imdecode(np.uint8(actual_frame), cv2.IMREAD_COLOR))
                out.release()
                text = text + ' Video frames saved in ' + filename_landmarks + '.avi file.'
            
        saved_landmarks['land'] = []
        saved_landmarks['sign'] = []
        saved_landmarks['vid'] = []
        
    return text, saved_landmarks