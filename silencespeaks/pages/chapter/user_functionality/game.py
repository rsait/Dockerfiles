# from dash import dcc, html, Input, Output, State, callback
# import dash_bootstrap_components as dbc
# import base64
# import dash
# import numpy as np
# from pages.chapter.user_functionality import help_functions
# import pickle as pkl
# from pages.chapter.user import navbar_user
# from scipy import stats

# dash.register_page(__name__, path='/user/game')

# config_names=np.array(['4','50','58','59','73','74','77','78'])
# obtained_random_config_name = np.random.choice(config_names)
# image_filename = 'pages/apps/images/'+obtained_random_config_name+'.png'
# encoded_image = base64.b64encode(open(image_filename, 'rb').read())


# gesture_names=np.array(['BIEN','CONTENTO','HOMBRE','MUJER','OYENTE'])
# gesture_now = np.random.choice(gesture_names)

# path = "/home/bee/Documents/multipage_app/apps/PRUEBITAS_ACC/"
# clf_path, features_index_conf, added_dist_conf = help_functions.best_clf(path, model_type='configs', flatten=True)
# clf = pkl.load(open(clf_path,"rb"))
# configurations = ['4','50','58','59','73','74','77','78']

# hmm_path, clf_path_config_HMM, features_index_hmm, added_dist_hmm = help_functions.best_clf(path, model_type='signs', flatten=True)
# clfs_hmm = pkl.load(open(hmm_path,"rb"))
# clf_config_HMM = pkl.load(open(clf_path_config_HMM,"rb"))
# signs = ['BIEN','CONTENTO','HOMBRE','MUJER','OYENTE']

# layout = html.Div([
#     navbar_user,
#     html.Div([
#         html.H3('What do you want to practice?'),
#         dcc.RadioItems(
#             id = 'radioItem-game',
#             options=[
#                 {'label':'Configurations','value':'conf'},
#                 {'label':'Gestures','value':'sign'},
#             ],
#             value='conf',
#             inputStyle={"margin-right": "5px", 'cursor': 'pointer', 'margin-left':'20px'}
#         ),
#         html.Br(),
#         dbc.Button('Start',
#             id='btn-game',
#             #children='Start',
#             n_clicks=0,
#             color='primary',
#             outline=True,
#             style={'width':'230px','margin-left':'20px'}
#         )
#     ], style={'display':'block',"margin-left": "18px"}),
#     html.Br(),        
#     html.Div([
#         # Camara mediapipe
#         html.Img(id='video',src="/user/configurations/video_mediapipe_feed"),
#         # Imagen de la configuración a practicar
#         html.Div([
#             html.Img(
#                 id='img-sign-game',
#                 src='data:image/png;base64,{}'.format(encoded_image.decode()), 
#                 style={'margin-left':'20px','height':'100%', 'width':'100%','border':'2px red solid', 'backgroundColor':'#EF989F', 'padding':'10px'}
#             ),
#             html.Br(),
#             # Botón de parar
#             dbc.Button('Refresh',
#                 id='btn-refresh',
#                 n_clicks=0,
#                 color='info',
#                 style={'margin-left':'20px','margin-top':'20px'}
#             ),
#         ], style={'display':'inline-block','height':'20%', 'width':'20%'}),
#         html.Br(),
#         dbc.Button('Stop',
#             id='btn-stop',
#             #children='Stop',
#             n_clicks=0,
#             color='secondary',
#             outline=True,
#             style={'width':'230px', 'margin-top':'20px'}
#         ),
#     ], id='div-config',style={'display':'none'}),


#     html.Div([
#         # Camara mediapipe
#         html.Img(id='video',src="/user/configurations/video_mediapipe_feed"),
#         html.Div([
#             # Video la configuración a practicar
#             html.Video(
#                 id='vid-sign-game',
#                 src='/static/'+gesture_now+'.mov',
#                 loop=True, autoPlay=True, controls=True, 
#                 style={'height':'100%', 'width':'100%','margin-left':'20px','border':'2px red solid', 'backgroundColor':'#EF989F', 'padding':'10px'}
#             ),
#             html.Br(),
#             # Botón de parar
#             dbc.Button('Refresh',
#                 id='btn-refresh',
#                 n_clicks=0,
#                 color='info',
#                 style={'margin-left':'20px','margin-top':'20px'}
#             ),
#         ], style={'display':'inline-block','height':'20%', 'width':'20%'}),
#         html.Br(),
#         dbc.Button('Stop',
#             id='btn-stop',
#             #children='Stop',
#             n_clicks=0,
#             color='secondary',
#             outline=True,
#             style={'width':'230px', 'margin-top':'20px'}
#         ),
#     ], id='div-gestos',style={'display':'none'}),
#     html.Div(
#         id='explanation-config-game', style={"margin-left": "18px"}
#     ),
#     dcc.Interval(
#         id='interval-pred',
#         interval=0.1 * 1000,  # in milliseconds
#         n_intervals=0
#     ),
#     dcc.Store('stored-game-predictions',data={'pred':np.empty(shape=(10))}),
#     dcc.Store('stored-game-conf-predict-proba', data={}),
#     dcc.Store('stored-selected-labels-game',data={'config':obtained_random_config_name,'sign':gesture_now}),
#     dcc.Store('cont-game',data={'cont_sign':0,'cont_config':0})
# ]),

# @callback([Output('div-config','style'), Output('div-gestos','style')],
#           [Input('btn-game','n_clicks'), Input('btn-stop','n_clicks')], 
#            State('radioItem-game','value'), prevent_initial_call=True)
# def which_div(n,n2, value):

#     show = {'display':'inline-block'}
#     hide = {'display':'none'}

#     ctx = dash.callback_context
#     if not ctx.triggered:
#         button_id = 'No clicks yet'
#     else:
#         button_id = ctx.triggered[0]['prop_id'].split('.')[0]

#     if button_id == 'btn-game':
#         if(value=='conf'):
#             sty_conf = show
#             sty_sign = hide
#         else:
#             sty_conf = hide
#             sty_sign = show
#     else:
#         sty_conf = hide
#         sty_sign = hide
    
#     return sty_conf, sty_sign

# @callback([Output('img-sign-game','src'), Output('img-sign-game','style'),
#            Output('vid-sign-game','src'), Output('vid-sign-game','style'),
#            Output('stored-selected-labels-game','data'), Output('cont-game','data'),
#            Output('stored-game-predictions','data'), Output('stored-game-conf-predict-proba', 'data')],
#            #Output('stored-predictions','data'), Output('stored-pred-proba-configs','data')],
#            [Input('interval-pred','n_intervals'), Input('btn-refresh','n_clicks')],
#            [State('img-sign-game','src'), State('img-sign-game','style'),
#            State('vid-sign-game','src'), State('vid-sign-game','style'),
#            State('div-config','style'), State('div-gestos','style'),
#            State('stored-selected-labels-game','data'), State('cont-game','data'),
#            State('stored-game-predictions','data'), State('stored-game-conf-predict-proba', 'data')])
# def predict(interval,btn_click,config_src, config_style, vid_src, vid_style, div_config_style, div_sign_style,labels,conts,data_preds,data_config_preds):
#     from global_ import landmarks as data_array

#     actual_config = labels['config']
#     actual_sign = labels['sign']
#     ctx = dash.callback_context
#     which_fired = ctx.triggered[0]['prop_id'].split('.')[0]
    
#     # REFRESH BUTTON
#     if(which_fired=='btn-refresh'):
#         if div_config_style['display'] != 'none':
#             new_config = np.random.choice(config_names)
#             config_src = 'data:image/png;base64,{}'.format(base64.b64encode(open('pages/apps/images/'+new_config+'.png', 'rb').read()).decode())
#             labels['config'] = new_config
#         else:
#             new_sign = np.random.choice(gesture_names)
#             vid_src = '/static/'+new_sign+'.mov'
#             labels['sign'] = new_sign
        
#         return config_src, config_style, vid_src, vid_style, labels, conts, data_preds, data_config_preds
    
#     # INTERVAL
#     if data_array is not None:
#         # PREDICT CONFIGURATION
#         if div_config_style['display'] != 'none':
#             data_to_predict = help_functions.transform_data(data_array, flatten=True)
#             data_with_distances = help_functions.add_distances_to_landmarks(data_to_predict,added_dist_conf)
#             test_data = data_with_distances[features_index_conf]

#             test_data = np.reshape(test_data,(1,-1))
#             predict_proba = clf.predict_proba(test_data)
#             predicted_config = np.argmax(predict_proba)
#             config = configurations[predicted_config]

#             output = str(config)
#             if output == actual_config:
#                 new_config = np.random.choice(config_names)
#                 config_src = 'data:image/png;base64,{}'.format(base64.b64encode(open('pages/apps/images/'+new_config+'.png', 'rb').read()).decode())
#                 config_style['border'] = '2px green solid'
#                 config_style['backgroundColor'] = '#E0F2F2'
#                 labels['config'] = new_config
#                 data_preds = {'pred':np.empty(shape=(10))}
#                 data_config_preds = {}
#             else:
#                 config_style['border'] = '2px red solid'
#                 config_style['backgroundColor'] = '#EF989F'
        
#         # PREDICT SIGN
#         elif div_sign_style['display'] != 'none':
#             data_to_predict = help_functions.transform_data(data_array, flatten=True)
#             data_with_distances = help_functions.add_distances_to_landmarks(data_to_predict,added_dist_hmm)
#             test_data = data_with_distances[features_index_hmm]

#             test_data = np.reshape(test_data,(1,-1))
#             predict_proba = clf.predict_proba(test_data)
#             data_config_preds[str(conts['cont_config'])] = predict_proba
#             conts['cont_config'] = (conts['cont_config'] + 1) % 25
        
#             array_pred_proba = np.array([np.array(data_config_preds[str(index)]).flatten() for index in range(len(data_config_preds))])
#             print(array_pred_proba)
#             print('CONTS: ',conts)
#             if (array_pred_proba.shape[0]==25):
#                 predictions_hmm = [model.score(array_pred_proba) for model in clfs_hmm]
#                 max_index = np.argmax(predictions_hmm)
#                 predicted_gesture = str(signs[max_index])
#                 data_preds['pred'][conts['cont_sign']] = predicted_gesture
#                 conts['cont_sign'] = (conts['cont_sign'] + 1) % 10
#                 result = stats.mode(data_preds['pred'])[0][0]
#                 print(data_preds)
#                 print('RESULT: ',result)
#                 if result == actual_sign:
#                     new_sign = np.random.choice(gesture_names)
#                     vid_src = '/static/'+new_sign+'.mov'
#                     vid_style['border'] = '2px green solid'
#                     vid_style['backgroundColor'] = '#E0F2F2'
#                     labels['sign'] = new_sign
#                 else:
#                     vid_style['border'] = '2px red solid'
#                     vid_style['backgroundColor'] = '#EF989F'

#     else: # NOT RIGHT HAND LANDMARK
#         config_style['border'] = '2px red solid'
#         config_style['backgroundColor'] = '#EF989F'
#         vid_style['border'] = '2px red solid'
#         vid_style['backgroundColor'] = '#EF989F'  

#     return config_src, config_style, vid_src, vid_style, labels, conts, data_preds, data_config_preds