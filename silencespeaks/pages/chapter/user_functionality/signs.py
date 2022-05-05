# from dash import dcc, html, Input, Output, State, callback
# import base64
# import dash
# import numpy as np
# from pages.chapter.user_functionality import help_functions
# import pickle as pkl
# from scipy import stats
# from pages.chapter.user import navbar_user


# dash.register_page(__name__, path='/user/signs')

# selected_sign = 'BIEN'
# image_filename = "pages/apps/images/" + selected_sign + ".png" 
# encoded_image = base64.b64encode(open(image_filename, 'rb').read())
# video_src = "/static/" + selected_sign + ".mov"

# path = "/home/bee/Documents/multipage_app/apps/PRUEBITAS_ACC/"
# hmm_path, clf_path, features_index, added_dist = help_functions.best_clf(path, model_type='signs', flatten=True)
# clfs_hmm = pkl.load(open(hmm_path,"rb"))
# clf = pkl.load(open(clf_path,"rb"))
# signs = ['BIEN','CONTENTO','HOMBRE','MUJER','OYENTE']

# layout = html.Div([
#     navbar_user,
#     html.Img(id="video", src='/user/configurations/video_mediapipe_feed'),
#     html.H2("""Which sign do you want to practice?""", style={'margin-right': '2em'}),
#     html.Div([
#         html.Div([
#             dcc.Dropdown(
#                 id='drop-signos',
#                 options=[
#                     {'label': 'BIEN', 'value':'BIEN'},
#                     {'label': 'MUJER', 'value':'MUJER'},
#                     {'label': 'HOMBRE', 'value':'HOMBRE'},
#                     {'label': 'CONTENTO', 'value':'CONTENTO'},
#                     {'label': 'OYENTE', 'value':'OYENTE'}
#                 ],
#                 value='BIEN',
#                 persistence=True
#             ),
#         ]),
#         html.Br(),
#         html.Div(id='text-sign-dropdown-output', style={'whiteSpace': 'pre'}),
#         html.Br(),
#         html.Div([
#             html.Img(id='img-sign',src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'height':'40%', 'width':'40%'}),
#             html.Video(id='vid-sign',src=video_src, controls=True, style={'height':'50%', 'width':'50%'})
#         ], style={'display':'flex'})  
#     ]),
#     dcc.Store('stored-selected-sign',data={'sign':selected_sign}),
#     dcc.Store('stored-predictions', data={'pred':np.empty(shape=(10))}),
#     dcc.Store('stored-pred-proba-configs',data={}),
#     dcc.Store('cont',data={'cont_sign':0,'cont_config':0}),
#     dcc.Interval('interval-predict-sign',interval=100,n_intervals=0)
# ])

# @callback([Output('img-sign','src'), Output('vid-sign','src'),Output('stored-selected-sign','data')],
#            Input('drop-signos','value'))
# def change_selected_sign(actual_sign):
#     image_filename = "pages/apps/images/" + actual_sign + ".png" 
#     encoded_image = base64.b64encode(open(image_filename, 'rb').read())
#     new_img_src = 'data:image/png;base64,{}'.format(encoded_image.decode())

#     new_video_src = "/static/" + actual_sign + ".mov"
#     return new_img_src, new_video_src, {'sign':actual_sign}

# @callback([Output('text-sign-dropdown-output','children'), Output('text-sign-dropdown-output','style'),
#            Output('stored-predictions','data'), Output('stored-pred-proba-configs','data'),Output('cont','data')],
#           Input('interval-predict-sign','n_intervals'),
#           [State('stored-selected-sign','data'),State('stored-predictions','data'),
#           State('stored-pred-proba-configs','data'), State('cont','data')])
# def predict(interval, selected_sign, preds, saved_pred_proba,conts):
#     cont_sign = conts['cont_sign']
#     cont_config = conts['cont_config']
#     sign = selected_sign['sign']
#     from global_ import landmarks as data_array

#     if data_array is not None:
#         # df_data = pd.DataFrame(landmarks)
#         # data_array = np.array(df_data)
#         data_to_predict = help_functions.transform_data(data_array, flatten=True)
#         data_with_distances = help_functions.add_distances_to_landmarks(data_to_predict,added_dist)
#         test_data = data_with_distances[features_index]

#         test_data = np.reshape(test_data,(1,-1))
#         predict_proba = clf.predict_proba(test_data)
#         # predicted_config = np.argmax(predict_proba)
#         # config = labels[predicted_config]
#         saved_pred_proba[str(cont_config)] = predict_proba
#         conts['cont_config'] = (cont_config + 1) % 25
    
#     array_pred_proba = np.array([np.array(saved_pred_proba[str(index)]).flatten() for index in range(len(saved_pred_proba))])
#     if (array_pred_proba.shape[0]==25):
#         predictions_hmm = [model.score(array_pred_proba) for model in clfs_hmm]
#         max_index = np.argmax(predictions_hmm)
#         predicted_gesture = str(signs[max_index])
#         preds['pred'][cont_sign] = predicted_gesture
#         conts['cont_sign'] = (cont_sign + 1) % 10
#         result = stats.mode(preds['pred'])[0][0]
#         if result == sign:
#             style = {'backgroundColor':'#99FF99'}
#         else:
#             style = {'backgroundColor':'#FF9999'}
#     else:
#         result = 'NOT SIGN PERFORMED YET'
#         style = {'backgroundColor':'#FFFFFF'}
#     #print('cont_config', cont_config)
#     #print('cont_sign',cont_sign)
#     return result, style, preds, saved_pred_proba, conts