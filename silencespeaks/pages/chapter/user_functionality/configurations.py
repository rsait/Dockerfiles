# from dash import dcc, html, Input, Output, State, callback
# import base64
# import dash
# import numpy as np
# from pages.chapter.user_functionality import help_functions
# from pages.chapter.user_functionality import plot_functions
# import pickle as pkl
# from pages.chapter.user import navbar_user

# dash.register_page(__name__, path='/user/configurations')

# layout = html.Div([
#     #html.Img(src="/user/configurations/video_mediapipe_feed"),
#     navbar_user,
#     html.Div([
#         html.Img(id='video', src='/user/configurations/video_mediapipe_feed'),
#         html.Div(
#             html.Div(
#                 dcc.Graph(
#                     id='graph-3d',
#                     figure=plot_functions.obtain_graph([]),
#                     responsive=True,
#                     style={
#                         "width": "100%",
#                         "height": "100%"
#                     }
#                 ),
#                 style={
#                     "width": "100%",
#                     "height": "100%",
#                 },
#             )
#         )],
#         style=dict(display='flex')
#     ),
#     html.Div([
#         html.H2("""Which configuration do you want to practice?""",
#                         style={'margin-right': '2em'}),
#         dcc.Dropdown(
#             id='drop_configuraciones',
#             options=[
#                 {'label': '4', 'value':'4'},
#                 {'label': '50', 'value':'50'},
#                 {'label': '58', 'value':'58'},
#                 {'label': '59', 'value':'59'},
#                 {'label': '73', 'value':'73'},
#                 {'label': '74', 'value':'74'},
#                 {'label': '77', 'value':'77'},
#                 {'label': '78', 'value':'78'}
#             ],
#             value='4',
#             persistence=True
#         ),
#         dcc.Store(id="stored-config-name", data={"name":'4'}),
#         html.Br(),
#         html.Div(id='textarea-state-example-output', style={'whiteSpace': 'pre'}),
#         html.Br(),
#         html.Img(id='img-config',src='data:image/png;base64,{}'.format(base64.b64encode(open("pages/apps/images/4.png", 'rb').read()).decode()))
#     ]),
#     dcc.Interval(id="a",interval=200,n_intervals=0)
# ])

# # Leer el fichero donde están guardados los accuracies
# # y coger el modelo con mejor resultado.
# # Mirar ese modelo qué features ha usado y guardarlo
# # Transformar los datos de entrada (features + wrist sustraction)
# # Realizar la clasificación
# # model, features = get_config_model()
# path = "/home/bee/Documents/multipage_app/apps/PRUEBITAS_ACC/"
# clf_path, features_index, added_dist = help_functions.best_clf(path, model_type='configs', flatten=True)
# clf = pkl.load(open(clf_path,"rb"))
# labels = ['4','50','58','59','73','74','77','78']

# @callback([Output('textarea-state-example-output','children'),Output('textarea-state-example-output','style'),
#            Output('graph-3d','figure')],
#            Input('a','n_intervals'),
#           [State('stored-config-name','data')]) #State('stored-landmarks','data'),
# def prediction(n,config):
#     #print(data)
#     from global_ import landmarks as data_array
#     actual_config = config["name"]
#     #print(data_array)
#     if data_array is not None:
#         #df_data = pd.DataFrame(data)
#         #data_array = np.array(df_data)
#         data_to_predict = help_functions.transform_data(data_array, flatten=True)
#         data_with_distances = help_functions.add_distances_to_landmarks(data_to_predict,added_dist)
#         test_data = data_with_distances[features_index]

#         test_data = np.reshape(test_data,(1,-1))
#         predict_proba = clf.predict_proba(test_data)
#         predicted_config = np.argmax(predict_proba)
#         config = labels[predicted_config]

#         output = str(config)
#         fig = plot_functions.obtain_graph(help_functions.transform_data(data_array, flatten=False))
#         if output == actual_config:
#             style = {'backgroundColor':'#99FF99'}
#         else:
#             style = {'backgroundColor':'#FF9999'}
#     else:
#         output = "Not right hand landmark visible"
#         style = {'backgroundColor':'#FFFFFF'}
#         fig = plot_functions.obtain_graph([])
#     return output, style, fig


# @callback([Output('stored-config-name','data'),Output('img-config','src')],
#           Input('drop_configuraciones','value'))
# def change_configuration_model(selected_config):
#     return {'name':str(selected_config)}, 'data:image/png;base64,{}'.format(base64.b64encode(open("pages/apps/images/" + selected_config +".png", 'rb').read()).decode())
