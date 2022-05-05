from dash import Dash, dcc, html, Input, Output, callback
import dash
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/")

# app = Dash(__name__, suppress_callback_exceptions=True)
# server = app.server


layout = html.Div([
    html.Div([
        dbc.Button("USER", outline=True, color="success",href="/chapter/user"),
        html.Div("Aqui puedo poner cositas???")
    ], style={'width':'49%', 'display':'inline-block'}, className="d-grid gap-2 col-6 mx-auto"),
    html.Div([
        dbc.Button('EXPERT', outline=True, color="danger", href='/chapter/expert'),
        html.Div("Y aquí? también? kkkkkkkkkkkk")
    ], style={'width':'49%', 'display':'inline-block'}, className="d-grid gap-2 col-6 mx-auto")
], style={'display':'flex'})


# if __name__ == '__main__':
#     app.run_server(debug=True)


