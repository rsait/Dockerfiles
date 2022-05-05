from dash import dcc, html
import dash_bootstrap_components as dbc
import dash

dash.register_page(__name__, path="/chapter/expert",icon="fas fa-user-cog")

dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Expert home", href="/chapter/expert"),
        dbc.DropdownMenuItem("Record dataset", href="/expert/record"),
        dbc.DropdownMenuItem("Train configuration model", href="/expert/configurations"),
        dbc.DropdownMenuItem("Train sign model", href="/expert/signs")
    ],
    nav = True,
    in_navbar = True,
    label = "Explore",
)
navbar_expert = dbc.NavbarSimple(
    children=[
        dropdown
    ],
    brand='TRAIN MODELS TO RECOGNIZE SPANISH SIGN LANGUAGE',
    brand_href='/chapter/expert',
    color="secondary",
    dark=True,
    className="mb-4",
)

layout = html.Div(
    [
        #navbar,
        navbar_expert,
        html.H1("CHOOSE IN THE EXPLORE MENU THE OPTION YOU WANT"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(html.Div([
                    html.H3("RECORD DATASET"),
                    html.Div('Record new instances to be used for creating the classification models.'),
                    html.Br(),
                    dbc.Button('RECORD',
                        href="/expert/record",
                        outline=True,
                        color='primary'                
                    )
                ], style={"border":"2px black solid","margin-left": "15px",'text-align':'justify', 'padding':'10px'} #'backgroundColor':'#E0F2F2',
                )),
                dbc.Col(html.Div([
                    html.H3("TRAIN CONFIGURATIONS"),
                    html.Div('Train new classification models to perform the recognition of different configurations of the Spanish Sign Language.'),
                    html.Br(),
                    dbc.Button('TRAIN CONFIGURATIONS',
                        href="/expert/configurations",
                        outline=True,
                        color='primary'
                    )
                ], style={"border":"2px black solid","margin-left": "10px", 'text-align':'justify', 'padding':'10px'} #,'backgroundColor':'#AEC0C9'
                )),
                dbc.Col(html.Div([
                    html.H3("TRAIN SIGNS"),
                    html.Div('Train new classification models to perform the recognition of different signs of the Spanish Sign Language.'),
                    html.Br(),
                    dbc.Button('TRAIN SIGNS',
                        href="/expert/signs",
                        outline=True,
                        color='primary'
                    )
                ], style={"border":"2px black solid","margin-left": "10px","margin-right": "15px", 'text-align':'justify', 'padding':'10px'} #'backgroundColor':'#E5E4DA',
                )),
            ]
        )
    ]
)
