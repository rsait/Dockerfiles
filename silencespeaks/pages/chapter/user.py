from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import base64
import dash

dash.register_page(__name__, path='/chapter/user',icon="fas fa-user")


dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("User home", href="/chapter/user"),
        dbc.DropdownMenuItem("Practice configurations", href="/user/configurations"),
        dbc.DropdownMenuItem("Practice signs", href="/user/signs"),
        dbc.DropdownMenuItem("Game", href="/user/game")
    ],
    nav = True,
    in_navbar = True,
    label = "Explore",
)
navbar_user = dbc.NavbarSimple(
    children=[
        dropdown
    ],
    brand='LEARN AND PRACTICE SPANISH SIGN LANGUAGE',
    brand_href='/chapter/user',
    color="secondary",
    dark=True,
    className="mb-4",
)

# logo_img = base64.b64encode(open("pages/apps/images/LSE_logo.jpg", 'rb').read())

# navbar = dbc.Navbar(
#     dbc.Container(
#         [
#             html.A(
#                 # Use row and col to control vertical alignment of logo / brand
#                 dbc.Row(
#                     [
#                         dbc.Col(html.Img(src='data:image/png;base64,{}'.format(logo_img.decode()), height="60px")),
#                         dbc.Col(dbc.NavbarBrand("LEARN AND PRACTICE SPANISH SIGN LANGUAGE", className="ml-2")),
#                     ],
#                     align="center",
#                     # no_gutters=True,
#                 ),
#                 href="/chapter/user",
#             ),
#             dbc.NavbarToggler(id="navbar-toggler2"),
#             dbc.Collapse(
#                 dbc.Nav(
#                     # right align dropdown menu with ml-auto className
#                     [dropdown], className="ml-auto", navbar=True
#                 ),
#                 id="navbar-collapse2",
#                 navbar=True,
#             ),
#         ]
#     ),
#     color="dark",
#     dark=True,
#     className="mb-4",
# )

layout = html.Div(
    [
        #navbar,
        navbar_user,
        html.H1("CHOOSE IN THE EXPLORE MENU THE LESSON YOU WANT TO PRACTICE"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(html.Div([
                    html.H3("PRACTICE: CONFIGURATIONS"),
                    html.Div('In LSE a sign is defined by four features: place, configuration, orientation and movement. In this first step, different configurations which form several signs can be practiced.'),
                    html.Br(),
                    html.Div('This activity is thought to practice 8 different configurations.'),
                    html.Br(),
                    dbc.Button('LEARN CONFIGURATIONS',
                        href="/user/configurations",
                        outline=True,
                        color='primary'                
                    )
                ], style={"border":"2px black solid","margin-left": "15px",'text-align':'justify', 'padding':'10px'} #'backgroundColor':'#E0F2F2',
                )),
                dbc.Col(html.Div([
                    html.H3("PRACTICE: SIGNS"),
                    html.Div('This activity is prepared to practice five different LSE signs: BIEN, CONTENTO, HOMBRE, MUJER and OYENTE.'),
                    html.Br(),
                    dbc.Button('LEARN SIGNS',
                        href="/user/signs",
                        outline=True,
                        color='primary'
                    )
                ], style={"border":"2px black solid","margin-left": "10px", 'text-align':'justify', 'padding':'10px'} #,'backgroundColor':'#AEC0C9'
                )),
                dbc.Col(html.Div([
                    html.H3("GAME"),
                    html.Div('Select between configurations or signs and try to perform them correctly.'),
                    html.Br(),
                    dbc.Button('GAME',
                        href="/user/game",
                        outline=True,
                        color='primary'
                    )
                ], style={"border":"2px black solid","margin-left": "10px","margin-right": "15px", 'text-align':'justify', 'padding':'10px'} #'backgroundColor':'#E5E4DA',
                )),
            ]
        )
    ]
)

# layout = html.Div([
#     html.H1("CHOOSE IN THE EXPLORE MENU THE LESSON YOU WANT TO PRACTICE"),
#     html.Br(),
#     dbc.Row(
#         [
#             dbc.Col(
#                 dbc.Button('LEARN CONFIGURATIONS',
#                     href="/user/configurations",
#                     outline=True,
#                     color='primary'                
#                 )
#             ),
#             dbc.Col(
#                 dbc.Button('LEARN SIGNS',
#                     href="/user/signs",
#                     outline=True,
#                     color='primary'
#                 )
#             ),
#             dbc.Col(
#                 dbc.Button('GAME',
#                     href="/user/game",
#                     outline=True,
#                     color='primary'
#                 )
#             ),
#         ]
#     )
# ])