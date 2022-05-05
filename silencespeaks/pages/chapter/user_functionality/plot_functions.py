import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def rotate_z(x, y, z, theta):
    w = x + 1j * y
    return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z


def obtain_graph(medoid):
    if len(medoid) == 0:
        empty_fig = go.Figure()
        empty_fig.update_scenes(xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False,
                                xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=False,
                                xaxis_showticklabels=False, yaxis_showticklabels=False, zaxis_showticklabels=False,
                                xaxis_showaxeslabels=False, yaxis_showaxeslabels=False, zaxis_showaxeslabels=False)
        return empty_fig

    fingers = ["Hand"] + ["Thumb"] * 4 + ["Index"] * 4 + ["Middle"] * 4 + ["Ring"] * 4 + ["Pinky"] * 4

    x = medoid[:, 0]
    y = medoid[:, 1]
    z = medoid[:, 2]
    df = pd.DataFrame(list(zip(x, y, z, fingers)), columns=["X", "Y", "Z", "finger"])

    fig1 = px.line_3d(df, x="X", y="Y", z="Z", color='finger', markers=True)

    other_x = [x[0], x[1], x[1], x[5], x[5], x[9], x[9], x[13], x[17], x[17], x[0]]
    other_y = [y[0], y[1], y[1], y[5], y[5], y[9], y[9], y[13], y[17], y[17], y[0]]
    other_z = [z[0], z[1], z[1], z[5], z[5], z[9], z[9], z[13], z[17], z[17], z[0]]
    df_other = pd.DataFrame(list(zip(other_x, other_y, other_z)), columns=["X", "Y", "Z"])

    fig2 = px.line_3d(df_other, x="X", y="Y", z="Z", markers=True)

    fig3 = go.Figure(data=fig2.data + fig1.data)
    fig3.update_scenes(xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False,
                       xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=False,
                       xaxis_showticklabels=False, yaxis_showticklabels=False, zaxis_showticklabels=False,
                       xaxis_showaxeslabels=False, yaxis_showaxeslabels=False, zaxis_showaxeslabels=False)

    # frames = []
    # x_eye = -1.25
    # y_eye = 2
    # z_eye = 0.5
    # fig3.update_layout(
    #     # title='Animation Test',
    #     width=600,
    #     height=600,
    #     scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
    #     updatemenus=[dict(type='buttons',
    #                       showactive=False,
    #                       y=1,
    #                       x=0.8,
    #                       xanchor='left',
    #                       yanchor='bottom',
    #                       pad=dict(t=45, r=10),
    #                       buttons=[dict(label='Play',
    #                                     method='animate',
    #                                     args=[None, dict(frame=dict(duration=5, redraw=True),
    #                                                      transition=dict(duration=0),
    #                                                      fromcurrent=True,
    #                                                      mode='immediate'
    #                                                      )]
    #                                     )
    #                                ]
    #                       )
    #                  ]
    # )

    # for t in np.arange(0, 6.26, 0.1):
    #     xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
    #     frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
    # fig3.frames = frames

    # fig3.update_layout(showlegend=False)
    # fig3.show()
    return fig3
