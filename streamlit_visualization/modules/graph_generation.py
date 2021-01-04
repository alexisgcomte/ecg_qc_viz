import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import SQIs_class as SQI


def fig_generation(df, start_frame, end_frame, fs=1000):

    fig = go.Figure()
    graph_df = df[(df.index >= start_frame) & (df.index < end_frame)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=graph_df.index/fs,
                            y=graph_df['ecg_signal'],
                            mode='lines',
                            name='ecg_signal'), secondary_y=False)

    fig.add_trace(go.Scatter(x=graph_df.index/fs,
                             y=graph_df['anno1'],
                             mode='lines',
                             name='anno1'), secondary_y=True)

    fig.add_trace(go.Scatter(x=graph_df.index/fs,
                             y=graph_df['anno2'],
                             mode='lines',
                             name='anno2'), secondary_y=True)

    fig.add_trace(go.Scatter(x=graph_df.index/fs,
                             y=graph_df['anno3'],
                             mode='lines',
                             name='anno3'), secondary_y=True)

    fig.add_trace(go.Scatter(x=graph_df.index/fs,
                             y=graph_df['cons'],
                             mode='lines',
                             name='cons'), secondary_y=True)

    fig.update_layout(template='plotly_white',
                      title='ECG viz',
                      xaxis_title='Seconds',
                      yaxis2=dict(range=[0, 3]))

    fig.update_yaxes(title_text="ECG", secondary_y=False)
    fig.update_yaxes(title_text="Classification", secondary_y=True)

    return fig

# 
# 
# def result_generation(df, start_frame, end_frame,
#                       window=1000, fs=1000):
# 
#     df_result = pd.DataFrame(columns=['pSQI', 'kSQI', 'basSQI', 'timestamp'])
#     input_df = df[(df.index >= start_frame) & (df.index < end_frame)]
#     timestamp = input_df.index[0]
# 
#     for i in range(int(round(input_df.shape[0] / window, 0))):
# 
#         start = i * window
#         end = start + window
#         sqi = SQI.SQI(input_df['ECG'][start:end].values, fs)
#         sqi.start_all()
#         df_result = df_result.append({'timestamp': (timestamp+i*window)/fs,
#                                       'classif_pSQI': sqi.classif_pSQI,
#                                       'classif_kSQI': sqi.classif_kSQI,
#                                       'classif_basSQI': sqi.classif_basSQI,
#                                       'pSQI_cr': sqi.pSQI_cr,
#                                       'kSQI_cr': sqi.kSQI_cr,
#                                       'basSQI_cr': sqi.basSQI_cr},
#                                      ignore_index=True)
# 
#     # Graph for classification
#     fig_classif = go.Figure()
#     fig_classif.add_trace(go.Scatter(x=df_result['timestamp'],
#                                      y=df_result['classif_pSQI'],
#                                      mode='lines',
#                                      name='pSQI'))
#     fig_classif.add_trace(go.Scatter(x=df_result['timestamp'],
#                                      y=df_result['classif_kSQI'],
#                                      mode='lines',
#                                      name='kSQI'))
#     fig_classif.add_trace(go.Scatter(x=df_result['timestamp'],
#                                      y=df_result['classif_basSQI'],
#                                      mode='lines',
#                                      name='basSQI'))
# 
#     fig_classif.update_layout(template='plotly_white',
#                               title='Classification',
#                               xaxis_title='Seconds')
# 
# #    # Graph for score
# #    fig_cr = make_subplots(specs=[[{"secondary_y": True}]])
# #    fig_cr.add_trace(go.Scatter(x=df_result['timestamp'],
# #                                y=df_result['pSQI_cr'],
# #                                mode='lines',
# #                                name='pSQI'), secondary_y=True)
#    fig_cr.add_trace(go.Scatter(x=df_result['timestamp'],
#                                y=df_result['kSQI_cr'],
#                                mode='lines',
#                                name='kSQI'), secondary_y=False)
#    fig_cr.add_trace(go.Scatter(x=df_result['timestamp'],
#                                y=df_result['basSQI_cr'],
#                                mode='lines',
#                                name='basSQI'), secondary_y=True)
#
#    fig_cr.update_layout(template='plotly_white',
#                         title='Scores',
#                         xaxis_title='Seconds',
#                         yaxis2=dict(range=[0, 1]))
#
#    fig_cr.update_yaxes(title_text="kSQI", secondary_y=False)
#    fig_cr.update_yaxes(title_text="pSQK/basSQI", secondary_y=True)

    return fig_classif
