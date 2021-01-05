import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import SQIs_class as SQI
from ecg_qc import ecg_qc
import math


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

    # Signal quality

    lib_path = '/home/aura-alexis/github/ecg_qc_viz/env/lib64/python3.6/site-packages/ecg_qc-1.0b1-py3.6.egg/ecg_qc'
    ecg_qc_test = ecg_qc(model='{}/ml/models/rfc.joblib'.format(lib_path))
    time_window = 9
    signal_quality_list = []
    classif_data = pd.DataFrame(range(0,graph_df.shape[0]), columns=['classif'])
    # print(graph_df.shape)

    for ecg_signal_index in range((math.floor(graph_df.shape[0]/(fs*time_window)) + 1)):

        start = ecg_signal_index*fs*time_window
        end = start + fs*time_window
        #  print(start, end)
        ecg_data = graph_df['ecg_signal'].iloc[start:end].values
        # sample = [[0.94, 0.59, 10.79, 0.51, 0.85, 2.97]]
        # print(ecg_qc_test.compute_sqi_scores(ecg_data))
        # print(len(ecg_data))
        # print( ecg_qc_test.predict_quality(ecg_qc_test.compute_sqi_scores(ecg_data)))

        # signal_quality = ecg_qc_test.get_signal_quality(ecg_data)
        signal_quality = ecg_qc_test.predict_quality(ecg_qc_test.compute_sqi_scores(ecg_data))
        signal_quality_list.append(signal_quality)
        classif_data.iloc[ecg_signal_index*fs*time_window:ecg_signal_index*fs*time_window+fs*time_window] = signal_quality[0]
        if signal_quality[0] == 1:
            print('1!')


    fig.add_trace(go.Scatter(x=graph_df.index/fs,
                            y=classif_data['classif'].values,
                            mode='lines',
                            name='classif ecg_qc'), secondary_y=True)

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
