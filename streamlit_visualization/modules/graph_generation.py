import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ecg_qc import ecg_qc
import math
import numpy as np


def ecg_graph_generation(df, start_frame, end_frame, fs=1000):

    fig = go.Figure()
    graph_df = df[(df.index >= start_frame) & (df.index < end_frame)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=graph_df.index/fs,
                             y=graph_df['ecg_signal'],
                             mode='lines',
                             name='ecg_signal'), secondary_y=False)

    classif_data = ecg_qc_predit(graph_df)

    fig.add_trace(go.Scatter(x=graph_df.index/fs,
                             y=classif_data['classif'].values,
                             mode='lines',
                             name='ecg_qc'), secondary_y=True)

    print(graph_df.index[0])

    fig.update_layout(template='plotly_white',
                      title='ECG viz',
                      xaxis_title='Seconds',
                      yaxis2=dict(range=[0, 1]),
                      xaxis=dict(showgrid=True,
                                 tickmode='linear',
                                 ticks="inside",
                                 tickson="boundaries",
                                 tick0=graph_df.index[0]/fs,
                                 ticklen=50,
                                 tickwidth=2,
                                 dtick=9))

    return fig


def heatmap_annot_generation(df, start_frame, end_frame, fs=1000):

    graph_df = df[(df.index >= start_frame) & (df.index < end_frame)]

    data = np.transpose(graph_df.iloc[:, 1:5].values)

    fig = go.Figure(data=go.Heatmap(
            x=graph_df.index/fs,
            y=graph_df.iloc[:, 1:5].columns,
            z=data,
            colorscale=[[0.0, "rgb(0,140,0)"],
                        [0.5, "rgb(255,165,0)"],
                        [1.0, "rgb(160,0,0)"]],
            zmin=1,
            zmax=3))

    fig.update_layout(
        title='Annotators',
        xaxis=dict(showgrid=True,
                    tickmode='linear',
                    ticks="inside",
                    tickson="boundaries",
                    tick0=graph_df.index[0]/fs,
                    ticklen=50,
                    tickwidth=2,
                    dtick=9))

    return fig


def heatmap_pred_generation(df, start_frame, end_frame, fs=1000):

    graph_df = df[(df.index >= start_frame) & (df.index < end_frame)]

    classif_data = ecg_qc_predit(graph_df)

    data = np.transpose(classif_data.values)

    fig = go.Figure(data=go.Heatmap(
            x=graph_df.index/fs,
            y=['pred'],
            z=data,
            colorscale=[[0.0, "rgb(120,0,0)"],
                        [1.0, "rgb(0,160,0)"]],
            zmin=0,
            zmax=1))

    fig.update_layout(
        title='Prediction',
        xaxis=dict(showgrid=True,
                    tickmode='linear',
                    ticks="inside",
                    tickson="boundaries",
                    tick0=graph_df.index[0]/fs,
                    ticklen=50,
                    tickwidth=2,
                    dtick=9))

    return fig


def ecg_qc_predit(dataset):

    lib_path = '/home/aura-alexis/github/ecg_qc_viz/env/lib64/python3.6/' + \
        'site-packages/ecg_qc-1.0b1-py3.6.egg/ecg_qc'
    ecg_qc_test = ecg_qc(model='{}/ml/models/rfc.joblib'.format(lib_path))
    time_window = 9
    fs = 1000

    classif_data = pd.DataFrame(
        range(0, dataset.shape[0]),
        columns=['classif'])

    for ecg_signal_index in range(
            math.floor(dataset.shape[0]/(fs * time_window)) + 1):

        start = ecg_signal_index*fs*time_window
        end = start + fs*time_window

        ecg_data = dataset['ecg_signal'].iloc[start:end].values

        signal_quality = ecg_qc_test.predict_quality(
            ecg_qc_test.compute_sqi_scores(ecg_data))
        classif_data.iloc[ecg_signal_index*fs*time_window:
                          ecg_signal_index*fs*time_window+fs*time_window] = \
            signal_quality[0]

    return classif_data
