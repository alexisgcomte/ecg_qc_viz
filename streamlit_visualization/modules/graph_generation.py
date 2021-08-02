import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ecg_qc import ecg_qc
import math
import numpy as np
import streamlit as st


def make_sqi_graph(df, sqis_date, fs=1000, tick_space=2):

    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=df.index/fs,
                             y=df['ecg_signal'],
                             mode='lines',
                             name='ecg_signal',
                             marker_color='rgba(44, 117, 255, .8)'),
                  secondary_y=False)
    sqi_names = ['q_sqi_score', 'c_sqi_score', 's_sqi_score',
                 'k_sqi_score', 'p_sqi_score', 'bas_sqi_score']

    for i in range(6):
        fig.add_trace(go.Scatter(
            x=df.index/fs,
            y=sqis_date[i],
            mode='lines',
            name=sqi_names[i]),
            secondary_y=True)

    fig.update_layout(template='plotly_white',
                      title='ECG viz',
                      xaxis_title='Seconds',
                      # yaxis2=dict(range=[0, 50]),
                      xaxis=dict(showgrid=True,
                                  tickmode='linear',
                                  ticks="inside",
                                  # ticklabelposition='inside top',
                                  tickson="boundaries",
                                  tick0=df.index[0]/fs,
                                  ticklen=10,
                                  tickwidth=1,
                                  dtick=tick_space,
                                  side='top'),
                      yaxis=dict(fixedrange=True),
                      yaxis2=dict(fixedrange=True),
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=-0.1,
                          xanchor="right",
                          x=1))
    st.plotly_chart(fig, use_container_width=True)


def ecg_graph_generation(df: pd.DataFrame,
                         start_frame: int,
                         end_frame: int,
                         tick_space: int = 9,
                         fs: int = 1_000,
                         wavelet_generation: bool = False,
                         source: str = 'Physionet',
                         time_window_ml=2,
                         time_window_cnn=2,
                         spectrum_max_hz=50) -> go.Figure:

    # ecg_qc predictions

    ecg_data = df['ecg_signal'].values
    classif_ecg_qc_ml_data, sqis_data = ecg_qc_predict(
        ecg_data=ecg_data,
        time_window_ml=time_window_ml,
        fs=fs,
        normalized=False)
    classif_ecg_qc_ml_data_norm, sqis_data_norm = ecg_qc_predict(
        ecg_data=ecg_data,
        time_window_ml=time_window_ml,
        fs=fs,
        normalized=True)
    
    if source == 'Physionet':

        # annotation converted in binary

        for column in df.columns[1:5]:
            df[column] = df[column].apply(
                lambda x: annot_classification_correspondance(x))

        data = np.transpose(df.iloc[:, 1:5].values)

        labels = list(df.iloc[:, 1:5].columns) + ['ecg_qc pred '] + \
            ['ecg_qc self_normalized ']

        data = [data[0],
                data[1],
                data[2],
                data[3],
                classif_ecg_qc_ml_data,
                # np.transpose(classif_ecg_qc_cnn_data.values)[0]]
                classif_ecg_qc_ml_data_norm]

        make_sqi_graph(df, sqis_data)
        make_sqi_graph(df, sqis_data_norm)

    if source == 'La Teppe':
        for column in df.columns[1:5]:
            df[column] = 1
        data = np.transpose(df.iloc[:, 1:5].values)
        labels = ['ecg_qc pred '] + \
            ['ecg_qc self normalized']

        # consolidation of data
        data = [classif_ecg_qc_ml_data,
                # np.transpose(classif_ecg_qc_cnn_data.values)[0]]
                classif_ecg_qc_ml_data]

    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Heatmap(x=df.index/fs,
                             y=labels,
                             z=data,
                             colorscale=[[0.0, "rgb(160,0,0)"],
                                         [1.0, "rgb(0,140,0)"]],
                             zmin=0,
                             zmax=1,
                             opacity=0.2,
                             showscale=False,
                             ygap=1),
                  secondary_y=True)

    fig.add_trace(go.Scatter(x=df.index/fs,
                             y=df['ecg_signal'],
                             mode='lines',
                             name='ecg_signal',
                             marker_color='rgba(44, 117, 255, .8)'),
                  secondary_y=False)

    fig.update_layout(template='plotly_white',
                      title='ECG viz',
                      xaxis_title='Seconds',
                      # yaxis2=dict(range=[0, 50]),
                      xaxis=dict(showgrid=True,
                                  tickmode='linear',
                                  ticks="inside",
                                  # ticklabelposition='inside top',
                                  tickson="boundaries",
                                  tick0=df.index[0]/fs,
                                  ticklen=10,
                                  tickwidth=1,
                                  dtick=tick_space,
                                  side='top'),
                      yaxis=dict(fixedrange=True),
                      yaxis2=dict(fixedrange=True)
                      )

    return fig


def ecg_qc_predict(ecg_data: np.ndarray,
                   time_window_ml: int = 9,
                   fs: int = 1_000,
                   normalized: bool = False) -> np.ndarray:

    # ecg_qc_test = ecg_qc()
    ecg_qc_test = ecg_qc(model='model_2s_rfc_normalized_premium.pkl',
                         normalized=normalized)

    classif_ecg_qc_data = np.zeros(len(ecg_data))
    sqis_data = [np.zeros(len(ecg_data)) for n in range(6)]

    for start in range(
            math.floor(len(ecg_data)/(fs * time_window_ml)) + 1):

        start = start * fs * time_window_ml
        end = start + fs * time_window_ml
        ecg_signal = np.array(ecg_data[start:end])

        signal_quality = ecg_qc_test.get_signal_quality(ecg_signal)
        classif_ecg_qc_data[start:end] = signal_quality

        sqis = ecg_qc_test.compute_sqi_scores(ecg_signal)[0]
        for n in range(6):
            sqis_data[n][start:end] = sqis[n]

    return classif_ecg_qc_data, sqis_data


def annot_classification_correspondance(classif: int) -> int:

    if classif == 2 or classif == 3:
        classif = 0

    return classif
