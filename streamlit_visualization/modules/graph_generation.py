import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ecg_qc import ecg_qc
import math
import numpy as np
import scaleogram as scg
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def ecg_graph_generation(df: pd.DataFrame,
                         start_frame: int,
                         end_frame: int,
                         tick_space: int = 9,
                         fs: int = 1000) -> go.Figure:

    graph_df = df[(df.index >= start_frame) & (df.index < end_frame)]

    # annotation converted in binary
    for column in graph_df.columns[1:5]:
        graph_df[column] = graph_df[column].apply(
            lambda x: annot_classification_correspondance(x))

    data = np.transpose(graph_df.iloc[:, 1:5].values)

    # ecg_qc preidction
    classif_ecg_qc_data = ecg_qc_predict(graph_df)
    classif_ecg_qc_cnn_data = ecg_qc_predict_cnn(graph_df)

    # consolidation of data
    data = [data[0],
            data[1],
            data[2],
            data[3],
            np.transpose(classif_ecg_qc_data.values)[0],
            np.transpose(classif_ecg_qc_cnn_data.values)[0]]
    labels = list(graph_df.iloc[:, 1:5].columns) + ['ecg_qc pred '] + \
        ['ecg_qc cnn ']

    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Heatmap(x=graph_df.index/fs,
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

    fig.add_trace(go.Scatter(x=graph_df.index/fs,
                             y=graph_df['ecg_signal'],
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
                                  tick0=graph_df.index[0]/fs,
                                  ticklen=10,
                                  tickwidth=1,
                                  dtick=tick_space,
                                  side='top'),
                      yaxis=dict(fixedrange=True),
                      yaxis2=dict(fixedrange=True)
                      )

    return fig


def ecg_qc_predict(dataset: pd.DataFrame) -> pd.DataFrame:

    ecg_qc_test = ecg_qc()
    time_window = 9
    fs = 1000

    classif_ecg_qc_data = pd.DataFrame(
        range(0, dataset.shape[0]),
        columns=['classif'])

    for ecg_signal_index in range(
            math.floor(dataset.shape[0]/(fs * time_window)) + 1):

        start = ecg_signal_index*fs*time_window
        end = start + fs*time_window

        ecg_data = dataset['ecg_signal'].iloc[start:end].values

        signal_quality = ecg_qc_test.predict_quality(
            ecg_qc_test.compute_sqi_scores(ecg_data))

        classif_ecg_qc_data.iloc[ecg_signal_index * fs * time_window:
                                 ecg_signal_index * fs * time_window +
                                 fs * time_window] = signal_quality

    return classif_ecg_qc_data


def annot_classification_correspondance(classif: int) -> int:

    if classif == 2 or classif == 3:
        classif = 0

    return classif


def ecg_qc_predict_cnn(dataset: pd.DataFrame) -> pd.DataFrame:

    ecg_qc_test = ecg_qc(model_type='cnn')
    time_window = 2
    fs = 1000

    classif_ecg_qc_data = pd.DataFrame(
        range(0, dataset.shape[0]),
        columns=['classif'])

    print(dataset.index[0])


    for ecg_signal_index in range(
            math.floor(dataset.shape[0]/(fs * time_window)) + 1):

        start = ecg_signal_index*fs*time_window
        end = start + fs*time_window

        ecg_data = dataset['ecg_signal'].iloc[start:end].values

        signal_quality = annot_classification_correspondance(
            ecg_qc_test.get_signal_quality(ecg_data))

        classif_ecg_qc_data.iloc[ecg_signal_index * fs * time_window:
                                 ecg_signal_index * fs * time_window +
                                 fs * time_window] = signal_quality

                

        # choose default wavelet function 
        scg.set_default_wavelet('morl')

        signal_length = 2000
        # range of scales to perform the transform
        scales = scg.periods2scales(np.arange(1, signal_length+1))
        # the scaleogram
        ax = scg.cws(ecg_data, scales=scales, figsize=(10, 4.0), coi = False, ylabel="Period", xlabel="Time",
                title="scaleogram from frame {} to {}, quality:{}".format(start + dataset.index[0]/fs, end + dataset.index[0]/fs, signal_quality))
        st.pyplot(ax.figure)


       # print("Default wavelet function used to compute the transform:", scg.get_default_wavelet(), "(",
       #     pywt.ContinuousWavelet(scg.get_default_wavelet()).family_name, ")")



    return classif_ecg_qc_data
