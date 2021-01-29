import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ecg_qc import ecg_qc
import math
import numpy as np
import scaleogram as scg
import streamlit as st
import matplotlib.pyplot as plt
from scipy import signal


def ecg_graph_generation(df: pd.DataFrame,
                         start_frame: int,
                         end_frame: int,
                         tick_space: int = 9,
                         fs: int = 1_000,
                         wavelet_generation: bool = False,
                         source: str = 'Physionet',
                         time_window_ml=9,
                         time_window_cnn=2,
                         spectrum_max_hz=50) -> go.Figure:

    # ecg_qc predictions

    ecg_data = df['ecg_signal'].values

    classif_ecg_qc_ml_data = ecg_qc_predict(ecg_data=ecg_data,
                                            time_window_ml=time_window_ml,
                                            fs=fs,
                                            wavelet_generation=False)

    generate_spectral_analysis(ecg_data=ecg_data,
                               start=0,
                               end=len(ecg_data),
                               fs=fs,
                               spectrum_max_hz=spectrum_max_hz)
    # Uncomment when cnn can load
    #
    # classif_ecg_qc_cnn_data = ecg_qc_predict_cnn(df,
    #                                              wavelet_generation=
    #                                              wavelet_generation,
    #                                              time_window_cnn=2,
    #                                               wave)

    # can be removed when CNN can run:

    if wavelet_generation is True:
        print(fs)
        generate_detailed_wavelet(ecg_data=ecg_data,
                                  time_window=2,
                                  start=0,
                                  end=0,
                                  spectrum_max_hz=spectrum_max_hz,
                                  fs=fs)

    if source == 'Physionet':

        # annotation converted in binary

        for column in df.columns[1:5]:
            df[column] = df[column].apply(
                lambda x: annot_classification_correspondance(x))

        data = np.transpose(df.iloc[:, 1:5].values)

        labels = list(df.iloc[:, 1:5].columns) + ['ecg_qc pred '] + \
            ['ecg_qc cnn ']

        data = [data[0],
                data[1],
                data[2],
                data[3],
                classif_ecg_qc_ml_data,
                # np.transpose(classif_ecg_qc_cnn_data.values)[0]]
                classif_ecg_qc_ml_data]

    if source == 'La Teppe':
        for column in df.columns[1:5]:
            df[column] = 1
        data = np.transpose(df.iloc[:, 1:5].values)
        labels = ['ecg_qc pred '] + \
            ['ecg_qc cnn ']

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
                   wavelet_generation: bool = False) -> np.ndarray:

    ecg_qc_test = ecg_qc()
    classif_ecg_qc_data = np.zeros(len(ecg_data))

    for start in range(
            math.floor(len(ecg_data)/(fs * time_window_ml)) + 1):

        start = start * fs * time_window_ml
        end = start + fs * time_window_ml

        signal_quality = ecg_qc_test.predict_quality(
            ecg_qc_test.compute_sqi_scores(ecg_data[start:end]))

        classif_ecg_qc_data[start:end] = signal_quality

        if wavelet_generation:
            generate_spectral_analysis(ecg_data=ecg_data,
                                       start=start,
                                       end=end,
                                       fs=fs,
                                       spectrum_max_hz=50,
                                       classif=signal_quality)

    return classif_ecg_qc_data


def annot_classification_correspondance(classif: int) -> int:

    if classif == 2 or classif == 3:
        classif = 0

    return classif


def ecg_qc_predict_cnn(dataset: pd.DataFrame,
                       time_window_cnn: int = 2,
                       fs: int = 1_000,
                       wavelet_generation: bool = False) -> pd.DataFrame:

    ecg_qc_test = ecg_qc(model_type='cnn')

    classif_ecg_qc_data = pd.DataFrame(
        range(0, dataset.shape[0]),
        columns=['classif'])

    for ecg_signal_index in range(
            math.floor(dataset.shape[0]/(fs * time_window_cnn)) + 1):

        start = ecg_signal_index * fs * time_window_cnn
        end = start + fs * time_window_cnn

        ecg_data = dataset['ecg_signal'].iloc[start:end].values

        signal_quality = annot_classification_correspondance(
            ecg_qc_test.get_signal_quality(ecg_data))

        classif_ecg_qc_data.iloc[start:end] = signal_quality

        if wavelet_generation:
            generate_spectral_analysis(ecg_data=ecg_data,
                                       start=start,
                                       end=end,
                                       fs=fs,
                                       spectrum_max_hz=50,
                                       classif=signal_quality)

    return classif_ecg_qc_data


def generate_spectral_analysis(ecg_data: list,
                               start: int,
                               end: int,
                               classif: str = 'NA',
                               spectrum_max_hz: int = 40,
                               fs: int = 1_000):

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 4), sharex=True)
    fig.subplots_adjust(hspace=.01)

    # ax0
    ax0.plot(np.arange(0, len(ecg_data)), ecg_data, linewidth=1)
    ax0.axis('off')

    #  ax2
    f, t, Sxx = signal.spectrogram(ecg_data,
                                   fs,
                                   # window=('tukey', 0.1),
                                   # nperseg=150)
                                   window=('tukey', 0.1),
                                   nperseg=int(round(fs/4, 0)))
    ax1.pcolormesh(t*fs, -f, Sxx, shading='flat')
    ax1.set_ylim(-spectrum_max_hz)
    ax1.set_ylabel('Hz (inv)')

    # ax1
    scg.set_default_wavelet('morl')

    signal_length = spectrum_max_hz
    # range of scales to perform the transform
    scales = scg.periods2scales(np.arange(1, signal_length+1))

    # the scaleogram
    # sampling_ratio = 4
    # sample_ecg = []
    # for element in range(int(round(len(ecg_data)/sampling_ratio, 0))):
    #     sample_ecg.append(np.mean(ecg_data[sampling_ratio * element:
    #                                        sampling_ratio * element +
    #                                        sampling_ratio]))
    scg.cws(ecg_data,
            scales=scales,
            figsize=(10, 2.0),
            coi=False,
            ylabel="Hz",
            xlabel='Frame',
            ax=ax2,
            cbar=None,
            title='')

    fig.suptitle('spectrum frequency from frame {} to {} - classif : {}'.
                 format(start, end, classif),
                 fontsize=10)

    st.pyplot(fig)


def generate_detailed_wavelet(ecg_data: np.ndarray,
                              start: int = 0,
                              end: int = 16_000,
                              spectrum_max_hz: int = 50,
                              fs: int = 1_000,
                              time_window: int = 2):

    for start in range(math.floor(len(ecg_data)/(fs * time_window)) + 1):

        start = start * fs * time_window
        end = start + fs * time_window

        data_to_generate = ecg_data[start:end]
        try:
            generate_spectral_analysis(ecg_data=data_to_generate,
                                       start=start,
                                       end=end,
                                       fs=fs,
                                       spectrum_max_hz=spectrum_max_hz,
                                       classif='NA')
        except Exception:
            pass
