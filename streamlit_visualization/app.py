from google.protobuf.symbol_database import Default
import streamlit as st
import pandas as pd
from modules.graph_generation import ecg_graph_generation
from joblib import load
from pyedflib import highlevel
from pyedflib import edfreader


st.set_page_config(page_title="ECG",
                   page_icon=":chart_with_upwards_trend:",
                   layout='wide',
                   initial_sidebar_state='auto')

edf_path = '/home/DATA/lateppe/RechercheDetectionCrise_PL/'
edf_files = ['PAT_10/EEG_122_s1.edf',
             'PAT_10/EEG_250_s20.edf',
             'PAT_7/EEG_491_s12.edf']

wavelet_generation = False
target_id = '103001_selection'


@st.cache()
def load_ecg():

    df_ecg = pd.read_pickle(
        'dataset_streamlit/df_ecg_{}.pkl'.format(
            target_id))
    return df_ecg


def load_edf(edf_file: str,
             channel: str,
             start: int = 0,
             n: int = 10_000):
    with edfreader.EdfReader(edf_file) as f:
        signals = f.readSignal(channel, start=start, n=n)
    df_ecg = pd.DataFrame(data=signals, columns=[channel])
    return df_ecg


# Loading in cache annotations

st.sidebar.header(body='Parameters')


@st.cache()
def load_annot():
    df_ann = pd.read_csv('dataset_streamlit/{}_ANN.csv'.format(target_id),
                         header=None)

    df_ann.columns = ['anno1_start_sample', 'anno1_end_sample', 'anno1_tag',
                      'anno2_start_sample', 'anno2_end_sample', 'anno2_tag',
                      'anno3_start_sample', 'anno3_end_sample', 'anno3_tag',
                      'cons_start_sample', 'cons_end_sample', 'cons_tag']

    targets = ['anno1_start_sample', 'anno1_end_sample',
               'anno2_start_sample', 'anno2_end_sample',
               'anno3_start_sample', 'anno3_end_sample',
               'cons_start_sample', 'cons_end_sample']

    for target in targets:
        df_ann[target] = df_ann[target] / fs

    df_ann = df_ann[
        (df_ann['cons_start_sample'] >= (df_ecg.index[0] / fs))
        ].fillna(0)

    df_ann.reset_index(drop=True, inplace=True)

    return df_ann

# To improve


source_selection = st.sidebar.selectbox('Chose source seleciton',
                                        options=['Physionet', 'La Teppe'],
                                        index=1)

if source_selection == 'La Teppe':
    edf_patient_selection = st.sidebar.selectbox('Chose patient',
                                                 options=edf_files,
                                                 index=0)

    edf_patient_selection = st.sidebar.text_input(
            'manual input',
            'PAT_10/EEG_122_s1.edf')
    edf_file = edf_path+edf_patient_selection
    headers = highlevel.read_edf_header(edf_file)
    channels = headers['channels']
    limited_channels = ['ECG1+ECG1-',
                        'EMG1+EMG1-',
                        'EMG6+EMG6-',
                        'EMG2G2',
                        'EMG3G2',
                        'EMG2G2-EMG3G2']

    limited_channels = [limited_channels[i] for i in range(
        len(limited_channels)) if limited_channels[i] in channels]

    channels_selection = st.sidebar.selectbox('select channel',
                                              options=limited_channels,
                                              index=0)

    start_selection = st.sidebar.text_input("Start of sample (k):", 0)
    start_selection = int(start_selection) * 1_000

    n_selection = st.sidebar.text_input("Size of sample (k):", 100)
    n_selection = int(n_selection) * 1_000

    if channels_selection == 'EMG2G2-EMG3G2':
        channel_1 = 'EMG2G2'
        channel_2 = 'EMG3G2'

        fs = headers['SignalHeaders'][
            channels.index(channel_1)]['sample_rate']
        assert fs == headers['SignalHeaders'][
            channels.index(channel_2)]['sample_rate']

        df_ecg_1 = load_edf(edf_file=edf_file,
                            channel=channels.index(channel_1),
                            start=start_selection,
                            n=n_selection)
        df_ecg_2 = load_edf(edf_file=edf_file,
                            channel=channels.index(channel_1),
                            start=start_selection,
                            n=n_selection)
        df_ecg = df_ecg_1 - df_ecg_2
    else:
        fs = headers['SignalHeaders'][
            channels.index(channels_selection)]['sample_rate']
        df_ecg = load_edf(edf_file=edf_file,
                          channel=channels.index(channels_selection),
                          start=start_selection,
                          n=n_selection)
    df_ecg.columns = ['ecg_signal']

else:
    fs = 1_000
    df_ecg = load_ecg()
    df_ann = load_annot()


# Subheader

st.sidebar.subheader(body='Frame selection')

frame_window_selection = st.sidebar.slider(
    label='Seconds to display:',
    min_value=0,
    max_value=180,
    step=2,
    value=16)

if source_selection == 'Physionet':

    def start_frame_definition():
        df_start_frame = load('streamlit_visualization/next.pkl')
        start_frame = df_start_frame.iloc[0][0]
        return df_start_frame, start_frame

    df_start_frame, start_frame = start_frame_definition()

    frame_start_selection = st.sidebar.slider(
        label='Start Frame:',
        min_value=int(round(df_ecg.index.values[0]/fs, 0)),
        max_value=int(round(df_ecg.index.values[-1]/fs, 0)),
        step=1,
        value=int(round(start_frame/fs, 0)))


else:
    frame_start_selection = st.sidebar.slider(
        label='Start Frame:',
        min_value=int(round(df_ecg.index.values[0]/fs, 0)),
        max_value=int(round(df_ecg.index.values[-1]/fs, 0)),
        step=1,
        value=0)


start_frame = frame_start_selection * fs

tick_space_selection = st.sidebar.slider(
    label='Tick spacing (seconds):',
    min_value=1,
    max_value=60,
    step=1,
    value=2)

if source_selection == 'Physionet':

    if st.sidebar.button('next'):
        start_frame += frame_window_selection * fs
        df_start_frame.iloc[0] = start_frame

    if st.sidebar.button('previous'):
        start_frame -= frame_window_selection * fs
        df_start_frame.iloc[0] = start_frame

    frame_selection = st.sidebar.selectbox('frame_selection',
                                           options=df_ann['cons_start_sample'],
                                           index=0)

    if st.sidebar.checkbox('Frame Selection'):
        start_frame = int(round(frame_selection * fs, 0))

    df_start_frame.to_pickle('streamlit_visualization/next.pkl')

end_frame = start_frame + frame_window_selection * fs

df_ecg = df_ecg[(df_ecg.index >= start_frame) & (df_ecg.index < end_frame)]

spectrum_max_hz_display = st.sidebar.slider(
    label='Spectrum Max Frequency:',
    min_value=50,
    max_value=2_000,
    step=50,
    value=50)

ecg_graph = st.empty()

ecg_graph.plotly_chart(
    ecg_graph_generation(df=df_ecg,
                         start_frame=start_frame,
                         end_frame=end_frame,
                         tick_space=tick_space_selection,
                         fs=fs,
                         source=source_selection,
                         spectrum_max_hz=spectrum_max_hz_display),
    use_container_width=True)
