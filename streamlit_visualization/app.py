import streamlit as st
import pandas as pd
from modules.graph_generation import (ecg_graph_generation,
                                      heatmap_annot_generation,
                                      heatmap_pred_generation)


target_id = 103001
fs = 1000

# Loading in cache ECG


@st.cache()
def load_ecg():
    if target_id == 103001:
        df_ecg = pd.read_pickle(
            'dataset_streamlit/df_ecg_103001_selection.pkl')
    else:
        df_ecg = pd.read_pickle(
            'dataset_streamlit/df_full_ecg_data_merge_{}.pkl'.format(
                target_id))
    return df_ecg


# Loading in cache annotations
if target_id == 103001:
    st.sidebar.header('103001 loaded')
else:
    st.sidebar.header('{} loaded'.format(target_id))


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


df_ecg = load_ecg()
df_ann = load_annot()

# Subheader

st.sidebar.header(body='Parameters')
st.sidebar.subheader(body='Frame selection')

frame_start_selection = st.sidebar.slider(
    label='Start Frame:',
    min_value=int(round(df_ecg.index.values[0]/fs, 0)),
    max_value=int(round(df_ecg.index.values[-1]/fs, 0)),
    step=1,
    value=int(round(df_ecg.index.values[0]/fs, 0)))

start_frame = frame_start_selection * fs

frame_window_selection = st.sidebar.slider(
    label='Seconds to display:',
    min_value=0,
    max_value=180,
    step=2,
    value=16)

frame_selection = st.sidebar.selectbox('frame_selection',
                                       options=df_ann['cons_start_sample'],
                                       index=0)

if st.sidebar.checkbox('Frame Selection'):
    start_frame = int(round(frame_selection * fs, 0))

st.sidebar.subheader(body='Score time window selection')

score_time_window_selection = st.sidebar.slider(
    label='Score time window (seconds):',
    min_value=1,
    max_value=60,
    step=1,
    value=9)

score_time_window = score_time_window_selection * fs
end_frame = start_frame + frame_window_selection * fs

st.plotly_chart(
    ecg_graph_generation(df_ecg, start_frame, end_frame, fs=fs))

st.plotly_chart(
    heatmap_annot_generation(df_ecg, start_frame, end_frame, fs=fs))

st.plotly_chart(
    heatmap_pred_generation(df_ecg, start_frame, end_frame, fs=fs))
