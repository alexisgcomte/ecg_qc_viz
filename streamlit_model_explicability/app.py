import streamlit as st
import pandas as pd
from modules.ecg_annoted_creation import load_ecg_validation, ecg_graph_generation, comment_parser, generate_decision_path
from joblib import load
from pyedflib import highlevel
from pyedflib import edfreader
from ecg_qc import ecg_qc


st.set_page_config(page_title="ECG",
                   page_icon=":chart_with_upwards_trend:",
                   layout='wide',
                   initial_sidebar_state='auto')

edf_path = '/home/DATA/lateppe/Recherche_ECG/20210211T1032/'
edf_files = ['PAT_2/EEG_3_s1.edf',
             'PAT_2/EEG_5_s1.edf',
             'PAT_3/EEG_23_s1.edf',
             'PAT_3/EEG_25_s1.edf',
             'PAT_3/EEG_6_s1.edf',
             'PAT_3/EEG_8_s1.edf',
             'PAT_4/EEG_11_s1.edf',
             'PAT_4/EEG_12_s1.edf',
             'PAT_4/EEG_13_s1.edf',
             'PAT_4/EEG_14_s1.edf',
             'PAT_4/EEG_15_s1.edf',
             'PAT_4/EEG_16_s1.edf',
             'PAT_4/EEG_26_s1.edf',
             'PAT_4/EEG_32_s1.edf',
             'PAT_4/EEG_9_s1.edf',
             'PAT_5/EEG_101_s1.edf',
             'PAT_5/EEG_104_s1.edf',
             'PAT_5/EEG_106_s1.edf',
             'PAT_5/EEG_110_s1.edf',
             'PAT_5/EEG_17_s1.edf',
             'PAT_5/EEG_20_s1.edf',
             'PAT_5/EEG_34_s1.edf',
             'PAT_5/EEG_97_s1.edf',
             'PAT_5/EEG_102_s1.edf',
             'PAT_5/EEG_105_s1.edf',
             'PAT_5/EEG_107_s1.edf',
             'PAT_5/EEG_111_s1.edf',
             'PAT_5/EEG_19_s1.edf',
             'PAT_5/EEG_33_s1.edf',
             'PAT_5/EEG_91_s1.edf',
             'PAT_6/EEG_70_s1.edf',
             'PAT_6/EEG_72_s1.edf',
             'PAT_6/EEG_73_s1.edf',
             'PAT_6/EEG_74_s1.edf',
             'PAT_6/EEG_77_s1.edf',
             'PAT_6/EEG_79_s1.edf',
             'PAT_6/EEG_79_s2.edf',
             'PAT_6/EEG_79_s3.edf',
             'PAT_6/EEG_79_s4.edf',
             'PAT_6/EEG_85_s1.edf',
             'PAT_6/EEG_85_s2.edf',
             'PAT_6/EEG_85_s3.edf',
             'PAT_6/EEG_85_s4.edf']

wavelet_generation = False
target_id = '103001_selection'


@st.cache()
def load_datas():
    # Loading ECGs

    df_ecg_train = pd.read_pickle(
        '/home/aura-research/ecg_qc_raw_data/df_consolidated_teppe.pkl')
    df_ecg_val = load_ecg_validation()
    df_ecg_val.columns = ['signal']

    # Loading related SQIs
    df_train_sqi = pd.read_csv('training_dataset/df_consolidated_ml.csv',
                               index_col=0)
    df_val_sqi = pd.read_csv('validation_dataset/df_ml_103001_2s.csv')
    # Setting same logic
    df_val_sqi['consensus'] = df_val_sqi['classif']

    # initalising ecg_qc_ with Decsion Tree Classifier
    ecg_qc_dtc = ecg_qc(model='training_dataset/model_dtc_direct.pkl',
                        normalized=False,
                        sampling_frequency=256)

    # Adding predictions to datasets
    for dataframe in [df_train_sqi, df_val_sqi]:
        dataframe['dtc_pred'] = [
            ecg_qc_dtc.predict_quality([x])
            for x in dataframe.iloc[:, 2:8].values]

    return ecg_qc_dtc, df_ecg_train, df_ecg_val, df_train_sqi, df_val_sqi


# Loading in cache annotations

ecg_qc_dtc, df_ecg_train, df_ecg_val, df_train_sqi, df_val_sqi = load_datas()

st.sidebar.header(body='Parameters')
source_selection = st.sidebar.selectbox('Chose source seleciton',
                                        options=['train', 'val'],
                                        index=0)
# Subheader

st.sidebar.subheader(body='Frame selection')
checkbox_only_wrong = st.sidebar.checkbox('only wrong values')

if source_selection == 'train':
    df_ecg = df_ecg_train
    df_sqi = df_train_sqi
    if checkbox_only_wrong:
        df_sqi = df_sqi[df_sqi['consensus'] != df_sqi['dtc_pred']]
    max_index = df_sqi.shape[0]
else:
    df_ecg = df_ecg_val
    df_sqi = df_val_sqi
    if checkbox_only_wrong:
        df_sqi = df_sqi[df_sqi['consensus'] != df_sqi['dtc_pred']]
    max_index = df_sqi.shape[0]

index_selection = st.sidebar.slider(
    label='Index selection:',
    min_value=0,
    max_value=max_index,
    step=1,
    value=0)

start_index = index_selection

if source_selection == 'train':
    fs = 256
    start_frame = df_sqi['timestamp_start'].values[start_index]
    end_frame = df_sqi['timestamp_end'].values[start_index]
else:
    fs = 1_000
    start_frame = df_sqi['timestamp_start'].values[start_index]/1000
    end_frame = df_sqi['timestamp_end'].values[start_index]/1000


st.text_input(comment_parser(df_sqi, start_index))

ecg_graph = st.empty()
ecg_graph.plotly_chart(
    ecg_graph_generation(df_ecg, start_frame, end_frame, fs=fs),
    use_container_width=True)

if st.checkbox('display decision path?'):
    generate_decision_path(ecg_qc_dtc, df_train_sqi, df_sqi.iloc[start_index, 2:8].values)
    st.image('temp_dtviz.svg',
             use_column_width=True)

