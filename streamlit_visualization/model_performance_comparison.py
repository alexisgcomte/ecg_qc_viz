import pandas as pd
from ecg_qc import ecg_qc
import math
from modules.graph_generation import annot_classification_correspondance
from tqdm import tqdm


fs = 1000

df_ecg = pd.read_pickle('dataset_streamlit/df_ecg_103001_selection.pkl')
df_ecg.head()

ecg_data = df_ecg['ecg_signal']

ecg_qc_ml = ecg_qc(model_type='xgb')
ecg_qc_cnn = ecg_qc(model_type='cnn')
time_window_ml = 9
time_window_cnn = 2
fs = 1000

df_results = df_ecg
df_results['ml'] = ''
df_results['cnn'] = ''


for ecg_signal_index in tqdm(range(
        math.floor(ecg_data.shape[0]/(fs * time_window_ml)) + 1)):
    start = ecg_signal_index*fs*time_window_ml
    end = start + fs*time_window_ml
    ml_prediction = ecg_qc_ml.get_signal_quality(ecg_data[start:end])
    df_results['ml'].iloc[start:end] = ml_prediction


for ecg_signal_index in tqdm(range(
        math.floor(ecg_data.shape[0]/(fs * time_window_cnn)) + 1)):
    try:
        start = ecg_signal_index*fs*time_window_cnn
        end = start + fs*time_window_cnn
        cnn_prediction = annot_classification_correspondance(ecg_qc_cnn.get_signal_quality(ecg_data[start:end]))
        df_results['cnn'].iloc[start:end] = cnn_prediction
    except:
        start = ecg_signal_index*fs*time_window_cnn
        end = start + fs*time_window_cnn
        df_results['cnn'].iloc[start:end] = 'na'

df_results.to_csv('dataset_streamlit/df_model_comparison_103001_selection.csv')
