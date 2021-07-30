import pandas as pd
from ecg_qc import ecg_qc
import math
from modules.graph_generation import annot_classification_correspondance
from tqdm import tqdm


fs = 1000

df_ecg = pd.read_pickle('dataset_streamlit/df_ecg_103001_selection.pkl')
df_ecg.head()

ecg_data = df_ecg['ecg_signal']

# 'env/lib/python3.6/site-packages/ecg_qc-1.0b4-py3.6.egg/ecg_qc/ml/models/model_2s_rfc_normalized_premium.pkl'
ecg_qc_ml = ecg_qc(model='env/lib/python3.6/site-packages/ecg_qc-1.0b4-py3.6.egg/ecg_qc/ml/models/model_2s_rfc_normalized_premium.pkl')
time_window_ml = 2
fs = 1000

df_results = df_ecg
df_results['ml'] = ''


for ecg_signal_index in tqdm(range(
        math.floor(ecg_data.shape[0]/(fs * time_window_ml)) + 1)):
    start = ecg_signal_index*fs*time_window_ml
    end = start + fs*time_window_ml
    ml_prediction = ecg_qc_ml.get_signal_quality(ecg_data[start:end])
    df_results['ml'].iloc[start:end] = ml_prediction


df_results.to_csv('dataset_streamlit/df_model_comparison_103001_selection.csv')
