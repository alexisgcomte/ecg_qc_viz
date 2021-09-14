from ecg_qc.ecg_qc import EcgQc
from pyedflib import edfreader
from joblib import numpy_pickle
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('streamlit_visualization/modules')
from import_ecg_segment import EdfLoader
import glob
from tqdm import tqdm


EDF_FILE_FOLDER = '/home/DATA/lateppe/RechercheDetectionCrise_PL/'
MODEL_TIME_INPUT_S = 2
'''
1. List edf files from a folder OK
2. Load them OK, process them, compute quality
3. Add to a dataFrame the result (time in sec, %), with path
'''


def list_edf_files_path(edf_file_folder: str = EDF_FILE_FOLDER) -> list:

    edf_files_path = glob.glob(f'{edf_file_folder}/*/*.edf',
                               recursive=True)
    return edf_files_path


if __name__ == '__main__':

    ecg_qc = EcgQc(model='env/lib/python3.6/site-packages/ecg_qc-1.0b4-py3.6.egg/ecg_qc/ml/models/dtc_2s.pkl')
    edf_files_path = list_edf_files_path()

    df_stats = pd.DataFrame(columns=['file',
                                     'patient',
                                     'proportion',
                                     'segment_counts',
                                     'good_quality',
                                     'bad_quality',
                                     'path'])

    for edf_file_path in tqdm(edf_files_path):
        try:

            # Load
            edf_file = os.path.basename(edf_file_path)
            patient_folder = ('PAT_' +
                              edf_file_path.split(edf_file)[0].split('PAT_')[1])

            # loader = EdfLoader(default_path=EDF_FILE_FOLDER,
            #                    edf_file=edf_file)
            loader = EdfLoader(default_path=EDF_FILE_FOLDER+patient_folder,
                               edf_file=edf_file)

            df_ecg = loader.convert_edf_to_dataframe('ECG1+ECG1-')
            sampling_frequency_hz = loader.sampling_frequency_hz

            # Process
            signal = list(df_ecg['signal'])
            segment_length = MODEL_TIME_INPUT_S * sampling_frequency_hz
            signal_segments = [signal[start_index*segment_length:
                                      (start_index+1)*segment_length] 
                               for start_index in range(
                                   int(len(signal)/segment_length))]

            qualities = [ecg_qc.get_signal_quality(signal_segment)
                         for signal_segment in signal_segments]

            df_stats = df_stats.append({
                'file': edf_file,
                'patient': patient_folder,
                'proportion': np.mean(qualities),
                'segment_counts': len(signal_segments),
                'good_quality': np.count_nonzero(np.array(qualities) == 1),
                'bad_quality': np.count_nonzero(np.array(qualities) == 0),
                'path': edf_file_path},
                ignore_index=True)
            df_stats.to_csv('exports/df_stats_global.csv', index=False)
        except:
            pass