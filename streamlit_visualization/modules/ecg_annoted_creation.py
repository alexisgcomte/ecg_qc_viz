import pandas as pd
import wfdb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

patient = '103001'
target_annot = ['cons']
sampling_frequency = 1000
input_data_folder = 'datasets/0_physionet_ecg'
output_folder = 'datasets/1_ecg_and_annotation_creation'


def ecg_annoted_creation(patient: str = patient,
                         target_annot: str = target_annot,
                         sampling_frequency: str = sampling_frequency,
                         input_data_folder: str = input_data_folder,
                         output_folder: str = output_folder
                         ):

    print('Starting ecg_annoted_creation for patient {}'.format(patient))

    # Loading datasets

    df_ann = pd.read_csv('{}/{}_ANN.csv'.format(input_data_folder, patient),
                         header=None)
    df_ann.columns = ['anno1_start_sample', 'anno1_end_sample', 'anno1_tag',
                      'anno2_start_sample', 'anno2_end_sample', 'anno2_tag',
                      'anno3_start_sample', 'anno3_end_sample', 'anno3_tag',
                      'cons_start_sample', 'cons_end_sample', 'cons_tag']

    ecg_columns = ['anno1_start_sample', 'anno1_end_sample',
                   'anno2_start_sample', 'anno2_end_sample',
                   'anno3_start_sample', 'anno3_end_sample',
                   'cons_start_sample', 'cons_end_sample']

    for column in ecg_columns:
        df_ann[column] = df_ann[column] / sampling_frequency

    ecg_data = wfdb.rdrecord('{}/{}_ECG'.format(input_data_folder, patient))
    df_ecg = pd.DataFrame(ecg_data.p_signal)
    df_ecg.columns = ['ecg_signal']
    df_ecg.index = df_ecg.index / sampling_frequency

    return df_ecg


def ecg_graph_generation(df: pd.DataFrame,
                         start_frame: pd.Timestamp,
                         end_frame: pd.Timestamp,
                         fs: int = 256) -> go.Figure:

    # ecg_qc predictions

    
    df_ecg = df[(df.index >= start_frame) & (df.index < end_frame)]

    fig = go.Figure()
    fig = make_subplots()
    fig.add_trace(go.Scatter(x=df_ecg.index,
                             y=df_ecg['signal'].values,
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
                                  tick0=df_ecg.index[0],
                                  ticklen=1,
                                  tickwidth=1,
                                  dtick=fs*8,
                                  side='top',
                                  showticklabels=False),
                      yaxis=dict(fixedrange=True),
                      yaxis2=dict(fixedrange=True)
                      )

    return fig


if __name__ == '__main__':

    ecg_annoted_creation(patient=patient,
                         target_annot=target_annot,
                         sampling_frequency=sampling_frequency,
                         input_data_folder=input_data_folder,
                         output_folder=output_folder)
