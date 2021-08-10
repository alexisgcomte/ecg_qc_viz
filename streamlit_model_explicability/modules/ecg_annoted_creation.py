import pandas as pd
import wfdb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import f1_score, accuracy_score
from dtreeviz.trees import dtreeviz


def load_ecg_validation(patient: str = '103001',
                        sampling_frequency: int = 1_000,
                        input_data_folder: str = 
                        '/home/aura-research/ecg_qc_raw_data'
                        ) -> pd.DataFrame:

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


def comment_parser(df: pd.DataFrame,
                   index: int):

    output_text = (
        f'Index : {index}'
        f'\n'
        f'Prediction: {df["dtc_pred"].values[index]}'
        f'\n'
        f'True value: {df["consensus"].values[index]}'
        f'\n'
        f'SQIs values: {df.iloc[index, 2:8]}'
        f'\n'
        f'f1_score : {f1_score(df["dtc_pred"].values, df["consensus"].values)}'
        f'\n'
        f'accuracy_score : {accuracy_score(df["dtc_pred"].values, df["consensus"].values)}'
        )
    return output_text


def generate_decision_path(ecg_qc_dtc,
                           df,
                           sqis):

    sqi_names = ['q_sqi_score', 'c_sqi_score', 's_sqi_score',
                 'k_sqi_score', 'p_sqi_score', 'bas_sqi_score']
    viz = dtreeviz(ecg_qc_dtc.model,
                   df.iloc[:, 2:8].values,
                   df['consensus'].values,
                   target_name='quality',
                   feature_names=sqi_names,
                   class_names=['noise', 'clean'],
                   X=sqis,
                   show_just_path=True)
    viz.save('temp_dtviz.svg')


if __name__ == '__main__':

    ecg_annoted_creation(patient=patient,
                         target_annot=target_annot,
                         sampling_frequency=sampling_frequency,
                         input_data_folder=input_data_folder,
                         output_folder=output_folder)
