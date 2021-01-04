import pandas as pd
import numpy as np
from scipy import signal
from scipy import stats
from math import nan
import biosppy
import biosppy.signals.ecg as bsp_ecg
import biosppy.signals.tools as bsp_tools
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from modules.rr_detection import compute_heart_rate, make_report


class SQI:

    def __init__(self, ecg_signal, sampling_frequency):

        self.ecg_signal =           ecg_signal
        self.sampling_frequency =   sampling_frequency
        self.qSQI_cr =              0
        self.pSQI_cr =              0
        self.sSQI_cr =              0
        self.kSQI_cr =              0
        self.basSQI_cr =            0
        self.casSQI_cr =            0
        self.classif_qSQI =         "Unknown"
        self.classif_pSQI =         "Unknown"
        self.sSQI_cr =              "Unknown"
        self.classif_kSQI =         "Unknown"
        self.classif_basSQI =       "Unknown"
        self.classif_cSQI =         "Unknown"

        return

    def filter_sig(self):

        order = int(1 * self.sampling_frequency)
        self.ecg_signal, _, _ = bsp_tools.filter_signal(signal=self.ecg_signal,
                                                     ftype='FIR',
                                                     band='bandpass',
                                                     order=order,
                                                     frequency=[3, 45],
                                                     sampling_rate=self.sampling_frequency)

    def hamilton(self):
        order = int(0.3 * self.sampling_frequency)
        filtered, _, _ = bsp_tools.filter_signal(signal=self.ecg_signal,
                                                     ftype='FIR',
                                                     band='bandpass',
                                                     order=order,
                                                     frequency=[3, 45],
                                                     sampling_rate=self.sampling_frequency)
        rpeaks, = bsp_ecg.hamilton_segmenter(signal=filtered, sampling_rate=self.sampling_frequency)
        rpeaks, = bsp_ecg.correct_rpeaks(signal=filtered, rpeaks=rpeaks, sampling_rate=self.sampling_frequency, tol=0.05)
        _, qrs_detections = bsp_ecg.extract_heartbeats(signal=filtered, rpeaks=rpeaks, sampling_rate=self.sampling_frequency,
                                                           before=0.2, after=0.4)
        return qrs_detections

    def similarity(self, detec_ref, detec_algo2, tolerance):
        common_detec = 0
        for fr in detec_ref:
            interval = range(fr-tolerance, fr+tolerance+1)
            corresponding_detec_algo2 = list(set(interval).intersection(detec_algo2))
            if len(corresponding_detec_algo2) > 0:
                common_detec += 1
    #    similarity_percentage = round(100*common_detec/len(detec_ref),2)
        return common_detec

    def qSQI(self):
        compute_hr = compute_heart_rate(fs=self.sampling_frequency)

        df_sample = pd.DataFrame(self.ecg_signal, columns=['ECG'])
        df_sample['timestamp'] = df_sample.index / self.sampling_frequency
        compute_hr.compute(df_input=df_sample)
        data = compute_hr.data
        qSQI_val = data['score']['corrcoefs']['hamilton'][2]
        self.qSQI_cr = round(qSQI_val, 2)
        return self.qSQI_cr

    def qSQI_ecg_classif(self):
        qSQI = round(self.qSQI_cr * 100, 2)

        if qSQI > 90:
            classif = 'optimal'
        elif qSQI >= 60 and qSQI <= 90:
            classif = 'suspicious'
        else:
            classif = 'unqualified'
        return classif

    def cSQI(self):
        compute_hr = compute_heart_rate(fs=self.sampling_frequency)

        df_sample = pd.DataFrame(self.ecg_signal, columns=['ECG'])
        df_sample['timestamp'] = df_sample.index / self.sampling_frequency
        compute_hr.compute(df_input=df_sample)
        data = compute_hr.data
        RRI_list = data['hamilton']['rr_intervals']
        self.cSQI_cr = round(np.std(RRI_list, ddof=1) / np.mean(RRI_list), 2)

        return self.cSQI_cr

    def cSQI_ecg_classif(self):
        if self.cSQI_cr < 0.45:
            classif = 'optimal'
        elif self.cSQI_cr >= 0.45 and self.cSQI_cr <= 0.64:
            classif = 'suspicious'
        else:
            classif = 'unqualified'
        return classif

    def sSQI(self):
        num = np.mean((self.ecg_signal - np.mean(self.ecg_signal))**3)
        sSQI = num / (np.std(self.ecg_signal, ddof=1)**3)
        self.sSQI_cr = round(sSQI, 2)

        return self.sSQI_cr

    def kSQI(self):
        num = np.mean((self.ecg_signal - np.mean(self.ecg_signal))**4)
        kSQI = num / (np.std(self.ecg_signal, ddof=1)**4)
        kSQI_fischer = kSQI - 3.0
        self.kSQI_cr = round(kSQI_fischer, 2)

        return self.kSQI_cr

    def sSQI_ecg_classif(self):
        return 'NA'

    def kSQI_ecg_classif(self):
        if self.kSQI_cr > 5:
            classif = 'optimal'
        else:
            classif = 'suspicious'
        return classif

    def pSQI_fft(self):

        N = len(self.ecg_signal)
        T = 1 / self.sampling_frequency

        yf = np.fft.fft(self.ecg_signal)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

        fft_results = []

        for offset in range(len(xf)):
            fft_results.append([xf[offset], np.abs(yf[offset])])

        pds_num = [np.abs(yf[idx]) for idx in range(len(xf)) if xf[idx]>=5 and xf[idx]<=15]
        pds_denom = [np.abs(yf[idx]) for idx in range(len(xf)) if xf[idx]>=5 and xf[idx]<=40]
        self.pSQI_cr = round(sum(pds_num) / sum(pds_denom), 2) 

        return self.pSQI_cr

    def pSQI(self):
        freq, pds = signal.periodogram(self.ecg_signal, self.sampling_frequency, scaling='spectrum')
        pds_num = [pds[idx] for idx in range(len(pds)) if freq[idx]>=5 and freq[idx]<=15]
        pds_denom = [pds[idx] for idx in range(len(pds)) if freq[idx]>=5 and freq[idx]<=40]
        self.pSQI_cr = round(sum(pds_num) / sum(pds_denom), 2) 


        return self.pSQI_cr

    def pSQI_ecg_classif(self, heart_rate=80):
        if heart_rate >= 60 and heart_rate <= 130:
            l1, l2, l3 = 0.5, 0.8, 0.4
        elif heart_rate > 130 and heart_rate <= 160:
            l1, l2, l3 = 0.4, 0.7, 0.3
        else: 
            print('heart rate is not an acceptable value')
        if self.pSQI_cr >= l1 and self.pSQI_cr <= l2:
            classif = 'optimal'
        elif self.pSQI_cr >= l3 and self.pSQI_cr < l1:
            classif = 'suspicious'
        else:
            classif = 'unqualified'
        return classif

    def basSQI_modified(self):
        N = len(self.ecg_signal)
        T = 1 / self.sampling_frequency

        yf = np.fft.fft(self.ecg_signal)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

        fft_results = []

        for offset in range(len(xf)):
            fft_results.append([xf[offset], np.abs(yf[offset])])

        pds_num = [np.abs(yf[idx]) for idx in range(len(xf)) if xf[idx]>=0 and xf[idx]<=1]
        pds_denom = [np.abs(yf[idx]) for idx in range(len(xf)) if xf[idx]>=0 and xf[idx]<=40]


        self.basSQI_cr = round(1 - (sum(pds_num) / sum(pds_denom)), 2)
    
        return self.basSQI_cr

    def basSQI(self):            
        freq, pds = signal.periodogram(x=self.ecg_signal, fs=self.sampling_frequency, scaling='spectrum')
        # plt.semilogy(freq, pds)
        # plt.show()
        pds_num = [pds[idx] for idx in range(len(pds)) if freq[idx]>=0 and freq[idx]<=1]
        pds_denom = [pds[idx] for idx in range(len(pds)) if freq[idx]>=0 and freq[idx]<=40] 

        self.basSQI_cr = round((1-sum(pds_num)) / sum(pds_denom), 2) 
        return self.basSQI_cr

    def basSQI_ecg_classif(self):
        if self.basSQI_cr >= 0.95 and self.basSQI_cr <= 1:
            classif = 'optimal'
        elif self.basSQI_cr >= 0.90 and self.basSQI_cr < 0.95:
            classif = 'suspicious'
        else:
            classif = 'unqualified'
        return classif

    def ecg_classif(self, qSQI_classif, pSQI_classif, kSQI_classif, basSQI_classif):
        criterion_classif = [qSQI_classif, pSQI_classif, kSQI_classif, basSQI_classif]
        nb_optimal = criterion_classif.count('optimal')
        nb_suspicious = criterion_classif.count('suspicious')
        nb_unqualified = criterion_classif.count('unqualified')
        if nb_optimal >= 3 and nb_unqualified == 0:
            ecg_classif = 'excellent'
        elif nb_unqualified >= 3 or (nb_unqualified == 2 and nb_suspicious >= 1) or (nb_unqualified == 1 and nb_suspicious == 3):
            ecg_classif = 'unacceptable'
        else:
            ecg_classif = 'barely acceptable'
        return ecg_classif

    def get_RRI(self, r_peaks):
        RRI = []
        for idx in range(1, len(r_peaks)):
            RRI.append(r_peaks[idx] - r_peaks[idx - 1])
        return RRI


    def start_all(self):

        results_list = []

        # -------- qSQI Classification ----------
        self.qSQI_cr =          self.qSQI()
        self.classif_qSQI =     self.qSQI_ecg_classif()

        results_list.append([self.qSQI_cr, self.classif_qSQI])


        # -------- pSQI Classification ----------
        self.pSQI_cr =          self.pSQI_fft()
        self.classif_pSQI =     self.pSQI_ecg_classif()

        results_list.append([self.pSQI_cr, self.classif_pSQI])

        # -------- sSQI Classification ----------
        self.sSQI_cr =          self.sSQI()
        self.classif_sSQI =     self.sSQI_ecg_classif()

        results_list.append([self.sSQI_cr, self.classif_sSQI])


        # -------- kSQI Classification ----------
        self.kSQI_cr =          self.kSQI()
        self.classif_kSQI =     self.kSQI_ecg_classif()

        results_list.append([self.kSQI_cr, self.classif_kSQI])


        # -------- basSQI Classification ----------
        self.basSQI_cr =        self.basSQI_modified()
        self.classif_basSQI =   self.basSQI_ecg_classif()

        results_list.append([self.basSQI_cr, self.classif_basSQI])

        # -------- cSQI Classification ----------
        self.cSQI_cr =        self.cSQI()
        self.classif_cSQI =   self.cSQI_ecg_classif()

        results_list.append([self.cSQI_cr, self.classif_cSQI])        
        
        return results_list