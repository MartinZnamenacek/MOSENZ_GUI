"""Class handling data access for manual data labeling"""

__author__      = "Veronika Kalouskova"
__copyright__   = "Copyright 2025, FBMI CVUT"

import os
import sys
import math
import pandas as pd
import numpy as np

from scipy.signal import butter, filtfilt


class DataHandler():

    def __init__(self, filename, fs, seg_len):
        labels_path = '../../data/labels/'
        self.FILE_IN = '../../data/' + filename
        
        self.FILE_OUT_LABELS = labels_path + filename.split('.')[0] + '_' + str(seg_len) + '.csv'
        self.FILE_OUT_SIGNALS = labels_path + filename.split('.')[0] + '_signals' + '.csv'

        self.df_in = self.read_file()

        self.filter_data(fs)
        self.label_phases()

        self.load_data(fs, seg_len, labels_path)

    #   Load output .csv file if it exists, if not create a new dataframe
    def load_data(self, fs, seg_len, labels_path):
        # If file exists, open it, if not, create a new one
        if os.path.exists(self.FILE_OUT_LABELS):
            self.df_out = pd.read_csv(self.FILE_OUT_LABELS, sep=';')
        else:
            # Create directory for labels if not present
            if not os.path.exists(labels_path): 
                os.makedirs(labels_path) 

            self.create_df(fs, seg_len)
            self.df_out.to_csv(self.FILE_OUT_LABELS, sep=';', index=False)  

        # Always save new filtered signals and labeled phases, if the file exists, overwrite, if not create one
        self.df_in[['Th_Ref_Filt', 'Th1_Filt', 'Th2_Filt', 'Phase']].to_csv(self.FILE_OUT_SIGNALS, sep=';', index=False)

    #   Create output dataframe
    def create_df(self, fs, seg_len):
        data_size = len(self.df_in['ECG'])

        seg_len_pts = seg_len * fs
        rows = math.floor(data_size / seg_len_pts)

        start = np.arange(0, data_size - seg_len_pts + 1, seg_len_pts)
        end = np.arange(seg_len_pts, data_size + 1, seg_len_pts)

        # Initialize pandas dataframe for output data
        self.df_out = pd.DataFrame(index=range(rows), columns=['Start', 'End', 'Confidence','Reference', 'Signal_1', 'Signal_2'])

        self.df_out['Start'] = start
        self.df_out['End'] = end

        self.df_out['Confidence'] = 1   # Default value is that labeling is confident
        
        self.df_out['Reference'] = 1    # Default value is that reference signal is visible
        self.df_out['Signal_1'] = 0     # Default value is that signal 1 is not visible
        self.df_out['Signal_2'] = 0     # Default value is that signal 2 is not visible

    #   Label thresholding where epsilon is a dead zone parameter
    def dead_zone_threshold(self, derivative, epsilon=0):
        labels = pd.Series(index=derivative.index, dtype='object')

        # Label phases
        labels[derivative > epsilon] = 1   # Inspiration (1) 
        labels[derivative < -epsilon] = 0  # Expiration (0) 
        labels[(derivative >= -epsilon) & (derivative <= epsilon)] = np.nan

        labels.ffill(inplace=True)

        return labels
    
    #   Hysteresis where exhalation and inhalation thresholds are parameters
    def hysteresis_threshold(self, derivative, inhalation_thresh=0, exhalation_thresh=0):
        labels = []

        # Start with either exhalation or inhalation
        phase = 0  

        # Label phases
        for d in derivative:
            if phase == 0 and d > inhalation_thresh:
                phase = 1
            elif phase == 1 and d < exhalation_thresh:
                phase = 0
            labels.append(phase)

        return pd.Series(labels)
    
    #   Label expiratory and inspiratory phases of the reference signal 
    def label_phases(self):
        derivative = self.df_in['Th_Ref_Filt'].diff()
        derivative[0] = derivative[1]

        # Experiment with smoothing the derivative
        derivative = derivative.rolling(window=500, center=True).mean()                    

        # Simple thresholding
        # labels = self.dead_zone_threshold(derivative, 0)

        # Dead zone thresholding
        labels = self.dead_zone_threshold(derivative, 0.04)

        # Hysteresis
        # labels = self.hysteresis_threshold(derivative, 0.2, 0.2)

        # Experiment with smoothing the labels
        # labels = round(labels.rolling(window=250, center=True).mean())

        self.df_in['Phase'] = labels
        self.df_in['Phase'] = self.df_in['Phase'].bfill()

    #   Define Butterworth filter for filtering out high frequency noise
    def butter_lowpass_filter(self, data, cutoff, fs, order=4):
        # Nyquist frequency
        nyq = 0.5 * fs  

        b, a = butter(order, cutoff / nyq, btype='low', analog=False)
        y = filtfilt(b, a, data)
        
        return y
    
    #   Filter data using a simple moving average filter, remove DC component from signal
    def filter_data(self, fs):
        window_size = 30

        # Remove DC component and invert signal
        self.df_in['Th1_Filt'] = -(self.df_in['Th1_Filt'] - self.df_in['Th1_Filt'].mean())
        self.df_in['Th2_Filt'] = -(self.df_in['Th2_Filt'] - self.df_in['Th2_Filt'].mean())

        # Moving average filter
        self.df_in['Th1_Filt'] = self.df_in['Th1_Filt'].rolling(window=window_size).mean()
        self.df_in['Th2_Filt'] = self.df_in['Th2_Filt'].rolling(window=window_size).mean()

        # Low pass Butterworth and MA filter
        self.df_in['Th_Ref_Filt'] = self.butter_lowpass_filter(self.df_in['Th_Ref'] , cutoff=1, fs=fs)
        self.df_in['Th_Ref_Filt'] = self.df_in['Th_Ref_Filt'].rolling(window=window_size).mean()

        # Align the dataframe again, remove rows with NaN values, and reset index after    
        self.df_in = self.df_in.dropna().reset_index(drop=True)

    #   Set value of reference column based on radio button selection
    def set_column_value(self, seg_curr, label, column):
        self.df_out.at[seg_curr, column]  = int(label)
        self.df_out.to_csv(self.FILE_OUT_LABELS, sep=';', index=False)  

    #   Get value of column at current selection  
    def get_column_value(self, seg_curr, column):
        if pd.isna(self.df_out.at[seg_curr, column]):
            self.get_column_value(seg_curr, 0, column)

        return self.df_out.at[seg_curr, column]

    #   Read input .csv file, handle possible exceptions
    def read_file(self):
        try:
            column_names = ['ECG', 'Th_Ref', 'Th1_Raw', 'Th2_Raw', 'Th1_Filt', 'Th2_Filt']

            input_data = pd.read_csv(self.FILE_IN, header=None, names=column_names, sep=';', usecols=[3, 4, 5, 6, 7, 8])

        except FileNotFoundError:
            print('File not found.')
            sys.exit(1)
        except pd.errors.ParserError:
            print('Parse error.')
            sys.exit(1)
        except pd.errors.EmptyDataError:
            print('Empty file.')
            sys.exit(1)
        
        return input_data
