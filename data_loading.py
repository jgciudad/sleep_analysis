from os import listdir
from os.path import join, basename, isfile, realpath, dirname
import numpy as np
import glob
from filters import ButterworthFilter
from pathlib import Path
import mne
import pandas as pd
import scipy
from mne.time_frequency import psd_array_multitaper
from datetime import datetime

config = {}
config['SAMPLING_RATE'] = 400
config['EPOCH_LENGTH'] = 4 # seconds

def listdir_nohidden(path):
    return [f for f in listdir(path) if not f.startswith('.')]

def load_kornum_labels(label_path):

    df = pd.read_csv(label_path, skiprows=9, engine='python', sep='\t', index_col=False) # ParserWarning can be ignored, it is working properly.
    df_stages = df.iloc[:, 4]
    df_time = df.iloc[:, 1]

    # transform integer labels to category name
    def int_class_to_string(row):
        if row == 1:
            return 'WAKE'
        if row == 2:
            return 'NREM'
        if row == 3:
            return 'REM'
        if (row != 1) & (row != 2) & (row != 3):
            return 'ARTIFACT'
        
    def convert_to_datetime(row):
        datetime_object = datetime.strptime(row[5:], '%H:%M:%S')

        return datetime_object

    stages = df_stages.apply(int_class_to_string)
    times = df_time.apply(convert_to_datetime)

    return stages.to_list(), times.to_list(), stages.index.values.tolist()


def read_kornum_recording(label_path: str):
    """reads data from rec_name.edf and rec_name.tsv

    Returns:
        (dict, list, list): features in the form of a dict with entries for each CHANNEL, labels as a list, list of
        start times of samples as indexes in features
    """

    labels, times, _ = load_kornum_labels(label_path)
    features_dict, _ = load_raw_kornum_recording(label_path)

    return features_dict, times, labels


def load_resample_edf(file_path, resample_rate=None):
    """
    :param file_path: path to the .edf recording
    :param resample_rate: new sampling rate in Hertz
    :return: numpy array with the signal
    """

    data = mne.io.read_raw_edf(file_path)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names

    if resample_rate:
        new_num_samples = raw_data.shape[1]/info['sfreq']*resample_rate
        if new_num_samples.is_integer() is False:
            raise Exception("New number of samples is not integer")

        raw_data = scipy.signal.resample(x=raw_data, num=int(new_num_samples), axis=1)

    return raw_data, channels


def load_raw_kornum_recording(label_path):

    signal_path = label_path[:-4] + '.edf'

    raw_data, channels = load_resample_edf(signal_path, resample_rate=None)

    eeg_filter = ButterworthFilter(order=2, fc=[0.3, 35], type="band")
    emg_filter = ButterworthFilter(order=4, fc=[10], type="highpass")

    features = {}

    for c, channel in enumerate(channels):
        
        if channel == 'EEG EEG1A-B':
            c_name = 'EEG1'
            c_filter = eeg_filter
        elif channel == 'EEG EEG2A-B':
            c_name = 'EEG2'
            c_filter = eeg_filter
        elif channel == 'EMG EMG':
            c_name = 'EMG'
            c_filter = emg_filter
            
        s = c_filter(raw_data[c, :], config['SAMPLING_RATE'])

        s = (s - np.mean(s, keepdims=True)) / np.std(s, keepdims=True)

        features[c_name] = np.reshape(s, (int(s.shape[0]/config['SAMPLING_RATE']/config['EPOCH_LENGTH']), -1))

    sample_start_times = np.arange(0, features[c_name].shape[0], config['SAMPLING_RATE']*config['EPOCH_LENGTH'], dtype=int)

    return features, sample_start_times.tolist()

print('STARTED')
start_time = datetime.now()


dataset_path = '/Users/tlj258/toy_dataset_tool2'
# cohort = 'Cohort_2'
# mouse = 'M8'
# day = 'D7'

dataframe_list = []

for cohort in listdir_nohidden(dataset_path):
    for mouse in listdir_nohidden(join(dataset_path, cohort)):
        for day in listdir_nohidden(join(dataset_path, cohort, mouse)):
            label_path = glob.glob(join(dataset_path, cohort, mouse, day, '*.tsv'))[0]

            features, times, labels = read_kornum_recording(label_path)

            # for channel in features.keys():
            channel = 'EEG1'
            for epoch, time, label, idx in zip(features[channel], times, labels, range(len(times))):
                print('Epoch {}/{}'.format(idx+1, len(times)))
                psd, freqs = psd_array_multitaper(epoch,
                                                    config['SAMPLING_RATE'],
                                                    fmin=0.0,
                                                    fmax=35.0,
                                                    n_jobs=-1,
                                                    adaptive=True,
                                                    normalization='full',
                                                    verbose=False)
                
                for power, freq in zip(psd, freqs):
                    epoch_dict = {
                        'mouse_id': mouse,
                        'time': time.time(),
                        'stage': label,
                        'experimental_day': day,
                        'cohort': cohort,
                        'channel': channel,
                        'frequency': freq,
                        'power': power
                        }

                    dataframe_list.append(epoch_dict)

df = pd.DataFrame(dataframe_list)
df.to_csv('power_data.csv', index=False)

end_time = datetime.now()
print('FINISHED')
print('Duration: {}'.format(end_time - start_time))