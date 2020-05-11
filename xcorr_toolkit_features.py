import os
import glob
import subprocess
import csv
import re
import pandas as pd
import pickle
import math
import numpy as np
import progressbar
import csv

import librosa
from Signal_Analysis.features.signal import get_HNR
from scipy.signal import find_peaks, peak_prominences, correlate, lfilter
from numpy import linalg as la
from pathlib import Path

def extract_features(wav_dir, feature_dir, data_sets, config_path, config_file, tasks, exe_location):
    for data_set in data_sets:
        if len(tasks) == 2:
            glob_path = f"{wav_dir}\\{data_set}\\**\\*.wav"
        else:
            glob_path = f"{wav_dir}\\{data_set}\\{tasks[0]}\\*.wav"

        for file_path in glob.glob(glob_path, recursive=True):
            [file_dir, file_name] = os.path.split(file_path)
            [_, file_task] = os.path.split(file_dir)
            target_dir = os.path.join(feature_dir, data_set)
            target_dir = os.path.join(target_dir, file_task)
            target_file = os.path.join(target_dir, f"{file_name[0: -4]}_{config_file}.csv")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # run praat or opensmile exe through shell (windows version)
            if exe_location.find('praat') > 0:
                subprocess.run(f"{exe_location} --run {config_path} {file_path} {target_file}")
            else:
                subprocess.run(f"{exe_location} -C {config_path} -I {file_path} -O {target_file}")

def extract_praat_features(wav_dir='C:\\AVEC2014\\audio\\wav', feature_dir='C:\\Features\\praat_opensmile_features',
                           project_directory='C:\\Darwin-Project', data_sets=['Development', 'Testing', 'Training'],
                           config_files=['cpp', 'ff', 'hnr'], tasks=['Northwind', 'Freeform']):

    praat_exe = f"{project_directory}\\data\\audio\\Praat\\praat_executable"
    for config_file in config_files:
        # path to config file, may differ from your location
        config_path = os.path.join(f"{project_directory}\\data\\audio\\Praat\\Scripts", config_file)
        extract_features(wav_dir, feature_dir, data_sets, config_path, config_file, tasks, praat_exe)

def extract_mfcc_features(wav_dir='C:\\AVEC2014\\audio\\wav', feature_dir='C:\\Features\\praat_opensmile_features',
                          project_directory='C:\\Darwin-Project', data_sets=['Development', 'Testing', 'Training'],
                          tasks=['Northwind', 'Freeform']):

    config_path = f"{project_directory}\\data\\audio\\openSMILE\\config\\mfcc\\mfcc.conf"
    open_smile_exe = f"{project_directory}\\data\\audio\\openSMILE\\msvcbuild\\SMILExtract_Release"
    extract_features(wav_dir, feature_dir, data_sets, config_path, 'mfcc', tasks, open_smile_exe)

def get_delta_mfcc(file, rec_len):
    number_of_mfcc = 16
    delta_mfcc_delay_parameter = 2
    recording_mfccs = np.ndarray((math.ceil(rec_len * 100), number_of_mfcc + 2), dtype=object)

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        for row in csv_reader:

            j = 0
            for collumn in row[0].split(';'):
                recording_mfccs[i, j] = collumn
                j += 1

            i += 1

    recording_mfccs = recording_mfccs[1:math.ceil(rec_len * 100), delta_mfcc_delay_parameter:number_of_mfcc + delta_mfcc_delay_parameter]
    recording_mfccs = recording_mfccs.astype(np.float)
    recording_mfccs = recording_mfccs[~np.isnan(recording_mfccs).any(axis=1)]
    number_of_mfcc_segments = recording_mfccs.shape[0]

    if delta_mfcc_delay_parameter > 0:

        recording_delta_mfccs_raw = np.hstack((
            np.repeat(np.reshape(recording_mfccs[:, 0], (number_of_mfcc_segments, 1)), delta_mfcc_delay_parameter, 1),
            recording_mfccs,
            np.repeat(np.reshape(recording_mfccs[:, -1], (number_of_mfcc_segments, 1)), delta_mfcc_delay_parameter, 1)
        ))

        recording_delta_mfccs = np.zeros((number_of_mfcc_segments, number_of_mfcc), float)
        for i in range(number_of_mfcc_segments):

            for j in range(number_of_mfcc):

                numerator = 0
                denominator = 0
                for k in range(delta_mfcc_delay_parameter):
                    numerator = numerator + (k + 1) * (recording_delta_mfccs_raw[i, j + k + 1 + delta_mfcc_delay_parameter] -
                                                       recording_delta_mfccs_raw[i, j - k - 1 + delta_mfcc_delay_parameter])
                    denominator = denominator + (k + 1) * (k + 1)

                recording_delta_mfccs[i, j] = numerator / (denominator * 2)
    return recording_delta_mfccs


def get_peak_prominences(file_path):
    with open(file_path) as csv_file:
        cpps = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            cpps.append(float(row[0]))
    return np.array(cpps[1:]), cpps[0]


def get_formants(file_path, rec_len):
    number_of_formants = 3
    formants = np.zeros((math.ceil(rec_len * 100), number_of_formants))
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = -1
        for row in csv_reader:
            if len(row) > 0:
                row_string = row[0]
                if row_string.find("frames [") >= 0:
                    i += 1
                    j = -1
                if row_string.find("formant [") >= 0:
                    j += 1
                if row_string.find("frequency = ") >= 0 and j in [1, 2, 3]:
                    formants[i - 1, j - 1] = float(row_string.split("frequency = ")[1])
    formants = np.array([i for i in formants if max(i) > 0])
    return formants


def get_hnr(file_path):
    hnr = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 0:
                if row[0].find("z [1] [") >= 0:
                    hnr.append(float(row[0].split("] = ")[1]))
    return np.array(hnr)


def get_xcorr_features(features, max_eig=None):
    f_xcorr = np.array([])
    f_cov = np.array([])

    for k in [1, 3, 7, 15]:
        f_c = np.array([features[k * i:k * i + 15, :].flatten() for i in range(int(features.shape[0] / k) - 15)])
        f_c += 0.0000000001
        f_eig = np.array(la.eig(np.corrcoef(f_c, rowvar=False))[0])[:max_eig].real
        f_xcorr = np.append(f_xcorr, f_eig)

        cov = np.cov(f_c, rowvar=False)
        det = np.linalg.det(cov)
        cov_power = np.log(np.sum(np.diagonal(cov)))
        cov_entropy = np.log(det) if det > 0 else 0
        f_cov = np.append(f_cov, [cov_power, cov_entropy])

    return np.append(f_xcorr, f_cov)


def save_feature_sets(csv_dir='C:\\Features\\praat_opensmile_features', project_dir='C:\\Darwin-Project',
                       target_dir='data\\audio\\features\\xcorr_toolkit', tasks=['Northwind', 'Freeform']):
    data = {}

    # build an organized dict with all data
    for task in tasks:
        glob_path = f"{csv_dir}\\**\\{task}\\*_cpp.csv"
        for file_path in glob.glob(glob_path, recursive=True):
            file_path = file_path.split('_cpp.csv')[0]
            [(partition, task)] = re.findall(r"(Development|Testing|Training)\\(Freeform|Northwind)", file_path)
            file_name = os.path.split(file_path)[1]

            if not partition in data: data[partition] = {}
            if not task in data[partition]: data[partition][task] = {}
            if not file_name in data[partition][task]:
                data[partition][task][file_name] = {}


            cpp, rec_len = get_peak_prominences(file_path + '_cpp.csv')
            formants = get_formants(file_path + '_ff.csv', rec_len)
            hnr = get_hnr(file_path + '_hnr.csv')
            delta_mfcc = get_delta_mfcc(file_path + '_mfcc.csv', rec_len)

            cpp, hnr, formants, delta_mfcc = equalise_arrays(cpp, hnr, formants, delta_mfcc)

            mfcc_xcorr = get_xcorr_features(delta_mfcc)
            data[partition][task][file_name]['mfcc_xcorr'] = mfcc_xcorr

            form_cpp = np.concatenate((formants, np.array([cpp]).T), axis=1)
            form_cpp_xcorr = get_xcorr_features(form_cpp)
            data[partition][task][file_name]['form_cpp_xcorr'] = form_cpp_xcorr

            cpp_hnr = np.array(list(zip(cpp, hnr)))
            cpp_hnr_xcorr = get_xcorr_features(cpp_hnr, max_eig=20)
            data[partition][task][file_name]['cpp_hnr_xcorr'] = cpp_hnr_xcorr

        # separate final features by partition and save in separate file
        for p in data:
            target_dir = os.path.join(project_dir, target_dir)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            f = open(os.path.join(target_dir, f"{p.lower()}_{task.lower()}.pkl"), "wb")
            pickle.dump(data[p], f)
            f.close()

def build_feature_sets(feature_dir):
    data = {}
    columns = []

    # build an organized dict with all data
    print('Extracting xcorr toolkit features from audio...')
    paths = list(feature_dir.glob('**\\*_cpp.csv'))
    for i in progressbar.progressbar(range(len(paths))):
        path = paths[i]
        path_str = str(path)[0:-8]
        regex = r"(Development|Testing|Training)\\(Freeform|Northwind)"
        [(partition, task)] = re.findall(regex, '\\'.join(path.parts))

        cpp, rec_len = get_peak_prominences(path_str + '_cpp.csv')
        formants = get_formants(path_str + '_ff.csv', rec_len)
        hnr = get_hnr(path_str + '_hnr.csv')
        delta_mfcc = get_delta_mfcc(path_str + '_mfcc.csv', rec_len)

        cpp, hnr, formants, delta_mfcc = equalise_arrays(cpp, hnr, formants, delta_mfcc)

        form_cpp = np.concatenate((formants, np.array([cpp]).T), axis=1)
        cpp_hnr = np.array(list(zip(cpp, hnr)))

        mfcc_xcorr = get_xcorr_features(delta_mfcc)
        form_cpp_xcorr = get_xcorr_features(form_cpp)
        cpp_hnr_xcorr = get_xcorr_features(cpp_hnr, max_eig=20)

        if not columns:
            columns = [f'mfcc_xcorr_{i}' for i in range(mfcc_xcorr.shape[0])]
            columns += [f'form_cpp_xcorr_{i}' for i in range(form_cpp_xcorr.shape[0])]
            columns += [f'cpp_hnr_xcorr_{i}' for i in range(cpp_hnr_xcorr.shape[0])]

        if not partition in data: data[partition] = {}
        if not task in data[partition]: data[partition][task] = {}
        if not path.name in data[partition][task]:
            data[partition][task][path.name] = pd.DataFrame(columns=columns)

        df = data[partition][task][path.name]
        df.loc[len(df)] = np.append(mfcc_xcorr, np.append(form_cpp_xcorr, cpp_hnr_xcorr))
    return data

def equalise_arrays(cpp, hnr, formants, delta_mfcc):

    feature_number = max(cpp.shape[0], hnr.shape[0], formants.shape[0], delta_mfcc.shape[0])
    for i in range(feature_number - cpp.shape[0]):
        cpp = np.append(cpp, 0)
    for i in range(feature_number - hnr.shape[0]):
        hnr = np.append(hnr, 0)
    for i in range(feature_number - formants.shape[0]):
        formants = np.vstack((formants, np.array([0, 0, 0])))
    for i in range(feature_number - delta_mfcc.shape[0]):
        delta_mfcc = np.vstack((delta_mfcc, [0 for i in range(16)]))

    return cpp, hnr, formants, delta_mfcc
