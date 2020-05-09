import os
import subprocess
import csv
import glob
import re
import pandas as pd
import pickle
import math
import numpy as np
import progressbar

import librosa
from Signal_Analysis.features.signal import get_HNR
from scipy.signal import find_peaks, peak_prominences, correlate, lfilter
from numpy import linalg as la
from pathlib import Path

def get_delta_mfcc(y, sr):
    fr = int(sr / 100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16, hop_length=fr)

    return librosa.feature.delta(mfcc)[:, 1:]

def get_prominences(y, sr):
    fr = int(sr / 100)

    peaks = [find_peaks(y[fr*i:fr*(i+1)], distance=fr)[0] + fr*i for i in range(int(y.shape[0]/fr))]
    peaks = np.array(list(map(lambda x: x[0] if x.shape[0] else 0, np.array(peaks).flatten())))

    return peak_prominences(y, peaks)[0]

def get_formants(y, sr):
    fr = int(sr / 100)
    frqs = np.zeros((int(y.shape[0] / fr), 3))

    for i in range(int(y.shape[0] / fr)):
        try:
            seg = y[i * fr: (i+1) * fr]

            # Get Hamming window.
            N = len(seg)
            w = np.hamming(N)

            # Apply window and high pass filter.
            seg = seg * w
            seg = lfilter([1], [1, 0.63], seg)

            n_order = int(2 + sr / 1000)
            A = librosa.lpc(seg, n_order)

            rts = np.roots(A)
            rts = [r for r in rts if np.imag(r) >= 0]

            # Get angles.
            angz = np.arctan2(np.imag(rts), np.real(rts))

            # Get frequencies.
            frqs[i] = np.array(sorted(angz * (sr / (2 * math.pi)))[:3])
        except FloatingPointError:
            continue

    return frqs

def get_hnr(y, sr):
    fr = int(sr / 100)
    return np.array([get_HNR(y[fr*i:fr*(i+1)], sr) for i in range(int(y.shape[0] / fr))])

def get_xcorr_features(features, max_eig=None):
    f_xcorr = np.array([])
    f_cov = np.array([])

    for k in [1, 3, 7, 15]:
        f_c = np.array([features[k*i:k*i + 15, :].flatten() for i in range(int(features.shape[0]/k) - 15)])
        f_eig = np.array(la.eig(np.corrcoef(f_c, rowvar=False))[0])[:max_eig].real
        f_xcorr = np.append(f_xcorr, f_eig)

        cov = np.cov(f_c, rowvar=False)
        det = np.linalg.det(cov)
        cov_power = np.log(np.sum(np.diagonal(cov)))
        cov_entropy = np.log(det) if det > 0 else 0
        f_cov = np.append(f_cov, [cov_power, cov_entropy])

    return np.append(f_xcorr, f_cov)

def build_feature_sets(wav_dir='AVEC2014', target_dir=''):
    data = {}

    # build an organized dict with all data
    for path in glob.glob(f"{wav_dir}/**/*.wav", recursive=True):
        [(partition, task)] = re.findall(r"(Development|Testing|Training)/(Freeform|Northwind)", path)
        file_name = os.path.split(path)[1]

        if not partition in data: data[partition] = {}
        if not task in data[partition]: data[partition][task] = {}
        if not file_name in data[partition][task]:
            data[partition][task][file_name] = {}

        y, sr = librosa.load(path)

        delta_mfcc = get_delta_mfcc(y, sr).T
        formants = get_formants(y, sr)
        cpp = get_prominences(y, sr)
        hnr = get_hnr(y, sr)

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
        f = open(os.path.join(target_dir, f"xcorr_audio_features_{p.lower()}.pkl"), "wb")
        pickle.dump(data[p], f)
        f.close()

def extract_features(audio_dir):
    data = {}
    columns = []
    # build an organized dict with all data
    print('Extracting xcorr features from audio...')
    paths = list(audio_dir.glob('**/*.wav'))
    for i in progressbar.progressbar(range(len(paths))):
        path = paths[i]
        regex = r"(Development|Testing|Training)/(Freeform|Northwind)"
        [(partition, task)] = re.findall(regex, '/'.join(path.parts))

        y, sr = librosa.load(path)

        delta_mfcc = get_delta_mfcc(y, sr).T
        formants = get_formants(y, sr)
        cpp = get_prominences(y, sr)
        hnr = get_hnr(y, sr)

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

def get_labels(partition, labels_dir='labels/AVEC2014_Labels'):
    labels_glob = os.path.join(labels_dir, f"{partition.capitalize()}_DepressionLabels/*.csv")
    labels = {}

    for path in glob.glob(labels_glob, recursive=True):
        file = open(path, "r")
        labels[re.search(r"\d{3}_\d", path).group()] = int(file.read())
        file.close()

    return np.array([labels[k] for k in sorted(labels)])

def get_features(partition):
    f_train = pickle.load(open(f"xcorr_audio_features_{partition}.pkl", "rb"))
    f_free = f_train["Freeform"]
    f_north = f_train["Northwind"]
    f_all = {re.search(r"\d{3}_\d+", k).group():f_free[k] for k in f_free}

    if partition == "training":
        # fix broken key
        f_all["205_1"] = f_all["205_2"]
        del f_all["205_2"]

    if partition == "development":
        # fix broken key
        f_all["205_2"] = f_all["205_1"]
        del f_all["205_1"]

    for k in f_north:
        fk = re.search(r"\d{3}_\d+", k).group()
        for f in f_north[k]:
            f_all[fk][f] = np.mean((f_all[fk][f], f_north[k][f]), axis=0)

    return np.array([f_all[k] for k in sorted(f_all)])

def parse_data_frame(self, data, task):
    data = data[task]
    df = pd.DataFrame([])

    for k in data:
        df[k] = np.hstack(tuple(data[k].values()))

    return df.T
