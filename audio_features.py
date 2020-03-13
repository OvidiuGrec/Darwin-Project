import os
import subprocess
import csv
import glob
import re
import pandas as pd
import pickle
import math
import numpy as np

from pydub import AudioSegment

def segment_wav_files(wav_dir='AVEC2014/audio', base_dir='segments', seg_len=3):
    sec_in_mili = 1000
    segment_length = int(seg_len * sec_in_mili) # 3 seconds * 1000 = 3000 miliseconds
    pre = ['', '0', '00']

    # iterate through all wav files from AVEC2014 dataset
    for path in glob.glob(f"{wav_dir}/**/*.wav", recursive=True):
        audio = AudioSegment.from_wav(path)
        duration = audio.duration_seconds

        [file_dir, file_name] = os.path.split(path)
        target_dir = os.path.join(base_dir, file_dir)
        target_dir = os.path.join(target_dir, file_name[0: -4])

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # build segments (3 seconds with 1 second overlap)
        # for entire wav file
        print(duration)
        for i in range(math.ceil((int(duration) - 2)/2)):
            segment_name = f'{file_name[0: -4]}_{pre[3 - len(str(i))]}{i}.wav'
            target_file = os.path.join(target_dir, segment_name)
            start = 2 * i * sec_in_mili
            end = start + segment_length

            # export segmented wav file
            audio[start:end].export(target_file, format="wav")

def extract_opensmile_features(wav_dir='AVEC2014/audio', base_dir='avec_features', config_file='avec2013.conf'):
    # path to config file, may differ from your location
    config_file = os.path.join('openSMILE/config', config_file)

    # extract AVEC2014 features from each audio segment
    # CAUTION: takes a long time to finish
    for path in glob.glob(f"{wav_dir}/**/*.wav", recursive=True):
        [file_dir, file_name] = os.path.split(path)
        target_dir = os.path.join(base_dir, file_dir)
        target_dir = target_dir.replace(f'{wav_dir}/', '')
        target_file = os.path.join(target_dir, f'{file_name[0: -4]}.csv')

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # run shell opensmile command (only works on linux for now)
        # Modify with your machine specific command line SMILExtract
        subprocess.run(['SMILExtract', '-C', config_file, '-I', path, '-O', target_file])

def csv_extract_headers(path):
    headers = []
    
    for line in open(path):
        header = re.findall(r"^@attribute (.+) ", line)
        if header and header[0] not in ['name', 'class']:
            headers.append(header[0])
    
    return headers
        
def csv_extract_features(path):
    audio_features = None
    with open(path) as csv_file:
        data = csv.reader(csv_file, delimiter=',')

        for row in data:
            if len(row) > 0 and row[0] == "'unknown'":
                audio_features = row[1:-1] 
                break

    return list(map(lambda x: float(x), audio_features))

def build_feature_sets(csv_dir='audio_features', output_dir='data/audio/features', feature_type='avec'):
    # extract feature names from a feature sample file
    feature_names = []
    data = {}
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # build an organized dict with all data
    for path in glob.glob(f"{csv_dir}/**/*.csv", recursive=True):
        [(partition, task)] = re.findall(r"(Development|Testing|Training)/(Freeform|Northwind)", path)
        file_name = ''
        if feature_type == 'avec':
            file_name = os.path.split(os.path.split(path)[0])[1]
        else:
            file_name = os.path.split(path)[1]
        
        if not feature_names: feature_names = csv_extract_headers(path)
        if not partition in data: data[partition] = {}
        if not task in data[partition]: data[partition][task] = {}
        if not file_name in data[partition][task]: 
            data[partition][task][file_name] = pd.DataFrame([], columns=feature_names)

        df = data[partition][task][file_name]
        df.loc[len(df)] = csv_extract_features(path)
    
    audio_features = {}
    for p in data:
        audio_features[p] = {}
        for t in data[p]:
            audio_features[p][t] = {}
            for d in data[p][t]:
                audio_features[p][t][d] = data[p][t][d].mean(axis=0)

    # separate final features by partition and save in separate file
    for p in audio_features:
        f = open(os.path.join(output_dir, f"{feature_type}_audio_features_{p.lower()}.pkl"), "wb")
        pickle.dump(audio_features[p], f)
        f.close()

def get_features(partition, features_dir='data/audio/features', feature_type='avec'):
    file_name = f"{feature_type}_audio_features_{partition}.pkl"
    
    f_train = pickle.load(open(os.path.join(features_dir, file_name), "rb"))
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
        f_all[fk] = f_all[fk].append(f_north[k])

    for k in f_all:
        f_all[k] = f_all[k].tolist()
    
    return np.array([f_all[k] for k in sorted(f_all)])

def get_labels(partition, labels_dir='data/labels/AVEC2014_Labels'):
    labels_glob = os.path.join(labels_dir, f"{partition.capitalize()}_DepressionLabels/*.csv")
    labels = {}

    for path in glob.glob(labels_glob, recursive=True):
        file = open(path, "r")
        labels[re.search(r"\d{3}_\d", path).group()] = int(file.read())
        file.close()
    
    return np.array([labels[k] for k in sorted(labels)])

def augment_features(features, labels, size):
    f_aug = np.zeros((size, features.shape[1]))
    l_aug = np.repeat([labels], int(size / len(labels)) + 1, axis=0).flatten()[:size]
    f_mean = np.mean(features, axis=0)
    
    for i in range(size):
        for j in range(features.shape[1]):
            f_aug[i, j] = features[i % len(labels), j] + f_mean[j] * np.random.rand() * 0.1
    return f_aug, l_aug