import os
import sys
import subprocess
import csv
import glob
import re
import pandas as pd
import pickle
import math
import numpy as np
import progressbar
import xcorr_audio_features as xcorr

from pathlib import Path
from pydub import AudioSegment
from helper import save_to_file, load_from_file

class AudioFeatures:
    def __init__(self, config):
        self.raw_audio_dir = Path(config['raw_audio_folder']) if 'raw_audio_folder' in config else None
        self.seg_audio_dir = Path(config['seg_audio_folder']) if 'seg_audio_folder' in config else None
        self.features_dir = Path(config['audio_folder'])

        self.feature_type = config['audio_features'].lower()
        if not self.feature_type in ['avec', 'xcorr', 'egemaps']:
            raise ValueError("feature_type should be either 'AVEC', 'XCORR' or 'EGEMAPS'")
        self.__setup_opensmile()

    def segment_audio_files(self, seg_len=3):
        sec_in_mili = 1000
        segment_length = int(seg_len * sec_in_mili) # 3 seconds * 1000 = 3000 miliseconds
        pre = ['', '0', '00']

        print('Segmenting audio files...')
        # iterate through all wav files from AVEC2014 dataset
        paths = list(self.raw_audio_dir.glob('**/*.wav'))
        for k in progressbar.progressbar(range(len(paths))):
            audio = AudioSegment.from_wav(paths[k])
            duration = audio.duration_seconds

            target_dir = self.seg_audio_dir / paths[k].relative_to(self.raw_audio_dir).stem
            if not target_dir.exists():
                target_dir.mkdir(parents=True)

            # build segments (3 seconds with 1 second overlap)
            # for entire wav file
            for i in range(math.ceil((int(duration) - 2)/2)):
                segment_name = f'{paths[k].name[0: -4]}_{pre[3 - len(str(i))]}{i}.wav'
                target_file = target_dir / segment_name
                start = 2 * i * sec_in_mili
                end = start + segment_length

                # export segmented wav file
                audio[start:end].export(target_file, format="wav")

    def extract_opensmile_features(self):
        # path to config file, may differ from your location
        if not self.__config_file: return

        print('Extracting features from audio files...')
        # extract AVEC2014 features from each audio segment
        # CAUTION: takes a long time to finish
        paths = list(self.__audio_dir.glob("**/*.wav"))
        for i in progressbar.progressbar(range(len(paths))):
            target_file = self.features_dir / self.feature_type / 'csv'
            target_file /= paths[i].relative_to(self.__audio_dir).with_suffix('.csv')

            if not target_file.exists():
                target_file.mkdir(parents=True)

            # run shell opensmile command (not tested on Windows)
            if sys.platform.startswith('linux'):
                subprocess.run(['SMILExtract', '-C', self.__config_file, '-I', paths[i], '-O', target_file])
            elif sys.platform.startswith('win32'):
                subprocess.run(['SMILExtract.exe', '-C', self.__config_file, '-I', paths[i], '-O', target_file])

    def build_feature_sets(self):
        print('Building audio feature sets...')
        if self.feature_type in ['avec', 'egemaps']:
            feature_sets = self.__build_feature_sets()
        else:
            feature_sets = xcorr.extract_features(self.raw_audio_dir)
        feature_sets = self.__mean_feature_sets(feature_sets)
        self.__save_feature_sets(feature_sets)

    def __build_feature_sets(self):
        # build an organized dict with all data
        data = {}
        feature_names = []
        paths = list((self.features_dir / self.feature_type / 'csv').glob('**/*.csv'))
        for i in progressbar.progressbar(range(len(paths))):
            path = paths[i]
            regex = r"(Development|Testing|Training)/(Freeform|Northwind)"
            [(partition, task)] = re.findall(regex, '/'.join(path.parts))
            file_name = path.parent.name if self.feature_type == 'avec' else path.name

            if not feature_names: feature_names = self.__csv_extract_headers(path)
            if not partition in data: data[partition] = {}
            if not task in data[partition]: data[partition][task] = {}
            if not file_name in data[partition][task]:
                data[partition][task][file_name] = pd.DataFrame([], columns=feature_names)

            df = data[partition][task][file_name]
            df.loc[len(df)] = self.__csv_extract_features(path)
        return data

    def __mean_feature_sets(self, data):
        for p in data:
            for t in data[p]:
                df = pd.DataFrame(columns=list(data[p][t].values())[0].columns)
                for d in data[p][t]:
                    df.loc[d] = data[p][t][d].mean(axis=0)
                data[p][t] = df
        return data

    def __save_feature_sets(self, data):
        if not self.features_dir.exists():
            self.features_dir.mkdir(parents=True)

        # separate final features by partition and save in separate file
        for p in data:
            for t in data[p]:
                file_name = f'{p.lower()}_{t.lower()}.pkl'
                target_dir = self.features_dir / self.feature_type
                save_to_file(target_dir, file_name, data[p][t])

    def get_features(self, partition, extraction=False):
        if extraction is True:
            if self.feature_type in ['avec', 'egemaps']:
                self.segment_audio_files()
                self.extract_opensmile_features()
            self.build_feature_sets()

        return self.__load_features(partition)

    def __load_features(self, partition):
        file_dir = self.features_dir / self.feature_type

        f_free = load_from_file(file_dir / f'{partition}_freeform.pkl')
        f_north = load_from_file(file_dir / f'{partition}_northwind.pkl')

        f_free.index = [re.search(r"\d{3}_\d+", i).group() for i in f_free.index]
        f_north.index = [re.search(r"\d{3}_\d+", i).group() for i in f_north.index]

        if partition == "training":
            # fix broken key
            f_north.loc["205_2"] = f_north.loc["205_1"]
            f_north = f_north.drop("205_1")

        if partition == "development":
            # fix broken key
            f_north.loc["205_1"] = f_north.loc["205_2"]
            f_north = f_north.drop("205_2")

        f_free.index = [f'{i}_Freeform' for i in f_free.index]
        f_north.index = [f'{i}_Northwin' for i in f_north.index]
        return pd.concat([f_free, f_north], sort=False)

    def __csv_extract_headers(self, path):
        headers = []

        for line in open(path):
            header = re.findall(r"^@attribute (.+) ", line)
            if header and header[0] not in ['name', 'class']:
                headers.append(header[0])

        return headers

    def __csv_extract_features(self, path):
        audio_features = None
        with open(path) as csv_file:
            data = csv.reader(csv_file, delimiter=',')

            for row in data:
                if len(row) > 0 and row[0] == "'unknown'":
                    audio_features = row[1:-1]
                    break

            return list(map(lambda x: float(x), audio_features))

    def __setup_opensmile(self):
        if self.feature_type == 'avec':
            self.__config_file = Path('tools/openSMILE/config/avec2013.conf')
            self.__audio_dir = self.seg_audio_dir
        elif self.feature_type == 'egemaps':
            self.__config_file = Path('tools/openSMILE/config/gemaps/eGeMAPSv01a.conf')
            self.__audio_dir = self.raw_audio_dir

    # def get_labels(partition, labels_dir='data/labels/AVEC2014_Labels'):
    #     labels_glob = os.path.join(labels_dir, f"{partition.capitalize()}_DepressionLabels/*.csv")
    #     labels = {}
    #
    #     for path in glob.glob(labels_glob, recursive=True):
    #         file = open(path, "r")
    #         labels[re.search(r"\d{3}_\d", path).group()] = int(file.read())
    #         file.close()
    #
    #     return np.array([labels[k] for k in sorted(labels)])
    #
    # def augment_features(features, labels, size):
    #     f_aug = np.zeros((size, features.shape[1]))
    #     l_aug = np.repeat([labels], int(size / len(labels)) + 1, axis=0).flatten()[:size]
    #     f_mean = np.mean(features, axis=0)
    #
    #     for i in range(size):
    #         for j in range(features.shape[1]):
    #             f_aug[i, j] = features[i % len(labels), j] + f_mean[j] * np.random.rand() * 0.1
    #     return f_aug, l_aug