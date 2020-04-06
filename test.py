import os
import glob
import subprocess

def extract_praat_features(wav_dir='AVEC2014/audio', base_dir='', config_files=['CPP', 'FF', 'HNR']):

    for config_file in config_files:
        # path to config file, may differ from your location
        config_file = os.path.join('Praat/Scripts', config_file)
        print(config_file)
        for path in glob.glob(f"{wav_dir}/**/*.wav", recursive=True):
            [file_dir, file_name] = os.path.split(path)
            target_dir = os.path.join(base_dir, file_dir)
            target_dir = target_dir.replace(f'{wav_dir}/', '')
            target_file = os.path.join(target_dir, f'{file_name[0: -4]}.csv')

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # run shell opensmile command (only works on linux for now)
            subprocess.run([praat_executable, '--run', config_file, path, target_file])

extract_praat_features()