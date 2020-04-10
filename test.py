import os
import glob
import subprocess

def extract_praat_features(wav_dir='C:\\AVEC2014\\audio', base_dir='C:\\Features\\praat_features', project_directory='C:\\Darwin-Project', config_files=['CPP', 'FF', 'HNR'], tasks=['north', 'free']):

    for config_file in config_files:
        # path to config file, may differ from your location
        config_file = os.path.join(f"{project_directory}\\data\\audio\\Praat\\Scripts", config_file)
        print(config_file)
        if len(tasks) == 2:
            glob_path = f"{wav_dir}\\**\\*.wav"
        else:
            glob_path = f"{wav_dir}\\{tasks[0]}\\*.wav"

        for path in glob.glob(glob_path, recursive=True):
            [file_dir, file_name] = os.path.split(path)
            [_, file_dir] = os.path.split(file_dir)
            target_dir = os.path.join(base_dir, file_dir)
            target_file = os.path.join(target_dir, f"{file_name[0: -4]}.csv")
            print(target_file)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # run shell praat command (windows version)
            subprocess.run([f"{project_directory}\\data\\audio\\Praat\\praat_executable", '--run', config_file, path, target_file])

extract_praat_features()